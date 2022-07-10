# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from gensim.models import KeyedVectors
import numpy as np


# baseline
class BertForSequenceClassification(nn.Module):
    def __init__(self, args):
        super(BertForSequenceClassification, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(args.hidden_dim, args.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class BertForQuestionAnswering(nn.Module):
    def __init__(self, args):
        super(BertForQuestionAnswering, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.qa_outputs = nn.Linear(args.hidden_dim, args.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        return start_logits, end_logits


# Hidden Representation Mix and Fusion (HRMF)
class BertHrmfForSequenceClassification(nn.Module):
    def __init__(self, args):
        super(BertHrmfForSequenceClassification, self).__init__()
        self.device = args.device
        self.num_heads = args.num_heads
        self.bert = BertModel.from_pretrained(args.bert_dir)
        for param in self.bert.parameters():
            param.requires_grad = True

        word_embedding_model = KeyedVectors.load_word2vec_format(args.word_embedding_weight_path)
        word_embedding_weight = word_embedding_model.vectors
        word_embedding_weight = np.append(word_embedding_weight, np.zeros((1, args.embedding_dim)), axis=0)
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding_weight))
        self.word_embedding.weight.requires_grad = False
        self.dim_transform = nn.Linear(args.embedding_dim, args.hidden_dim)
        self.act = nn.Tanh()
        self.dim_retain = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.mix_lambda = nn.Parameter(torch.tensor(args.mix_lambda))
        self.mu = nn.Parameter(torch.tensor(args.mu))

        self.query = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.key = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.value = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.query_imp = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.key_imp = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.value_imp = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fusion_linear = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.dense = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.activation = nn.Tanh()

        self.dropout = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(args.hidden_dim, args.num_labels)

    def forward(self, input_ids, word_index, word_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]

        word_embedding = self.word_embedding(word_ids)
        word_embedding = self.dim_transform(word_embedding)
        word_embedding = self.act(word_embedding)
        word_embedding = self.dim_retain(word_embedding)

        # 获得字词相似度
        char_word_cosine_similarity_list = \
            self.get_char_word_cosine_similarity(sequence_output, word_index, word_embedding)

        # 字的隐层表示加入word embedding
        new_sequence_output = self.hidden_representation_add_word_embedding(
            sequence_output, word_index, word_embedding, char_word_cosine_similarity_list)

        # 获得混合后的字的隐层表示和_attention_mask
        hidden_representation_mix, _attention_mask = self.get_hidden_representation_mix_and_attention_mask(
            new_sequence_output, word_index, char_word_cosine_similarity_list)

        # 获得融合后的每个token新的隐层表示
        first_hidden_states = self.multi_head_self_attention(
            hidden_representation_mix, self.query, self.key, self.value)
        second_hidden_states = self.multi_head_self_attention(
            hidden_representation_mix, self.query_imp, self.key_imp, self.value_imp, _attention_mask)
        fusion_hidden_states = self.hidden_states_fusion(first_hidden_states, second_hidden_states)

        pooled_output = self.get_pooled_output(fusion_hidden_states)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def get_pooled_output(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

    # 计算字词相似度return [[tensor, tensor, ···, tensor], [tensor, tensor, ···, tensor], ···, [tensor, tensor, ···, tensor]]
    def get_char_word_cosine_similarity(self, x, word_index, word_embedding):
        char_word_cosine_similarity_list = []  # return
        for idx in range(x.size(0)):
            # 相似度存储列表:[tensor, tensor, ···, tensor]
            cw_cosine_similarity_list = []
            for word_idx, (start, end) in enumerate(word_index[idx], 0):
                if start == 0 and end == 0:
                    break
                cw_cs_list = []
                for hr_idx in range(start, end):
                    # 获得词语内字词相似度
                    cosine_similarity = self.get_cosine_similarity(x[idx, hr_idx], word_embedding[idx, word_idx])
                    cw_cs_list.append(cosine_similarity)
                cw_cosine_similarity_list.append(torch.tensor(cw_cs_list))

            char_word_cosine_similarity_list.append(cw_cosine_similarity_list)

        return char_word_cosine_similarity_list

    # 计算混合后的字的隐层表示和_attention mask
    def get_hidden_representation_mix_and_attention_mask(self, x, word_index, char_word_cosine_similarity_list):
        hidden_representation_mix = x.clone()
        _attention_mask = torch.ones((x.size(0), self.num_heads, x.size(1), x.size(1))).to(self.device)
        f_lambda = self.information_retention_scale_function(self.mix_lambda)
        for idx in range(x.size(0)):
            for word_idx, (start, end) in enumerate(word_index[idx], 0):
                if start == 0 and end == 0:
                    break
                if end - start == 1:
                    # 如果只有一个字,那么这个字本身即为词语内最重要的字
                    _attention_mask[idx, :, :, start] = 0
                    continue
                # most_important_char_idx
                mic_idx = torch.argmax(char_word_cosine_similarity_list[idx][word_idx])
                # 其它所有字关注最重要的字
                hidden_representation_mix[idx, start:end, :] = \
                    (1 - ((1 - f_lambda) / (end - start - 1))) * x[idx, start:end, :] + \
                    (1 - f_lambda) / (end - start - 1) * x[idx, start + mic_idx, :]
                # 最重要的字关注其它所有字
                hidden_representation_mix[idx, start + mic_idx, :] = \
                    f_lambda * x[idx, start + mic_idx, :] + \
                    (1 - f_lambda) / (end - start - 1) * torch.sum(x[idx, start:end, :], dim=-2) - \
                    (1 - f_lambda) / (end - start - 1) * x[idx, start + mic_idx, :]
                # 计算_attention mask,只保留最重要的字的注意力
                _attention_mask[idx, :, :, start + mic_idx] = 0
        _attention_mask = _attention_mask.masked_fill(_attention_mask == 1, -float('inf'))
        return hidden_representation_mix, _attention_mask

    # 多头注意力
    def multi_head_self_attention(self, x, q, k, v, attention_mask=None):
        batch_size, seq_len, embed_dim = x.size()
        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, 'embed_dim must be divisible by num_heads'

        q = q(x).view(batch_size, seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k(x).view(batch_size, seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v(x).view(batch_size, seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, v).permute(0, 2, 1, 3)
        # context_layer = context_layer.contiguous().view(batch_size, seq_len, embed_dim)
        context_layer = context_layer.reshape(batch_size, seq_len, embed_dim)
        return context_layer

    # hidden states fusion
    def hidden_states_fusion(self, first_hidden_states, second_hidden_states):
        fusion_hidden_states = self.mu * first_hidden_states + (1 - self.mu) * second_hidden_states
        fusion_hidden_states = self.fusion_linear(fusion_hidden_states)
        fusion_hidden_states = self.activation(fusion_hidden_states)
        return fusion_hidden_states

    @staticmethod
    def get_cosine_similarity(tensor_x, tensor_y):
        normalized_tensor_x = tensor_x / tensor_x.norm(dim=-1, keepdim=True)
        normalized_tensor_y = tensor_y / tensor_y.norm(dim=-1, keepdim=True)
        cosine_similarity = (normalized_tensor_x * normalized_tensor_y).sum(dim=-1)
        return cosine_similarity

    def hidden_representation_add_word_embedding(self, x, word_index, word_embedding, char_word_cosine_similarity_list):
        for idx in range(x.size(0)):
            for word_idx, (start, end) in enumerate(word_index[idx], 0):
                if start == 0 and end == 0:
                    break
                # 字的隐层表示按照相似度归一化权重加入word embedding
                char_word_cosine_similarity = char_word_cosine_similarity_list[idx][word_idx].to(self.device)
                normalized_char_word_cosine_similarity = F.softmax(char_word_cosine_similarity, dim=-1)
                normalized_char_word_cosine_similarity = normalized_char_word_cosine_similarity.view(-1, 1)
                x[idx, start:end, :] += normalized_char_word_cosine_similarity * word_embedding[idx, word_idx]
        return x

    @staticmethod
    def information_retention_scale_function(x):
        return torch.exp(x - 1)


class BertHrmfForQuestionAnswering(nn.Module):
    def __init__(self, args):
        super(BertHrmfForQuestionAnswering, self).__init__()
        self.device = args.device
        self.num_heads = args.num_heads
        self.bert = BertModel.from_pretrained(args.bert_dir)
        for param in self.bert.parameters():
            param.requires_grad = True

        word_embedding_model = KeyedVectors.load_word2vec_format(args.word_embedding_weight_path)
        word_embedding_weight = word_embedding_model.vectors
        word_embedding_weight = np.append(word_embedding_weight, np.zeros((1, args.embedding_dim)), axis=0)
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding_weight))
        self.word_embedding.weight.requires_grad = False
        self.dim_transform = nn.Linear(args.embedding_dim, args.hidden_dim)
        self.act = nn.Tanh()
        self.dim_retain = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.mix_lambda = nn.Parameter(torch.tensor(args.mix_lambda))
        self.mu = nn.Parameter(torch.tensor(args.mu))

        self.query = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.key = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.value = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.query_imp = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.key_imp = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.value_imp = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fusion_linear = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.activation = nn.Tanh()

        self.qa_outputs = nn.Linear(args.hidden_dim, args.num_labels)

    def forward(self, input_ids, word_index, word_ids, attention_mask, token_type_ids,
                start_positions=None, end_positions=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]

        word_embedding = self.word_embedding(word_ids)
        word_embedding = self.dim_transform(word_embedding)
        word_embedding = self.act(word_embedding)
        word_embedding = self.dim_retain(word_embedding)

        # 获得字词相似度
        char_word_cosine_similarity_list = \
            self.get_char_word_cosine_similarity(sequence_output, word_index, word_embedding)

        # 字的隐层表示加入word embedding
        new_sequence_output = self.hidden_representation_add_word_embedding(
            sequence_output, word_index, word_embedding, char_word_cosine_similarity_list)

        # 获得混合后的字的隐层表示和_attention_mask
        hidden_representation_mix, _attention_mask = self.get_hidden_representation_mix_and_attention_mask(
            new_sequence_output, word_index, char_word_cosine_similarity_list)

        # 获得融合后的每个token新的隐层表示
        first_hidden_states = self.multi_head_self_attention(
            hidden_representation_mix, self.query, self.key, self.value)
        second_hidden_states = self.multi_head_self_attention(
            hidden_representation_mix, self.query_imp, self.key_imp, self.value_imp, _attention_mask)
        fusion_hidden_states = self.hidden_states_fusion(first_hidden_states, second_hidden_states)

        logits = self.qa_outputs(fusion_hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        return start_logits, end_logits

    # 计算字词相似度return [[tensor, tensor, ···, tensor], [tensor, tensor, ···, tensor], ···, [tensor, tensor, ···, tensor]]
    def get_char_word_cosine_similarity(self, x, word_index, word_embedding):
        char_word_cosine_similarity_list = []  # return
        for idx in range(x.size(0)):
            # 相似度存储列表:[tensor, tensor, ···, tensor]
            cw_cosine_similarity_list = []
            for word_idx, (start, end) in enumerate(word_index[idx], 0):
                if start == 0 and end == 0:
                    break
                cw_cs_list = []
                for hr_idx in range(start, end):
                    # 获得词语内字词相似度
                    cosine_similarity = self.get_cosine_similarity(x[idx, hr_idx], word_embedding[idx, word_idx])
                    cw_cs_list.append(cosine_similarity)
                cw_cosine_similarity_list.append(torch.tensor(cw_cs_list))

            char_word_cosine_similarity_list.append(cw_cosine_similarity_list)

        return char_word_cosine_similarity_list

    # 计算混合后的字的隐层表示和_attention mask
    def get_hidden_representation_mix_and_attention_mask(self, x, word_index, char_word_cosine_similarity_list):
        hidden_representation_mix = x.clone()
        _attention_mask = torch.ones((x.size(0), self.num_heads, x.size(1), x.size(1))).to(self.device)
        f_lambda = self.information_retention_scale_function(self.mix_lambda)
        for idx in range(x.size(0)):
            for word_idx, (start, end) in enumerate(word_index[idx], 0):
                if start == 0 and end == 0:
                    break
                if end - start == 1:
                    # 如果只有一个字,那么这个字本身即为词语内最重要的字
                    _attention_mask[idx, :, :, start] = 0
                    continue
                # most_important_char_idx
                mic_idx = torch.argmax(char_word_cosine_similarity_list[idx][word_idx])
                # 其它所有字关注最重要的字
                hidden_representation_mix[idx, start:end, :] = \
                    (1 - ((1 - f_lambda) / (end - start - 1))) * x[idx, start:end, :] + \
                    (1 - f_lambda) / (end - start - 1) * x[idx, start + mic_idx, :]
                # 最重要的字关注其它所有字
                hidden_representation_mix[idx, start + mic_idx, :] = \
                    f_lambda * x[idx, start + mic_idx, :] + \
                    (1 - f_lambda) / (end - start - 1) * torch.sum(x[idx, start:end, :], dim=-2) - \
                    (1 - f_lambda) / (end - start - 1) * x[idx, start + mic_idx, :]
                # 计算_attention mask,只保留最重要的字的注意力
                _attention_mask[idx, :, :, start + mic_idx] = 0
        _attention_mask = _attention_mask.masked_fill(_attention_mask == 1, -float('inf'))
        return hidden_representation_mix, _attention_mask

    # 多头注意力
    def multi_head_self_attention(self, x, q, k, v, attention_mask=None):
        batch_size, seq_len, embed_dim = x.size()
        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, 'embed_dim must be divisible by num_heads'

        q = q(x).view(batch_size, seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k(x).view(batch_size, seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v(x).view(batch_size, seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, v).permute(0, 2, 1, 3)
        # context_layer = context_layer.contiguous().view(batch_size, seq_len, embed_dim)
        context_layer = context_layer.reshape(batch_size, seq_len, embed_dim)
        return context_layer

    # hidden states fusion
    def hidden_states_fusion(self, first_hidden_states, second_hidden_states):
        fusion_hidden_states = self.mu * first_hidden_states + (1 - self.mu) * second_hidden_states
        fusion_hidden_states = self.fusion_linear(fusion_hidden_states)
        fusion_hidden_states = self.activation(fusion_hidden_states)
        return fusion_hidden_states

    @staticmethod
    def get_cosine_similarity(tensor_x, tensor_y):
        normalized_tensor_x = tensor_x / tensor_x.norm(dim=-1, keepdim=True)
        normalized_tensor_y = tensor_y / tensor_y.norm(dim=-1, keepdim=True)
        cosine_similarity = (normalized_tensor_x * normalized_tensor_y).sum(dim=-1)
        return cosine_similarity

    def hidden_representation_add_word_embedding(self, x, word_index, word_embedding, char_word_cosine_similarity_list):
        for idx in range(x.size(0)):
            for word_idx, (start, end) in enumerate(word_index[idx], 0):
                if start == 0 and end == 0:
                    break
                # 字的隐层表示按照相似度归一化权重加入word embedding
                char_word_cosine_similarity = char_word_cosine_similarity_list[idx][word_idx].to(self.device)
                normalized_char_word_cosine_similarity = F.softmax(char_word_cosine_similarity, dim=-1)
                normalized_char_word_cosine_similarity = normalized_char_word_cosine_similarity.view(-1, 1)
                x[idx, start:end, :] += normalized_char_word_cosine_similarity * word_embedding[idx, word_idx]
        return x

    @staticmethod
    def information_retention_scale_function(x):
        return torch.exp(x - 1)
