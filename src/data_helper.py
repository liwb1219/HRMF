# coding=utf-8
import torch
from torch.utils.data import Dataset, DataLoader
import re
from transformers import BertTokenizer
import ast
from gensim.models import KeyedVectors


def get_seg_text(word_index_list, text_list):
    assert len(word_index_list) == len(text_list), \
        'Index error, len(word_index_list)={}, text_list={}'.format(len(word_index_list), len(text_list))
    seg_text_list = []
    for i in range(len(word_index_list)):
        seg_text = []
        for (s, e) in word_index_list[i]:
            seg_text.append(text_list[i][s:e])
        seg_text_list.append(seg_text)
    return seg_text_list


def get_input_tuple(word_index_list, text_list, seq_len, tokenizer, CLS=101, SEP=102):
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    new_word_index_list = []
    for idx in range(len(word_index_list)):
        input_ids = [CLS]
        new_word_index = [(0, 1)]
        index = 1
        for (s, e) in word_index_list[idx]:
            word_ids = tokenizer.encode(text_list[idx][s:e], add_special_tokens=False)
            new_word_index.append((index, index + len(word_ids)))
            index += len(word_ids)
            input_ids += word_ids
        input_ids += [SEP]
        new_word_index.append((index, index + 1))
        while len(new_word_index) < seq_len + 2:
            new_word_index.append((0, 0))

        attention_mask = [1 for _ in range(len(input_ids))]
        token_type_ids = [0 for _ in range(len(input_ids))]

        padding = [0 for _ in range(seq_len + 2 - len(input_ids))]
        input_ids += padding
        attention_mask += padding
        token_type_ids += padding

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        token_type_ids_list.append(token_type_ids)
        new_word_index_list.append(new_word_index)

    return input_ids_list, attention_mask_list, token_type_ids_list, new_word_index_list


class DatasetReader(Dataset):
    def __init__(self, config, dataset_path, word_index_path):
        label_list = []
        text_list = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for sample in f:
                data_list = sample.split('\t', 1)
                if data_list[0] == 'label':
                    continue
                label_list.append(int(data_list[0]))
                filtered_text = re.sub(r'\s+|\r', '', data_list[1])
                filtered_text = filtered_text[0:config.max_length]
                text_list.append(filtered_text)

        word_index_list = []
        with open(word_index_path, 'r', encoding='utf-8') as f:
            for line in f:
                word_index = ast.literal_eval(line)
                word_index_list.append(word_index)

        # 计算word_ids
        word_ids_list = []
        model = KeyedVectors.load_word2vec_format(config.word_embedding_dir)
        vocab = model.wv.vocab
        word_to_ids = dict()
        for x, word in enumerate(vocab, 0):
            word_to_ids[word] = x
        unk_ids = len(vocab)
        seg_text_list = get_seg_text(word_index_list, text_list)
        for idx, seg_text in enumerate(seg_text_list, 0):
            word_ids = [unk_ids for _ in range(config.max_length + 2)]
            for idy, word in enumerate(seg_text, 1):
                if word in vocab:
                    word_ids[idy] = word_to_ids[word]
            word_ids_list.append(word_ids)

        # 初始化分词器
        tokenizer = BertTokenizer.from_pretrained(config.model_dir)

        # 获得input tuple
        input_ids_list, attention_mask_list, token_type_ids_list, new_word_index_list = \
            get_input_tuple(word_index_list, text_list, config.max_length, tokenizer)

        self.input_ids = torch.tensor(input_ids_list)
        self.attention_mask = torch.tensor(attention_mask_list)
        self.token_type_ids = torch.tensor(token_type_ids_list)
        self.word_index = torch.tensor(new_word_index_list)
        self.word_ids = torch.tensor(word_ids_list)
        self.labels = torch.tensor(label_list)

    def __getitem__(self, item):
        return self.input_ids[item], self.attention_mask[item], self.token_type_ids[item], \
               self.word_index[item], self.word_ids[item], self.labels[item]

    def __len__(self):
        return len(self.input_ids)


def create_dataloader(config):
    train_dataset = DatasetReader(config=config,
                                  dataset_path=config.train_data_dir,
                                  word_index_path=config.train_word_index)
    valid_dataset = DatasetReader(config=config,
                                  dataset_path=config.valid_data_dir,
                                  word_index_path=config.valid_word_index)
    test_dataset = DatasetReader(config=config,
                                 dataset_path=config.test_data_dir,
                                 word_index_path=config.test_word_index)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size)

    return train_loader, valid_loader, test_loader
