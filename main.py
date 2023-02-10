# coding=utf-8
import torch
import torch.nn as nn
import os
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from src.config.config import parse_args
from src.log.logger import Logger
from src.model.ptm_hrmf_model import PTMHRMF
from src.data_helper import create_dataloader

# 全局变量
args = parse_args()
for arg in vars(args):
    print(arg + ':', getattr(args, arg))
logger = Logger(args).create_logger()


def evaluate(valid_loader, model, criterion, epoch):
    model.eval()
    predict_list = []
    label_list = []
    correct = 0
    total = 0
    with torch.no_grad():
        running_loss = 0
        for data in valid_loader:
            input_ids, attention_mask, token_type_ids, word_index, word_ids, label = data
            input_ids, attention_mask, token_type_ids, word_index, word_ids, label = \
                input_ids.to(args.device), attention_mask.to(args.device), token_type_ids.to(args.device), \
                word_index.to(args.device), word_ids.to(args.device), label.to(args.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            word_index=word_index, word_ids=word_ids)

            loss = criterion(outputs, label)
            pred = torch.max(outputs, dim=1)[1]
            correct += (pred == label).sum().item()
            total += label.size(0)
            for i in range(len(label)):
                label_list.append(label[i].item())
                predict_list.append(pred[i].item())

            running_loss += loss.item()

        acc = accuracy_score(label_list, predict_list)

        logger.info('loss {}, accuracy {}% [{} / {}]'.format(
            running_loss / len(valid_loader), round(acc * 100, 3), correct, total))

        # model save
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        model_name = '%d' % (epoch + 1) + '_ptm_' + str(round(acc, 4)) + '.bin'
        model_save_path = os.path.join(args.save_dir, model_name)
        torch.save(model.state_dict(), model_save_path)


def main():
    model = PTMHRMF(args).to(args.device)
    criterion = nn.CrossEntropyLoss()

    train_loader, valid_loader, test_loader = create_dataloader(args)

    total_steps = len(train_loader) * args.epochs
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_ratio * total_steps, num_training_steps=total_steps)

    for epoch in range(args.epochs):
        with tqdm(total=len(train_loader), desc='Epoch {}'.format(epoch + 1), ncols=100) as pbar:
            for batch_idx, data in enumerate(train_loader, 0):
                model.train()
                input_ids, attention_mask, token_type_ids, word_index, word_ids, label = data
                input_ids, attention_mask, token_type_ids, word_index, word_ids, label = \
                    input_ids.to(args.device), attention_mask.to(args.device), token_type_ids.to(args.device), \
                    word_index.to(args.device), word_ids.to(args.device), label.to(args.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                word_index=word_index, word_ids=word_ids)
                loss = criterion(outputs, label)
                pbar.set_postfix({'loss': '{0:1.6f}'.format(loss.item() / len(input_ids))})
                pbar.update(1)
                optimizer.zero_grad()  # 梯度清零
                loss.backward()  # 反向传播求解梯度
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
                optimizer.step()  # 梯度下降,更新权重参数
                scheduler.step()  # 跟新learning rate

        evaluate(valid_loader, model, criterion, epoch)
        evaluate(test_loader, model, criterion, epoch)


if __name__ == '__main__':
    main()
