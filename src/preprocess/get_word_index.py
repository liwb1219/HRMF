# coding=utf-8
import pkuseg
import LAC
from snownlp import SnowNLP
import re
from tqdm import tqdm


# global variable
pku_seg = pkuseg.pkuseg()
lac = LAC.LAC(mode='seg')


# 指定分词器获得单篇文章的分词索引,返回分词索引
def get_word_index(text, tokenizer='pku_seg'):
    if tokenizer == 'pku_seg':
        seg_text_list = pku_seg.cut(text)
    elif tokenizer == 'lac':
        seg_text_list = lac.run(text)
    elif tokenizer == 'snow':
        seg_text_list = SnowNLP(text).words
    else:
        seg_text_list = []

    # word index
    word_index = []
    index = 0
    for word in seg_text_list:
        word_index.append((index, index + len(word)))
        index += len(word)

    return word_index


# 将多个分词器对单篇文章的分词索引进行聚合,返回聚合后的分词索引
def get_multi_word_index(multi_word_index, end_index):
    index_dict = {}
    for (s, e) in multi_word_index:
        if (s, e) in index_dict:
            index_dict[(s, e)] += 1
        else:
            index_dict[(s, e)] = 1

    new_index_dict = {}
    for (s, e), cnt in index_dict.items():
        new_index_dict.setdefault(s, []).append((e, cnt))

    word_index = []
    start = 0
    while True:
        new_index_dict[start].sort(key=lambda x: (x[1], x[0]), reverse=True)
        end = new_index_dict[start][0][0]
        word_index.append((start, end))
        start = end
        if start == end_index:
            break
    return word_index


# 保存分词索引为txt文件
def save_word_index(file_path, data_list):
    with open(file_path, 'w', encoding='utf-8') as f:
        for data in data_list:
            f.write(str(data))
            f.write('\n')


# 获得分词后的文本
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


def preprocess_sentences_data(dataset_path, seq_len):
    text_list = []
    word_index_list = []
    row_num = sum([1 for _ in open(dataset_path, 'r', encoding='utf-8')])
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for sample in tqdm(f, total=row_num, ncols=100):
            data_list = sample.split('\t', 1)
            if data_list[0] == 'label':
                continue
            filtered_text = re.sub(r'\s+|\r', '', data_list[1])
            filtered_text = filtered_text[0:seq_len]
            text_list.append(filtered_text)

            pkuseg_word_index = get_word_index(filtered_text, 'pku_seg')
            lac_word_index = get_word_index(filtered_text, 'lac')
            snow_word_index = get_word_index(filtered_text, 'snow')

            multi_word_index = pkuseg_word_index + lac_word_index + snow_word_index

            assert pkuseg_word_index[-1][1] == lac_word_index[-1][1], 'Index calculation error'
            assert pkuseg_word_index[-1][1] == snow_word_index[-1][1], 'Index calculation error'

            end_index = pkuseg_word_index[-1][1]
            word_index = get_multi_word_index(multi_word_index, end_index)

            word_index_list.append(word_index)

    return text_list, word_index_list


if __name__ == '__main__':
    train_dataset_path = '../../data/chnsenticorp/train.tsv'
    valid_dataset_path = '../../data/chnsenticorp/valid.tsv'
    test_dataset_path = '../../data/chnsenticorp/test.tsv'

    train_word_index_path = '../../data/chnsenticorp/train_word_index.txt'
    valid_word_index_path = '../../data/chnsenticorp/valid_word_index.txt'
    test_word_index_path = '../../data/chnsenticorp/test_word_index.txt'

    word_set_path = '../../data/chnsenticorp/word_set.txt'

    # 获得文本列表,聚合后的分词索引列表并保存
    # train
    train_text_list, train_word_index_list = preprocess_sentences_data(dataset_path=train_dataset_path, seq_len=256)
    save_word_index(train_word_index_path, train_word_index_list)
    # valid
    valid_text_list, valid_word_index_list = preprocess_sentences_data(dataset_path=valid_dataset_path, seq_len=256)
    save_word_index(valid_word_index_path, valid_word_index_list)
    # test
    test_text_list, test_word_index_list = preprocess_sentences_data(dataset_path=test_dataset_path, seq_len=256)
    save_word_index(test_word_index_path, test_word_index_list)

    # 获得词表
    word_index_list = train_word_index_list + valid_word_index_list + test_word_index_list
    text_list = train_text_list + valid_text_list + test_text_list
    seg_text_list = get_seg_text(word_index_list, text_list)
    word_set = set()
    for seg_text in seg_text_list:
        for word in seg_text:
            word_set.add(word)
    print(word_set)
    with open(word_set_path, 'w', encoding='utf-8') as f:
        f.write(str(word_set))
    print(len(word_set))
