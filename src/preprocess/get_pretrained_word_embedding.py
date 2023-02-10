# coding=utf-8
from gensim.models import KeyedVectors
import ast


if __name__ == '__main__':
    Tencent_AILab_ChineseEmbedding_path = '../../../tencent-ailab-embedding-zh-d200-v0.1.0/tencent-ailab-embedding-zh-d200-v0.1.0.txt'
    word_set_path = '../../data/chnsenticorp/word_set.txt'
    word_embedding_path = '../../data/chnsenticorp/pretrained_word_embedding.txt'

    model = KeyedVectors.load_word2vec_format(Tencent_AILab_ChineseEmbedding_path)
    vocab = model.wv.vocab
    with open(word_set_path, 'r', encoding='utf-8') as f:
        for line in f:
            word_set = ast.literal_eval(line)
    print('Total words: ', len(word_set))
    word_vector_list = []
    for word in word_set:
        if word in model:
            word_vector = word + ' ' + ' '.join(list(map(str, model[word])))
            word_vector_list.append(word_vector)

    print('word vector: ', len(word_vector_list))
    with open(word_embedding_path, 'w', encoding='utf-8') as f:
        f.write(str(len(word_vector_list)) + ' 200' + '\n')
        for word_vector in word_vector_list:
            f.write(word_vector + '\n')
