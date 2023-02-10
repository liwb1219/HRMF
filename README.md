# Requirement
pytorch 1.9<br>
pip install transformers<br>
pip install gensim==3.8.3<br>
pip install pkuseg<br>
pip install LAC<br>
pip install snownlp<br>
pretrained word embedding need: tencent-ailab-embedding-zh-d200-v0.1.0.txt<br>
GPU: NVIDIA GeForce RTX 3090

# Example
cd src/preprocess<br>
python get_word_index.py  # Get the start and end index of each word after text segmentation.<br>
python get_pretrained_word_embedding.py  # Filter out the word embedding required by this data from the pre-trained word embedding.<br>
cd ../..<br>
python main.py

# Tips
1.src/data_helper.py中的get_input_tuple函数中的CLS和SEP参数需要按照对应模型进行修改，BERT/BERT-wwm是101和102，ERNIE是1和2<br>
2.本示例给出单个句子的处理方法，对于sentences pair和MRC任务，需要在两个句子之间插入[SEP]<br>
3.部分数据集中有隐藏字符，模型没有处理start==end的情况，需自行删除（当遇到start==end的情况，会报错）<br>

# Citation
`@inproceedings{li2022exploiting,<br>
  title={Exploiting Word Semantics to Enrich Character Representations of Chinese Pre-trained Models},<br>
  author={Li, Wenbiao and Sun, Rui and Wu, Yunfang},<br>
  booktitle={Natural Language Processing and Chinese Computing: 11th CCF International Conference, NLPCC 2022, Guilin, China, September 24--25, 2022, Proceedings, Part I},<br>
  pages={3--15},<br>
  year={2022},<br>
  organization={Springer}<br>
}`
