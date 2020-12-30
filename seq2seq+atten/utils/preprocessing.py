import os
import sys
import math
from collections import Counter
import numpy as np

from gensim.models import KeyedVectors
import torch
import nltk
import jieba

def load_data(in_file): #数据加载
    text = []
    summary = []
    num_examples = 0
    with open(in_file,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip().split("||")

            summary.append( list(jieba.cut(line[0])))
            # split chinese sentence into characters
            text.append(list(jieba.cut(line[1])))
    return text,summary


def stoi(sentences, max_words=50000): #构建词典
    word_count = Counter()
    for sentence in sentences:
        for s in sentence:
            word_count[s] += 1
    ls = word_count.most_common(max_words)
    total_words = len(ls) + 3

    word_stoi = {w[0]: index+3 for index, w in enumerate(ls)}
    word_stoi['<EOS>'] = 0
    word_stoi['<BOS>'] = 1
    word_stoi['<UNK>']=2
    return word_stoi, total_words

def embed_matrix(filename,ward_dict): #从已有的词向量文件中读取词向量
    fp=open(filename,'r',encoding='utf-8')
    line=fp.readline().strip()
    total_wards,embed_dim=line.split()
    print("词向量文件中有总共有词{total_wards}，维度为{embed_dim}")
    matrix = np.random.normal(size=(len(ward_dict),300))
    while line:
        ward=line.split()[0]
        if ward in ward_dict:
            vec=line.split()[1:]
            matrix[ward_dict[ward]]=vec
        line=fp.readline().strip()

    fp.close()
    matrix[ward_dict['<BOS>']]=0.5

    matrix[ward_dict['<EOS>']]=0

    matrix=torch.from_numpy(matrix)
    return matrix

def sentoi(text_sentences,summary_sentences,text_stoi,summary_stoi,sort_by_len=True): #将句子用id来替换,并将按照句子长度从短到长排序
    '''
        Encode the sequences.
    '''
    text_sentences_id=[]
    for sent in text_sentences:
        sent_id=[text_stoi['<BOS>']]
        for word in sent:
            try:
                sent_id.append(text_stoi[word])
            except KeyError:
                pass
        sent_id.append(text_stoi['<EOS>'])
        text_sentences_id.append(sent_id)

    summary_sentences_id=[]
    for sent in summary_sentences:
        sent_id=[summary_stoi['<BOS>']]
        for ward in sent:
            try:
                sent_id.append(summary_stoi[ward])
            except KeyError:
                sent_id.append(summary_stoi['<UNK>']) #在词典中没有，那么就添加未知标签
        sent_id.append(summary_stoi['<EOS>'])
        summary_sentences_id.append(sent_id)


    # sort sentences by english lengths
    def len_argsort(seq):
        return sorted(range(len(seq)),key=lambda x: len(seq[x]))

    # 把中文和英文按照同样的顺序排序
    if sort_by_len:
        sorted_index = len_argsort(text_sentences_id)
        text_sentences_id = [text_sentences_id[i] for i in sorted_index]
        summary_sentences_id = [summary_sentences_id[i] for i in sorted_index]
    return text_sentences_id,summary_sentences_id

def get_minibatches(n, minibatch_size, shuffle=True): #从n里面取minibatch_size个索引
    idx_list = np.arange(0, n, minibatch_size) # 步长为minibatch_size
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches

def prepare_data(seqs): #seq=[[语句],[语句],...]
    lengths = [len(seq) for seq in seqs]  #[语句的长度,语句的长度,...]
    n_samples = len(seqs) #样本的个数
    max_len = np.max(lengths)  #取当前批次的数据当中最长的句子长度作为这批数据的长度

    x = np.zeros((n_samples, max_len)).astype('int32') #x.shape=(样本个数,这一批句子的最长的长度)
    x_lengths = np.array(lengths).astype("int32")
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
    return x, x_lengths #x_mask

def gen_examples(en_sentences, cn_sentences, batch_size):
    minibatches = get_minibatches(len(en_sentences), batch_size) #生成索引
    all_ex = []
    for minibatch in minibatches:
        mb_en_sentences = [en_sentences[t] for t in minibatch]
        mb_cn_sentences = [cn_sentences[t] for t in minibatch]

        mb_x, mb_x_len = prepare_data(mb_en_sentences) #mb_x为这一批句子,mb_x_len为mb_x中每个句子的长度
        mb_y, mb_y_len = prepare_data(mb_cn_sentences)
        all_ex.append((mb_x, mb_x_len, mb_y, mb_y_len))
    return all_ex

