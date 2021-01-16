from jieba import posseg
import jieba
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
from tqdm import tqdm
import csv
import os
import pickle
from collections import defaultdict
import numpy as np
import datetime
import logging
import jieba
jieba.enable_paddle()  # 激活

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# 读取文件
def read_file(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip('\n'))
    return lines

def read_tsv(input_file, quotechar=None):
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        # sum = 2400 #选24k数据训练模型
        # count = 0
        for line in reader:
            # if count < sum:
            lines.append(line)
                # count =count + 1
        return lines

# 保存词向量文件
def dump_pkl(vocab, pkl_path, overwrite=True):
    """
    存储文件
    :param pkl_path:
    :param overwrite:
    :return:
    """
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return
    if pkl_path:
        with open(pkl_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.dump(vocab, f, protocol=0)
        print("save %s ok." % pkl_path)


# 定义分词模式
def segment(sentence, cut_type='word', pos=False):
    seg_words = []
    seg_pos = []

    if cut_type == 'word':
        if pos == True:
            seg_word_pos = posseg.lcut(sentence,use_paddle=True)
            for word, pos in seg_word_pos:
                seg_words.append(word)
                seg_pos.append(pos)
            return seg_words, seg_pos
        elif pos == False:
            seg_words = jieba.lcut(sentence)
            return seg_words

    if cut_type == 'char':
        if pos == True:
            for char in sentence:
                seg_word_pos = posseg.lcut(char)
                for word, pos in seg_word_pos:
                    seg_words.append(word)
                    seg_pos.append(pos)
            return seg_words, seg_pos
        elif pos == False:
            for char in sentence:
                seg_words.append(char)
            return seg_words

# 读取原始数据,输出X Y

"""
其中
train_big.csv代表PART I中所有数据，段为ID、summary、source_text
eval.csv代表PART III中score为3-5分的数据，段为ID、score、summary、source_text

数据处理成src + tgt格式
"""
def parse_data(train_path, test_path):
    train_df = read_tsv(train_path)
    train_words = []
    for train in train_df:
        src = train[2]
        tgt = train[1]
        train_words.append(src + tgt)

    test_df =  read_tsv(test_path)
    test_words = []
    for test in test_df:
        src = test[3]
        tgt = test[2]
        test_words.append(src + tgt)

    return train_words, test_words

# 输入X Y,输出分词后的x y文件
def save_data(train_words, test_words, path_train, path_test, path_stopwords):
    stop_words = read_file(path_stopwords)
    with open(path_train, 'w', encoding='utf-8-sig') as f1:
        count1 = 0
        writer = csv.writer(f1)
        for line in tqdm(train_words):
            if isinstance(line, str):
                line = line.strip().replace(' ', '')
                seg_words = segment(line, cut_type='word', pos=True) # 如果pos=True,seg_words[0]为切分的词,seg_words[1]为切分的词的词性
                seg_words = [word for word in seg_words[0] if word not in stop_words]
                if len(seg_words) > 0:
                    seg_words = ' '.join(seg_words)
                    writer.writerow([seg_words])
                    count1 += 1
        print('len of train is {}'.format(count1))

    with open(path_test, 'w', encoding='utf-8-sig') as f2:
        count2 = 0
        writer = csv.writer(f2)
        for line in tqdm(test_words):
            if isinstance(line, str):
                line = line.strip().replace(' ', '')
                seg_words = segment(line, cut_type='word', pos=True) # 如果pos=True,seg_words[0]为切分的词,seg_words[1]为切分的词的词性
                seg_words = [word for word in seg_words[0] if word not in stop_words]
                if len(seg_words) > 0:
                    seg_words = ' '.join(seg_words)
                    writer.writerow([seg_words])
                else:
                    writer.writerow([seg_words])
                count2 += 1
        print('len of train_y is {}'.format(count2))
    # 输入分词后的x y文件，输出合并后的文件

def save_sentences(path_train, path_test, path_train_union_test):
    # 读两个文件并union
    sentences = read_file(path_train)
    sentences = sentences + read_file(path_test)

    # 将合并语料写出到文件
    #  w:    打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
    with open(path_train_union_test, 'w', encoding='utf-8-sig', newline='') as f:
        for sentence in sentences:
            f.write(sentence)


# 由于语料生成词典
def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            w, i = line
            f.write("%s\t%d\n" % (w, i))


def build_vocab(items, sort=True, min_count=0, lower=False):
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    result = []
    if sort:
        # sort by count
        dic = defaultdict(int)
        for item in items:
            for i in item.split(" "):
                i = i.strip()
                if not i: continue
                i = i if not lower else item.lower()
                dic[i] += 1
        # sort
        dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        for i, item in enumerate(dic):
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
        # sort by items
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)

    vocab = [(w, i) for i, w in enumerate(result)]
    reverse_vocab = [(i, w) for i, w in enumerate(result)]
    return vocab, reverse_vocab


# 训练词向量，并保存词向量
def build_w2v(path_train_union_test, path_words_vectors, w2v_model_path='/home/penglu/LewPeng/TranSummary/lcsts_data/word2vec/embedding/w2v.model', min_count=10):
    w2v = Word2Vec(sentences=LineSentence(path_train_union_test), size=512, window=5, min_count=min_count, iter=5)
    w2v.save(w2v_model_path)

    model = Word2Vec.load(w2v_model_path)
    model.wv.save_word2vec_format('/home/penglu/LewPeng/TranSummary/lcsts_data/word2vec/embedding/w2v.vector')

    logging.info('语料数：{}'.format(model.corpus_count))
    logging.info('词表长度：{}'.format(len(model.wv.vocab)))

    model = KeyedVectors.load(w2v_model_path)


    words_vectors = {}
    for word in model.wv.vocab:
        words_vectors[word] = model[word]

    dump_pkl(words_vectors, path_words_vectors, overwrite=True)

if __name__=='__main__':
    # start = datetime.datetime.now()
    # raw_path = '/home/penglu/LewPeng/TranSummary/lcsts_data/word2vec'
    # logging.info(raw_path)
    # logging.info("reading PART I、III(score 3-5) data !!!")
    # train_words, test_words = parse_data('/home/penglu/LewPeng/GDE/embedding/word_data/train_big.csv',
    #                                      '/home/penglu/LewPeng/GDE/embedding/word_data/eval.csv')
    # logging.info("reading end !!!")
    # logging.info("save cut words result !!!")
    # save_data(train_words, test_words, raw_path + '/data/train.txt',
    #       raw_path + '/data/test.txt',
    #     raw_path + '/stopwords/baidu_stopwords.txt')
    # logging.info("cut over !!!")
    # logging.info("union all data to train word2vec !!!")
    # save_sentences(raw_path + '/data/train.txt',
    #                raw_path + '/data/test.txt',
    #            raw_path + '/data/train_union_test.txt')
    #
    # logging.info("union end !!!")
    # logging.info("create vocab dict !!!")
    # lines = read_file(raw_path + '/data/train_union_test.txt')
    # vocab, reverse_vocab = build_vocab(lines)
    #
    # save_word_dict(vocab, raw_path + '/data/dict_from_lcsts.pkl')
    # logging.info("saved vocab dict !!!")
    # logging.info("begin to train LCSTS_W2V embedding !!!")
    # build_w2v(raw_path + '/data/train_union_test.txt',
    #       raw_path + '/embedding/LCSTS_Words_Vectors.txt')
    # logging.info("end to train LCSTS_W2V embedding !!!")
    # end = datetime.datetime.now()
    # logging.info("Word2Vec training time: {}".format(end - start))

    model = Word2Vec.load('/home/penglu/LewPeng/TranSummary/lcsts_data/word2vec/embedding/w2v.model')
    print(model.most_similar(u'中国',topn=10))
    word_2x = np.load('/home/penglu/LewPeng/TranSummary/lcsts_data/word2vec/embedding/w2v.model.trainables.syn1neg.npy')
    print(model.most_similar(u'教育', topn=10))
