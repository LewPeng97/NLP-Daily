import pickle
import os
import operator
from os.path import exists
from tqdm import tqdm
from prepare_data import load_lines
from configs import *
import csv
import json
import jieba.posseg as pseg
import jieba
jieba.enable_paddle()  # 激活



def load_lcsts_csv(file_path, word_counts, mode='train'):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        count = 0
        for line in reader:
            # if count < 2400:
            lines.append(line)
                # count += 1
    if mode == 'train':
        for data in tqdm(lines):
            src = data[2]
            seg_list_text = pseg.cut(src, use_paddle=True)
            src = " ".join(["{0}".format(w, t) for w, t in seg_list_text])
            count_words(word_counts, src)  # 统计词频
            # src = " ".join(seg_list_text)
            tgt = data[1]

            seg_list_text = pseg.cut(tgt, use_paddle=True)
            tgt = " ".join(["{0}".format(w, t) for w, t in seg_list_text])
            count_words(word_counts, tgt)
                        # tgt = " ".join(seg_list_text)
            tgt = '<s> ' + tgt + ' </s>'
            train.write(json.dumps(tgt + '<summ-content>' + src, ensure_ascii=False) + '\n')
        train.close()
    elif mode == 'test':
        for data in tqdm(lines):
            src = data[3]
            seg_list_text = pseg.cut(src, use_paddle=True)
            src = " ".join(["{0}".format(w, t) for w, t in seg_list_text])
            count_words(word_counts, src)
             # src = " ".join(seg_list_text)
            tgt = data[2]
            seg_list_text = pseg.cut(tgt, use_paddle=True)
            tgt = " ".join(["{0}".format(w, t) for w, t in seg_list_text])
            count_words(word_counts, tgt)
             # tgt = " ".join(seg_list_text)
            tgt = '<s> ' + tgt + ' </s>'
            test.write(json.dumps(tgt + '<summ-content>' + src, ensure_ascii=False) + '\n')
        test.close()
    else:
        for data in tqdm(lines):
            src = data[3]
            seg_list_text = pseg.cut(src, use_paddle=True)
            src = " ".join(["{0}".format(w, t) for w, t in seg_list_text])
            count_words(word_counts, src)
             # src = " ".join(seg_list_text)
            tgt = data[2]
            seg_list_text = pseg.cut(tgt, use_paddle=True)
            tgt = " ".join(["{0}".format(w, t) for w, t in seg_list_text])
            count_words(word_counts, tgt)
             # tgt = " ".join(seg_list_text)
            tgt = '<s> ' + tgt + ' </s>'
            valid.write(json.dumps(tgt + '<summ-content>' + src, ensure_ascii=False) + '\n')
        valid.close()

def write_to_pkl(file_path,mode='train'):
    xy_list = load_lines(file_path, mode+'.txt', configs)
    if mode == 'train':
        print("训练集大小 : ", len(xy_list))
        print("以二进制写入训练集...")
    else:
        print("测试集 or 验证集大小 : ", len(xy_list))
        print("以二进制写入测试集...")
    if not exists(processed_path):
        os.makedirs(processed_path)
    pickle.dump(xy_list, open(processed_path + mode +".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    return xy_list

def count_words(worddict,sent_seg):
    for w in sent_seg.split(' '):
        if w in worddict:
            worddict[w] += 1
        else:
            worddict[w] = 1


def load_dict(d_path, f_name, dic, dic_list):
    f_path = d_path + f_name
    f = open(f_path, "r")
    for line in f:
        line = line.strip('\n').strip('\r')
        if line:
            tf = line.split()
            if len(tf) == 2:
                dic[tf[0]] = int(tf[1])
                dic_list.append(tf[0])
            else:
                print("warning in vocab:", line)
    return dic, dic_list


def to_dict(xys, dic):
    # dict should not consider test set!!!!!
    for xy in xys:
        sents, summs = xy
        y = summs[0]
        for w in y:
            if w in dic:
                dic[w] += 1
            else:
                dic[w] = 1

        x = sents[0]
        for w in x:
            if w in dic:
                dic[w] += 1
            else:
                dic[w] = 1
    return dic

# 由于语料生成词典
def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write("%s\t%d\n" % (word, vocab[word])) #以词、出现次数写入文件中

def write_for_vocab(file_path,train_xy_list):
    print("fitering and building dict...")
    all_dic1 = {}
    all_dic2 = {}
    dic_list = []
    all_dic1, dic_list = load_dict(file_path, "vocab", all_dic1, dic_list)
    all_dic2 = to_dict(train_xy_list, all_dic2)
    for w, tf in all_dic2.items():
        if w not in all_dic1:
            all_dic1[w] = tf

    candiate_list = dic_list[0:configs.PG_DICT_SIZE]  # 50000
    candiate_set = set(candiate_list) #删除重复数据

    dic = {}
    w2i = {}
    i2w = {}
    w2w = {}

    for w in [configs.W_PAD, configs.W_UNK, configs.W_BOS, configs.W_EOS]:
        w2i[w] = len(dic)
        i2w[w2i[w]] = w
        dic[w] = 10000
        w2w[w] = w

    for w, tf in all_dic1.items():
        if w in candiate_set:
            w2i[w] = len(dic)
            i2w[w2i[w]] = w
            dic[w] = tf
            w2w[w] = w
        else:
            w2w[w] = configs.W_UNK
    hfw = []
    sorted_x = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)
    for w in sorted_x:
        hfw.append(w[0])

    # Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
    assert len(hfw) == len(dic)
    assert len(w2i) == len(dic)

    print("dump 5w dict word...")
    save_word_dict(dic, processed_path + "dic_5w.pkl")
    print("dump dict...")
    pickle.dump([all_dic1, dic, hfw, w2i, i2w, w2w], open(processed_path + "dic.pkl", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)
if __name__ == '__main__':
    word_counts = {}  # 构建词表
    print(os.getcwd())
    raw_path = '/home/penglu/LewPeng/TranSummary/lcsts_data/'
    processed_path = raw_path + '/processed_data/'
    if not exists(raw_path):
        os.makedirs(raw_path)
    lcsts_train_path = '/home/penglu/LewPeng/Transformer-for-Summarization/data/processed_data/train_big.csv'
    lcsts_test_path = '/home/penglu/LewPeng/Transformer-for-Summarization/data/processed_data/eval.csv'

    train = open(raw_path + 'train.txt', 'w', encoding='utf-8')
    test = open(raw_path + 'test.txt', 'w', encoding='utf-8')
    valid = open(raw_path + 'valid.txt', 'w', encoding='utf-8')
    configs = DeepmindConfigs()



    print("trainset...")
    load_lcsts_csv(lcsts_train_path, word_counts, mode='train')
    print("dump train...")
    train_xy_list = write_to_pkl(raw_path, mode='train',)

    print("testset...")
    load_lcsts_csv(lcsts_test_path, word_counts, mode='test')
    print("dump test...")
    write_to_pkl(raw_path, mode='test')

    print("validset...")
    load_lcsts_csv(lcsts_test_path, word_counts, mode='valid')
    print("dump valid...")
    write_to_pkl(raw_path, mode='valid')

    print("bulid vocab...")
    print("所有词的总数:", len(word_counts))
    word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    print(word_counts)
    vocab_file = open(processed_path + '/vocab', 'w', encoding='utf-8')
    for word, id in word_counts:
        vocab_file.write(word + ' ' + str(id) + '\n')
    vocab_file.close()
    write_for_vocab(processed_path,train_xy_list)