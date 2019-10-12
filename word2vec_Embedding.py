
import  numpy as np
import  os.path

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import logging
import  gensim
from gensim.models import word2vec
"""
将多个词典合在一起


"""
#数据加载
def load_data(path):
    text = []
    with open(path,'r',encoding='utf-8-sig') as f:
        while True:
            lines = f.readline()
            if not lines:
                break
                pass
            txt = lines.split()
            text.append(txt)
            pass
        # text = np.array(text)# 将数据从list类型转换为array类型。
    # text = [i for item in text for i in item] #将二维列表转化为一维

    return text
#训练word2vec
def train_word2vec(input):
    #设置输出日志
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentence = input
    n_dim=300
    #训练skip-gram模型
    model = word2vec.Word2Vec(sentence,size=n_dim,min_count=1,sg=1)
    model.wv.save_word2vec_format('./data/law_word2Vec_2.model',binary=True)
#测试word2vec
def test_word2vec(path):
    # 加载bin格式的模型
    wordVec = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    s1 = wordVec.similarity('版权','证券')
    print('词相似度：\n'+str(s1))


if __name__ == '__main__':
    data_path_1 = 'E://LewPeng/Code/Python Sample/sifa/data/dict/婚姻类词典.txt'
    data_path_2 = 'E://LewPeng/Code/Python Sample/sifa/data/dict/法律通用词典.txt'
    word2vec_path = './data/law_word2Vec_2.model'
    text_1 = load_data(data_path_1)
    # print(text_1)
    text_2 = load_data(data_path_2)
    text = text_1 + text_2#两种词典合起来的列表
    print(text)
    print('去除重复词语前数目：'+str(len(text))+'\n')
    # text = set(text) #去除重复词语
    # print('去除重复词语后数目：'+str(len(text))+'\n')
    # text = ''.join('%s'%id for id in text) #list包含数字，不能直接转化成字符串。
    # for content in text:
    #     print(''.join(content))
    train_word2vec(text)
    test_word2vec(word2vec_path)


