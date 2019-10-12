import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import logging
import  gensim
from gensim.models import word2vec
from gensim import models

def main():
    data_path_1 = './data/dict/法律通用词典.txt'
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence(data_path_1)
    model = word2vec.Word2Vec(sentences, size=250,min_count=1)
    # 保存模型，供日后使用
    model.save('./data/word2vec.model')
def load(path):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = models.Word2Vec.load(path)
    res = model.similarity('守法','法院')
    print(res)

if __name__ == "__main__":
    # main()
    load('./data/word2vec.model')