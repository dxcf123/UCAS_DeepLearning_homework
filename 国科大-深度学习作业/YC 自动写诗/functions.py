import numpy as np
import json
from gensim.models import Word2Vec
import os

class wordprocess:
    def __init__(self):
        self.poetry = []
        self.max_len = int
        # 生成X_train和Y_train
        self.X_train = []
        self.Y_train = []
        self.word2vec = {}
        self.key2index = {}
        self.index2key = {}
        self.poemextract()
        self.load_or_train()
        self.dataset_prepare()

    def poemextract(self, MAX_LEN = 33, MIN_LEN = 2, DISALLOWED_WORDS = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']', '？', '；']):
        # 打开并读取JSON文件
        with open(r'C:\Users\IG2017-Laptop-017\source\repos\qzwx0908\DL testworks\UCAS_DeepLearning_homework\国科大-深度学习作业\YC 自动写诗\poet.song.1000.json', 'r', encoding='utf-8') as file:
            poems = json.load(file)
        for line in poems:
            # 利用正则表达式拆分 标题 和 内容
            fields = line['paragraphs']
            # 跳过异常数据
            if len(fields) != 2:
                continue
            # 得到诗词内容（后面不需要标题）
            content = fields
            # 过滤数据：跳过内容过长、过短、存在禁用符的诗词
            if len(content) > MAX_LEN - 2 or len(content) < MIN_LEN:
                continue
            if any(word in content for word in DISALLOWED_WORDS):
                continue
            
            content = ''.join(content)
            self.poetry.append(["<bos>"]+['' if c == '\n' else c for c in content]+["<eos>"]) # 最后要记得删除换行符
        
    def load_or_train(self):
        modelpath = r'C:\Users\IG2017-Laptop-017\source\repos\qzwx0908\DL testworks\UCAS_DeepLearning_homework\国科大-深度学习作业\YC 自动写诗\mytransformer_w2v'
        if os.path.exists(modelpath):
            model = Word2Vec.load(modelpath)
            print('成功加载现有模型')
        else:
            model = Word2Vec(self.poetry, vector_size=100, window=5, min_count=3, workers=4)
            model.save(modelpath)
            print('已训练完模型并保存到model和目录')

        np.random.seed(123)
        model.wv.add_vector("<unknown>", [np.random.random() for i in range(100)])
        self.word2vec = model.wv
        self.key2index = model.wv.key_to_index
        self.index2key = model.wv.index_to_key

    def get_word_vec(self):
        return self.word2vec
    
    def dataset_prepare(self):
        for poem in self.poetry:
            sentence_sequence = [self.word2vec[p] if p in self.key2index else self.word2vec["<unknown>"] for p in poem]#word_tranform(poem, self.word2vec)
            sentence_index = [self.key2index[p] if p in self.key2index else self.key2index["<unknown>"]for p in poem]#word_tranform(poem, self.key2index)
            for i in range(1, len(sentence_sequence)):
                self.X_train.append(sentence_sequence[:i])  # 输入是句子中从开始到第i-1个词
                #self.Y_train.append([np.concatenate((pad, [sentence_index[i]]))]) #输出为第i个词
                self.Y_train.append([sentence_index[i]])

        #将X_train, Y_train填充到相同长度并转为numpy数组
        self.max_len = max(len(x) for x in self.X_train)
        self.X_train = np.array([np.pad(x, ((0, self.max_len - len(x)), (0, 0)), 'constant', constant_values=0) for x in self.X_train])
        self.Y_train = [np.pad(y, ((self.max_len - 1, 0)), 'constant', constant_values=0) for y in self.Y_train]
        self.Y_train = np.array(self.Y_train).reshape([len(self.Y_train), self.max_len, 1])
        pass

if __name__ == '__main__':
    wp = wordprocess()