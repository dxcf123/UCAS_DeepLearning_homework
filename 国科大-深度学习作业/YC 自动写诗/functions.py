import numpy as np
import json
from gensim.models import Word2Vec
import os

class wordprocess:
    def __init__(self):
        self.poem7words = []
        self.poem5words = []
        self.max_len = int
        # 生成X_train和Y_train
        self.X_train = []
        self.Y_train = []
        self.model = None
        self.poemextract()
        self.load_or_train()
        self.dataset_prepare()

    def poemextract(self):
        # 打开并读取JSON文件
        with open(r'C:\Users\IG2017-Laptop-017\source\repos\qzwx0908\DL testworks\UCAS_DeepLearning_homework\国科大-深度学习作业\YC 自动写诗\poet.song.1000.json', 'r', encoding='utf-8') as file:
            poems = json.load(file)
        for p in poems:
            da = ' '.join(''.join(p['paragraphs']))
            if len(da) == 47:
                if da[10] != '，' or da[22] != '。' or da[34] != '，' or da[-1] != '。':
                    continue
                else:
                    self.poem5words.append(da)
            if len(da) == 63:
                if da[14] != '，' or da[30] != '。' or da[46] != '，' or da[-1] != '。':
                    continue
                self.poem7words.append(da)
        
    def get_word_vec(self):
        return self.model.wv
    
    def load_or_train(self):
        modelpath = r'C:\Users\IG2017-Laptop-017\source\repos\qzwx0908\DL testworks\UCAS_DeepLearning_homework\国科大-深度学习作业\YC 自动写诗\mytransformer_w2v'
        if os.path.exists(modelpath):
            self.model = Word2Vec.load(modelpath)
            print('成功加载现有模型')
        else:
            self.model = Word2Vec(self.poem7words + self.poem5words, vector_size=100, window=5, min_count=1, workers=4)
            self.model.save(modelpath)
            print('已训练完模型并保存到self.model和目录')
    
    def get_dataset(self):
        return self.poem7words, self.poem5words

    def dataset_prepare(self):
        def word_tranform(sentence, word_to_index):
            sentence_processed = sentence.split()
            return [word_to_index[word] for word in sentence_processed]

        word_vec = self.get_word_vec()
        word_2_index = word_vec.key_to_index

        for poem in self.poem5words:
            sentence_sequence = word_tranform(poem, word_vec)
            sentence_index = word_tranform(poem, word_2_index)
            for i in range(1, len(sentence_sequence)):
                pad = np.zeros(i-1)
                self.X_train.append(sentence_sequence[:i])  # 输入是句子中从开始到第i-1个词
                self.Y_train.append([np.concatenate((pad, [sentence_index[i]]))]) #输出为第i个词

        #将X_train, Y_train填充到相同长度并转为numpy数组
        self.max_len = max(len(x) for x in self.X_train)
        self.X_train = np.array([np.pad(x, ((0, self.max_len - len(x)), (0, 0)), 'constant', constant_values=0) for x in self.X_train])
        self.Y_train = [np.pad(y, ((0, 0), (0, self.max_len - len(y[0]))), 'constant', constant_values=0) for y in self.Y_train]
        self.Y_train = np.array(self.Y_train).reshape([1564, 23, 1])