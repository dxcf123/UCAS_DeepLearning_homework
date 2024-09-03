import numpy as np
import os 
import keras
import json
from gensim.models import Word2Vec

class wordprocess:
    def __init__(self):
        self.poem7words = []
        self.poem5words = []
        self.poemextract()

    def poemextract(self):
        # 打开并读取JSON文件
        with open('/workspaces/UCAS_DeepLearning_homework/国科大-深度学习作业/YC 自动写诗/poet.song.1000.json', 'r') as file:
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
        model = Word2Vec(self.poem7words + self.poem5words, vector_size=100, window=5, min_count=1, workers=4)
        return model.wv
    
    def get_dataset(self):
        return self.poem7words, self.poem5words