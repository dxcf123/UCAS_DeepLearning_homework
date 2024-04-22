import os
import json
import classmodel
import numpy as np
import torch
from gensim.models import Word2Vec
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def process_1(path, path2):  # 将古诗文件中的内容提取出来，并保存到poem列表中
    # path：古诗文件路径  这个路径下包含json文件
    # path2：保存古诗的路径

    poem_5 = []  # 保存5言古诗的列表
    poem_7 = []  # 保存7言古诗的列表
    jsons = os.listdir(path)  # 读取古诗路径
    for js in jsons:  # 读取json文件
        with open(os.path.join(path, js), 'r', encoding='utf-8') as f:  # 读取古诗文件
            data = json.load(f)

        for i in data:  # 将古诗内容保存到poem列表中，这里只处理五言与七言古诗
            da = ' '.join(''.join(i['paragraphs']))  # 读取古诗,将每首故事组成一个字符串
            if len(da) == 47:
                if da[10] != '，' or da[22] != '。' or da[34] != '，' or da[-1] != '。':
                    continue
                else:
                    poem_5.append(da)
            if len(da) == 63:
                if da[14] != '，' or da[30] != '。' or da[46] != '，' or da[-1] != '。':
                    continue
                poem_7.append(da)
    with open(f'{path2}/poem_5.txt', 'w', encoding='utf-8') as f:  # 将古诗保存为txt文件
        for i in poem_5:
            f.write(i + '\n')
    with open(f'{path2}/poem_7.txt', 'w', encoding='utf-8') as f:
        for i in poem_7:
            f.write(i + '\n')


def process_2(path):  # 使用word2vec训练词向量
    # path 古诗路径
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read().split('\n')  # 读取数据并每行切分

    # data 古诗数据 vector_size 词向量维度 min_count 忽略出现次数少于1的单词 workers 线程数
    model = Word2Vec(data, vector_size=100, min_count=1, workers=6, epochs=20)
    model.save(f'{path[:-4]}_W.bin')
    # 词向量   model.syn1neg
    # key_to_index  model.wv.key_to_index
    # index_to_key  model.wv.index_to_key


def gen_poetry(wordsize, index_2_word, type1, wvec, model):  # 生成古诗
    """
    古诗生成
    :param wordsize: 词表大小
    :param index_2_word: 用于将索引转换为词语
    :param type1: 需要生成的字符数
    :param wvec:词嵌入向量
    :param model: 模型
    :return:
    """
    result = ""  # 保存生成的故事
    wordindex = np.random.randint(0, wordsize, 1)[0]  # 随机生成一个索引
    result += index_2_word[wordindex]  # 查询该索引对应的字符用于古诗的第一个字
    h0, c0 = None, None  # LSTM的两个中间矩阵
    model.eval()  # 模型变为评估模式
    for i in range(type1):  # 循环生成字符
        wordemd = torch.tensor(wvec[wordindex]).reshape(1, 1, -1)  # 提取第一个字符的词嵌入向量
        pre, h0, c0 = model(wordemd, h0, c0)  # 传入模型预测
        wordindex = int(torch.argmax(pre))  # 将模型预测结果转换为索引
        pre = index_2_word[wordindex]  # 获取结果
        result += pre
    print(''.join(result.split(' ')))
    return result  # 返回生成的古诗


def train(path1, path2, peo, batchsize):
    """
    模型训练
    :param path1: 原json数据文件
    :param path2:  处理后的文件夹
    :param peo: 5 代表五言古诗，7 代表七言古诗
    :param batchsize: 批量
    :return:
    """
    process_1(path1, path2)
    process_2(os.path.join(path2, f'poem_{peo}.txt'))
    data = Word2Vec.load(os.path.join(path2, f'poem_{peo}_W.bin'))
    wvec = data.syn1neg  # 词嵌入矩阵
    word2index = data.wv.key_to_index  # 获取词语到索引对应关系
    with open(os.path.join(path2, f'poem_{peo}.txt'), 'r', encoding='utf-8') as f:  # 读取古诗文件
        data2 = f.read().split('\n')  # 训练数据
        data2 = data2[:len(data2) - len(data2) % 100]

    dataset = classmodel.myDataset(data2, wvec, word2index)  # 创建古诗数据集
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    out_put, emd_num = wvec.shape

    hidden_num = 600  # 隐藏层神经元个数
    num_layer = 2  # 层数
    lr = 3e-4
    model = classmodel.PoemLstm(emd_num, hidden_num, out_put, num_layer)  # 创建模型
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器
    model.to('cuda')
    model.train()  # 训练模式
    epoch = 24
    loss_all = []
    for j in range(epoch):  # 训练
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            optimizer.zero_grad()
            pre, h0, c0 = model(inputs)
            loss = model.loss(pre, labels.reshape(-1))
            loss.backward()
            optimizer.step()
            if i % 60 == 0:
                loss_all.append(float(loss))
                print(j, i, loss, end='\n')
                gen_poetry(len(word2index), data.wv.index_to_key, (peo + 1) * 8 - 2, wvec, model)
                model.train()
        # if (j + 1) % 10 == 0:
        #     lr = lr * 0.8
        #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器
        # if (j + 1) % 100 == 0:
        #     torch.save(model.state_dict(), f'./data/poem__{peo}.pth')
    with open('loss.txt','w',encoding='utf-8') as f:
        f.write(str(loss_all))
    return model
