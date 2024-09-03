from functions import wordprocess 
import transformer
import numpy as np
from tensorflow.keras.utils import to_categorical

wp = wordprocess()
word_vec = wp.get_word_vec()
poem7words, poem5words = wp.get_dataset()
vocab_size = len(word_vec.key_to_index)  # 词汇表大小

def word_sequence_to_index(sentence, word_to_index):
    sentence_processed = sentence.split()
    return [word_to_index[word] for word in sentence_processed]

# 生成X_train和Y_train
X_train = []
Y_train = []

for poem in poem5words:
    sentence_indices = word_sequence_to_index(poem, word_vec)
    key_indices = word_sequence_to_index(poem, word_vec.key_to_index)
    for i in range(1, len(sentence_indices)):
        X_train.append(sentence_indices[:i])  # 输入是句子中从开始到第i-1个词
        Y_train.append(sentence_indices[i]) #输出为第i个词

#将X_train填充到相同长度并转为numpy数组
max_len = max(len(x) for x in X_train)
X_train = np.array([np.pad(x, ((0, max_len - len(x)), (0, 0)), 'constant', constant_values=0) for x in X_train])
Y_train = to_categorical(Y_train, num_classes=vocab_size)

input_shape = X_train[0].shape 
num_heads = 4
ff_dim = 256  # Feed Forward层的维度
num_blocks = 2  # Transformer Block的数量

#构建Transformer模型
model = transformer.build_autoregressive_transformer(input_shape, num_heads, ff_dim, num_blocks, vocab_size)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 打印模型结构
model.summary()
model.fit(X_train, Y_train, epochs=5, validation_split=0.1)