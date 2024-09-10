import tensorflow as tf
from keras import layers
import numpy as np
from functions import wordprocess

class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, model_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.model_dim = model_dim
        self.pos_encoding = self.positional_encoding(max_len, model_dim)

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "max_len": self.max_len,
            "model_dim": self.model_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_angles(self, pos, i, model_dim):
        angle_rates = 1.0 / tf.pow(10000.0, (2 * tf.cast(i // 2, tf.float32)) / tf.cast(model_dim, tf.float32))
        return tf.cast(pos, tf.float32) * angle_rates

    def positional_encoding(self, max_len, model_dim):
        angle_rads = self.get_angles(
            pos=tf.range(max_len)[:, tf.newaxis],
            i=tf.range(model_dim)[tf.newaxis, :],
            model_dim=model_dim
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# Transformer Block
def transformer_block(inputs, num_heads, ff_dim, dropout=0.1):
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention_output = layers.Dropout(dropout)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    ffn_output = layers.Dense(ff_dim, activation="relu")(attention_output)
    ffn_output = layers.Dense(inputs.shape[-1])(ffn_output)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    sequence_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
    
    return sequence_output

# 自回归Transformer模型
def build_autoregressive_transformer(input_shape, num_heads, ff_dim, num_blocks):
    inputs = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0)(inputs)
    x = PositionalEncoding(input_shape[0], input_shape[1])(inputs)
    
    for _ in range(num_blocks):
        x = transformer_block(x, num_heads, ff_dim)

    x = layers.Dense(len(word_2_index))(x)
    outputs = layers.Softmax()(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def train():
    input_shape = X_train[0].shape 
    num_heads = 4
    ff_dim = 256  # Feed Forward层的维度
    num_blocks = 2  # Transformer Block的数量

    #构建Transformer模型
    model = build_autoregressive_transformer(input_shape, num_heads, ff_dim, num_blocks)
    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # 打印模型结构
    model.fit(X_train, Y_train, epochs=8, validation_split=0.1)
    model.save(r'C:\Users\IG2017-Laptop-017\source\repos\qzwx0908\DL testworks\UCAS_DeepLearning_homework\国科大-深度学习作业\YC 自动写诗\my_transformer_model.keras')

def generate_text(my_model, seed_text, word_2_vec, index_2_word, max_length):
    """
    基于种子文本生成文本。
    :param model: 训练好的Transformer模型
    :param word_vec: 词到索引的映射字典
    :param index_to_word: 索引到词的映射字典
    :param seed_text: 用于生成的初始种子文本
    :param max_length: 生成文本的最大长度
    :return: 生成的文本
    """

    # 将种子文本转换为向量

    generated_text = [t for t in seed_text]

    for _ in range(max_length):
        # 获取所有输出的词向量
        input_vector = []
        for word in generated_text:
            word_v = word_2_vec[word]
            input_vector.append(word_v)
        input_vector = np.array([np.pad(input_vector, ((0, max_length - len(input_vector)), (0, 0)), 'constant', constant_values=0)])
          
        # 模型预测下一个词的概率分布
        prediction = my_model.predict(input_vector)

        # 选择概率最高的词
        predicted_index = np.argmax(prediction[0][-1])
        predicted_word = index_2_word[predicted_index]

        # 将生成的词添加到结果中
        generated_text.append(predicted_word)
        # 如果预测的词是结束标记，则停止生成
        if predicted_word == '<eos>':  # 假设 '。' 是结束标记
            break

    return ''.join(generated_text)

if __name__ == '__main__':
    wp = wordprocess()
    word_2_vec = wp.get_word_vec()
    word_2_index = word_2_vec.key_to_index
    index_2_word = word_2_vec.index_to_key
    X_train = wp.X_train
    Y_train = wp.Y_train
    train()
    