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
    inputs = tf.keras.layers.Masking(mask_value=0)(inputs)
    x = PositionalEncoding(input_shape[0], input_shape[1])(inputs)
    
    for _ in range(num_blocks):
        x = transformer_block(x, num_heads, ff_dim)

    x = layers.Dense(len(word_2_index))(x)
    outputs = layers.Softmax()(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def train():

    input_shape = X_train[0].shape 
    num_heads = 8
    ff_dim = 512  # Feed Forward层的维度
    num_blocks = 3  # Transformer Block的数量

    #构建Transformer模型
    model = build_autoregressive_transformer(input_shape, num_heads, ff_dim, num_blocks)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # 打印模型结构
    model.fit(X_train, Y_train, epochs=20, validation_split=0.1)
    model.save('/workspaces/UCAS_DeepLearning_homework/国科大-深度学习作业/YC 自动写诗/my_transformer_model.keras')

def generate_text(my_model, seed_text, word_2_vec, index_2_word, max_length=50):
    """
    基于种子文本生成文本。

    :param model: 训练好的Transformer模型
    :param word_vec: 词到索引的映射字典
    :param index_to_word: 索引到词的映射字典
    :param seed_text: 用于生成的初始种子文本
    :param max_length: 生成文本的最大长度
    :return: 生成的文本
    """
    input_sequence = []

    # 将种子文本转换为向量
    for word in seed_text:
        if word in word_2_vec:
            ori_input = word_2_vec[word]
            ori_input = tf.expand_dims(ori_input, axis=0)
            ori_input = np.array([np.pad(ori_input, ((0, 23 - len(ori_input)), (0, 0)), 'constant', constant_values=0)])
            input_sequence.append(ori_input)

    generated_text = seed_text

    for _ in range(max_length):
        # 获取所有输出的词向量
        input_vector = np.array([my_model.layers[0](input_sequence)]).reshape(1, -1, input_vector.shape[-1])

        # 模型预测下一个词的概率分布
        predictions = my_model.predict(input_vector)

        # 选择概率最高的词
        predicted_index = np.argmax(predictions)
        predicted_word = index_2_word[predicted_index]

        # 将生成的词添加到结果中
        generated_text += predicted_word

        # 更新输入序列，将预测的词添加到输入序列中
        input_vector = np.append(input_sequence, predicted_index)

        # 如果预测的词是结束标记，则停止生成
        if predicted_word == '。':  # 假设 '。' 是结束标记
            break

    return generated_text

if __name__ == '__main__':
    wp = wordprocess()
    word_2_vec = wp.get_word_vec()
    word_2_index = word_2_vec.key_to_index
    index_2_word = word_2_vec.index_to_key
    X_train = wp.X_train
    Y_train = wp.Y_train
    train()