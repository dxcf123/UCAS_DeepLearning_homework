import tensorflow as tf
from keras import layers
import numpy as np

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

class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, model_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(max_len, model_dim)

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

# 自回归Transformer模型
def build_autoregressive_transformer(input_shape, num_heads, ff_dim, num_blocks, vocab_size):
    inputs = layers.Input(shape=input_shape)
    x = PositionalEncoding(input_shape[0], input_shape[1])(inputs)
    
    for _ in range(num_blocks):
        x = transformer_block(x, num_heads, ff_dim)

    x = layers.Dense(vocab_size)(x)
    outputs = layers.Softmax()(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model