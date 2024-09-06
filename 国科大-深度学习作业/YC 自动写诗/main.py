from keras.utils import CustomObjectScope
from keras.models import load_model
from transformer import PositionalEncoding, generate_text
import functions
import numpy as np
import os
import tensorflow as tf

path ='/workspaces/UCAS_DeepLearning_homework/国科大-深度学习作业/YC 自动写诗/my_transformer_model.keras'
if os.path.exists(path):
    with CustomObjectScope({'PositionalEncoding': PositionalEncoding}):
        model = load_model('/workspaces/UCAS_DeepLearning_homework/国科大-深度学习作业/YC 自动写诗/my_transformer_model.keras')
        print("模型装载成功")
        model.summary()

wp = functions.wordprocess()
word_vec = wp.get_word_vec()
original_input = word_vec['日']
original_input = tf.expand_dims(original_input, axis=0)
input = np.array([np.pad(original_input, ((0, 23 - len(original_input)), (0, 0)), 'constant', constant_values=0)])
output = model.predict(input)
output = np.argmax(output[0], axis = -1)
output = word_vec.index_to_key[output]
print(output)

generated_sequence = generate_text(model, "白", word_vec, word_vec.index_to_key)
print(w for w in generated_sequence if generated_sequence)
