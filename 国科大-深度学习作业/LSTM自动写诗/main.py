import torch

import function
print(torch.cuda.is_available())
path1 = r'tangshi'  # 古诗所在的文件夹
path2 = r'data'  # 存放处理后的数据

# 返回模型，可以进行后处理
#  参数：训练集文件夹路径，处理后数据存放路径，5言/7言古诗，每个批次的大小
model = function.train(path1, path2, 7, 64)
# torch.save(model.state_dict(), './data/poem_7_model.pth')
