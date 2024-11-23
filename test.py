import os

import numpy as np

path1 = 'saved/collect_data.npy'

data1 = np.load(path1)
print(data1.shape)

path2 = '/Users/jayden/Desktop/未命名文件夹/'
data2 = []
for i in os.listdir(path2):
    if i.endswith('.npy'):
        data = np.load(path2 + i)
        print(data.shape)
        data2.append(data)

datas = data1
for data in data2:
    datas = np.concatenate((datas, data), axis=0)
print(datas.shape)

datas = datas[:50000, :]
print(datas.shape)

np.save(path1, datas)
