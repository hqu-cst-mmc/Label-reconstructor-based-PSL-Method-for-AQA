import numpy as np
import random

train_file = './data_files/training_idx.npy'
train_ori = np.load(train_file)

## 随机选择200个idx
train_ori = train_ori.tolist()
train1 = random.sample(train_ori,200)
# print(train1)

def diff_list(x):
    ret = []
    for i in x:
        if isinstance(i, list):
            for j in diff_list(i):
                ret.append(j)
        else:
            ret.append(i)
    return ret

train1 = diff_list(train1) ## 200

train_ori = diff_list(train_ori)

# print('train_ori:',train_ori)
train2 = list(set(train_ori).difference(set(train1))) ## 100
# print('train2:',train2)
# print('train1:',train1)

## 存入两个idx文件 200/100
np.save('./data_files/train1_idx.npy', train1)
np.save('./data_files/train2_idx.npy', train2)

