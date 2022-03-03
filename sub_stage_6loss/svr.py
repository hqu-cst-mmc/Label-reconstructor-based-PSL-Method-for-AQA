# -*- coding: utf-8 -*-
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import numpy as np

import os
from os import listdir
from os.path import isfile, join, isdir
from scipy.stats import spearmanr

import warnings
warnings.filterwarnings("ignore")


feature_path = './p3d_features'
# train_file = './data_files/training_idx.npy'
# test_file = './data_files/testing_idx.npy'
# label_file = './data_files/difficulty_level.npy'
label_file = './data_files/overall_scores.npy'
train_file = './data_files/all_train_v2.npy'
test_file = './data_files/all_test_v2.npy'


def to_array(l):
	output = np.zeros_like(len(l))
	for i in l:
		print(i)
		output = np.append(output, i)
	return output[1:]


train_data = np.array(np.load(train_file,encoding="latin1"))
test_data = np.array(np.load(test_file,encoding="latin1"))

# stage-wise feature
# param: 1-19-1
x_train = np.array(train_data[0])
x_test = np.array(test_data[0])

#stage-wise score
# x_train = np.array(train_data[1])
# x_test = np.array(test_data[1])

# print('x_train:',x_train.shape)
# print('test_data:',test_data.shape)


# if using the concated feature, uncomment this
# concated stage-wise feature
# x_train = np.array(train_data[-1])
# x_test = np.array(test_data[-1])

y_train = np.array(train_data[2])
y_test = np.array(test_data[2])



for idx, i in enumerate(x_train):
	for idy, j in enumerate(i):
		x_train[idx][idy] = round(float(j),1)

x_train = np.array(list(x_train), dtype=np.float)
y_train = np.array(y_train, dtype=np.float)

for idx, i in enumerate(x_test):
	for idy, j in enumerate(i):
		x_test[idx][idy] = round(float(j),1)

x_test = np.array(list(x_test), dtype=np.float)
y_test = np.array(y_test, dtype=np.float)

print('x_train:',x_train.shape)
print('x_test:',x_test.shape)

# for i in range(len(x_train)):
# 	print x_train[i], y_train[i]
print(len(x_train), len(y_train))

clf = LinearRegression()
clf.fit(x_train, y_train)
y_predit = clf.predict(x_test)

rho, p_val = spearmanr(y_test, y_predit)
print('LR:',rho)

# clf = SVR(C=100, epsilon=0.01)
# clf.fit(x_train, y_train)
# y_predit = clf.predict(x_test)

clf = SVR(C=221, epsilon=25,kernel='linear')
clf.fit(x_train, y_train)
y_predit = clf.predict(x_test)

rho, p_val = spearmanr(y_test, y_predit)
print ('lr:',rho)

clf = SVR(C=100, epsilon=0.001)
clf.fit(x_train, y_train)
y_predit = clf.predict(x_test)

rho, p_val = spearmanr(y_test, y_predit)
print ('SVR:',rho)

# for i in range(len(y_test)):
# 	print (y_test[i],'-',y_predit[i])

#c_range = [60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260]
# e_range = [7, 8,9,10,11,12,13,14]
#c_range = [1, 1000]
# e_range = [7,8,9,10,11,12,13,14,15,16,17,18,19,0.1,0.01,0.001]
k_range = ['rbf', 'linear']# 'poly',
g_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
c_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
y_predit_max = 0
epsilon_max = 0
C_max = 0
gamma_max = 0
# # for i in range(10,1000,10):
# # for i in range(1,100):
for i in c_range:
	for j in range(1,50):
		for g in g_range:
			for k in k_range:
				clf = SVR(C=i, epsilon=j, gamma=g, kernel=k)
				clf.fit(x_train, y_train)
				y_predit = clf.predict(x_test)

				rho, p_val = spearmanr(y_test, y_predit)
				if(rho > y_predit_max):
					y_predit_max = rho
					C_max = i
					epsilon_max = j
					gamma_max = g
				# kernel = k
print(C_max,epsilon_max,gamma_max,y_predit_max)
			# print (k, i, j, round(rho, 3))
			# logging.info("k:{0}, C:{1}, e:{2}, rho:{3}".format(k, i, j, rho))
			# print (k, i, j, rho)
# print(y_predit)

