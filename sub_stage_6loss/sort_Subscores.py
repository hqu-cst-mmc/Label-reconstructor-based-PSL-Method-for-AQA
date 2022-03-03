import numpy as np
import torch

subscores = np.load('./data_files/sub_scores.npy',allow_pickle=True)
# print(subscores)
subscores = subscores[np.lexsort(subscores[:,::-1].T)]
np.save('./data_files/training_sub_scores.npy',subscores)
# print(subscores)