import numpy as np
import torch
# dir = 'layer_ablation(5FC)'
# load_train_dir = './results'+dir +'training_sub_scores.npy'
train_subscores = np.load('./results/layer_ablation(1FC)/training_sub_scores.npy',allow_pickle=True)
test_subscores = np.load('./results/layer_ablation(1FC)/testing_sub_scores.npy',allow_pickle=True)
sub_scores = np.concatenate((train_subscores,test_subscores),axis=0)
sub_scores = sub_scores[np.lexsort(sub_scores[:,::-1].T)]
print(sub_scores)
np.save('./results/layer_ablation(1FC)/sub_scores.npy',sub_scores)
