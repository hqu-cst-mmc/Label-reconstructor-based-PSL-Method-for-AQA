import torch
import torch.nn as nn
import numpy as np

class LBE(nn.Module):
    def __init__(self):
        super(LBE, self).__init__()
        self.min = np.amin()
        self.max = np.amax()
    def forward(self, x):
        min = self.min(x)
        max = self.max(x)
        return (x-min)/(max-min)