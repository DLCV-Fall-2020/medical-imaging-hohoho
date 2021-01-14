import torch
import torch.nn as nn
import numpy as np

from utils.others import weight_smooth 

a = np.random.randn(20,5)
a = np.array([[1,0,0,1,0],[1,1,0,0,1],[1,0,1,1,0]])
b = weight_smooth(a, 3)
print(b.shape)

