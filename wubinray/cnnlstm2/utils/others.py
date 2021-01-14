import math
import numpy as np

def weight_smooth(label, index_select):
    t,cls = label.shape
    weight = [math.pow(2,-np.abs(i-index_select)) for i in range(t)]
    weight = np.array(weight).reshape(1,t)
    
    label = np.dot(weight, label) / weight.sum()
    label = (label > 0.5).astype(int)
    label = label.reshape(cls)
    return label 

class Averager():
    def __init__(self):
        self.clean()

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
    
    def clean(self):
        self.n = 0
        self.v = 0

