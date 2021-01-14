import torch
import torch.nn as nn
'''
from models.model import HemoCnnLstm

model = HemoCnnLstm()
model.cuda()
'''
a = torch.ones(4,5,64,64)
#out = model(a.cuda())

try:
    from apex import amp
    import apex 
    apex_support = True
except:
    print("\t[Info] apex is not supported")
    apex_support = False 

print(apex_support)
a = torch.ones(3,10)
a = a.cuda()
b = apex.normalization.FusedLayerNorm(10).cuda()
c = b(a)

