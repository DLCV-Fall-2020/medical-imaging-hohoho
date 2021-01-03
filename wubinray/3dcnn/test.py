import torch
import torch.nn as nn 
from collections import Counter

from data_aug.dataset_wrapper import DatasetWrapper
from models.resnet3d import *
from models.model3d import HemoModel3d

if __name__=='__main__':
    #model = resnet10_3d(**dict(in_channels=1))
    model = HemoModel3d()
    model.train()
    model.cuda()
    img_size=256
    a = torch.ones(16,1,40,img_size,img_size).cuda()
    b = model(a)
    print(b.shape)
    print("finish")
'''
if __name__=='__main__':
    
    dataset = DatasetWrapper('/media/disk1/aa/Blood_data/train/', 32)
    train_loader,_ = dataset.get_dataloaders()
    t_ls = []
    for d in train_loader.dataset.data:
        t_ls.append(len(d[1]))
    print(len(train_loader.dataset),sorted(Counter(t_ls).items()))
    
    print(len(train_loader))
    for idx,(imgs, lbls, mask) in enumerate(train_loader):
        print(imgs.shape, lbls.shape, mask.shape)
        input()
'''
