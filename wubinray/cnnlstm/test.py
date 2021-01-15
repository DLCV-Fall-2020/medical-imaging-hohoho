from data_aug.dataset_wrapper import DatasetWrapper, BloodDataset_Test
import torch
import torch.nn as nn
'''
from models.model import HemoCnnLstm

model = HemoCnnLstm()
model.cuda()
'''

dataset = DatasetWrapper("/media/disk1/aa/Blood_data/train/", 4)
train,_ = dataset.get_dataloaders()
dataset = train.dataset 

#dataset = BloodDataset_Test("/media/disk1/aa/Blood_data/test")

for i in range(0,1200,30):
    print(dataset.data[i][1])
    input()

