from collections import Counter
from models.densenet import HemoDenseNet121
from data_aug.dataset_wrapper import DatasetWrapper, BloodDataset, DataLoader
import torch 

if __name__=='__main__':
    model = HemoDenseNet121()
    a = torch.randn(3,3,512,512)
    b = model.featrue(a)
    print(b.shape)
