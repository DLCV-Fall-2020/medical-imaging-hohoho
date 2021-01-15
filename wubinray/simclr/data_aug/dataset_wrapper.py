from data_aug.gaussian_blur import GaussianBlur

#####
from PIL import Image, ImageOps

import os
import glob 
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms 

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets

np.random.seed(87)
torch.manual_seed(87)

img_size=256
img_size=512 

# ich,ivh,sah,sdh,edh

class BloodDataset(Dataset):
    def __init__(self, dirs, ch=3):
        self.ch = ch 
        self.trans = SimCLRTrans(ch)
        
        self.data = []
        
        for _dir in dirs: 
            _fnames = os.listdir(_dir)
            _fnames = [(int(f[f.find('_')+1:f.find('.jpg')]), f)
                        for f in _fnames]
            _fnames = sorted(_fnames, key=lambda x:x[0])
            _fnames = [f[1] for f in _fnames]

            for i in range(0, len(_fnames)-self.ch):
                self.data.append((_dir,
                                  _fnames[i:i+self.ch])) # (1,ch)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _dir, _fnames = self.data[idx]
        if self.ch == 1:
            img = Image.open(f"{_dir}/{_fnames[0]}")
        else:
            stack = []
            for i in range(self.ch):
                img = Image.open(f"{_dir}/{_fnames[i]}")
                img = img.resize((img_size,img_size))
                img = np.array(img).astype(np.uint8)
                stack.append(img)
            stack = np.stack(stack, axis=-1)
            img = Image.fromarray(stack)
        
        img1,img2 = self.trans(img)
        return torch.tensor(img1), torch.tensor(img2)

    def collate_fn(self, samples):
        batch_imgs1, batch_imgs2 = [], []

        for img1, img2 in samples:
            batch_imgs1.append(img1)
            batch_imgs2.append(img2)
        
        batch_imgs1 = torch.stack(batch_imgs1, dim=0) # (b,ch,h,w)
        batch_imgs2 = torch.stack(batch_imgs2, dim=0) # (b,ch,h,w)
        return batch_imgs1, batch_imgs2 
    
class DatasetWrapper(object):

    def __init__(self, path, bsize, valid_size=0.15, ch=1):
        self.path = path 
        self.bsize = bsize
        self.valid_size = valid_size
        self.ch = ch 
    
    def get_dataloaders(self):
        # split train dirs
        train_dirs = os.listdir(self.path+"/train")
        train_dirs = [self.path+"/train/"+d for d in train_dirs]
        
        test_dirs = os.listdir(self.path+"/test")
        test_dirs = [self.path+"/test/"+d for d in test_dirs]
        
        dirs = train_dirs + test_dirs 
        np.random.shuffle(dirs)
        split = int(len(dirs)*(1-self.valid_size))
        train_dirs, valid_dirs = dirs[:split], dirs[split:]
    
        # dataset
        train_dataset = BloodDataset(train_dirs, self.ch)
        valid_dataset = BloodDataset(valid_dirs, self.ch)

        # dataloader
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.bsize,
                                  collate_fn=train_dataset.collate_fn,
                                  num_workers=6,
                                  drop_last=True,
                                  shuffle=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=self.bsize,
                                  collate_fn=valid_dataset.collate_fn,
                                  num_workers=6,
                                  drop_last=True)
        return train_loader, valid_loader 

class SimCLRTrans(object):
    def __init__(self, ch=1):
        self.trans = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomRotation(90, fill=(0,)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply(
                        [transforms.ColorJitter(0.1,0.1,0.1,0)]
                    ,p=0.8),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*ch,
                                     std=[0.5]*ch)
            ])

    def __call__(self, sample):
        xi = self.trans(sample)
        xj = self.trans(sample)
        return xi,xj

