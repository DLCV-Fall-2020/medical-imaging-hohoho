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

# ich,ivh,sah,sdh,edh

class BloodDataset_Test(Dataset):
    def __init__(self, path, ch=1):
        self.path = path
        self.ch = ch 
        self.trans = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*self.ch,std=[0.5]*self.ch)
            ]) 
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass

class BloodDataset(Dataset):
    def __init__(self, path, dirs, trans, t=32):
        train_df = pd.read_csv(path.rstrip('/')+".csv")
        
        self.path = path
        self.trans = trans 
        self.t = t
        
        self.data = []
        
        for _dir in dirs:
            sub_df = train_df.loc[train_df['dirname']==_dir]
            
            _fnames, lbls = sub_df['ID'], sub_df.to_numpy()[:,2:]
            
            self.data.append((_dir, _fnames, lbls)) #(1,t,(t,class))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _dir, _fnames, label = self.data[idx]
        stack = []
        for f in _fnames:
            img_path = self.path + f"{_dir}/{f}"
            img = Image.open(img_path).resize((img_size,img_size))
            img = self.trans(img)
            stack.append(img)
        stack = np.stack(stack, axis=1)
        
        label = label.astype(np.bool)

        stack = torch.tensor(stack) # (1,t,h,w)
        label = torch.tensor(label) # (t,class)
        return stack, label

    def collate_fn(self, samples):
        t_max = max([stack.shape[1] for stack,_ in samples])
        
        batch_imgs, batch_lbls, batch_mask = [],[],[]
        for stack, label in samples:
            # stack:(1,t,h,w), label:(t,class)
            t, cls = label.shape 
            
            # img
            img = torch.zeros(1,t_max,img_size,img_size)
            img[:,:t] = stack
            
            # lbl
            lbl = torch.zeros(t_max, cls).bool()
            lbl[:t] = label 
            
            # mask
            mask = [1]*t + [0]*(t_max-t)
            mask = torch.tensor(mask)

            batch_imgs.append(img)
            batch_lbls.append(lbl)
            batch_mask.append(mask)
        
        batch_imgs = torch.stack(batch_imgs,dim=0) # (b,1,t_max,h,w)
        batch_lbls = torch.stack(batch_lbls,dim=0) # (b,t_max,class)
        batch_mask = torch.stack(batch_mask,dim=0) # (b,t_max)
        return batch_imgs, batch_lbls, batch_mask 
    
    @staticmethod 
    def get_transform(ch=1):
        train_trans = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                        transforms.ColorJitter(0.1,0.1,0.1,0),
                    ],p=0.4),
                #GaussianBlur(kernel_size=int(0.1 * img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*ch, std=[0.5]*ch)
            ])
        test_trans = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*ch, std=[0.5]*ch)
            ])
        return train_trans, test_trans

class DatasetWrapper(object):

    def __init__(self, path, bsize, valid_size=0.15, ch=1):
        self.path = path 
        self.bsize = bsize
        self.valid_size = valid_size
        self.ch = ch 
    
    def get_dataloaders(self):
        # split train dirs
        dirs = os.listdir(self.path)
        np.random.shuffle(dirs)
        split = int(len(dirs)*(1-self.valid_size))
        train_dirs, valid_dirs = dirs[:split], dirs[split:]
        
        # data aug
        train_trans, valid_trans = BloodDataset.get_transform(self.ch)
    
        # dataset
        train_dataset = BloodDataset(self.path, train_dirs, train_trans, self.ch)
        valid_dataset = BloodDataset(self.path, valid_dirs, valid_trans, self.ch)

        # dataloader
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.bsize,
                                  collate_fn=train_dataset.collate_fn,
                                  num_workers=6,
                                  shuffle=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=self.bsize,
                                  collate_fn=valid_dataset.collate_fn,
                                  num_workers=6)
        return train_loader, valid_loader 

class SimCLRTrans(object):
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, sample):
        xi = self.trans(sample)
        xj = self.trans(sample)
        return xi,xj

