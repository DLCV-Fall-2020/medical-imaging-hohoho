#from data_aug.gaussian_blur import GaussianBlur
from utils.others import weight_smooth

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

class BloodDataset_Test(Dataset):
    def __init__(self, path, trans, t=32):
        self.path = path
        self.trans = trans 
        self.t = t

        self.data = [] # [ (1,t,1), ... ]

        dirs = sorted(os.listdir(path))

        for _dir in dirs:
            _fnames = sorted(os.listdir(f"{path}/{_dir}"))

            for i in range(len(_fnames)):
                src = max(0, i-t//2)
                end = src + t
                self.data.append((_dir, _fnames[src:end], _fnames[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _dir, _fnames, _fname = self.data[idx]
        stack = []
        for f in _fnames:
            img_path = self.path + f"{_dir}/{f}"
            img = Image.open(img_path).resize((img_size,img_size))
            img = self.trans(img)
            stack.append(img)
        stack = np.stack(stack, axis=1)
        stack = torch.tensor(stack) # (1,t,h,w)
        
        index_select = _fnames.index(_fname)
        
        return stack, index_select, _fname 

    def collate_fn(self, sample):
        pass 
    
    @staticmethod
    def get_transform(ch=1):
        trans = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*self.ch,
                                     std=[0.5]*self.ch)
            ]) 
        return trans 
 

class BloodDataset(Dataset):
    def __init__(self, path, dirs, trans, t=3):
        assert t%2==1 
        train_df = pd.read_csv(path.rstrip('/')+".csv")
        #train_df = pd.read_csv(path.rstrip('/')+"_clean.csv")

        self.path = path
        self.trans = trans 
        self.t = t
        
        self.data = [] # [ (1,t,(t,class)), ... ] 
        
        for _dir in dirs:
            sub_df = train_df.loc[train_df['dirname']==_dir]
            
            _fnames, lbls = sub_df['ID'].tolist(), sub_df.to_numpy()[:,2:]
            
            max_len = len(_fnames)
            for i in range(0, max_len):
                src = max(0, i-t//2)
                end = src + t
                self.data.append((_dir, 
                            _fnames[i], _fnames[src:end], lbls[src:end]))
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _dir, _fname, _fnames, label = self.data[idx]
        stack = []
        for f in _fnames:
            img_path = self.path + f"{_dir}/{f}"
            img = Image.open(img_path).resize((img_size,img_size))
            img = self.trans(img)
            stack.append(img)
        stack = np.stack(stack, axis=1)
        
        # label smooth
        index_select = _fnames.index(_fname)        
        label = label.astype(np.bool)
        label = weight_smooth(label, index_select) # (t,class)->(1,class)
        
        # return: stack(1,t,h,w) lable(1,class)
        stack = torch.tensor(stack)
        label = torch.tensor(label)
        return stack, label

    def collate_fn(self, samples):
        t_max = max([stack.shape[1] for stack,_ in samples])
        
        batch_imgs, batch_lbls, batch_mask = [],[],[]
        for stack, label in samples:
            # stack:(1,t,h,w), label:(class)
            t = stack.size(1)
            
            # img
            img = torch.zeros(1,t_max,img_size,img_size)
            img[:,:t] = stack
            
            # lbl
            lbl = label

            batch_imgs.append(img)
            batch_lbls.append(lbl)
        
        batch_imgs = torch.stack(batch_imgs,dim=0) # (b,1,t_max,h,w)
        batch_lbls = torch.stack(batch_lbls,dim=0) # (b,class)
        return batch_imgs, batch_lbls#, batch_mask 
    
    @staticmethod 
    def get_transform(ch=1):
        train_trans = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomRotation(30, fill=(0,)), 
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                        transforms.ColorJitter(0.1,0.1,0.1,0)
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

    def __init__(self, path, bsize, valid_size=0.15, ch=1, t=10):
        self.path = path 
        self.bsize = bsize
        self.valid_size = valid_size
        self.ch = ch 
        self.t = t
    
    def get_dataloaders(self):
        # split train dirs
        dirs = os.listdir(self.path)
        np.random.shuffle(dirs)
        split = int(len(dirs)*(1-self.valid_size))
        train_dirs, valid_dirs = dirs[:split], dirs[split:]
        
        # data aug
        train_trans, valid_trans = BloodDataset.get_transform(self.ch)
    
        # dataset
        train_dataset = BloodDataset(self.path, train_dirs, train_trans,
                                    self.t)
        valid_dataset = BloodDataset(self.path, valid_dirs, valid_trans,
                                    self.t)

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
