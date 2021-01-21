from data_aug.gaussian_blur import GaussianBlur

#####
from PIL import Image, ImageOps

import os
import glob 
import math 
import torch
import pickle
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
    def __init__(self, path, ch):
        self.path = path
        self.ch = ch 
        self.trans = 0# self.get_transform(ch)

        self.data = [] # [ (1,t), ... ]

        dirs = sorted(os.listdir(path))
        for _dir in dirs:
            _fnames = os.listdir(f"{path}/{_dir}")
            _fnames = [(int(f[f.find('_')+1:f.find('.jpg')]), f) 
                        for f in _fnames]
            _fnames = sorted(_fnames, key=lambda x:x[0])
            _fnames = [f[1] for f in _fnames]
            
            self.data.append((_dir, _fnames))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _dir, _fnames = self.data[idx]
        if self.ch==1:
            stack = []
            for f in _fnames:
                img_path = self.path + f"{_dir}/{f}"
                img = Image.open(img_path).resize((img_size,img_size))
                img = self.trans(img)
                stack.append(img)
            stack = np.stack(stack, axis=1)
            stack = torch.tensor(stack) # (1,t,h,w)
        else:
            stack = []
            for i,f in enumerate(_fnames):
                tmp = []
                for j in range(i-self.ch//2, i+math.ceil(self.ch/2)):
                    if j<0 or j>=len(_fnames):
                        j=i
                    img_path = self.path + f"/{_dir}/{_fnames[j]}"
                    img = Image.open(img_path)
                    img = img.resize((img_size, img_size))
                    tmp.append(img)
                tmp = np.stack(tmp, axis=-1)
                img = Image.fromarray(tmp)
                img = self.trans(img)
                stack.append(img)
            stack = np.stack(stack, axis=0) # (t,ch,h,w)

        _dir = [_dir]*len(_fnames)

        return stack, _dir, _fnames 

    def collate_fn(self, samples):
        t_max = max([stack.shape[0] for stack,_,_ in samples])
        
        batch_imgs, batch_mask, batch_dir, batch_fnames = [],[],[],[]
        for stack, _dir, _fnames in samples:
            # stack:(t,ch,h,w), label:(t,class)
            t = stack.size(0)
            
            # img
            img = torch.zeros(t_max,self.ch,img_size,img_size)
            img[:t] = stack
            
            # mask
            mask = [[True]]*t + [[False]]*(t_max-t)
            mask = torch.tensor(mask)

            batch_imgs.append(img)
            batch_mask.append(mask)
            batch_dir.append(_dir)
            batch_fnames.append(_fnames)
        
        batch_imgs = torch.stack(batch_imgs,dim=0)
        batch_mask = torch.stack(batch_mask,dim=0)
        # imgs(b,t_max,ch,h,w) mask(b,t_max) dir(b,t) fnames(b,t)
        return batch_imgs, batch_mask, batch_dir, batch_fnames 

class BloodDataset(Dataset):
    def __init__(self, path, dirs, ch=3, valid_mode=True):
        train_df = pd.read_csv(path.rstrip('/')+".csv")

        self.ch = ch 
        self.path = path
        self.trans = Trans(valid_mode, ch)
        
        df = pd.read_csv(path.rstrip('/')+".csv")

        self.data = [] # [ (1,t,(t,class)), ... ] 
        
        for _dir in dirs:
            sub_df = train_df.loc[train_df['dirname']==_dir]
            
            _fnames, _lbls = sub_df['ID'].tolist(), sub_df.to_numpy()[:,2:]
            
            self.data.append((_dir, _fnames, _lbls)) # (1,t,(t,class))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _dir, _fnames, label = self.data[idx]
        
        stack = []
        for i,f in enumerate(_fnames):
            tmp = []
            for j in range(i-self.ch//2, i+math.ceil(self.ch/2)):
                if j<0 or j>=len(_fnames):
                    j=i
                img_path = self.path + f"/{_dir}/{_fnames[j]}"
                img = Image.open(img_path)
                img = img.resize((img_size, img_size))
                tmp.append(img)
            tmp = np.stack(tmp, axis=-1)
            img = Image.fromarray(tmp)
            img = self.trans(img)
            stack.append(img)
        stack = np.stack(stack, axis=0)
        stack = torch.tensor(stack)

        label = label.astype(np.bool) 
        label = torch.tensor(label) 

        return stack, label # (t,ch,h,w) (t,class)

    def collate_fn(self, samples):
        t_max = max([stack.shape[0] for stack,_ in samples])
        
        batch_imgs, batch_lbls, batch_mask = [],[],[]
        for stack, label in samples:
            # stack:(t,ch,h,w), label:(t,class)
            t, cls = label.shape 
            
            # img
            img = torch.zeros(t_max,self.ch,img_size,img_size)
            img[:t] = stack
            
            # lbl
            lbl = torch.zeros(t_max, cls).bool()
            lbl[:t] = label 
            
            # mask
            mask = [[True]]*t + [[False]]*(t_max-t)
            mask = torch.tensor(mask)

            batch_imgs.append(img)
            batch_lbls.append(lbl)
            batch_mask.append(mask)
        
        batch_imgs = torch.stack(batch_imgs,dim=0) # (b,t_max,ch,h,w)
        batch_lbls = torch.stack(batch_lbls,dim=0) # (b,t_max,class)
        batch_mask = torch.stack(batch_mask,dim=0) # (b,t_max)
        return batch_imgs, batch_lbls, batch_mask 

class DatasetWrapper(object):

    def __init__(self, path, bsize, valid_size=0.15, ch=3,
                    train_valid_split_pkl=None):
        assert train_valid_split_pkl is not None 
        self.path = path 
        self.ch = ch
        self.bsize = bsize
        self.valid_size = valid_size
        self.train_valid_split_pkl = train_valid_split_pkl 
    
    def get_dataloaders(self):
        # load pre split train valid dirs
        print("\t[Info] load pre split train valid set")
        with open(self.train_valid_split_pkl+"/train_set.pkl", 'rb') as f:
            train_dirs = pickle.load(f)
        with open(self.train_valid_split_pkl+"/valid_set.pkl", 'rb') as f:
            valid_dirs = pickle.load(f)
        print(f"\t\t len train:{len(train_dirs)} valid:{len(valid_dirs)}")
    
        # dataset
        train_dataset = BloodDataset(self.path+"/train", train_dirs, self.ch)
        valid_dataset = BloodDataset(self.path+"/train", valid_dirs, self.ch, valid_mode=True)

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

class Trans(object):
    def __init__(self, valid_mode=True, ch=3):
        train_trans = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomRotation(20, fill=(0,)), 
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
        self.trans = test_trans if valid_mode else train_trans 

    def __call__(self, sample):
        xi = self.trans(sample)
        return xi

