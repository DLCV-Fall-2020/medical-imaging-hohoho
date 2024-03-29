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

class BloodDataset_Test_TTA(Dataset): # TestTimeAugment(3)
    def __init__(self, path, ch=3, num_tta=3):
        self.path = path
        self.ch = ch 
        self.num_tta = num_tta 
        self.trans = self.get_transform(ch)

        self.data = [] # [ (1,t,1), ... ]

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
        stack_tta = []
        for i in range(self.num_tta):
            if self.ch==1:
                stack = []
                for f in _fnames:
                    img_path = self.path + f"{_dir}/{f}"
                    img = Image.open(img_path).resize((img_size,img_size))
                    img = self.trans(img)
                    stack.append(img)
                stack = np.stack(stack, axis=1)
            else:
                stack = []
                for i,f in enumerate(_fnames):
                    tmp = []
                    for j in range(i-self.ch//2, i+math.ceil(self.ch/2)):
                        if j<0 or j>=len(_fnames):
                            j=i 
                        img_path = self.path + f"/{_dir}/{_fnames[j]}"
                        img = Image.open(img_path)
                        img = img.resize((img_size,img_size))
                        tmp.append(img)
                    tmp = np.stack(tmp, axis=-1)
                    img = Image.fromarray(tmp)
                    img = self.trans(img)
                    stack.append(img)
                stack = np.stack(stack, axis=0)
            stack_tta.append(stack)
            
        stack_tta = np.concatenate(stack_tta)
        stack = torch.tensor(stack) # (num_tta,t,ch,h,w)
        
        _dir = [_dir]*len(_fnames)

        return stack, _dir, _fnames   

    def collate_fn(self, samples):
        t_max = max([stack.shape[1] for stack,_,_ in samples])
        
        batch_imgs, batch_mask, batch_dir, batch_fnames = [],[],[],[]
        for stack, _dir, _fnames in samples:
            # stack:(num_tta,t,ch,h,w), label:(t,class)
            t = stack.size(1)
            
            # img
            img = torch.zeros(self.num_tta,t_max,img_size,img_size)
            img[:,:t] = stack
            
            # mask
            mask = [[True]]*t + [[False]]*(t_max-t)
            mask = torch.tensor(mask)

            batch_imgs.append(img)
            batch_mask.append(mask)
            batch_dir.append(_dir)
            batch_fnames.append(_fnames)
        
        batch_imgs = torch.cat(batch_imgs,dim=0)
        batch_mask = torch.stack(batch_mask,dim=0)
        # imgs(b*num_tta,t_max,ch,h,w) mask(b,t_max) dir(b,t) fnames(b,t)
        return batch_imgs, batch_mask, batch_dir, batch_fnames 
    
    @staticmethod
    def get_transform(ch=1):
        trans = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*ch,
                                     std=[0.5]*ch)
            ]) 
        return trans 
 

class BloodDataset_Test(Dataset):
    def __init__(self, path, ch):
        self.path = path
        self.ch = ch 
        self.trans = self.get_transform(ch)

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
    
    @staticmethod
    def get_transform(ch=1):
        trans = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*ch,
                                     std=[0.5]*ch)
            ]) 
        return trans 
 

class BloodDataset(Dataset):
    def __init__(self, path, dirs, trans, ch=1):
        train_df = pd.read_csv(path.rstrip('/')+".csv")
        #train_df = pd.read_csv(path.rstrip('/')+"_clean.csv")

        self.path = path
        self.trans = trans 
        self.ch = ch 
        
        self.data = [] # [ (1,t,(t,class)), ... ] 
        
        for _dir in dirs:
            sub_df = train_df.loc[train_df['dirname']==_dir]
            
            _fnames, lbls = sub_df['ID'].tolist(), sub_df.to_numpy()[:,2:]
            
            self.data.append((_dir, _fnames, lbls)) # (1,t,(t,class))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _dir, _fnames, label = self.data[idx]
        
        if self.ch==1:
            stack = []
            for f in _fnames:
                img_path = self.path + f"{_dir}/{f}"
                img = Image.open(img_path).resize((img_size,img_size))
                img = self.trans(img)
                stack.append(img)
            stack = np.stack(stack, axis=1)
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
            stack = np.stack(stack, axis=0)

        label = label.astype(np.bool)
        
        stack = torch.tensor(stack) # (t,ch,h,w)
        label = torch.tensor(label) # (t,class)
        return stack, label

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
    
    @staticmethod 
    def get_transform(ch=1):
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
        return train_trans, test_trans

class DatasetWrapper(object):

    def __init__(self, path, ch, bsize, valid_size=0.15, 
                    train_valid_split_pkl=None):
        self.path = path 
        self.ch = ch
        self.bsize = bsize
        self.valid_size = valid_size
        self.train_valid_split_pkl = train_valid_split_pkl 
    
    def get_dataloaders(self):
        # split train valid dirs
        if self.train_valid_split_pkl is None:
            dirs = os.listdir(self.path)
            np.random.shuffle(dirs)
            split = int(len(dirs)*(1-self.valid_size))
            train_dirs, valid_dirs = dirs[:split], dirs[split:]
        elif os.path.exists(self.train_valid_split_pkl):
            print("\t[Info] load pre split train valid set")
            ''''
            with open(self.train_valid_split_pkl, 'rb') as f:
                dirs = pickle.load(f)
                train_dirs, valid_dirs = dirs['train'], dirs['valid']
            '''
            with open(self.train_valid_split_pkl+"/train_set.pkl", 'rb') as f:
                train_dirs = pickle.load(f)
            with open(self.train_valid_split_pkl+"/valid_set.pkl", 'rb') as f:
                valid_dirs = pickle.load(f)
            print(f"\t\t len train:{len(train_dirs)} valid:{len(valid_dirs)}")
        else:
            raise FileNotFoundError(f"{self.train_valid_split_pkl} not exis")
        
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

