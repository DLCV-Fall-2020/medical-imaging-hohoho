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

class BloodDataset_Test(Dataset):
    def __init__(self, path, ch=1):
        self.path = path
        self.ch = ch 
        self.trans = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*self.ch,std=[0.5]*self.ch)
            ]) 
        
        self.data = []
        self.dirs = glob.glob(path+"/*")
        for _p_dir in sorted(self.dirs):
            _id = _p_dir.replace(self.path, "").lstrip("/")
            imgs = glob.glob(_p_dir+"/*.jpg")
            for img in sorted(imgs):
                _dir = _p_dir.replace(path,"") # dir ex:ID_3bffxae
                _fname = img.replace(_p_dir, "").lstrip("/") # fname ex:3bffxae_3.jpg
                _uid = _fname[:_fname.find('_')] # uid ex:3bffxae
                _idx = int(_fname[_fname.find('_')+1:_fname.find('.jpg')]) # idx ex:3
                self.data.append((_dir, _fname, _uid, _idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _dir, _fname, _uid, _idx = self.data[idx]
        if self.ch==1:
            img_path = os.path.join(self.path,_dir,_fname)
            img = Image.open(img_path).resize((img_size,img_size))
            img = self.trans(img)
            return img, _dir, _fname
        else:
            stack = []
            for i in range(self.ch):
                img_path = os.path.join(self.path,_dir,f"{_uid}_{_idx+i-self.ch//2}.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.path,_dir,_fname)
                img = Image.open(img_path).resize((img_size,img_size))
                img = np.array(img).astype(np.uint8)
                stack.append(img)
            stack = np.stack(stack, axis=-1)
            img = Image.fromarray(stack)
            img = self.trans(img)
            return img, _dir, _fname

class BloodDataset(Dataset):
    def __init__(self, path, dirs, trans, ch=1):
        assert ch%2==1

        df = pd.read_csv(path.rstrip('/')+".csv")
        
        self.path = path
        self.dirs = dirs 
        self.trans = trans 
        self.ch = ch 
        
        self.data = []
        self.label = []
        for _dir in dirs:
            sub_df = df.loc[df['dirname']==_dir]
            for row in sub_df.to_numpy():
                _dir, _fname = row[:2]
                self.data.append((_dir, # ex: ID_3bfxedafae
                                  _fname, # ex: 3bfxedafae_3.jpg
                                  _fname[:_fname.find('_')], # ex:3bfxedafae
                                  int(_fname[_fname.find('_')+1:_fname.find('.jpg')])) # ex:3
                                )
                self.label.append(row[2:])
        self.label = np.stack(self.label, axis=0)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _dir, _fname, _uid, _idx = self.data[idx]
        if self.ch==1:
            img_path = os.path.join(self.path,_dir,f"{_uid}_{_idx}.jpg")
            img = Image.open(img_path).resize((img_size,img_size))
            img = self.trans(img)
            return img, self.label[idx].astype(np.bool)
        else:
            stack = []
            for i in range(self.ch):
                img_path = os.path.join(self.path,_dir,f"{_uid}_{_idx+i-self.ch//2}.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.path,_dir,f"{_uid}_{_idx}.jpg")
                img = Image.open(img_path).resize((img_size,img_size))
                img = np.array(img).astype(np.uint8)
                stack.append(img)
            stack = np.stack(stack, axis=-1)
            img = Image.fromarray(stack)
            img = self.trans(img)
            return img, self.label[idx].astype(np.bool)

    def collate_fn(self, samples):
        batch_imgs, batch_lbls = [],[]
        for img, lbl in samples:
            batch_imgs.append(img.unsqueeze(0))
            batch_lbls.append(lbl)
        batch_imgs = torch.cat(batch_imgs,dim=0)
        batch_lbls = torch.tensor(batch_lbls).float()
        return batch_imgs, batch_lbls
    
    @staticmethod 
    def get_transform(ch=1):
        train_trans = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomApply([
                #        transforms.ColorJitter()
                #    ],p=0.1),
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
        split = int(len(dirs) * (1-self.valid_size))
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

