from data_aug.gaussian_blur import GaussianBlur

#####
from PIL import Image, ImageOps

import os
import glob 
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

class BloodDataset(Dataset):
    def __init__(self, path, dirs, ch=3):
        self.path = path 
        self.ch = ch 
        self.trans = SupContrastTrans(ch)

        df = pd.read_csv(path.rstrip("/")+".csv")
        
        self.data = []
        
        for _dir in dirs: 
            _fnames = os.listdir(_dir)
            _fnames = [(int(f[f.find('_')+1:f.find('.jpg')]), f)
                        for f in _fnames]
            _fnames = sorted(_fnames, key=lambda x:x[0])
            _fnames = [f[1] for f in _fnames]
           
            sub_df = df.loc[df['dirname']==_dir.replace(path,"").replace("/","")]

            for i in range(0, len(_fnames)-self.ch):
                _fnm = _fnames[i:i+self.ch]
                _lbl = sub_df.loc[sub_df['ID'].isin(_fnm)].to_numpy()[:,-5:]
                _lbl = (_lbl[0]*0.6 + _lbl[1]*1 + _lbl[2]*0.6)/2.2
                _lbl = _lbl.astype(np.int)
                self.data.append((_dir,
                                  _fnm,
                                  _lbl)) # (1,ch,(ch,5))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _dir, _fnames, _lbl = self.data[idx]
        stack = []
        for i in range(self.ch):
            img = Image.open(f"{_dir}/{_fnames[i]}")
            img = img.resize((img_size,img_size))
            img = np.array(img).astype(np.uint8)
            stack.append(img)
        stack = np.stack(stack, axis=-1)
        img = Image.fromarray(stack)
        
        img = self.trans(img)
        return torch.tensor(img), torch.tensor(_lbl)

    def collate_fn(self, samples):
        batch_imgs, batch_lbls = [], []
        for img,lbl in samples:
            batch_imgs.append(img)
            batch_lbls.append(lbl)

        batch_imgs = torch.stack(batch_imgs, dim=0) # (b,ch,h,w)
        batch_lbls = torch.stack(batch_lbls, dim=0) # (b,5)
        return batch_imgs, batch_lbls 
    
class DatasetWrapper(object):

    def __init__(self, path, bsize, valid_size=0.15, ch=3):
        assert 0.0<valid_size and valid_size<1.0
        self.path = path 
        self.bsize = bsize
        self.valid_size = valid_size
        self.ch = ch 
    
    def get_dataloaders(self):
        # split train dirs
        train_dirs = os.listdir(self.path+"/train")
        train_dirs = [self.path+"/train/"+d for d in train_dirs]
        
        #test_dirs = os.listdir(self.path+"/test")
        #test_dirs = [self.path+"/test/"+d for d in test_dirs]
        
        dirs = train_dirs # + test_dirs
        np.random.shuffle(dirs)
        split = int(len(dirs)*(1-self.valid_size))
        train_dirs, valid_dirs = dirs[:split], dirs[split:]
        
        # dump train valid split set 
        print("\t[Info] dump train valid set")
        os.system("mkdir -p ./checkpoints")
        with open("./checkpoints/train_set.pkl", "wb") as f:
            pickle.dump(train_dirs, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open("./checkpoints/valid_set.pkl", "wb") as f:
            pickle.dump(valid_dirs, f, protocol=pickle.HIGHEST_PROTOCOL)
    
        # dataset
        train_dataset = BloodDataset(self.path+"/train", train_dirs, 
                                     self.ch)
        valid_dataset = BloodDataset(self.path+"/train", valid_dirs, 
                                     self.ch)
        
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

class SupContrastTrans(object):
    def __init__(self, ch=3):
        self.trans = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomRotation(45, fill=(0,)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply(
                        [transforms.ColorJitter(0.1,0.1,0.1,0)]
                    ,p=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*ch,
                                     std=[0.5]*ch)
            ])

    def __call__(self, sample):
        xi = self.trans(sample)
        return xi

