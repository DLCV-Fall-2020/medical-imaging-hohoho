from data_aug.dataset_wrapper import DatasetWrapper
from models.model3d import HemoModel3d 
from loss.supconloss import SupConLoss
from utils.args import parse_args
from utils.warmup import WarmupScheduler
from utils.metrics import hemorrhage_metrics
from utils.others import Averager

import os,sys
import numpy as np 
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import wandb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(87)
torch.manual_seed(87)
torch.cuda.manual_seed_all(87)

def train(args, dataset):

    # dataloader 
    train_loader, valid_loader = dataset.get_dataloaders()
    
    # model
    model = HemoModel3d(args.backbone, args.ch,
                        args.hidd_dim, args.proj_dim).to(args.device)

    # loss
    supcon_loss = SupConLoss(args.temperature)

    # optimizer 
    optimizer = optim.SGD(model.parameters(), args.lr)

    # lr scheduler
    step_after = optim.lr_scheduler.CosineAnnealingLR(
                                optimizer, T_max=args.epochs, 
                                eta_min=args.eta_min, last_epoch=-1)
    lr_scheduler = WarmupScheduler(optimizer, multiplier=1,
                                total_epoch=args.warmup_epochs,
                                after_scheduler=step_after)

    # start training
    best_valid_loss = np.inf
    for epoch in range(1, args.epochs+1):
        print(f" Epoch {epoch}")

        lr_scheduler.step()

        train_loss, valid_loss = [Averager() for i in range(2)]
        
        # train
        model.train()
        for idx, (imgs, lbls, mask) in enumerate(train_loader):
            b,t = imgs.size(0),imgs.size(2)
            imgs = imgs.to(args.device)
            lbls = lbls.to(args.device)
            mask = mask.to(args.device)

            _, zs = model(imgs) #(b,t,hid)

            loss = supcon_loss(zs.view(b*t, -1),
                               lbls.view(b*t, -1),
                               mask.view(b*t, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.add(loss.item())
            print("\t[%d/%d] loss:%.2f"%(
                    idx+1,len(train_loader),train_loss.item()),
                end='  \r')
        print("\t Train Loss:%.4f "%(train_loss.item()))

        # valid
        model.eval()
        for idx, (imgs, lbls, mask) in enumerate(valid_loader):
            b,t = imgs.size(0),imgs.size(2)
            imgs = imgs.to(args.device)
            lbls = lbls.to(args.device)
            mask = mask.to(args.device)

            with torch.no_grad():
                _, zs = model(imgs) #(b,t,hid)

            loss = supcon_loss(zs.view(b*t, -1),
                               lbls.view(b*t, -1),
                               mask.view(b*t, 1))

            valid_loss.add(loss.item())
            print("\t[%d/%d] loss:%.2f"%(
                    idx+1,len(valid_loader),valid_loss.item()),
                end='  \r')
        print("\t Valid Loss:%.4f "%(valid_loss.item()))

if __name__=='__main__':
    args = parse_args()

    #wandb.init(config=args, project="CT_Hemorrhage", name=f"3D_Cnn_CLR")

    dataset = DatasetWrapper("/media/disk1/aa/Blood_data/train/",
                            args.bsize,
                            args.valid_size,
                            args.ch)
    train(args, dataset)
    
