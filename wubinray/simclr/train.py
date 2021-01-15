from data_aug.dataset_wrapper import DatasetWrapper
#from models.model3d import HemoModel3d 
from models.resnet import HemoResNet18
#from loss.supconloss import SupConLoss
from loss.nt_xent import NTXentLoss 
from optimizer.ranger2020 import Ranger
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

try:
    from apex import amp
    apex_support = True
except:
    print("\t[Info] apex is not supported")
    apex_support = False 

def train(args, dataset):

    # dataloader 
    train_loader, valid_loader = dataset.get_dataloaders()
    
    # model
    model = HemoResNet18(args.ch, args.proj_dim)
    model.to(args.device)

    # loss
    ntxent_loss = NTXentLoss(args.device, args.bsize, args.temperature)

    # optimizer 
    #optimizer = optim.SGD(model.parameters(), args.lr)
    optimizer = Ranger(model.parameters(), args.lr)

    # lr scheduler
    step_after = optim.lr_scheduler.CosineAnnealingLR(
                                optimizer, T_max=args.epochs, 
                                eta_min=args.eta_min, last_epoch=-1)
    lr_scheduler = WarmupScheduler(optimizer, multiplier=1,
                                total_epoch=args.warmup_epochs,
                                after_scheduler=step_after)
    # ap_fix16
    use_fp16 = apex_support and args.fp16_precision
    if use_fp16:
        print("\t[Info] Use fp16_precision")
        model, optimizer = amp.initialize(model, optimizer,
                opt_level='O2', keep_batchnorm_fp32=True, verbosity=0)

    # start training
    best_valid_loss = np.inf
    for epoch in range(1, args.epochs+1):
        print(f" Epoch {epoch}")

        lr_scheduler.step()

        train_loss, valid_loss = [Averager() for i in range(2)]
        
        # train
        model.train()
        for idx, (imgs1, imgs2) in enumerate(train_loader):
            b = imgs1.size(0)
            imgs1 = imgs1.to(args.device)
            imgs2 = imgs2.to(args.device)
            
            zis, zjs = model(imgs1), model(imgs2) # (b,z)
            zis, zjs = F.normalize(zis,2,1), F.normalize(zjs,2,1)

            loss = ntxent_loss(zis, zjs)

            optimizer.zero_grad()
            if use_fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            train_loss.add(loss.item())
            print("\t[%d/%d] loss:%.2f"%(
                    idx+1,len(train_loader),train_loss.item()),
                end='  \r')
        print("\t Train Loss:%.4f "%(train_loss.item()))

        # valid
        model.eval()
        for idx, (imgs1, imgs2) in enumerate(train_loader):
            b = imgs1.size(0)
            imgs1 = imgs1.to(args.device)
            imgs2 = imgs2.to(args.device)
            
            with torch.no_grad():
                zis, zjs = model(imgs1), model(imgs2) # (b,z)
                zis, zjs = F.normalize(zis,2,1), F.normalize(zjs,2,1)

            loss = ntxent_loss(zis, zjs)

            valid_loss.add(loss.item())
            print("\t[%d/%d] loss:%.2f"%(
                    idx+1,len(valid_loader),valid_loss.item()),
                end='  \r')
        print("\t Valid Loss:%.4f "%(valid_loss.item()))

        if valid_loss.item() < best_valid_loss:
            best_valid_loss = valid_loss.item()
            path=f"./checkpoints/{args.backbone}"
            os.system(f"mkdir -p {path}")
            torch.save(model.state_dict(), f"{path}/best.pth")
            print("\t save weight")

if __name__=='__main__':
    args = parse_args()

    #wandb.init(config=args, project="CT_Hemorrhage", name=f"SimCLR")

    dataset = DatasetWrapper("/media/disk1/aa/Blood_data/",
                            args.bsize,
                            args.valid_size,
                            args.ch)
    train(args, dataset)
    
