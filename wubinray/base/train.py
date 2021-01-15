from data_aug.dataset_wrapper import DatasetWrapper
from models.resnet import HemoResNet18, HemoResNet50, HemoResnext50 
from loss.focal_loss import AsymmetricLoss, AsymmetricLossOptimized
from optimizer.ranger2020 import Ranger 
from utils.others import Averager
from utils.args import parse_args 
from utils.warmup import WarmupScheduler 
from utils.metrics import hemorrhage_metrics
from utils.metrics import accuracy_multi as accuracy 

import os, sys
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import wandb 
import warnings
warnings.filterwarnings('ignore')

np.random.seed(987)
torch.manual_seed(987)
torch.cuda.manual_seed_all(987)

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
    if args.backbone=='resnet18':
        model = HemoResNet18(in_channels=args.ch,n_classes=5).to(args.device)
    elif args.backbone=='resnet50':
        model = HemoResNet50(in_channels=args.ch,n_classes=5).to(args.device)

    # loss
    pos_weight = torch.tensor([7.54, 11.22, 7.64, 5.07, 24.03])
    #loss_f = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(args.device)
    loss_f = AsymmetricLossOptimized(gamma_pos=0, gamma_neg=4, pos_weight=pos_weight)

    # optimizer & lr schedule
    #optimizer = optim.Adam(model.parameters(), args.lr, 
    #                    weight_decay=args.weight_decay)
    optimizer = Ranger(model.parameters(), args.lr)
    #optimizer = optim.SGD(model.parameters(), args.lr)

    step_after = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=15, 
                            eta_min=args.eta_min, last_epoch=-1)
    lr_scheduler = WarmupScheduler(optimizer, multiplier=1,
                                     total_epoch = args.warmup_epochs,
                                     after_scheduler = step_after)
    
    # ap_fix16
    use_fp16 = apex_support and args.fp16_precision
    if use_fp16:
        print(" Use fp16_precision")
        model, optimizer = amp.initialize(model, optimizer,
                opt_level='O2', keep_batchnorm_fp32=True, verbosity=0)

    # start training
    best_valid_f2 = 0
    for epoch in range(1,args.epochs+1):
        print(f" Epoch {epoch}")
        
        #if epoch > 0.7*args.epochs:
        lr_scheduler.step()
        
        train_loss, valid_loss, train_acc, train_recall, train_f2=\
                [Averager() for i in range(5)]
        
        # training
        model.train()
        
        for idx, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(args.device), lbls.to(args.device)
            
            preds = model(imgs)

            loss = loss_f(preds, lbls)

            optimizer.zero_grad()
            if use_fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            
            preds = torch.sigmoid(preds)
            metric = hemorrhage_metrics(preds.cpu().detach().numpy(),
                                        lbls.cpu().detach().numpy())
            train_loss.add(loss.item())
            train_acc.add(metric['acc'])
            train_recall.add(metric['recall'])
            train_f2.add(metric['f2'])
            #wandb.log({'train_loss':loss.item(), 'train_acc': metric['acc'],
            #    'train_recall': metric['recall'], 'train_f2': metric['f2']})
            print("\t[%d/%d] loss:%.2f acc:%.2f recall:%.2f f2:%.2f" % (
                    idx, len(train_loader), train_loss.item(), 
                    train_acc.item(), train_recall.item(), train_f2.item()),
                end='  \r')
        print("\t Train loss:%.4f, acc:%.3f, recall:%.3f, f2:%.3f" % (
            train_loss.item(), train_acc.item(), train_recall.item(), train_f2.item()))

        # validating
        model.eval()
       
        val_pred, val_lbls = [], []
        for idx, (imgs, lbls) in enumerate(valid_loader):
            imgs, lbls = imgs.to(args.device), lbls.to(args.device)
           
            with torch.no_grad():
                preds = model(imgs)

            loss = loss_f(preds, lbls)
           
            preds = torch.sigmoid(preds)
            val_pred.append(preds.cpu().detach().numpy())
            val_lbls.append(lbls.cpu().detach().numpy())

            valid_loss.add(loss.item())
            print("\t[%d/%d] loss:%.2f" % (
                idx, len(valid_loader), valid_loss.item()),
                end='  \r')
        
        val_metric = hemorrhage_metrics(np.concatenate(val_pred),
                                        np.concatenate(val_lbls))
        #wandb.log({'valid_loss':loss.item(), 'valid_acc': val_metric['acc'],
        #           'valid_recall': val_metric['recall'], 
        #           'valid_f2': val_metric['f2']})
        print("\t Valid loss:%.4f, acc:%.3f, recall:%.3f, f2:%.3f" % 
                (valid_loss.item(), val_metric['acc'], val_metric['recall'], val_metric['f2']))

        if val_metric['f2'] > best_valid_f2:
            best_valid_f2 = val_metric['f2']
            path = f"./checkpoints/{args.backbone}"
            os.system(f'mkdir -p {path}')
            torch.save(model.state_dict(),f"{path}/best.pth")
            print("\t save weight")


if __name__=='__main__':

    args = parse_args()

    #wandb.init(config=args, project="CT_Hemorrhage", name=f"base_ch{args.ch}")
    
    dataset = DatasetWrapper("/media/disk1/aa/Blood_data/train/", 
                             args.bsize, 
                             args.valid_size,
                             args.ch)

    train(args, dataset)


