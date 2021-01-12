from data_aug.dataset_wrapper import DatasetWrapper
from models.model import HemoCnnLstm 
#from loss.supconloss import SupConLoss
from loss.focal_loss import AsymmetricLossOptimized
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
    model = HemoCnnLstm(args.backbone, args.t, n_classes=args.n_classes)
    model.to(args.device)

    # loss
    pos_weight = torch.tensor([7.54, 11.22, 7.64, 5.07, 24.03])
    loss_f = AsymmetricLossOptimized(gamma_pos=0, gamma_neg=4, 
                                  pos_weight=pos_weight)
    #loss_f = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(args.device)

    # optimizer 
    optimizer1 = optim.AdamW(model.parameters(), args.lr, 
                            weight_decay=args.weight_decay)
    #optimizer1 = optim.Adam(model.parameters(), args.lr, 
    #                        weight_decay=args.weight_decay)
    #optimizer2 = optim.SGD(model.parameters(), args.lr,
    #                        args.momentum)
    optimizer2 = optimizer1 
    optimizer = optimizer1

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
        print(" [Info] Use fp16_precision")
        model, optimizer = amp.initialize(model, optimizer,
                opt_level='O2', keep_batchnorm_fp32=True, verbosity=0)
    
    # start training
    best_valid_f2 = -np.inf
    for epoch in range(1, args.epochs+1):
        print(f" Epoch {epoch}")

        lr_scheduler.step()

        train_loss, valid_loss, train_acc, train_recall, train_f2 =\
                    [Averager() for i in range(5)]

        # change optimizer stage policy
        if epoch > args.epochs*0.7:
            lr_scheduler.after_scheduler.optimizer = optimizer2
        
        # train
        model.train()
        for idx, (imgs, lbls) in enumerate(train_loader):
            b,t = imgs.size(0),imgs.size(2)
            imgs = imgs.to(args.device)
            lbls = lbls.to(args.device)
           
            preds = model(imgs)
            
            loss = loss_f(preds, lbls.float())

            optimizer.zero_grad()
            if use_fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            
            preds = torch.sigmoid(preds)
            metric = hemorrhage_metrics(preds.cpu().detach().numpy(),
                                        lbls.cpu().detach().numpy())
            
            train_loss.add(loss.item())
            train_acc.add(metric['acc'])
            train_recall.add(metric['recall'])
            train_f2.add(metric['f2'])
            #wandb.log({'train_loss':loss.item(),'train_acc':metric['acc'],
            #    'train_recall':metric['recall'],'train_f2':metric['f2']})
            print("\t[%d/%d] loss:%.2f acc:%.2f recall:%.2f f2:%.2f"%(
                   idx+1,len(train_loader),train_loss.item(),
                   train_acc.item(), train_recall.item(), train_f2.item()),
                end='  \r')
        print("\t Train Loss:%.4f acc:%.3f recall:%.3f f2:%.3f"%(
            train_loss.item(), train_acc.item(), train_recall.item(),
            train_f2.item()))

        # valid
        model.eval()
        val_pred, val_lbls = [], []
        for idx, (imgs, lbls) in enumerate(valid_loader):
            b,t = imgs.size(0),imgs.size(2)
            imgs = imgs.to(args.device)
            lbls = lbls.to(args.device)

            with torch.no_grad():
                preds = model(imgs)

            loss = loss_f(preds, lbls.float())
            
            preds = torch.sigmoid(preds)
            val_pred.append(preds.cpu().detach().numpy())
            val_lbls.append(lbls.cpu().detach().numpy())
            valid_loss.add(loss.item())
            print("\t[%d/%d] loss:%.2f"%(
                    idx+1,len(valid_loader),valid_loss.item()),
                end='  \r')

        metric = hemorrhage_metrics(np.concatenate(val_pred),
                                      np.concatenate(val_lbls))
        #wandb.log({'valid_loss':loss.item(),'valid_acc':metric['acc'],
        #    'valid_recall':metric['recall'],'valid_f2':metric['f2']})
        print("\t Valid Loss:%.4f acc:%.3f recall:%.3f f2:%.3f"%(
                valid_loss.item(), metric['acc'], metric['recall'], 
                metric['f2']))

        if metric['f2'] > best_valid_f2:
            best_valid_f2 = metric['f2']
            path=f"./checkpoints/{args.backbone}"
            os.system(f"mkdir -p {path}")
            torch.save(model.state_dict(), f"{path}/best.pth")
            print("\t save weight")

if __name__=='__main__':
    args = parse_args()

    #wandb.init(config=args, project="CT_Hemorrhage", name=f"CNN_LSTM")

    dataset = DatasetWrapper("/media/disk1/aa/Blood_data/train/",
                            args.bsize,
                            args.valid_size,
                            args.ch,
                            args.t)
    train(args, dataset)
    
