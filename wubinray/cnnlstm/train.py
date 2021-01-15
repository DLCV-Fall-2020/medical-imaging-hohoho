from data_aug.dataset_wrapper import DatasetWrapper
from models.model import HemoCnnLstm 
#from loss.supconloss import SupConLoss
from loss.focal_loss import AsymmetricLossOptimized
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
 
    # loss
    #loss_f = AsymmetricLossOptimized(gamma_pos=0, gamma_neg=2, clip=0.01)
    pos_weight = torch.tensor([7.54, 11.22, 7.64, 5.07, 24.03]) / 1.5
    loss_f = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(args.device)
   
    # model
    model = HemoCnnLstm(args.backbone, args.n_classes, args.load_pretrained)
    model.to(args.device)
    for param in model.backbone.parameters():
        param.requires_grad = False 

    # optimizer 
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    plist = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]
    #optimizer = Ranger(plist, args.lr)
    optimizer = optim.Adam(plist, args.lr)
    #optimizer = Ranger(models.parameters(), args.lr)
    #optimizer = optim.Adam(models.parameters(), args.lr)

    # lr scheduler
    step_after = optim.lr_scheduler.CosineAnnealingLR(
                                optimizer, T_max=30, 
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
    best_valid_f2 = -np.inf
    for epoch in range(1, args.epochs+1):
        print(f" Epoch {epoch}")

        lr_scheduler.step()

        train_loss, valid_loss, train_acc, train_recall, train_precision,\
            train_f2 = [Averager() for i in range(6)]
        
        # tune backbone ?? 
        if epoch > args.warmup_epochs:
            for param in model.backbone.parameters():
                param.requires_grad = True 

        # train
        model.train()
        for idx, (imgs, lbls, mask) in enumerate(train_loader):
            b,t,cls = imgs.size(0),imgs.size(2),lbls.size(2)
            imgs = imgs.to(args.device)
            lbls = lbls.to(args.device)
            mask = mask.to(args.device)
            
            preds = model(imgs)
            
            preds = torch.masked_select(preds, mask).view(-1,cls)
            lbls = torch.masked_select(lbls, mask).view(-1,cls)
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
                                        lbls.cpu().detach().numpy(),
                                        args.threshold)
            
            train_loss.add(loss.item())
            train_acc.add(metric['acc'])
            train_precision.add(metric['precision'])
            train_recall.add(metric['recall'])
            train_f2.add(metric['f2'])
            wandb.log({'train_loss':loss.item(),'train_acc':metric['acc'],
                'train_recall':metric['recall'],'train_f2':metric['f2']})
            print("\t[%d/%d] loss:%.2f acc:%.2f prec:%.2f rec:%.2f f2:%.2f"%(
                   idx+1,len(train_loader),train_loss.item(),
                   train_acc.item(), train_precision.item(), 
                   train_recall.item(), train_f2.item()),
                end='  \r')
        print("\t Train Loss:%.4f acc:%.3f prec:%.3f rec:%.3f f2:%.3f"%(
            train_loss.item(), train_acc.item(), train_precision.item(), 
            train_recall.item(), train_f2.item()))

        # valid
        model.eval()
        val_pred, val_lbls = [], []
        for idx, (imgs, lbls, mask) in enumerate(valid_loader):
            b,t,cls = imgs.size(0),imgs.size(2),lbls.size(2)
            imgs = imgs.to(args.device)
            lbls = lbls.to(args.device)
            mask = mask.to(args.device)
           
            with torch.no_grad():
                preds = model(imgs)
            
            preds = torch.masked_select(preds, mask).view(-1,cls)
            lbls = torch.masked_select(lbls, mask).view(-1,cls)
            loss = loss_f(preds, lbls.float())

            preds = torch.sigmoid(preds)
            val_pred.append(preds.cpu().detach().numpy())
            val_lbls.append(lbls.cpu().detach().numpy())
            valid_loss.add(loss.item())
            print("\t[%d/%d] loss:%.2f"%(
                    idx+1,len(valid_loader),valid_loss.item()),
                end='  \r')

        metric = hemorrhage_metrics(np.concatenate(val_pred),
                                    np.concatenate(val_lbls),
                                    args.threshold)
        wandb.log({'valid_loss':loss.item(),'valid_acc':metric['acc'],
             'valid_recall':metric['recall'],'valid_f2':metric['f2']})
        print("\t Valid Loss:%.4f acc:%.3f prec:%.3f rec:%.3f f2:%.3f"%(
                valid_loss.item(), metric['acc'], metric['precision'],
                metric['recall'], metric['f2']))

        if metric['f2'] > best_valid_f2:
            best_valid_f2 = metric['f2']
            path=f"./checkpoints/{args.backbone}"
            os.system(f"mkdir -p {path}")
            torch.save(model.state_dict(), f"{path}/best.pth")
            print("\t save weight")

if __name__=='__main__':
    args = parse_args()

    wandb.init(config=args, project="CT_Hemorrhage", name=f"CNN_LSTM_fine-tune")

    dataset = DatasetWrapper("/media/disk1/aa/Blood_data/train/",
                            args.bsize,
                            args.valid_size,
                            args.train_valid_split_pkl)
    train(args, dataset)
    
