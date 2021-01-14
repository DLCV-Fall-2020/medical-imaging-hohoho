from data_aug.dataset_wrapper import BloodDataset_Test, DataLoader
from models.model import HemoCnnLstm

import csv
import argparse
import numpy as np
import pandas as pd 

import torch 
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--ch', type=int, default=1,
                    help='input channels how many picture')
parser.add_argument('--n_classes', type=int, default=5,
                    help='n classes')
parser.add_argument('--backbone', type=str, default='resnet18',
                    help='backbone used')
parser.add_argument('--model_path', type=str, default='./checkpoints/resnet18/best.pth',
                    help='trained model pth path')
parser.add_argument('--pred_csv_path', type=str, default='./pred.csv',
                    help='predicted csv path')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='model predict threshold')
parser.add_argument('--device', type=str, default='cuda:1',
                    help='cuda:0, cuda:1, cpu')
args = parser.parse_args()

@torch.no_grad()
def inference():
    
    test_trans = BloodDataset_Test.get_transform()
    test_dataset = BloodDataset_Test("/media/disk1/aa/Blood_data/test/",
                                    test_trans)
    test_loader = DataLoader(test_dataset, batch_size=4, 
                             collate_fn = test_dataset.collate_fn, 
                             num_workers = 5, shuffle=False)
    
    model = HemoCnnLstm(args.backbone, args.n_classes)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.to(args.device)
    model.eval()

    pt_name_list, image_name_list = [], []
    prediction = []
        
    for idx, (img, mask, _dir, _fnames) in enumerate(test_loader):
        bsize = img.size(0)

        logits = model(img.to(args.device))
        logits = torch.sigmoid(logits)
        
        for b in range(bsize):
            pt_name_list += _dir[b]
            image_name_list += _fnames[b]
            prediction.append(logits.cpu().numpy()[b][:len(_fnames[b])])
        
        print(f"\t[{idx+1}/{len(test_loader)}]", end=' \r')
    
    prediction = np.concatenate(prediction)
    prediction = np.array(prediction > args.threshold, dtype=int)
    df = pd.DataFrame({"dirname":pt_name_list,"ID":image_name_list,
                        "ich":prediction[:,0],"ivh":prediction[:,1],
                        "sah":prediction[:,2],"sdh":prediction[:,3],
                        "edh":prediction[:,4]})
    df.to_csv(args.pred_csv_path, index=False)

if __name__=='__main__':
    inference()

