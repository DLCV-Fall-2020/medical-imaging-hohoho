from data_aug.dataset_wrapper import BloodDataset_Test, DataLoader
from models.resnet import HemoResNet18

import csv
import argparse
import numpy as np
import pandas as pd 

import torch 
import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--ch', type=int, default=3,
                    help='input channels how many picture')
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

    test_dataset = BloodDataset_Test(path="/media/disk1/aa/Blood_data/test/", ch=args.ch)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers = 8)
    
    model = HemoResNet18(in_channels=args.ch, n_classes=5)
    model.load_state_dict(torch.load(args.model_path,map_location='cpu'))
    model.eval().to(args.device)

    pt_name_list, image_name_list = [], []
    prediction = []
        
    for idx, (img, _dir, _fname) in enumerate(test_loader):

        logits = model(img.to(args.device))
        logits = torch.sigmoid(logits)
        
        pt_name_list += _dir
        image_name_list += _fname 
        prediction.append(logits.cpu().numpy())
        
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

