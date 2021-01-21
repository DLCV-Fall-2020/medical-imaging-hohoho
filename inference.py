from src.util import get_pt_id_list, prediction_to_csv, img_id_to_pt_id
from torch.utils.data import DataLoader
from src.dataset import LSTMHemorrhageDataset, collatefn, collatefn_end_to_end
from src.models import HemoLSTMBasic, HemoResNet18
from src.split import RandomSplitPt
from src.metric import hemorrhage_metrics
import numpy as np
import torch
import torch.nn as nn
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='testing data folder path', type=str)
    parser.add_argument('--embedding', help='embedding pkl path', type=str)
    parser.add_argument('--device', help='device', type=str, default="cuda")
    parser.add_argument('--lstm', help='path to lstm model', type=str, default="./hemo_lstm_best_0.78162.pth")
    parser.add_argument('--threshold', help='threshold', type=float, default=0.45)
    parser.add_argument('--output_csv', help='path of the output scv', type=str)
    args = parser.parse_args()

    
    collate_fn = collatefn

    test_id_list = get_pt_id_list(args.data)
    test_dataset = LSTMHemorrhageDataset(pt_id_list = test_id_list, embedding_root = args.embedding, mode = "test")
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn, shuffle=False, num_workers = 16, pin_memory=True)

    n_classes = 5
    device = args.device
    model = HemoLSTMBasic(embed_size=512, LSTM_UNITS=32)
    model.load_state_dict(torch.load(args.lstm))
    model.to(device)

    model.eval()
    pt_name_list = []
    image_name_list = []
    prediction = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader, 1):
            print(f"Process {i} / {len(test_loader)}    ", end="\r")
            data = batch["embeddings"].to(device, dtype=torch.float)

            mask = batch['mask'].to(device, dtype=torch.int)

            logits = model(data)

            maskidx = mask.view(-1) == 1
            logits = logits.view(-1, n_classes)[maskidx]
            logits = torch.sigmoid(logits)
            logits = logits.cpu().numpy()
            
            prediction.append(logits)
            pt_name_list += list(map(img_id_to_pt_id, np.array(batch["img_ids"]).flatten()[maskidx.cpu().numpy()]))
            image_name_list += list(np.array(batch["img_ids"]).flatten()[maskidx.cpu().numpy()])
    prediction = np.concatenate((prediction))

    prediction_to_csv(pt_name_list, image_name_list, prediction, 
                      output_csv_name=args.output_csv, 
                      threshold=args.threshold, to_kaggle=True, remove_defunct=True)