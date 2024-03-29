import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import yaml
import pickle5 as pickle
from datetime import datetime
from utils.util import get_pt_id_list, get_weight
from utils.dataset import LSTMHemorrhageDataset, collatefn, collatefn_end_to_end
from utils.models import HemoLSTMBasic, HemoResNet18
from utils.split import RandomSplitPt
from utils.metric import hemorrhage_metrics
from utils.draw import draw_compare_train_val, draw_metric

if __name__ == "__main__":
    # init experiment
    now = datetime.now()
    experiment_root = "./experiment"
    experiment_id = now.strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(experiment_root, experiment_id)
    os.makedirs(experiment_dir)

    # read config
    with open("./config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    training_config = config["LSTM_TRAINING"]

    # copy config
    with open(os.path.join(experiment_dir, "config.yaml"), 'w') as f:
        documents = yaml.dump(config, f)

    # split set
    if training_config["FIX_SPLIT"]:
        with open(training_config["TRAIN_SET_PATH"], "rb" ) as f: 
            train_pt = np.array(pickle.load(f))
        with open(training_config["VAL_SET_PATH"], "rb" ) as f: 
            val_pt = np.array(pickle.load(f))
    else:
        random_split_pt = RandomSplitPt(train_data_root = training_config["TRAIN_DATA_PATH"])
        train_pt, val_pt = random_split_pt.randomly_split(test_size = 0.1)
    
    if training_config["END2END"]:
        collate_fn = collatefn_end_to_end
    else:
        collate_fn = collatefn
        
    train_dataset = LSTMHemorrhageDataset(pt_id_list = train_pt, 
                                          embedding_root = training_config["TRAIN_EMBEDDING"], 
                                          label_csv_path = training_config["TRAIN_LABEL_CSV"], 
                                          mode = "train",
                                          end2end = training_config["END2END"],
                                          data_root = training_config["IMG_ROOT"])

    train_loader = DataLoader(train_dataset, batch_size=training_config["BATCH_SIZE"],collate_fn=collate_fn, shuffle=True, num_workers = 16, pin_memory=True)

    val_dataset = LSTMHemorrhageDataset(pt_id_list = val_pt, 
                                        embedding_root = training_config["TRAIN_EMBEDDING"], 
                                        label_csv_path = training_config["TRAIN_LABEL_CSV"], 
                                        mode = "val",
                                        end2end = training_config["END2END"],
                                        data_root = training_config["IMG_ROOT"])
    val_loader = DataLoader(val_dataset, batch_size=training_config["BATCH_SIZE"],collate_fn=collate_fn, shuffle=False, num_workers = 16, pin_memory=True)

    n_classes = training_config["N_CLASS"]
    device = training_config["DEVICE"]
    weight = get_weight(train_csv_path=training_config["TRAIN_LABEL_CSV"], pt_restriction=train_pt)
    pos_weight = torch.Tensor(weight).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = HemoLSTMBasic(embed_size=training_config["LSTM_EMBEDDING_SIZE"], LSTM_UNITS=training_config["LSTM_UNITS"])

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    plist = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': training_config["DECAY"]},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    if training_config["OPTIMIZER"] == "Adam":
        optimizer = torch.optim.Adam(plist, lr=training_config["LEARNING_RATE"])
    elif training_config["OPTIMIZER"] == "SGD":
        optimizer = torch.optim.SGD(plist, lr=training_config["LEARNING_RATE"], momentum=0.9, nesterov=True)
    
    if training_config["SCHEDULER"]:
        scheduler = StepLR(optimizer, 1, gamma=training_config["LR_GAMMA"], last_epoch=-1)

    model.to(device)

    # init cnn_backbone if end2end 
    if training_config["END2END"]:
        if training_config["CNN_BACKBONE_TYPE"] == "resnet18":
            cnn_model = HemoResNet18(in_channels = 3, n_classes=5)
            cnn_model.load_state_dict(torch.load(training_config["CNN_BACKBONE_PATH"]))
            # to get embedding
            model_latent = nn.Sequential(*list(cnn_model.base_model.children())[:-1])
            cnn_model.base_model = model_latent
            #################
        cnn_model.to(training_config["DEVICE"])
        cnn_model.eval()

    # start training
    best_val_f2 = 0
    records = []
    for epoch in range(1, training_config["MAX_EPOCH"] + 1):
        model.train()
        train_pred = []
        train_true = []
        train_loss_accum = 0
        for i, batch in enumerate(train_loader, 1):
            print(f"Process {i} / {len(train_loader)}    ", end="\r")
            data = batch["embeddings"].to(device, dtype=torch.float)

            if training_config["END2END"]:
                # (PT, CT_LEN, 3, 512, 512) -> (PT, CT_LEN, 512)
                with torch.no_grad():
                    pts_embedding = []
                    for one_pt_data in data:
                        embeddings = cnn_model(one_pt_data)
                        pts_embedding.append(torch.squeeze(torch.squeeze(embeddings,-1),-1))
                    data = torch.stack(pts_embedding)

            label = batch["labels"].to(device, dtype=torch.float)
            mask = batch['mask'].to(device, dtype=torch.int)
        
            logits = model(data)
            
            maskidx = mask.view(-1) == 1
            label = label.view(-1, n_classes)[maskidx]
            logits = logits.view(-1, n_classes)[maskidx]
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            train_pred.append(torch.sigmoid(logits).cpu().detach().numpy())
            train_true.append(label.cpu().numpy())

        if training_config["SCHEDULER"]:
            scheduler.step()

        model.eval()
        val_pred = []
        val_true = []
        val_loss_accum = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader, 1):
                print(f"Process {i} / {len(val_loader)}    ", end="\r")
                data = batch["embeddings"].to(device, dtype=torch.float)

                if training_config["END2END"]:
                    # (PT, CT_LEN, 3, 512, 512) -> (PT, CT_LEN, 512)
                    pts_embedding = []
                    for one_pt_data in data:
                        embeddings = cnn_model(one_pt_data)
                        pts_embedding.append(torch.squeeze(torch.squeeze(embeddings,-1),-1))
                    data = torch.stack(pts_embedding)

                label = batch["labels"].to(device, dtype=torch.float)
                mask = batch['mask'].to(device, dtype=torch.int)
        
                logits = model(data)
            
                maskidx = mask.view(-1) == 1
                label = label.view(-1, n_classes)[maskidx]
                logits = logits.view(-1, n_classes)[maskidx]
                loss = criterion(logits, label)
                
                val_loss_accum += loss.item()
                val_pred.append(torch.sigmoid(logits).cpu().detach().numpy())
                val_true.append(label.cpu().numpy())
        
        train_pred = np.concatenate((train_pred))
        train_true = np.concatenate((train_true))
        val_pred = np.concatenate((val_pred))
        val_true = np.concatenate((val_true))
        
        train_metric = hemorrhage_metrics(train_pred, train_true)
        val_metric = hemorrhage_metrics(val_pred, val_true)
        
        if val_metric['f2'] > best_val_f2:
            best_val_f2 = val_metric['f2']
            print(F"BEST AT epoch {epoch:3d} || VAL F2 = {val_metric['f2']:.4f}")
            torch.save(model.state_dict(), os.path.join(experiment_dir, "lstm_best.pth"))
            
            if epoch >= 5:
                draw_metric(train_pred, train_true, val_pred, val_true, metric="acc", save=os.path.join(experiment_dir, f"checkpoint_{epoch}_acc.png"))
                draw_metric(train_pred, train_true, val_pred, val_true, metric="f2", save=os.path.join(experiment_dir, f"checkpoint_{epoch}_{train_metric['f2']:.4f}_{best_val_f2:.4f}_f2.png"))
                draw_metric(train_pred, train_true, val_pred, val_true, metric="recall", save=os.path.join(experiment_dir, f"checkpoint_{epoch}_recall.png"))
                draw_metric(train_pred, train_true, val_pred, val_true, metric="precision", save=os.path.join(experiment_dir, f"checkpoint_{epoch}_precision.png"))
        
        records.append([train_metric['acc'], train_metric['f2'], train_loss_accum, val_metric['acc'], val_metric['f2'], val_loss_accum])
        print(f"[epoch {epoch:3d}] TRAIN acc {train_metric['acc']:.4f} f2 {train_metric['f2']:.4f} loss {train_loss_accum:.4f} || VAL acc {val_metric['acc']:.4f} f2 {val_metric['f2']:.4f} loss {val_loss_accum:.4f}")
    
    records = np.array(records)
    draw_compare_train_val(records[:,0], records[:,3], save=os.path.join(experiment_dir, f"overall_acc.png"))
    draw_compare_train_val(records[:,1], records[:,4], ylabel="f2", title="f2", save=os.path.join(experiment_dir, f"overall_f2.png"))
    draw_compare_train_val(records[:,2], records[:,5], ylabel="Loss", title="Loss", save=os.path.join(experiment_dir, f"overall_loss.png"))




