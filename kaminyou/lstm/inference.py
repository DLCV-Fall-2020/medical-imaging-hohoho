from utils.util import get_pt_id_list, prediction_to_csv, img_id_to_pt_id
from torch.utils.data import DataLoader
from utils.dataset import LSTMHemorrhageDataset, collatefn
from utils.models import HemoLSTMBasic
from utils.split import RandomSplitPt
from utils.metric import hemorrhage_metrics
import yaml
import numpy as np
import torch
import torch.nn as nn

if __name__ == "__main__":
    # read config
    with open("./config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    testing_config = config["LSTM_INFERENCE"]

    test_id_list = get_pt_id_list(testing_config["TEST_DATA_PATH"])
    test_dataset = LSTMHemorrhageDataset(pt_id_list = test_id_list, embedding_root = testing_config["TEST_EMBEDDING"], mode = "test")
    test_loader = DataLoader(test_dataset, batch_size=testing_config["BATCH_SIZE"],collate_fn=collatefn, shuffle=False, num_workers = 16, pin_memory=True)

    n_classes = testing_config["N_CLASS"]
    device = testing_config["DEVICE"]
    model = HemoLSTMBasic(embed_size=512)
    model.load_state_dict(torch.load(testing_config["MODEL_PATH"]))
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
                      output_csv_name=testing_config["OUTPUT_CSV"], 
                      threshold=testing_config["THRESHOLD"], to_kaggle=True, remove_defunct=True)