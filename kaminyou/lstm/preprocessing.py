from utils.dataset import ReHemorrhageDataset
from utils.models import HemoResNet18
from utils.util import get_pt_id_list
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch
import yaml
import pickle

if __name__ == "__main__":
    
    # config
    with open("./config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    preprocessing_config = config["PREPROCESSING"]

    train_id_list = get_pt_id_list(preprocessing_config["TRAIN_DATA_PATH"])
    test_id_list = get_pt_id_list(preprocessing_config["TEST_DATA_PATH"])

    val_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                            std=[0.5, 0.5, 0.5])
                                    ])
    train_dataset = ReHemorrhageDataset(train_id_list, data_root = preprocessing_config["TRAIN_DATA_PATH"], stack_img = True, mode="test",augmentation=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers = 16, pin_memory=True)

    test_dataset = ReHemorrhageDataset(test_id_list, data_root = preprocessing_config["TEST_DATA_PATH"], stack_img = True, mode="test",augmentation=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers = 16, pin_memory=True)

    if preprocessing_config["BACKBONE_MDOEL_NAME"] == "resnet18":
        model = HemoResNet18(in_channels = 3, n_classes=5)
        model.load_state_dict(torch.load(preprocessing_config["BACKBONE_MODEL_PATH"]))
        # to get embedding
        model_latent = nn.Sequential(*list(model.base_model.children())[:-1])
        model.base_model = model_latent
        #################
    model.to(preprocessing_config["DEVICE"])

    model.eval()
    train_embedding_dict = {}
    test_embedding_dict = {}
    with torch.no_grad():
        for i, (pt_name, img_name, data) in enumerate(train_loader, 1):
            print(f"Process {i} / {len(train_loader)}    ", end="\r")
            data = data.to(preprocessing_config["DEVICE"])

            embedding = np.squeeze(np.squeeze(model(data).cpu().numpy(), axis=-1), axis=-1)

            for s_name, s_img_name, s_embedding in zip(pt_name, img_name, embedding):
                if not s_name in train_embedding_dict:
                    train_embedding_dict[s_name] = {}
                train_embedding_dict[s_name][s_img_name] = s_embedding

        for i, (pt_name, img_name, data) in enumerate(test_loader, 1):
            print(f"Process {i} / {len(test_loader)}    ", end="\r")
            data = data.to(preprocessing_config["DEVICE"])

            embedding = np.squeeze(np.squeeze(model(data).cpu().numpy(), axis=-1), axis=-1)

            for s_name, s_img_name, s_embedding in zip(pt_name, img_name, embedding):
                if not s_name in test_embedding_dict:
                    test_embedding_dict[s_name] = {}
                test_embedding_dict[s_name][s_img_name] = s_embedding
    
    with open(preprocessing_config["TRAIN_EMBEDDING_SAVE"], 'wb') as handle:
        pickle.dump(train_embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(preprocessing_config["TEST_EMBEDDING_SAVE"], 'wb') as handle:
        pickle.dump(test_embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
