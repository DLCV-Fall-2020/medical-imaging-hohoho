from src.dataset import ReHemorrhageDataset
from src.models import HemoResNet18
from src.util import get_pt_id_list
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch
import pickle
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='testing data folder path', type=str)
    parser.add_argument('--backbone', help='backbone model path', default="./hemo_resnet18_0.75001.pth", type=str)
    parser.add_argument('--device', help='device', type=str, default="cuda")
    parser.add_argument('--save', help='path to save embedding', type=str)
    args = parser.parse_args()

    test_id_list = get_pt_id_list(args.data)

    val_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                            std=[0.5, 0.5, 0.5])
                                    ])
    test_dataset = ReHemorrhageDataset(test_id_list, data_root = args.data, stack_img = True, mode="test",augmentation=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers = 16, pin_memory=True)

    model = HemoResNet18(in_channels = 3, n_classes=5)
    model.load_state_dict(torch.load(args.backbone))
    # to get embedding
    model_latent = nn.Sequential(*list(model.base_model.children())[:-1])
    model.base_model = model_latent
    #################
    model.to(args.device)

    model.eval()
    test_embedding_dict = {}
    with torch.no_grad():
        for i, (pt_name, img_name, data) in enumerate(test_loader, 1):
            print(f"Process {i} / {len(test_loader)}    ", end="\r")
            data = data.to(args.device)

            embedding = np.squeeze(np.squeeze(model(data).cpu().numpy(), axis=-1), axis=-1)

            for s_name, s_img_name, s_embedding in zip(pt_name, img_name, embedding):
                if not s_name in test_embedding_dict:
                    test_embedding_dict[s_name] = {}
                test_embedding_dict[s_name][s_img_name] = s_embedding
    
        
    with open(args.save, 'wb') as handle:
        pickle.dump(test_embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
