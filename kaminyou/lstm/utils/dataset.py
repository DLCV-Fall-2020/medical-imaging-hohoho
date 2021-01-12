import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils.augmentation import get_default_transform
import pickle

def collatefn(batch):
    """
    each batch: pt_id, image_ids, embeddings, labels(in train mode)
    """
    maxlen = max([l["embeddings"].shape[0] for l in batch])
    embdim = batch[0]["embeddings"].shape[1]
    
    if "labels" in batch[0]:
        withlabel = True
        labdim= batch[0]["labels"].shape[1]
        
    else:
        withlabel = False
        
    for b in batch:
        masklen = maxlen-len(b["embeddings"])
        b["embeddings"] = np.vstack((np.zeros((masklen, embdim)), b["embeddings"]))
        
        b["img_ids"] = ["-1"] * masklen + b["img_ids"]
        b["mask"] = np.ones((maxlen))
        b["mask"][:masklen] = 0.
        if withlabel:
            b['labels'] = np.vstack((np.zeros((maxlen-len(b['labels']), labdim)), b['labels']))
            
    outbatch = {'embeddings' : torch.tensor(np.vstack([np.expand_dims(b['embeddings'], 0) \
                                                for b in batch])).float()}  
    outbatch['mask'] = torch.tensor(np.vstack([np.expand_dims(b['mask'], 0) \
                                                for b in batch])).float()
    outbatch['img_ids'] = [b['img_ids'] for b in batch]
    outbatch['pt'] = [b['pt'] for b in batch]
    if withlabel:
        outbatch['labels'] = torch.tensor(np.vstack([np.expand_dims(b['labels'], 0) for b in batch])).float()
    return outbatch

class LSTMHemorrhageDataset(Dataset):
    def __init__(self, pt_id_list, embedding_root = "./train_embedding.pkl", label_csv_path = "./../../Blood_data/train.csv", mode = "train"):
        self.pt_id_list = pt_id_list
        self.embedding_root = embedding_root
        self.label_csv_path = label_csv_path
        self.mode = mode
        self.__read_embedding()
        
        if (mode == "train") or (mode == "val"):
            self.__read_label_csv()
        
    def __read_embedding(self):
        with open(self.embedding_root, "rb" ) as f: 
            self.embedding = pickle.load(f)

    def __read_label_csv(self):
        self.all_label_df = pd.read_csv(self.label_csv_path)
    
    def __pt_id_generate_img_id(self, pt_id, order):
        id_itself = pt_id.split("_")[1]
        img_id = id_itself + "_" + str(order) + ".jpg"
        return img_id
    
    def query_label(self, pt_name, img_id):
        label = self.all_label_df.query(f'dirname == "{pt_name}" and ID == "{img_id}"').values[0][2:]
        return np.array(label, dtype=int)
    
    def __getitem__(self, index):
        pt = self.pt_id_list[index]
        pt_embedding = self.embedding[pt]
        CT_len = len(pt_embedding)

        stack_accum = 0
        img_order_idx = 0
        img_id_stack = []
        pt_embedding_stack = []
        label_stack = []
        while stack_accum < CT_len:
            img_id = self.__pt_id_generate_img_id(pt, img_order_idx)
            if img_id in pt_embedding:
                pt_embedding_stack.append(pt_embedding[img_id])
                img_id_stack.append(img_id)

                if not self.mode == "test":
                    label = self.query_label(pt, img_id)
                    label_stack.append(label)

                stack_accum += 1
                img_order_idx += 1

        pt_embedding_stack = np.vstack(pt_embedding_stack)
        
        
        if not self.mode == "test":
            label_stack = np.vstack(label_stack)
            return {"pt":pt, "img_ids":img_id_stack, "embeddings":pt_embedding_stack, "labels":label_stack}
        else:
            return {"pt":pt, "img_ids":img_id_stack, "embeddings":pt_embedding_stack}

    def __len__(self):
        return len(self.pt_id_list)

class ReHemorrhageDataset(Dataset):
    def __init__(self, pt_id_list, data_root = "./Blood_data/train/", label_csv_path = "./Blood_data/train.csv", mode = "train", stack_img=False, augmentation=None, return_level=False):
        self.pt_id_list = pt_id_list
        self.data_root = data_root
        self.label_csv_path = label_csv_path
        self.mode = mode
        self.stack_img = stack_img
        self.augmentation = augmentation
        self.return_level = return_level
        
        self.__get_all_images_with_pt_id()
        
        if (mode == "train") or (mode == "val"):
            self.__read_label_csv()

        if not self.augmentation:
            self.augmentation = get_default_transform(mode = self.mode, RGB = True)
        
    def __get_all_images_with_pt_id(self):
        all_images_path = []
        pt_images_available_dict = {}
        
        for pt in self.pt_id_list:
            pt_images = os.listdir(os.path.join(self.data_root, pt))
            pt_images = sorted(pt_images, key = lambda i: self.__get_order(i))
            pt_images_available_dict[pt] = []
            
            for single_img in pt_images:
                all_images_path.append([pt, single_img])
                pt_images_available_dict[pt].append(self.__get_order(single_img))
                
        self.all_images_path = all_images_path
        self.pt_images_available_dict = pt_images_available_dict
        
    def __read_label_csv(self):
        self.all_label_df = pd.read_csv(self.label_csv_path)
    
    def query_label(self, pt_name, img_id):
        label = self.all_label_df.query(f'dirname == "{pt_name}" and ID == "{img_id}"').values[0][2:]
        return np.array(label, dtype=int)
    
    def img_name_change_order(self, img_name, new_order):
        prefix = img_name.split("_")[0]
        suffix = img_name.split(".")[1]
        return prefix + "_" + str(new_order) + "." + suffix
        
    def __reverse_order(self, pt_name, order):
        return max(self.pt_images_available_dict[pt_name]) - order

    def __read_img(self, img_path):
        if self.stack_img:
            img = Image.open(img_path)
        else:
            img = Image.open(img_path).convert("RGB")
        if not img.size == (512,512):
            img = img.resize((512,512))
        return img
    
    def __get_order(self, img_name):
        try:
            order = int(img_name.split("_")[1].split(".")[0])
            return order
        except:
            return 0
    
    def __getitem__(self, index):
        pt_name, img_name = self.all_images_path[index]
        img = self.__read_img(os.path.join(self.data_root, pt_name, img_name))
        
        mid_img_order = self.__get_order(img_name)

        if self.stack_img:
            
            top_img_order = (mid_img_order - 1) if (mid_img_order - 1) in self.pt_images_available_dict[pt_name] else mid_img_order
            bottom_img_order = (mid_img_order + 1) if (mid_img_order + 1) in self.pt_images_available_dict[pt_name] else mid_img_order
            
            img_top = self.__read_img(os.path.join(self.data_root, pt_name, self.img_name_change_order(img_name, top_img_order)))
            img_bottom = self.__read_img(os.path.join(self.data_root, pt_name, self.img_name_change_order(img_name, bottom_img_order)))
            
            stack = np.stack((np.array(img_top), np.array(img), np.array(img_bottom)), axis=-1) # (512, 512, channel)
            img = Image.fromarray(stack.astype(np.uint8))

        # augmentation
        img = self.augmentation(img)
            
        if (self.mode == "train") or (self.mode == "val"):
            label = self.query_label(pt_name, img_name)
            if self.return_level:
                reverse_order = self.__reverse_order(pt_name, mid_img_order)
                return pt_name, img_name, img, label, mid_img_order, reverse_order
            else:        
                return pt_name, img_name, img, label
        
        else:
            if self.return_level:
                reverse_order = self.__reverse_order(pt_name, mid_img_order)
                return pt_name, img_name, img, mid_img_order, reverse_order
            else:
                return pt_name, img_name, img
        
    def __len__(self):
        return len(self.all_images_path)