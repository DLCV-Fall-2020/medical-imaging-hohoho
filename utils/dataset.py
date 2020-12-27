import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
class HemorrhageDataset(Dataset):
    def __init__(self, pt_id_list, data_root = "./Blood_data/train/", label_csv_path = "./Blood_data/train.csv", mode = "train", stack_img=False):
        self.pt_id_list = pt_id_list
        self.data_root = data_root
        self.label_csv_path = label_csv_path
        self.mode = mode
        self.stack_img = stack_img
        
        self.__get_all_images_with_pt_id()
        
        if (mode == "train") or (mode == "val"):
            self.__read_label_csv()
        
        self.transform_RGB_train = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        
        self.transform_RGB_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        
        self.transform_gray_train = transforms.Compose([
            #transforms.RandomRotation(20),
            #transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.transform_gray_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
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
        
    def __read_img(self, img_path):
        return Image.open(img_path)
    
    def __get_order(self, img_name):
        try:
            order = int(img_name.split("_")[1].split(".")[0])
            return order
        except:
            return 0
    
    def __getitem__(self, index):
        pt_name, img_name = self.all_images_path[index]
        img = self.__read_img(os.path.join(self.data_root, pt_name, img_name))
        
        if self.stack_img:
            mid_img_order = self.__get_order(img_name)
            
            top_img_order = (mid_img_order - 1) if (mid_img_order - 1) in self.pt_images_available_dict[pt_name] else mid_img_order
            bottom_img_order = (mid_img_order + 1) if (mid_img_order + 1) in self.pt_images_available_dict[pt_name] else mid_img_order
            
            img_top = self.__read_img(os.path.join(self.data_root, pt_name, img_name.replace(str(mid_img_order), str(top_img_order))))
            img_bottom = self.__read_img(os.path.join(self.data_root, pt_name, img_name.replace(str(mid_img_order), str(bottom_img_order))))
            
            stack = np.stack((np.array(img_top), np.array(img), np.array(img_bottom)), axis=-1) # (512, 512, channel)
            img = Image.fromarray(stack.astype(np.uint8))
            
        if self.mode == "train":
            label = self.query_label(pt_name, img_name)
            
            if self.stack_img:
                img = self.transform_RGB_train(img)
            else:
                img = self.transform_gray_train(img)
            
            return img, label
        
        elif self.mode == "val":
            label = self.query_label(pt_name, img_name)
            
            if self.stack_img:
                img = self.transform_RGB_test(img)
            else:
                img = self.transform_gray_test(img)
            
            return img, label
        
        else:
            if self.stack_img:
                img = self.transform_RGB_test(img)
            else:
                img = self.transform_gray_test(img)
            return img
        
    def __len__(self):
        return len(self.all_images_path)