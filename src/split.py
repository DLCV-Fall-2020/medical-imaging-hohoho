import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class RandomSplitPt(object):
    def __init__(self, train_data_root = "./Blood_data/train", small = False, small_csv="./Blood_data/train_small.csv"):
        self.train_data_root = train_data_root
        self.small = small
        
        if self.small:
            self.small_csv = small_csv
        
        self.__get_all_pt()
        
    def __get_all_pt(self):
        if self.small:
            self.pt_id_list = pd.read_csv(self.small_csv)["dirname"].unique()
        else:
            all_pt = os.listdir(self.train_data_root)
            all_pt.sort() # to reproduce
            self.pt_id_list = np.array(all_pt)
    
    def randomly_split(self, test_size = 0.25, random_state=42):
        train_ID_list, val_ID_list = train_test_split(self.pt_id_list, test_size=test_size, random_state=random_state)
        return train_ID_list, val_ID_list
    
    def kFold(self, k = 4, seed = 42):
        np.random.seed(seed)
        self.pt_id_list = np.random.permutation(self.pt_id_list)
        each_time_num = len(self.pt_id_list) // k + 1
        for i in range(k):
            val_flags = np.zeros(len(self.pt_id_list), dtype=bool)
            val_flags[i * each_time_num:(i+1) * each_time_num] = True
            val_ID_list = self.pt_id_list[val_flags]
            train_ID_list = self.pt_id_list[~val_flags]
            yield train_ID_list, val_ID_list

if __name__ == "__main__":
    print("Test randomly_split function")
    random_split_pt = RandomSplitPt(train_data_root = "../Blood_data/train", small_csv="../Blood_data/train_small.csv")
    train, val = random_split_pt.randomly_split()
    print(len(train))
    print(len(val))

    print("Test kFold function")
    random_split_pt = RandomSplitPt(train_data_root = "../Blood_data/train", small_csv="../Blood_data/train_small.csv")
    all_id = []
    for train, val in random_split_pt.kFold():
        all_id += list(val)
    print(len(all_id))
    print(len(np.unique(np.array(all_id))))