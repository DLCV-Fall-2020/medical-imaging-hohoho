from collections import Counter

from data_aug.dataset_wrapper import DatasetWrapper, BloodDataset, DataLoader

if __name__=='__main__':
    dataset = DatasetWrapper("/media/disk1/aa/Blood_data/train/", 32)
    dataset.get_dataloaders()
