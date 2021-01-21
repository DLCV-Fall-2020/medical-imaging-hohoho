import warnings 
warnings.filterwarnings("ignore")

from data_aug.dataset_wrapper import DatasetWrapper

dataset = DatasetWrapper("/media/disk1/aa/Blood_data/", 3, train_valid_split_pkl="./checkpoints/")
print('aa')
train_loader, valid_loader = dataset.get_dataloaders()

print('bb')
for img, lbl in train_loader:
    print(img.shape, lbl.shape)
    input()

