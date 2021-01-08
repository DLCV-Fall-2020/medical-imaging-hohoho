import pandas as pd
import numpy as np

train = pd.read_csv('../Blood_data/train.csv')
train_small = pd.read_csv('../Blood_data/train_small.csv')

anomaly = pd.read_csv('../data_filtered/anomaly_ID_partial_noncontinuous.csv')
print(anomaly.head())

ind1 = np.unique(anomaly['ID'])
train_new = train[~train['dirname'].isin(ind1)]
train_new.to_csv('../data_filtered/train_filtered.csv', index=False)

ind2 = np.unique(anomaly['ID'])
train_small_new = train_small[~train_small['dirname'].isin(ind2)]
train_small_new.to_csv('../data_filtered/train_small_filtered.csv',index=False)
