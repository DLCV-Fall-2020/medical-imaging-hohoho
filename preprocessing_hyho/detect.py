import pandas as pd
import numpy as np
train_df = pd.read_csv("../Blood_data/train.csv")
#detect_pattern = ["101","1001","10001"]
detect_pattern = ["101", "1001"]
count = 0

with open('../data_filtered/anomaly_ID_partial_noncontinuous.csv','w') as f:
    f.write('ID,subtype,pattern\n')
    for subtype in ["ich", "ivh", "sah", "sdh", "edh"]:
        print(f"============ Subtype {subtype} ===============")
        for name, group in train_df.groupby("dirname"):
        
            pattern = ''.join(map(str, group[subtype].values))
            for dp in detect_pattern:
                if dp in pattern:
                    f.write(name+','+subtype+','+dp+'\n')
                    #print(group)
                    print(name)
                    print(pattern)
                    count += 1
print(count)
f.close()


print(f'total patients {len(np.unique(train_df.dirname))}')

anomaly = pd.read_csv('../data_filtered/anomaly_ID_partial_noncontinuous.csv')

unique_id = np.unique(anomaly['ID'])
print(f'anomalous patients {len(unique_id)}')

print(anomaly.groupby('pattern').count()['ID'])


