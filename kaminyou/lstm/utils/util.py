import csv
import os
import numpy as np
import pandas as pd

def get_pt_id_list(pt_folder_path):
    return os.listdir(pt_folder_path)

def prediction_to_csv(pt_name_list, image_name_list, prediction, output_csv_name=None,threshold=0.5, to_kaggle=False, remove_defunct=True):
    assert output_csv_name
    prediction = np.array(prediction > threshold, dtype=int)
    prediction_df = pd.DataFrame({"dirname":pt_name_list,"ID":image_name_list,
                                "ich":prediction[:,0],"ivh":prediction[:,1],"sah":prediction[:,2],
                                "sdh":prediction[:,3],"edh":prediction[:,4]})
    prediction_df.to_csv(output_csv_name, index=False)
    if to_kaggle:
        out_kaggle_name = output_csv_name.replace(".csv", "_KAGGLE.csv")
        ta_convert(output_csv_name, out_kaggle_name)
        print(f"Save as {out_kaggle_name}")
        if remove_defunct:
            os.remove(output_csv_name)
    else:
        print(f"Save as {output_csv_name}")

    
def csv_to_kaggle_ver(csv_name, ignore_ta=True, remove_defunct=True):
    kaggle_csv_name = csv_name.replace(".csv", "_KAGGLE.csv")
    if ignore_ta:
        ta_convert(csv_name, kaggle_csv_name)
        if remove_defunct:
            os.remove(csv_name)

    else:
        os.system(f"python3 to_kaggle.py --pred_csv_path {csv_name} --out_csv_path {kaggle_csv_name}")

def ta_convert(in_csv, out_csv):
    kaggle_eval_classes = ['ich', 'ivh', 'sah', 'sdh', 'edh']
    with open(in_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        output_rows = []
        for row_idx, row in enumerate(csv_reader):
            if row_idx == 0:
                continue
            for cls_idx, cls in enumerate(kaggle_eval_classes):
                ID_single = row[1].split('.')[0] + '_' + cls
                output_row = [ID_single, row[cls_idx + 2]]
                output_rows.append(output_row)

                
    with open(out_csv, mode='w') as csv_file:
        fieldnames = ['ID', 'prediction']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for row in output_rows:
            writer.writerow({'ID': row[0], 'prediction': row[1]})

def get_weight(train_csv_path="./Blood_data/train.csv", pt_restriction=[]):
    train_df = pd.read_csv(train_csv_path)
    if len(pt_restriction) > 0:
        assert isinstance(pt_restriction, np.ndarray)
        df_pt_select = pd.DataFrame({"pt":pt_restriction})
        train_df = pd.merge(train_df, df_pt_select, how="inner", left_on="dirname", right_on="pt", left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None).drop(columns="pt")
    pos_each_class = train_df[["ich", "ivh", "sah", "sdh", "edh"]].sum().values
    neg_each_class = len(train_df) - pos_each_class
    weight = neg_each_class / pos_each_class
    return weight

def img_id_to_pt_id(img_id):
    img_id_itself = img_id.split("_")[0]
    return "ID_" + img_id_itself