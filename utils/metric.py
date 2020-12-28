import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }

def hemorrhage_metrics(pred, target, threshold = 0.5):
    pred = np.array(pred > threshold, dtype=float)
    TP = ((pred == 1) & (target == 1)).sum()
    FP = ((pred == 1) & (target == 0)).sum()
    TN = ((pred == 0) & (target == 0)).sum()
    FN = ((pred == 0) & (target == 1)).sum()
    precision = TP / (TP+FP+1e-8)
    recall = TP / (TP+FN+1e-8)
    acc = (TP + TN) /(TP+TN+FP+FN)
    f2 = (5*precision*recall)/(4*precision + recall)
    return {"precision":precision, "recall":recall, "acc":acc, "f2":f2}