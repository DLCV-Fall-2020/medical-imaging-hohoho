import matplotlib.pyplot as plt
from utils.metric import hemorrhage_metrics
import numpy as np

def draw_compare_train_val(train, val, xlabel="Epoch", ylabel="Accuracy", title="Accuracy", save=None):
    plt.plot(train, label="train")
    plt.plot(val, label="val")
    plt.title(title)
    plt.xlabel=xlabel
    plt.ylabel=ylabel
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(save)
    else:
        plt.show()

def draw_metric(train_pred, train_true, val_pred, val_true, metric="f2", save=None):
    testing_range = np.arange(0,0.999,0.1)
    assert metric in ["acc","f2", "precision", "recall"]

    train_result = []
    val_result = []
    for i in testing_range:
        train_metric = hemorrhage_metrics(train_pred, train_true, threshold=i)
        train_result.append(train_metric[metric])
    for i in testing_range:
        val_metric = hemorrhage_metrics(val_pred, val_true, threshold=i)
        val_result.append(val_metric[metric])

    plt.plot(testing_range, train_result, label="train")
    plt.plot(testing_range, val_result, label="val")
    plt.xlabel("Threshold")
    plt.ylabel(metric)
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(save)
    else:
        plt.show()