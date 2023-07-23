import torch.nn as nn
import numpy as np
from sklearn import metrics


def accuracy(ypred, ytrue):
    ypred = ypred.detach().cpu().numpy()
    ytrue = ytrue.detach().cpu().numpy()
    return metrics.accuracy_score(ytrue, ypred.argmax(axis=1))


def confusion_matrix(ypred, ytrue):
    '''
    0: 'Health', 1: 'Other', 2: 'Rust'
    '''
    ypred = ypred.detach().cpu().numpy()
    ytrue = ytrue.detach().cpu().numpy()
    cm = metrics.confusion_matrix(ytrue, ypred.argmax(axis=1), labels=[0, 1, 2])
    return cm