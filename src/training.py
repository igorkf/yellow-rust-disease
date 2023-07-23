import torch
import torch.nn as nn
import numpy as np

from metrics import accuracy, confusion_matrix


def train_one_step(model, batch, loss_fn, optimizer, device):
    optimizer.zero_grad()
    x = batch[0].to(device).float()
    ytrue = batch[1].to(device)
    ypred = model(x)
    loss = loss_fn(ypred, ytrue)
    loss.backward()
    optimizer.step()

    metrics = {}
    metrics['acc'] = accuracy(ypred, ytrue)
    metrics['cm'] = confusion_matrix(ypred, ytrue)

    # n_null_imgs = (x.detach().cpu().numpy().reshape(64, -1).sum(axis=1) == 0).sum()
    # print(f'# Null imgs in train: {n_null_imgs}/{x.shape[0]}')

    return loss, metrics


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    avg_metrics = {'acc': 0, 'cm': np.zeros((3, 3))}
    for idx, batch in enumerate(data_loader):
        loss, metrics = train_one_step(model, batch, loss_fn, optimizer, device)
        total_loss += loss
        for k in metrics:
            avg_metrics[k] += metrics[k]
    avg_loss = total_loss / (idx + 1)
    avg_metrics['acc'] = avg_metrics['acc'] / (idx + 1)
    return avg_loss, avg_metrics


def validate_one_step(model, batch, loss_fn, device):
    x = batch[0].to(device).float()
    ytrue = batch[1].to(device)
    ypred = model(x)
    loss = loss_fn(ypred, ytrue)

    metrics = {}
    metrics['acc'] = accuracy(ypred, ytrue)
    metrics['cm'] = confusion_matrix(ypred, ytrue)

    # n_null_imgs = (x.detach().cpu().numpy().reshape(64, -1).sum(axis=1) == 0).sum()
    # print(f'# Null imgs in val: {n_null_imgs}/{x.shape[0]}')

    return loss, metrics


def validate_one_epoch(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    avg_metrics = {'acc': 0, 'cm': np.zeros((3, 3))}
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            loss, metrics = validate_one_step(model, batch, loss_fn, device)
            total_loss += loss
            for k in metrics:
                avg_metrics[k] += metrics[k]
    avg_loss = total_loss / (idx + 1)
    avg_metrics['acc'] = avg_metrics['acc'] / (idx + 1)
    return avg_loss, avg_metrics