import torch
from torch import nn


def accuracy(ypred, ytrue):
    soft = nn.functional.softmax(ypred, dim=1)
    probs, preds = soft.topk(1, dim=1)
    tp = (ytrue == preds.ravel()).sum().item()
    acc = tp / ytrue.size(0)
    return acc


def train_one_step(model, batch, loss_fn, optimizer, device):
    optimizer.zero_grad()
    x = batch[0].to(device).float()
    ytrue = batch[1].to(device)
    ypred = model(x)
    acc = accuracy(ypred, ytrue)
    loss = loss_fn(ypred, ytrue)
    loss.backward()
    optimizer.step()
    return loss, acc


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    total_acc = 0
    for idx, batch in enumerate(data_loader):
        loss, acc = train_one_step(model, batch, loss_fn, optimizer, device)
        total_loss += loss
        total_acc += acc
    avg_loss = total_loss / (idx + 1)
    avg_acc = total_acc / (idx + 1)
    return avg_loss, avg_acc


def validate_one_step(model, batch, loss_fn, device):
    x = batch[0].to(device).float()
    ytrue = batch[1].to(device)
    ypred = model(x)
    acc = accuracy(ypred, ytrue)
    loss = loss_fn(ypred, ytrue)
    return loss, acc


def validate_one_epoch(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            loss, acc = validate_one_step(model, batch, loss_fn, device)
            total_loss += loss
            total_acc += acc
    avg_loss = total_loss / (idx + 1)
    avg_acc = total_acc / (idx + 1)
    return avg_loss, avg_acc