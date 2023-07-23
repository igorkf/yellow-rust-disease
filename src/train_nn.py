import os
from datetime import datetime
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
import numpy as np

from utils import set_seed
from preprocessing import make_df
from dataset import TIFDataset
from model import ResNet18, HuEtAl
from training import train_one_epoch, validate_one_epoch


parser = argparse.ArgumentParser()
parser.add_argument('--bs')
args = parser.parse_args()
args.bs = int(args.bs)

PATH = 'data'
OUTPUT = 'output/models'
LABEL2INT = {'Health': 0, 'Other': 1, 'Rust': 2}
INT2LABEL = {0: 'Health', 1: 'Other', 2: 'Rust'}
BANDS = 125
# RANDOM_BANDS = BANDS // 2
N_FOLDS = 5
RANDOM_SEED = 42
IMG_SIZE = (64, 64)
EPOCHS = 100

MODEL = 'resnet18'
# MODEL = 'convnext_large'


def channel_shuffle(img: torch.tensor, p):
    if random.random() < p:
        cs = nn.ChannelShuffle(img.shape[1] // 2)
        img = cs(img)
    return img


class EarlyStopper:
    '''
    From https://stackoverflow.com/a/73704579
    '''
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


if __name__ == '__main__':
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    OUT = f'output/models/{timestamp}_{MODEL}_bs{args.bs}'
    os.mkdir(OUT)

    set_seed(RANDOM_SEED)

    device = (
        'cuda:0'
        if torch.cuda.is_available()
        else 'cpu'
    )
    if device == 'cpu':
        NUM_WORKERS = 1
    else:
        NUM_WORKERS = 4
    print('Device:', device)
    print('Num workers:', NUM_WORKERS)

    df = make_df(PATH, test=False)
    df['label'] = df['label'].map(LABEL2INT)
    
    acc_folds = np.zeros(N_FOLDS)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for fold, (train, val) in enumerate(skf.split(df.drop('label', axis=1), df['label'])):
        print(f'\n-------- FOLD {fold} --------')

        # criteria
        loss_fn = torch.nn.CrossEntropyLoss()

        # define model
        if MODEL == 'hu':
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                # transforms.Lambda(lambda img: channel_shuffle(img, p=0.5))
            ])
            model = HuEtAl(img_size=IMG_SIZE, input_channels=BANDS, n_classes=len(LABEL2INT))
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0, verbose=True)
            # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=20, verbose=True)
        elif MODEL == 'resnet18':
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
            ])
            model = ResNet18(BANDS)

            # freeze high level features' layers: https://datascience.stackexchange.com/a/77587/97330
            model.model.layer3.requires_grad_(False)
            model.model.layer4.requires_grad_(False)

            # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)        
            # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0, verbose=True)
            scheduler = EarlyStopper(patience=int(EPOCHS * 0.20), min_delta=0.01)
        else:
            raise Exception(f'Model {MODEL} not implemented')
        
        model = model.to(device)

        # split train/val
        df_train = df.loc[train]
        train_ds = TIFDataset(df_train, bands=BANDS, transform=transform)
        train_dataloader = DataLoader(train_ds, batch_size=args.bs, num_workers=NUM_WORKERS, shuffle=True)
        df_val = df.loc[val]
        val_ds = TIFDataset(df_val, bands=BANDS, transform=None)
        val_dataloader = DataLoader(val_ds, batch_size=args.bs, num_workers=NUM_WORKERS, shuffle=False)

        # https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, verbose=True)
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, verbose=True)

        best_acc = 0.0
        for epoch in range(EPOCHS):
            train_loss, train_metrics = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device)
            val_loss, val_metrics = validate_one_epoch(model, val_dataloader, loss_fn, device)
            train_acc = train_metrics['acc']
            val_acc = val_metrics['acc']

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_model_path = os.path.join(
                    OUT,
                    f'fold{fold}_epoch{epoch + 1}_acc{val_acc:.6f}'
                )
                print('Saving:', best_model_path)
                torch.save(model.state_dict(), best_model_path)

            print(f'\n[EPOCH] {epoch + 1}/{EPOCHS}')
            print(f'[TRAIN] Loss: {train_loss:.6f}, Acc: {train_acc:.6f}')
            print(train_metrics["cm"], '\n  Predicted')
            print(f'[VAL  ] Loss: {val_loss:.6f}, Acc: {val_acc:.6f}')
            print(val_metrics["cm"], '\n  Predicted')
            print(f'[BEST VAL] Epoch: {best_epoch + 1}, Acc: {best_acc:.6f}')
        
            if scheduler.early_stop(val_loss):      
                print('Early stopping...')   
                break

            # print learnable band weights
            # for name, param in model.named_parameters():
            #     if name == 'W':
            #         w = param.data.cpu().numpy()
            #         print('Learnable band weights:')
            #         print(pd.DataFrame(w).describe().round(4).T)

        acc_folds[fold] = best_acc

    mean_acc = acc_folds.mean().round(6) 

    print('-' * 30)
    print('Fold results:')
    print('ACC:', acc_folds.round(6))
    print('Mean ACC:', mean_acc, 'Std ACC:', acc_folds.std())
   
    mean_acc = str(mean_acc).replace('.', '_') 
    os.rename(OUT, f'{OUT}_acc{mean_acc}')

