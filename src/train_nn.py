from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
import torchvision.transforms.functional as TF
import albumentations as A
from sklearn.model_selection import StratifiedKFold
import numpy as np

from utils import set_seed
from preprocessing import make_df
from dataset import TIFDataset
from model import ResNet18
from training import train_one_epoch, validate_one_epoch


PATH = 'data'
LABEL2INT = {'Health': 0, 'Other': 1, 'Rust': 2}
INT2LABEL = {0: 'Health', 1: 'Other', 2: 'Rust'}
BANDS = 125
RANDOM_BANDS = BANDS // 2
IN_CHANNELS = 3
N_FOLDS = 5
RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 100
SHOW_LR = True


if __name__ == '__main__':
    set_seed(RANDOM_SEED)

    device = (
        'cuda:0'
        if torch.cuda.is_available()
        else 'cpu'
    )
    print(f'Device:', device)

    df = make_df(PATH, test=False)
    df['label'] = df['label'].map(LABEL2INT)
    # df_test = make_df(PATH, test=True)
    
    acc_folds = np.zeros(N_FOLDS)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (train, val) in enumerate(skf.split(df.drop('label', axis=1), df['label'])):
        print(f'\n-------- FOLD {fold} --------')
        
        # transforms
        # def custom_transform(img):
        #     return f(img)

        # transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomVerticalFlip(p=0.5),
        #     transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        #     # transforms.Lambda(custom_transform)
        # ])
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussianBlur(blur_limit=(5, 9), sigma_limit=(0.1, 5.0)),
            A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5),
            # A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=0.5),
        ])

        # split train/val
        df_train = df.loc[train]
        train_ds = TIFDataset(df_train, bands=BANDS, transform=transform)
        train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
        df_val = df.loc[val]
        val_ds = TIFDataset(df_val, bands=BANDS, transform=None)
        val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

        # define model
        model = ResNet18(BANDS, in_channels=IN_CHANNELS)
        model = model.to(device)

        loss_fn = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)        
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        best_acc = 0.0
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device)
            val_loss, val_acc = validate_one_epoch(model, val_dataloader, loss_fn, device)

            print(f'\n[EPOCH] {epoch + 1}/{EPOCHS}')
            if SHOW_LR:
                print('[LR]', scheduler.get_last_lr()[0])
            print(f'[TRAIN] Loss: {train_loss:.6f}, Acc: {train_acc:.6f}')
            print(f'[VAL  ] Loss: {val_loss:.6f}, Acc: {val_acc:.6f}')

            scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                model_path = f'output/models/resnet18_fold{fold}_epoch{epoch + 1}_acc{val_acc:.6f}'
                print('Saving:', model_path)
                torch.save(model.state_dict(), model_path)
        
        acc_folds[fold] = best_acc

    print('-' * 30)
    print('Fold results:')
    print('ACC:', acc_folds.round(6))
    print('Mean ACC:', acc_folds.mean(), 'Std ACC:', acc_folds.std())
