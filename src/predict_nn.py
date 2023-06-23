import os
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np

from preprocessing import make_df
from utils import set_seed
from dataset import TIFDataset
from model import ResNet18

from training import accuracy


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
MODELS = [
    'resnet18_fold0_epoch61_acc0.709821',
    'resnet18_fold1_epoch51_acc0.765625',
    'resnet18_fold2_epoch46_acc0.707589',
    'resnet18_fold3_epoch78_acc0.695312',
    'resnet18_fold4_epoch15_acc0.684152'
]


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
    df_test = make_df(PATH, test=True)
    df_test['label'] = df_test['label'].map(LABEL2INT)   
    test_ds = TIFDataset(df_test, bands=BANDS, transform=None)
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False) 

    test_probs = np.zeros((len(df_test), 3))
    acc_folds = np.zeros(N_FOLDS)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for fold, (_, val) in enumerate(skf.split(df.drop('label', axis=1), df['label'])):
        print(f'\n-------- FOLD {fold} --------')

        df_val = df.loc[val]
        val_ds = TIFDataset(df_val, bands=BANDS, transform=None)
        val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

        # define model
        model = ResNet18(BANDS, IN_CHANNELS)
        model_path = os.path.join('output', 'models', MODELS[fold])
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model = model.to(device)
        model.eval()

        # predict on val
        total_acc = 0.0
        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
                x = batch[0].to(device).float()
                ytrue = batch[1].to(device)
                ypred = model(x)
                acc = accuracy(ypred, ytrue)
                total_acc += acc
        avg_acc = total_acc / (idx + 1)
        print(f'[VAL] Acc: {avg_acc:.6f}')
        acc_folds[fold] = avg_acc

        # predict on test
        probs = []
        with torch.no_grad():
            for idx, batch in enumerate(test_dataloader):
                x = batch[0].to(device).float()
                ypred = model(x)
                p = nn.functional.softmax(ypred, dim=1).cpu().numpy()
                probs.append(p)
        probs = np.concatenate(probs)
        test_probs += probs / len(MODELS)  # blend

    print('-' * 30)
    print('Fold results:')
    print('ACC:', acc_folds.round(6))
    print('Mean ACC:', acc_folds.mean(), 'Std ACC:', acc_folds.std())

    print(test_probs.shape, test_probs.min(), test_probs.max())
    preds_idx = test_probs.argmax(axis=1)
    print(Counter(preds_idx))
    preds = [INT2LABEL[x] for x in preds_idx]
    print(Counter(preds))

    # submission
    df_test['label'] = preds
    df_test['path'] = df_test['path'].str.split('/').str[-1]
    df_test = df_test.rename(columns={'path': 'Id', 'label': 'Category'})
    df_test.to_csv('output/submission.csv', index=False)

 
