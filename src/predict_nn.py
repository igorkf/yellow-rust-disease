import os
from collections import Counter
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold

from preprocessing import make_df
from utils import set_seed
from dataset import TIFDataset
from model import ResNet, HuEtAl
from metrics import accuracy


def extract_best_models(dir):
    files = os.listdir(dir)
    folds = set(sorted([int(x[4]) for x in files if 'fold' in x]))
    best_models = []
    for fold in folds:
        files_fold = [x for x in files if f'fold{fold}' in x]
        best_acc = 0
        best_model = ''
        for file in files_fold:
            acc = float(file.split('acc')[-1])
            if acc > best_acc:
                best_acc = acc
                best_model = file
        best_models.append(best_model)
    return best_models


parser = argparse.ArgumentParser()
parser.add_argument('--dir')
parser.add_argument('--ignore', type=int, nargs='*', default=[])
parser.add_argument('--b0', type=int, default=0)
parser.add_argument('--b1', type=int, default=125)
parser.add_argument('--s', type=int, default=64)
args = parser.parse_args()

PATH = 'data'
OUTPUT = 'output/models'
LABEL2INT = {'Health': 0, 'Other': 1, 'Rust': 2}
INT2LABEL = {0: 'Health', 1: 'Other', 2: 'Rust'}
BANDS = np.arange(args.b0, args.b1)
# RANDOM_BANDS = BANDS // 2
N_FOLDS = 5
RANDOM_SEED = 42
IMG_SIZE = (args.s, args.s)


if __name__ == '__main__':
    print('Model:', args.dir)
    MODEL = args.dir.split('_')[1]
    BATCH_SIZE = int(args.dir.split('_')[2].replace('bs', ''))
    MODELS = extract_best_models(os.path.join(OUTPUT, args.dir))
    print('bs:', BATCH_SIZE)

    set_seed(RANDOM_SEED)

    device = (
        'cuda:0'
        if torch.cuda.is_available()
        else 'cpu'
    )
    if device == 'cpu':
        NUM_WORKERS = 1
    else:
        NUM_WORKERS = 8
    print('Device:', device)
    print('Num workers:', NUM_WORKERS)

    df = make_df(PATH, test=False)
    df['label'] = df['label'].map(LABEL2INT)

    df_test = make_df(PATH, test=True)
    df_test['label'] = df_test['label'].map(LABEL2INT)   
    test_ds = TIFDataset(df_test, bands=BANDS, transform=None)
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False) 

    test_probs = np.zeros((len(df_test), 3))
    oof_probs = []
    oof_trues = []
    acc_folds = []
    accs = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for fold, (_, val) in enumerate(skf.split(df.drop('label', axis=1), df['label'])):
        print(f'\n-------- FOLD {fold} --------')

        if fold in args.ignore:
            print('Ignoring fold...')
            continue

        # pick val
        df_val = df.loc[val]
        val_ds = TIFDataset(df_val, bands=BANDS, transform=None)
        val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

        print(f'-------- {MODELS[fold]} --------')

        # define model
        if MODEL == 'hu':
            model = HuEtAl(img_size=IMG_SIZE, input_channels=BANDS, n_classes=len(LABEL2INT))
        elif MODEL in ['resnet18', 'resnet34', 'resnet50']:
            model = ResNet(bands=BANDS, backbone=MODEL)
        else:
            raise Exception(f'Model {MODEL} not implemented.')

        # load model
        model_path = os.path.join(OUTPUT, args.dir, MODELS[fold])
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model = model.to(device)
        model.eval()

        # predict on val
        total_acc = 0.0
        probs_val = []
        trues_val = []
        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
                x = batch[0].to(device).float()
                ytrue = batch[1].to(device)
                trues_val.append(ytrue.detach().cpu().numpy())
                ypred = model(x)
                p = nn.functional.softmax(ypred, dim=1).detach().cpu().numpy()
                probs_val.append(p)
                acc = accuracy(ypred, ytrue)
                total_acc += acc
        trues_val = np.concatenate(trues_val)
        oof_trues.append(trues_val)
        probs_val = np.concatenate(probs_val)
        oof_probs.append(probs_val)
        avg_acc = total_acc / (idx + 1)
        print(f'[VAL] Acc: {avg_acc:.6f}')
        acc_folds.append(avg_acc)

        # predict on test
        probs = []
        with torch.no_grad():
            for idx, batch in enumerate(test_dataloader):
                x = batch[0].to(device).float()
                ypred = model(x)
                p = nn.functional.softmax(ypred, dim=1).detach().cpu().numpy()
                probs.append(p)
        probs = np.concatenate(probs)
        test_probs += probs / len(MODELS)  # blend

    # concatenate oofs
    oof_trues = np.concatenate(oof_trues)
    oof_probs = np.concatenate(oof_probs)

    # save actual and probs
    np.save(os.path.join(OUTPUT, args.dir, 'oof_trues.npy'), oof_trues)
    np.save(os.path.join(OUTPUT, args.dir, 'oof_probs.npy'), oof_probs)
    np.save(os.path.join(OUTPUT, args.dir, 'test_probs.npy'), test_probs)
    
    # show results
    acc_folds = np.array(acc_folds)
    print('-' * 30)
    print('Fold results:')
    print('ACC:', acc_folds.round(6))
    print('Mean ACC:', acc_folds.mean(), 'Std ACC:', acc_folds.std())

    # summarise
    print(test_probs.shape, test_probs.min(), test_probs.max())
    preds_idx = test_probs.argmax(axis=1)
    print(Counter(preds_idx))
    test_preds = [INT2LABEL[x] for x in preds_idx]
    print(Counter(test_preds))

    # submission
    df_test['label'] = test_preds
    df_test['path'] = df_test['path'].str.split('/').str[-1]
    df_test = df_test.rename(columns={'path': 'Id', 'label': 'Category'})
    if len(args.ignore) > 0:
        ignore = '_ignore' + '-'.join([str(x) for x in args.ignore])
    else: 
        ignore = ''
    assert len(df_test) == 300
    df_test.to_csv(
        os.path.join(OUTPUT, args.dir, f'submission{ignore}.csv'), 
        index=False
    )
