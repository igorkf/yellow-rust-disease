import os
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from preprocessing import make_df
from utils import set_seed
from dataset import TIFDataset
from model import ResNet18


PATH = 'data'
LABEL2INT = {'Health': 0, 'Other': 1, 'Rust': 2}
INT2LABEL = {0: 'Health', 1: 'Other', 2: 'Rust'}
BANDS = 125
RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4
MODELS = [
    '20230624_195734_fold0_epoch385_acc0.723214',
    '20230624_200822_fold1_epoch22_acc0.744420',
    '20230624_211024_fold2_epoch305_acc0.742188',
    '20230624_215235_fold3_epoch336_acc0.723214',
]


if __name__ == '__main__':
    set_seed(RANDOM_SEED)

    device = (
        'cuda:0'
        if torch.cuda.is_available()
        else 'cpu'
    )
    print(f'Device:', device)

    df_test = make_df(PATH, test=True)
    df_test['label'] = df_test['label'].map(LABEL2INT)   
    test_ds = TIFDataset(df_test, bands=BANDS, transform=None)
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False) 

    test_probs = []
    acc_folds = [float(x.split('acc')[-1]) for x in MODELS]
    for model_name in MODELS:
        print(f'\n-------- {model_name} --------')

        # define model
        model = ResNet18(BANDS)
        model_path = os.path.join('output', 'models', model_name)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model = model.to(device)
        model.eval()
        
        # predict on test
        probs = []
        with torch.no_grad():
            for idx, batch in enumerate(test_dataloader):
                x = batch[0].to(device).float()
                ypred = model(x)
                p = nn.functional.softmax(ypred, dim=1).cpu().numpy()
                probs.append(p)
        probs = np.concatenate(probs)
        test_probs.append(probs)

    # blend
    print('weights:', acc_folds)
    acc = np.mean(acc_folds)
    print('mean w:', acc, 'std w:', np.std(acc_folds))
    test_probs = np.average(
        np.array(test_probs),
        weights=acc_folds,
        axis=0
    ) 
    print(test_probs.shape, test_probs.min(), test_probs.max())
    preds_idx = test_probs.argmax(axis=1)
    print(Counter(preds_idx))
    preds = [INT2LABEL[x] for x in preds_idx]
    print(Counter(preds))

    # submission
    df_test['label'] = preds
    df_test['path'] = df_test['path'].str.split('/').str[-1]
    df_test = df_test.rename(columns={'path': 'Id', 'label': 'Category'})
    df_test.to_csv(f'output/submission_acc{acc}.csv', index=False)
 
