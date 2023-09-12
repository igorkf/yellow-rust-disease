import os
from glob import glob
import re

import pandas as pd


def make_df(basedir: str, test: bool):
    df = []
    if not test:
        for label in ['Health', 'Other', 'Rust']:
            dir_ = os.path.join(basedir, 'train', label, '*')
            files = glob(dir_)
            files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])  # sorting as img1, img2, img3, ...
            df_temp = pd.DataFrame()
            df_temp['path'] = files
            df_temp['label'] = label
            df.append(df_temp)
    else:
        dir_ = os.path.join(basedir, 'val', 'val', '*')
        files = glob(dir_)
        df_temp = pd.DataFrame()
        df_temp['path'] = files
        df_temp['label'] = pd.NA
        df.append(df_temp)
    df = pd.concat(df, axis=0, ignore_index=True)
    return df
