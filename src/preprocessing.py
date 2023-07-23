import os
from glob import glob

import numpy as np
import pandas as pd
import skimage
from tqdm import tqdm
from scipy.ndimage import uniform_filter


def make_df(basedir: str, test: bool):
    df = []
    if not test:
        for label in ['Health', 'Other', 'Rust']:
            dir_ = os.path.join(basedir, 'train', label, '*')
            files = glob(dir_)
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


def feature_extraction(img: np.ndarray, patch_size: list, band: int):
    M, N = patch_size
    img_band = uniform_filter(img[:, :, band], size=3, mode='reflect')  # running average (filter)
    img_band = np.expand_dims(img[:, :, band], axis=-1)
    patches = [img_band[x:x+M, y:y+N] for x in range(0, img_band.shape[0], M) for y in range(0, img_band.shape[1], N)]
    features = {}
    for i, patch in enumerate(patches):
        features[f'p5_{band}_{i}'] = np.quantile(patch, 0.05)
        features[f'q1_{band}_{i}'] = np.quantile(patch, 0.25)
        features[f'q2_{band}_{i}'] = np.quantile(patch, 0.50)
        features[f'q3_{band}_{i}'] = np.quantile(patch, 0.75)
        features[f'p95_{band}_{i}'] = np.quantile(patch, 0.95)
        features[f'std_{band}_{i}'] = np.std(patch)
        features[f'min_{band}_{i}'] = np.min(patch)
        features[f'max_{band}_{i}'] = np.max(patch)
    return features


def build_dataset(df: pd.DataFrame, patch_size: tuple, bands: list):
    data = []
    for path, label in tqdm(df[['path', 'label']].values):
        img = skimage.io.imread(path)
        features = {}
        for band in bands:
            features.update({'label': label})
            features.update(feature_extraction(img, patch_size, band))
        data.append(features)
    return pd.DataFrame(data)

