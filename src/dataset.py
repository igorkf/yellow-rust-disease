import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from skimage.io import imread
from skimage.util import img_as_float
import numpy as np


def normalize_between_channels(img):
    '''
    Divides by (std + eps) because some images are totally zero even in test
    '''
    eps = 1e-3
    return (img - img.mean()) / (img.std() + eps)


class TIFDataset(Dataset):
    def __init__(self, df, bands: np.array, transform=None):
        self.df = df
        self.bands = bands
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        label = self.df.iloc[idx]['label']
        label = torch.tensor(label)

        path = self.df.iloc[idx]['path']
        img = img_as_float(imread(path))[:, :, self.bands]
        img = ToTensor()(img)
        # img = torch.tensor(img)
        # img = normalize_between_channels(img)
        # img = img.permute(2, 0, 1)  # [h, w, c] -> [c, h, w]

        if self.transform:
            img = self.transform(img)

        return img, label