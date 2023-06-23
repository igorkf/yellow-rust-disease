import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import skimage
from skimage.util import img_as_float


class TIFDataset(Dataset):
    def __init__(self, df, bands, transform=None):
        self.df = df
        self.bands = bands
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['label']
        label = torch.tensor(label)
        img = img_as_float(skimage.io.imread(path))[:, :, :self.bands]
        if self.transform:
            img = self.transform(image=img)['image']  # from albumentations
        img = ToTensor()(img)  # from torchvision
        return img, label