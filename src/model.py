import math

import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, bands, in_channels=3):
        super(ResNet18, self).__init__()

        self.model = models.resnet18()

        # original definition of the first layer on the resnet class
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.conv0 = nn.Conv2d(bands, in_channels, 1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 3)  # output 3 classes

    def forward(self, x):
        x = self.conv0(x)
        return self.model(x)


class HuEtAl(nn.Module):
    """
    Adapted from: https://github.com/nshaud/DeepHyperX/blob/master/models.py#L243

    Deep Convolutional Neural Networks for Hyperspectral Image Classification
    Wei Hu, Yangyu Huang, Li Wei, Fan Zhang and Hengchao Li
    Journal of Sensors, Volume 2015 (2015)
    https://www.hindawi.com/journals/js/2015/258619/
    """

    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between -0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels * self.h * self.w)
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
        return x.numel()

    def __init__(self, img_size, input_channels, n_classes, kernel_size=None, pool_size=None):
        super(HuEtAl, self).__init__()
        if kernel_size is None:
            # [In our experiments, k1 is better to be [ceil](n1/9)]
            kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
            # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
            # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
            pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels
        self.h, self.w = img_size

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv1 = nn.Conv1d(1, 20, kernel_size)
        self.pool1 = nn.MaxPool1d(pool_size)

        self.conv2 = nn.Conv1d(20, 10, kernel_size)
        self.pool2 = nn.MaxPool1d(pool_size)

        self.features_size = self._get_final_flattened_size()
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        
        # learnable band weights
        # ws = torch.randint(0, 2, (input_channels, )).float()  # naturals on {0, 1}
        # ws = torch.rand(input_channels)  # reals on [0, 1)
        # self.W = torch.nn.Parameter(ws, requires_grad=True)

        self.apply(self.weight_init)

    def forward(self, x, extract_features=False):
        # channel-wise multiplication (https://discuss.pytorch.org/t/how-to-channel-wise-multiply-an-matrix-and-a-vector/61670)
        # x = x * self.W.unsqueeze(dim=-1).unsqueeze(dim=-1) 

        b = x.size(0)
        x = x.view(b, 1, -1)  # [b, 1, input_channels * h * w]

        x = self.conv1(x)  # [b, 20, (input_channels * h * w) - kernel_size + 1]
        x = torch.tanh(self.pool1(x))  # [b, 20, ((input_channels * h * w) - kernel_size + 1) // pool_size]

        x = self.conv2(x)  # [b, 10, (input_channels * h * w) - kernel_size + 1]
        x = torch.tanh(self.pool2(x))  # [b, 10, ((input_channels * h * w) - kernel_size + 1) // (pool_size ^ 2)]

        x = x.view(-1, self.features_size)  # [b, features_size]
        x = torch.tanh(self.fc1(x))  # [b, features_size, 100]
        
        if extract_features:
            return x
        
        x = self.fc2(x)  # [b, n_classes]
        
        return x
