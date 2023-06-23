import torch
from torch import nn
from torchvision import models


# class VGG11(nn.Module):
#     def __init__(self, bands, in_channels=3):
#         super(VGG11, self).__init__()
        
#         self.model = models.vgg11()
#         self.conv0 = nn.Conv2d(bands, in_channels, 1)
#         num_ftrs = self.model.classifier[0].in_features
#         self.model.classifier[0] = nn.Linear(num_ftrs, 3)  # output 3 classes
  
#     def forward(self, x):
#         x = self.conv0(x)
#         return self.model(x)


class ResNet18(nn.Module):
    def __init__(self, bands):
        super(ResNet18, self).__init__()

        self.model = models.resnet18()
        # first layer is nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.conv1 = nn.Conv2d(bands, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 3)  # output 3 classes

        # learnable band weights
        # ws = torch.randint(0, 2, (bands, )).float()  # naturals on {0, 1}
        # ws = torch.rand(bands)  # reals on [0, 1)
        # self.W = torch.nn.Parameter(ws, requires_grad=True)

    def forward(self, x):
        # channel-wise multiplication (https://discuss.pytorch.org/t/how-to-channel-wise-multiply-an-matrix-and-a-vector/61670)
        # x = x * self.W.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return self.model(x)