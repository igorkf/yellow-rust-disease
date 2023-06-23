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
    def __init__(self, bands, in_channels=3):
        super(ResNet18, self).__init__()

        self.model = models.resnet18()

        # original definition of the first layer on the renset class
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.conv0 = nn.Conv2d(bands, in_channels, 1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 3)  # output 3 classes

    def forward(self, x):
        x = self.conv0(x)
        return self.model(x)