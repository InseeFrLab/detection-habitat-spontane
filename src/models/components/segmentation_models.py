import torch
import torchvision
from torch import nn, optim


class DeepLabv3Module(nn.Module):
    """ """

    def __init__(self):
        """ """
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(
            weights="DeepLabV3_ResNet101_Weights.DEFAULT"
        )
        # 1 classe !
        self.model.classifier[4] = nn.Conv2d(
            256, 2, kernel_size=(1, 1), stride=(1, 1)
        )

    def forward(self, x):
        """ """
        return self.model(x)["out"]
