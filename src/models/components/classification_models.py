import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet50_Weights


class ResNet50Module(nn.Module):
    """
    Binary classification model based on DeepLabv3.
    """

    def __init__(self, nchannel = 3 ):
        super().__init__()
        self.model = torchvision.models.resnet50(
                    weights=ResNet50_Weights.DEFAULT
                )

        self.model.fc  = nn.Linear(in_features=2048, out_features=2, bias=True)

        if nchannel != 3:
            self.model.conv1 = nn.Conv2d(
                nchannel,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )


    def forward(self, input):
        return input.view(input.size(0), -1)

