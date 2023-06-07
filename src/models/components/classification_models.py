import torch
from torch import nn
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
        self.model.fc  = nn.Linear(2048, 2)
        self.model.sigmoid = nn.Sigmoid()

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
        # Get the predicted class labels
        output = self.model(input)
        predicted_classes = (torch.max(output, dim = 1).indices).clone().detach()
        predicted_classes = predicted_classes.type(torch.float)
        return predicted_classes
