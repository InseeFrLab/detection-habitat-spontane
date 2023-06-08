import torch
from torch import nn
import torchvision
from torchvision.models.resnet import ResNet50_Weights
import torch.multiprocessing as multiprocessing

# Increase the shared memory limit
multiprocessing.set_sharing_strategy('file_system')

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
        self.softmax = nn.Softmax(dim=1)

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

        output = self.model(input)
        probabilities = torch.softmax(output, dim = 1)

        return probabilities
