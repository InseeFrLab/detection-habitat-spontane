import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet50_Weights

class ResNet50Module(nn.Module):
    """
    Binary classification model based on DeepLabv3.
    """

    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(
                    weights=ResNet50_Weights.DEFAULT
                )
        num_features = model.fc.in_features
        nn.Linear(num_features, 2)


    def forward(self, input):
        return input.view(input.size(0), -1)
