import torch
from torch import nn
import torchvision
from torchvision.models.resnet import ResNet50_Weights
import torch.multiprocessing as multiprocessing

# Increase the shared memory limit
multiprocessing.set_sharing_strategy('file_system')


class ResNet50Module(nn.Module):
    """
    Finetuned ResNet50 model for binary classification.

    The model is based on the ResNet50 architecture and has been trained on a
    specific task to classify inputs into two labels.

    Args:
        n_channel: (int) number of channels of the input image

    Returns:
        torch.Tensor: The output tensor containing the probabilities
        for each class.
    """

    def __init__(self, nchannel=3):
        super().__init__()
        # Load the pre-trained ResNet50 model
        self.model = torchvision.models.resnet50(
                    weights=ResNet50_Weights.DEFAULT
                )

        # Replace the last fully connected layer
        self.model.fc = nn.Linear(2048, 2)
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
        """
        Performs the forward pass of the model.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output probabilities after applying the
            softmax activation.
        """
        output = self.model(input)
        probabilities = torch.softmax(output, dim=1)

        return probabilities
