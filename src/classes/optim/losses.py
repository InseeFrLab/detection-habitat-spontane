import torch
from torch import nn


class CrossEntropy(nn.Module):
    """
    PyTorch module that calculates the cross-entropy loss
    between the predicted output and the target output.
    
    """
    def __init__(self):
        """
        Constructor.

        """
        super(CrossEntropy, self).__init__()

    def forward(self, output, target):
        """
        Calculates the cross-entropy loss between the predicted 
        output and the target output.

        Args:
            output (torch.Tensor): The predicted output.
            target (torch.Tensor): The target output.

        Returns:
            torch.Tensor: The calculated cross-entropy loss.

        """
        target = torch.LongTensor(target)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        return loss


# Exemple de CustomLoss
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        target = torch.LongTensor(target)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        mask = target == 9  # penalisation des 9 par exemples
        high_cost = (loss * mask.float()).mean()
        return loss + high_cost
