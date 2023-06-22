import torch
from torch import nn


# Une IOU loss différentiable. !! https://home.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
class SoftIoULoss(nn.Module):
    """
    Soft IoU (Intersection over Union) loss that is differentiable.

    The SoftIoULoss computes the Intersection over Union (IoU) between
    the predicted output and the target for each sample in the batch.
    It is designed to be differentiable, allowing it to be used as a loss
    function for training neural networks.

    Args:
        weight (torch.Tensor, optional): A weight tensor to apply to the loss.
        Defaults to None.
        size_average (bool, optional): Whether to average the loss over the
        batch. Defaults to True.
        n_classes (int, optional): The number of classes. Defaults to 2.

    Returns:
        torch.Tensor: The computed loss value.
    """
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(SoftIoULoss, self).__init__()

    def forward(self, output, target):
        """
        Performs the forward pass of the SoftIoULoss.

        Args:
            output (torch.Tensor): The predicted output tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The computed loss value.
        """
        output = output
        target_one_hot = to_one_hot(target, 2)
        N = output.size()[0]
        inputs = torch.softmax(output, dim=1)
        inter = inputs * target_one_hot
        inter = inter.view(N, 2, -1).sum(2)  # 2 classes
        union = inputs + target_one_hot - (inputs*target_one_hot)
        union = union.view(N, 2, -1).sum(2)
        loss = inter/union
        return -loss.mean()


class CrossEntropySelfmade(nn.Module):
    """
    PyTorch module that calculates the cross-entropy loss
    between the predicted output and the target output.

    """
    def __init__(self):
        """
        Constructor.

        """
        super(CrossEntropySelfmade, self).__init__()

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
        target = target.to("cpu")
        output = output.to("cpu")

        target = torch.LongTensor(target)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        return loss


# Exemple de CustomLoss
class CustomLoss(nn.Module):
    """
    Custom loss function for training a neural network.

    The CustomLoss computes the loss between the predicted output
    and the target.
    It combines the Cross Entropy Loss with a high-cost penalty for
    specific target values.

    Returns:
        torch.Tensor: The computed loss value.
    """
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        """
        Performs the forward pass of the CustomLoss.

        Args:
            output (torch.Tensor): The predicted output tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The computed loss value.
        """
        target = torch.LongTensor(target)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        mask = target == 9  # Penalize target values of 9
        high_cost = (loss * mask.float()).mean()
        return loss + high_cost


def to_one_hot(tensor, nClasses):
    n, h, w = tensor.size()
    one_hot = torch.zeros(n, nClasses, h, w, device="cuda:0").scatter_(1, tensor.view(n, 1, h, w), 1)
    return one_hot
