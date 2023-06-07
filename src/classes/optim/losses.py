import torch
import torch.functional as F
from torch import nn


# Une IOU loss différentiable. !! https://home.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
class SoftIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(SoftIoULoss, self).__init__()

    def forward(self, output, target):
        output = output
        target_one_hot = to_one_hot(target, 2)
        N = output.size()[0]
        inputs = torch.softmax(output,dim=1)
        inter = inputs * target_one_hot
        inter = inter.view(N,2,-1).sum(2) # 2 classes
        union= inputs + target_one_hot - (inputs*target_one_hot)
        union = union.view(N,2,-1).sum(2)
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
        target = target.to("cpu")
        output = output.to("cpu")

        target = torch.LongTensor(target)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        return loss

class BCELossSelfmade(nn.Module):
    """
    PyTorch module that calculates the BCE loss
    between the predicted output and the target output.
    
    """
    def __init__(self):
        """
        Constructor.

        """
        super(BCELossSelfmade, self).__init__()

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

        criterion = nn.BCELoss()
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
        
def to_one_hot(tensor,nClasses):
    
    n,h,w = tensor.size()
    one_hot = torch.zeros(n,nClasses,h,w,device = "cuda:0").scatter_(1,tensor.view(n,1,h,w),1)
    return one_hot