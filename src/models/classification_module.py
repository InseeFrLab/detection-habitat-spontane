from typing import Dict, Union

import pytorch_lightning as pl
import torch
from torch import nn, optim

from classes.optim.evaluation_model import calculate_pourcentage_loss, proportion_ones


class ClassificationModule(pl.LightningModule):

    """
    Pytorch Lightning Module for ResNet50.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Union[nn.Module],
        optimizer: Union[optim.SGD, optim.Adam],
        optimizer_params: Dict,
        scheduler: Union[optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.ReduceLROnPlateau],
        scheduler_params: Dict,
        scheduler_interval: str,
    ):
        """
        Initialize TableNet Module.
        Args:
            model
            loss
            optimizer
            optimizer_params
            scheduler
            scheduler_params
            scheduler_interval
        """
        super().__init__()

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.scheduler_interval = scheduler_interval

    def forward(self, batch):
        """
        Perform forward-pass.
        Args:
            batch (tensor): Batch of images to perform forward-pass.
        Returns (Tuple[tensor, tensor]): Table, Column prediction.
        """
        return self.model(batch)

    def training_step(self, batch, batch_idx, device="cpu"):
        """
        Training step.
        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.
        Returns: Tensor
        """
        images, labels, dic = batch
        output = self.forward(images)

        output = output.to(device)
        labels = labels.to(device)

        target = labels.long()

        targets_one_hot = torch.zeros(target.shape[0], 2)
        targets_one_hot = targets_one_hot.scatter_(1, target.unsqueeze(1), 1)

        loss = self.loss(output, targets_one_hot)

        prop_ones = proportion_ones(labels)

        self.log("train_loss", loss, on_epoch=True)
        print(prop_ones)

        return loss

    def validation_step(self, batch, batch_idx, device="cpu"):
        """
        Validation step.
        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.
        Returns: Tensor
        """

        # TODO 99% Identique à training step, à optim
        images, labels, dic = batch
        output = self.forward(images)

        output = output.to(device)
        labels = labels.to(device)

        target = labels.long()

        targets_one_hot = torch.zeros(target.shape[0], 2)
        targets_one_hot = targets_one_hot.scatter_(1, target.unsqueeze(1), 1)

        loss = self.loss(output, targets_one_hot)

        loss_pourcentage = calculate_pourcentage_loss(output, labels)
        prop_ones = proportion_ones(labels)

        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_missclassed", loss_pourcentage, on_epoch=True)
        print(prop_ones)

        return loss

    def test_step(self, batch, batch_idx, device="cpu"):
        """
        Test step.
        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.
        Returns: Tensor
        """
        # TODO 99% Identique à training step, à optim
        images, labels, dic = batch
        output = self.forward(images)

        output = output.to(device)
        labels = labels.to(device)

        target = labels.long()

        targets_one_hot = torch.zeros(target.shape[0], 2)
        targets_one_hot = targets_one_hot.scatter_(1, target.unsqueeze(1), 1)

        loss = self.loss(output, targets_one_hot)

        return loss

    def configure_optimizers(self):
        """
        Configure optimizer for pytorch lighting.
        Returns: optimizer and scheduler for pytorch lighting.
        """
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        scheduler = self.scheduler(optimizer, **self.scheduler_params)
        scheduler = {
            "scheduler": scheduler,
            "monitor": self.scheduler_params["monitor"],
            "interval": self.scheduler_interval,
        }

        return [optimizer], [scheduler]


# lang = input("What's the programming language you want to learn? ")

# images, labels, dic = batch
# output = self.forward(images)

# output = output.to(device)
# labels = labels.to(device)

# target = labels.long()

# targets_one_hot = torch.zeros(target.shape[0], 2)
# targets_one_hot = targets_one_hot.scatter_(1, target.unsqueeze(1), 1)

# loss = self.loss(output, targets_one_hot)

# match step:
#     case "training":
#         prop_ones = proportion_ones(labels)

#         self.log("train_loss", loss, on_epoch=True)
#         print(prop_ones)

#     case "validation":
#         prop_ones = proportion_ones(labels)
#         loss_pourcentage = calculate_pourcentage_loss(output, labels)

#         self.log("validation_loss", loss, on_epoch=True)
#         self.log("validation_missclassed", loss_pourcentage, on_epoch=True)
#         print(prop_ones)

#     case "test":
#         self.log("test_loss", loss, on_epoch=True)

#     case _:
