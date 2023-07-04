from typing import Dict, Union

import pytorch_lightning as pl
import torch
from torch import nn, optim
import numpy as np

from sklearn.metrics import (
        recall_score,
        precision_score,
        accuracy_score,
        f1_score)

from classes.optim.evaluation_model import (
    calculate_pourcentage_loss,
    proportion_ones
)


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
        scheduler: Union[
            optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.ReduceLROnPlateau
        ],
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

    def training_step(self, batch, batch_idx):
        """
        Training step.
        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.
        Returns: Tensor
        """
        images, labels, dic = batch
        output = self.forward(images)

        output = output.to("cpu")
        labels = labels.to("cpu")

        target = labels.long()

        targets_one_hot = torch.zeros(target.shape[0], 2)
        targets_one_hot = targets_one_hot.scatter_(1, target.unsqueeze(1), 1)

        loss = self.loss(output, targets_one_hot)

        targets_one_hot = targets_one_hot.numpy()
        targets_one_hot = np.where(targets_one_hot >= 0.5, 1, 0)
        output = output.numpy()

        precision = precision_score(targets_one_hot, output)
        recall = recall_score(targets_one_hot, output)
        accuracy = accuracy_score(targets_one_hot, output)
        f1 = f1_score(targets_one_hot, output)

        prop_ones = proportion_ones(labels)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_precision", precision, on_epoch=True)
        self.log("train_recall", recall, on_epoch=True)
        self.log("train_accuracy", accuracy, on_epoch=True)
        self.log("train_f1_score", f1, on_epoch=True)
        print(prop_ones)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.
        Returns: Tensor
        """
        images, labels, dic = batch
        output = self.forward(images)

        output = output.to("cpu")
        labels = labels.to("cpu")

        target = labels.long()

        targets_one_hot = torch.zeros(target.shape[0], 2)
        targets_one_hot = targets_one_hot.scatter_(1, target.unsqueeze(1), 1)

        loss = self.loss(output, targets_one_hot)

        targets_one_hot = targets_one_hot.numpy()
        targets_one_hot = np.where(targets_one_hot >= 0.5, 1, 0)
        output = output.numpy()

        precision = precision_score(targets_one_hot, output)
        recall = recall_score(targets_one_hot, output)
        accuracy = accuracy_score(targets_one_hot, output)
        f1 = f1_score(targets_one_hot, output)

        loss_pourcentage = calculate_pourcentage_loss(output, labels)
        prop_ones = proportion_ones(labels)

        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_missclassed", loss_pourcentage, on_epoch=True)
        self.log("validation_precision", precision, on_epoch=True)
        self.log("validation_recall", recall, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)
        self.log("validation_f1_score", f1, on_epoch=True)
        print(prop_ones)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step.
        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.
        Returns: Tensor
        """
        images, labels, dic = batch
        output = self.forward(images)

        output = output.to("cpu")
        labels = labels.to("cpu")

        target = labels.long()

        targets_one_hot = torch.zeros(target.shape[0], 2)
        targets_one_hot = targets_one_hot.scatter_(1, target.unsqueeze(1), 1)

        loss = self.loss(output, targets_one_hot)

        targets_one_hot = targets_one_hot.numpy()
        targets_one_hot = np.where(targets_one_hot >= 0.5, 1, 0)
        output = output.numpy()

        precision = precision_score(targets_one_hot, output)
        recall = recall_score(targets_one_hot, output)
        accuracy = accuracy_score(targets_one_hot, output)
        f1 = f1_score(targets_one_hot, output)

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_precision", precision, on_epoch=True)
        self.log("test_recall", recall, on_epoch=True)
        self.log("test_accuracy", accuracy, on_epoch=True)
        self.log("test_f1_score", f1, on_epoch=True)

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
            "monitor": "validation_loss",
            "interval": self.scheduler_interval,
        }

        return [optimizer], [scheduler]
