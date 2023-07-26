"""
"""
from typing import Dict, Union

import pytorch_lightning as pl
from torch import nn, optim

from classes.optim.evaluation_model import calculate_IOU


class SegmentationModule(pl.LightningModule):
    """
    Pytorch Lightning Module for DeepLabv3.
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
        loss = self.loss(output, labels)

        self.log("train_loss", loss, on_epoch=True)

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
        loss = self.loss(output, labels)
        IOU = calculate_IOU(output, labels)

        self.log("validation_IOU", IOU, on_epoch=True)
        self.log("validation_loss", loss, on_epoch=True)

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

        loss = self.loss(output, labels)
        self.log("test_loss", loss, on_epoch=False)

        IOU = calculate_IOU(output, labels)
        self.log("test IOU", IOU, on_epoch=False)

        return IOU

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
