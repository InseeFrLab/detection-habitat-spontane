"""
"""
import torchvision
from typing import Union, Dict
import torch
from torch import optim, nn
import pytorch_lightning as pl


class DeepLabv3Module(pl.LightningModule):
    """
    Pytorch Lightning Module for DeepLabv3.
    """

    def __init__(
        self,
        optimizer: Union[optim.SGD, optim.Adam],
        optimizer_params: Dict,
        scheduler: Union[optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.ReduceLROnPlateau],
        scheduler_params: Dict,
        scheduler_interval: str,
    ):
        """
        Initialize TableNet Module.
        Args:
            optimizer
            optimizer_params
            scheduler
            scheduler_params
            scheduler_interval
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = torchvision.models.segmentation.deeplabv3_resnet101(
            weights='DeepLabV3_ResNet101_Weights.DEFAULT'
        )
        # 1 classe !
        self.model.classifier[4] = nn.Conv2d(
            256,
            2,
            kernel_size=(1, 1),
            stride=(1, 1))

        self.loss = nn.CrossEntropyLoss()

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
        samples, labels = batch
        labels = labels.type(torch.LongTensor)
        output = self.forward(samples)["out"]

        loss = self.loss(output, labels)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.
        Returns: Tensor
        """
        samples, labels = batch
        labels = labels.type(torch.LongTensor)
        output = self.forward(samples)["out"]

        loss = self.loss(output, labels)
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
        samples, labels = batch
        labels = labels.type(torch.LongTensor)
        output = self.forward(samples)["out"]

        loss = self.loss(output, labels)
        self.log("test_loss", loss, on_epoch=True)
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
            "interval": self.scheduler_interval
        }

        return [optimizer], [scheduler]
