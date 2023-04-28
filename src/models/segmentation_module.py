"""
"""
import os
from typing import Dict, Union

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim

from classes.data.labeled_satellite_image \
    import SegmentationLabeledSatelliteImage
from classes.data.satellite_image import SatelliteImage
from classes.optim.evaluation_model import calculate_IOU
from utils.plot_utils import \
    plot_list_segmentation_labeled_satellite_image


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
        self.list_labeled_satellite_image = []

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
        self.log("test_loss", loss, on_epoch=True)

        IOU = calculate_IOU(output, labels)
        self.log("test IOU", IOU, on_epoch=True)

        self.evaluate_on_example(batch_idx, output, images, dic)

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

    def evaluate_on_example(self, batch_idx, output, images, dic):
        """
        Evaluate model output on a batch of examples\
        and generate visualizations. the set data set contains all the patch\
        of a selected image, the whole image and the associated\
        model prediction will be saved in mlflow at the end.

        Args:
            batch_idx (int): Batch index.
            output (Tensor): Model output.
            images (Tensor): Input images.
            dic (dict): Dictionary containing image paths.

        Returns:
            None
        """
        preds = torch.argmax(output, axis=1)
        batch_size = images.shape[0]

        for idx in range(batch_size):
            pthimg = dic["pathimage"][idx]
            n_bands = images.shape[1]

            satellite_image = SatelliteImage.from_raster(
                file_path=pthimg, dep=None, date=None, n_bands=n_bands
            )
            satellite_image.normalize()

            img_label_model = SegmentationLabeledSatelliteImage(
                satellite_image, np.array(preds[idx].to("cpu")), "", None
            )

            self.list_labeled_satellite_image.append(img_label_model)

        if (batch_idx + 1) % batch_size == 0:
            fig1 = plot_list_segmentation_labeled_satellite_image(
                self.list_labeled_satellite_image, np.arange(n_bands)
            )

            if not os.path.exists("img/"):
                os.makedirs("img/")

            bounds = satellite_image.bounds
            bottom = str(bounds[1])
            right = str(bounds[2])

            plot_file = "img/" + bottom + "_" + right + ".png"
            fig1.savefig(plot_file)

            mlflow.log_artifact(plot_file, artifact_path="plots")
