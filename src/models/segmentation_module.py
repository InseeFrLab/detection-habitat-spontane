"""
"""
from typing import Dict, Union
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn, optim
from models.components.segmentation_models import DeepLabv3Module
from utils.satellite_image import SatelliteImage
from utils.labeled_satellite_image import SegmentationLabeledSatelliteImage
import numpy as np
import mlflow
from utils.scores import calculate_IOU

# si je veux passer au niveau d'abstraction au dessus : paramétrer la loss
class SegmentationModule(pl.LightningModule):
    """
    Pytorch Lightning Module for DeepLabv3.
    """
    def __init__(
        self,
        model: nn.Module,
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
            optimizer
            optimizer_params
            scheduler
            scheduler_params
            scheduler_interval
        """
        super().__init__()
        
        self.model = model
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
        images, labels, dic = batch
    
        output = self.forward(images)
    
        loss = self.loss(output, labels)
    
        self.log("train_loss", loss,on_epoch=True)

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
        IOU = calculate_IOU(output,labels)
        
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
        
        preds = torch.argmax(output,axis = 1)
        
        # Calculate model mask for the first element
        #idx  = 0
        for idx in range(images.shape[0]):
            pthimg = dic["pathimage"][idx]
            pthlabel = dic["pathlabel"][idx]

            satellite_image = SatelliteImage.from_raster(
                file_path = pthimg,
                dep = None,
                date = None,
                n_bands= 3)

            #img_label_gt= SegmentationLabeledSatelliteImage(satellite_image,np.load(pthlabel),"",None)
            img_label_model = SegmentationLabeledSatelliteImage(satellite_image,np.array(preds[idx].to("cpu")),"",None)

            #fig1 = img_label_gt.plot([0,1,2])
            fig1 = img_label_model.plot([0,1,2])
            plot_file = "img/"+pthimg+idx".png"
            fig1.savefig(plot_file)
            mlflow.log_artifact(plot_file, artifact_path="plots")

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
