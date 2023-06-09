"""
Detection Module.
"""
from typing import Dict, Union

import pytorch_lightning as pl
import torch
from torch import nn, optim
from torchvision.models.detection._utils import Matcher
from torchvision.ops.boxes import box_iou


class DetectionModule(pl.LightningModule):
    """
    Pytorch Lightning Module for object detection.
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
        self.list_labeled_satellite_image = []

    def forward(self, batch, targets=None):
        """
        Perform forward-pass. Torchvision FasterRCNN returns the loss
        during training and the boxes during eval.

        Args:
            batch (tensor): Batch of images to perform forward-pass.
        Returns (Tuple[tensor, tensor]): Table, Column prediction.
        """
        self.model.model.eval()
        return self.model.model(batch)

    def training_step(self, batch, batch_idx):
        """
        Training step.
        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.
        Returns: Tensor
        """
        images = batch[0]

        targets = []
        for boxes in batch[1]:
            target = {}
            target["boxes"] = boxes
            target["labels"] = torch.ones(len(target["boxes"])).long().to(self.device)
            targets.append(target)

        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.model.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", loss)
        return {"loss": loss, "log": loss_dict}

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.
        Returns: Tensor
        """
        images, boxes, metadata = batch
        pred_boxes = self.forward(images)

        accuracy = torch.mean(
            torch.stack(
                [
                    self.accuracy(b, pb["boxes"], iou_threshold=0.5)
                    for b, pb in zip(boxes, pred_boxes)
                ]
            )
        )
        self.log("validation_accuracy", accuracy, on_epoch=True)
        return accuracy

    def test_step(self, batch, batch_idx):
        """
        Test step.
        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.
        Returns: Tensor
        """
        images, boxes, metadata = batch
        pred_boxes = self.forward(images)
        accuracy = torch.mean(
            torch.stack(
                [
                    self.accuracy(b, pb["boxes"], iou_threshold=0.5)
                    for b, pb in zip(boxes, pred_boxes)
                ]
            )
        )
        return accuracy

    def configure_optimizers(self):
        """
        Configure optimizer for pytorch lighting.
        Returns: optimizer and scheduler for pytorch lighting.
        """
        monitor = self.scheduler_params.pop("monitor")
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        scheduler = self.scheduler(optimizer, **self.scheduler_params)
        scheduler = {
            "scheduler": scheduler,
            "monitor": monitor,
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
        raise NotImplementedError()

    def accuracy(self, src_boxes, pred_boxes, iou_threshold=1.0):
        """
        Computes accuracy metric between true and predicted boxes.
        """
        total_gt = len(src_boxes)
        total_pred = len(pred_boxes)
        if total_gt > 0 and total_pred > 0:
            # Define the matcher and distance matrix based on iou
            matcher = Matcher(iou_threshold, iou_threshold, allow_low_quality_matches=False)
            match_quality_matrix = box_iou(src_boxes, pred_boxes)

            results = matcher(match_quality_matrix)
            true_positive = torch.count_nonzero(results.unique() != -1)
            matched_elements = results[results > -1]

            # in Matcher, a predicted element can be matched only twice
            false_positive = torch.count_nonzero(results == -1) + (
                len(matched_elements) - len(matched_elements.unique())
            )
            false_negative = total_gt - true_positive
            return true_positive / (true_positive + false_positive + false_negative)
        elif total_gt == 0:
            if total_pred > 0:
                return torch.tensor(0.0)
            else:
                return torch.tensor(1.0)
        elif total_gt > 0 and total_pred == 0:
            return torch.tensor(0.0)
