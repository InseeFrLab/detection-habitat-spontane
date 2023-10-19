import torch.nn as nn
from torch.nn import CrossEntropyLoss

from classes.optim.losses import CrossEntropySelfmade, SoftIoULoss
from data.components.change_detection_dataset import (
    ChangeDetectionDataset,
    ChangeIsEverywhereDataset,
)
from data.components.classification_patch import PatchClassification
from data.components.dataset import (
    ObjectDetectionPleiadeDataset,
    SegmentationPleiadeDataset,
    SegmentationSentinelDataset,
)
from models.classification_module import ClassificationModule
from models.components.classification_models import ResNet50Module
from models.components.detection_models import FasterRCNNModule
from models.components.segmentation_models import DeepLabv3Module
from models.detection_module import DetectionModule
from models.segmentation_module import SegmentationModule

dataset_dict = {
    "PLEIADE": SegmentationPleiadeDataset,
    "CLASSIFICATION": PatchClassification,
    "SENTINEL": SegmentationSentinelDataset,
    "CHANGEISEVERYWHERE": ChangeIsEverywhereDataset,
    "CHANGEDETECTIONDATASET": ChangeDetectionDataset,
    "OBJECTDETECTION": ObjectDetectionPleiadeDataset,
}

module_dict = {
    "deeplabv3": DeepLabv3Module,
    "resnet50": ResNet50Module,
    "fasterrcnn": FasterRCNNModule,
}

loss_dict = {
    "softiou": SoftIoULoss,
    "crossentropy": CrossEntropyLoss,
    "crossentropyselmade": CrossEntropySelfmade,
    "lossbinaire": nn.BCELoss,
}

task_to_lightningmodule = {
    "segmentation": SegmentationModule,
    "classification": ClassificationModule,
    "change-detection": SegmentationModule,
    "detection": DetectionModule,
}
