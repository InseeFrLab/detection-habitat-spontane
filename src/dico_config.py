import torch.nn as nn
import torch.optim
from torch.nn import CrossEntropyLoss

from classes.optim.evaluation_model import (
    evaluer_modele_sur_jeu_de_test_classification_pleiade,
    evaluer_modele_sur_jeu_de_test_segmentation_pleiade,
    evaluer_modele_sur_jeu_de_test_segmentation_sentinel,
)
from classes.optim.losses import CrossEntropySelfmade, SoftIoULoss
from data.components.change_detection_dataset import ChangeIsEverywhereDataset, ChangeDetectionDataset
from data.components.classification_patch import PatchClassification
from data.components.dataset import PleiadeDataset, SentinelDataset
from models.classification_module import ClassificationModule
from models.components.classification_models import ResNet50Module
from models.components.segmentation_models import DeepLabv3Module
from models.segmentation_module import SegmentationModule

dataset_dict = {
    "PLEIADE": PleiadeDataset,
    "CLASSIFICATION": PatchClassification,
    "SENTINEL": SentinelDataset,
    "CHANGEISEVERYWHERE": ChangeIsEverywhereDataset,
    "CHANGEDETECTIONDATASET": ChangeDetectionDataset,
}

module_dict = {
    "deeplabv3": DeepLabv3Module,
    "deeplabv3_RGB_MOCO": DeepLabv3Module,
    "resnet50": ResNet50Module,}

optimizer_dict = {
    "SGD" : torch.optim.SGD,
    "Adam" : torch.optim.Adam,
}

scheduler_dict = {
    "ReduceLROnPlateau" : torch.optim.lr_scheduler.ReduceLROnPlateau,
    "OneCycleLR" : torch.optim.lr_scheduler.OneCycleLR,
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
}

task_to_evaluation = {
    "PLEIADESsegmentation": evaluer_modele_sur_jeu_de_test_segmentation_pleiade,
    "PLEIADESclassification": evaluer_modele_sur_jeu_de_test_classification_pleiade,
    "SENTINEL2segmentation": evaluer_modele_sur_jeu_de_test_segmentation_sentinel,
    "SENTINEL1-2segmentation": evaluer_modele_sur_jeu_de_test_segmentation_sentinel,
    "SENTINEL2-RVBsegmentation": evaluer_modele_sur_jeu_de_test_segmentation_sentinel,
    # "change-detection": evaluer_modele_sur_jeu_de_test_change_detection_pleiade

}
