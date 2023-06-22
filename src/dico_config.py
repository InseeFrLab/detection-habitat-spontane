from classes.optim.losses import CrossEntropySelfmade
from torch.nn import CrossEntropyLoss
import torch.nn as nn

from data.components.dataset import PleiadeDataset
from data.components.classification_patch import PatchClassification
from models.components.segmentation_models import DeepLabv3Module
from models.components.classification_models import ResNet50Module
from classes.optim.losses import SoftIoULoss
from models.segmentation_module import SegmentationModule
from models.classification_module import ClassificationModule
from classes.optim.evaluation_model import (
    evaluer_modele_sur_jeu_de_test_segmentation_pleiade,
    evaluer_modele_sur_jeu_de_test_classification_pleiade
)


dataset_dict = {
                "PLEIADE": PleiadeDataset,
                "CLASSIFICATION": PatchClassification
        }

module_dict = {
        "deeplabv3": DeepLabv3Module,
        "resnet50": ResNet50Module
        }

loss_dict = {
            "softiou": SoftIoULoss,
            "crossentropy": CrossEntropyLoss,
            "crossentropyselmade": CrossEntropySelfmade,
            "lossbinaire": nn.BCELoss
        }

task_to_lightningmodule = {
            "segmentation": SegmentationModule,
            "classification": ClassificationModule
        }

task_to_evaluation = {
            "segmentation": evaluer_modele_sur_jeu_de_test_segmentation_pleiade,
            "classification": evaluer_modele_sur_jeu_de_test_classification_pleiade
        }
