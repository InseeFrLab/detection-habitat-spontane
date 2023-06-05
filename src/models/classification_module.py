import os
from typing import Dict, Union

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim

from classes.data.labeled_satellite_image import SegmentationLabeledSatelliteImage
from classes.data.satellite_image import SatelliteImage
from classes.optim.evaluation_model import calculate_IOU
from utils.plot_utils import plot_list_segmentation_labeled_satellite_image


class ClassificationModule(pl.LightningModule):
