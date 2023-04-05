"""
"""
import random
from typing import List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from albumentations import Compose
from torch.utils.data import DataLoader, Dataset, random_split

from labeled_satellite_image import (
    DetectionLabeledSatelliteImage,
    SegmentationLabeledSatelliteImage,
)

# ce module pourrait etre generaliste si tout est separé en image et label (penser à utiliser os.walk pour les arborescences particulières)
# le data set est créé séparément et ne doit retourn er qu'une image
class SegmentationPleiadeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        mono_image_dataset : Dataset,
        transforms_preprocessing: Optional[Compose] = None,
        transforms_augmentation: Optional[Compose] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        validation_prop: float = 0.2,
        test_prop = 0.1,
        bands_indices: Optional[List[int]] = None,
    ):
        """
        Data Module constructor.
        Args:
            mono_image_dataset : a data set which return one single image (even if it contained more than 3 channels ), and a label
            transforms_preprocessing (Optional[Compose]): Compose object
                from albumentations applied on validation and test datasets.
            transforms_augmentation (Optional[Compose]): Compose object
                from albumentations applied on training dataset.
            batch_size (int): Batch size.
            num_workers (int): Number of workers to process data.
            bands_indices (List): List of indices of bands to plot.
                The indices should be integers between 0 and the
                number of bands - 1.
        """
        super().__init__()
        
        self.mono_image_dataset = mono_image_dataset
        self.transforms_preprocessing = transforms_preprocessing
        self.transforms_augmentation = transforms_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_prop = validation_prop
        self.test_prop = test_prop
        self.bands_indices = bands_indices
        self.dataset_train: Optional[Dataset] = None
        self.dataset_val: Optional[Dataset] = None
        self.dataset_test: Optional[Dataset] = None
        
        
    
    def setup(self, stage: str = None) -> None:
        
        if not self.dataset_train and not self.dataset_val and not self.dataset_test: 
            val_size = int(self.validation_prop * len(self.mono_image_dataset))
            test_size = int(self.test_prop * len(self.mono_image_dataset))
            train_size = len(self.mono_image_dataset) - val_size - test_size
            
            self.dataset_train, self.dataset_val, self.dataset_test = random_split(self.mono_image_dataset, [train_size, val_size, test_size])
            self.dataset_train.transforms = self.transforms_augmentation
            self.dataset_val.transforms = self.transforms_preprocessing
            self.dataset_test.transforms = self.transforms_preprocessing
            
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        """
        Create Dataloader.
        Returns: DataLoader
        """
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        """
        Create Dataloader.
        Returns: DataLoader
        """
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        """Create Dataloader.
        Returns: DataLoader
        """
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
            
            
            
            
            
            