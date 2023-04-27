"""
"""
from typing import List, Optional

import pytorch_lightning as pl
from albumentations import Compose
from torch.utils.data import DataLoader, Dataset, random_split


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        transforms_preprocessing: Optional[Compose] = None,
        transforms_augmentation: Optional[Compose] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        validation_prop: float = 0.2,
        dataset_test: Optional[Dataset] = None,
        bands_indices: Optional[List[int]] = None,
    ):
        """
        Data Module constructor.
        Args:
            mono_image_dataset : a data set which return one single imag\
            (even if it contained more than 3 channels ), and a label
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

        self.dataset = dataset
        self.transforms_preprocessing = transforms_preprocessing
        self.transforms_augmentation = transforms_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_prop = validation_prop
        self.bands_indices = bands_indices
        self.dataset_train: Optional[Dataset] = None
        self.dataset_val: Optional[Dataset] = None
        self.dataset_test = dataset_test

    def setup(self, stage: str = None) -> None:
        val_size = int(self.validation_prop * len(self.dataset))
        train_size = len(self.dataset) - val_size

        self.dataset_train, self.dataset_val = random_split(
            self.dataset, [train_size, val_size]
        )

        # la fonction random_split wrap le dataset dans un objet
        # \dont l'attribut est .dataset ...
        self.dataset_train.dataset.transforms = self.transforms_augmentation
        self.dataset_val.dataset.transforms = self.transforms_preprocessing
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
            batch_size=2,  # 8*8  = 64 patchs
            num_workers=self.num_workers,
        )
