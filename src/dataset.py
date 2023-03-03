"""
"""
import random
import numpy as np
from typing import List, Union, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from labeled_satellite_image import SegmentationLabeledSatelliteImage, DetectionLabeledSatelliteImage
import pytorch_lightning as pl
from albumentations import Compose


# TODO: pour le moment que pour la segmentation
class SatelliteDataset(Dataset):
    """
    Custom Dataset class.
    """

    def __init__(
        self,
        labeled_images: Union[List[SegmentationLabeledSatelliteImage], List[DetectionLabeledSatelliteImage]],
        transforms: Optional[Compose] = None,
        bands_indices: Optional[List[int]] = None
    ):
        """
        Constructor.

        Args:
            labeled_images (List): _description_
            transforms (Optional[Compose]): Compose object from albumentations.
            bands_indices (List): List of indices of bands to plot.
                The indices should be integers between 0 and the
                number of bands - 1.
        """
        self.labeled_images = labeled_images
        self.transforms = transforms
        self.bands_indices = bands_indices

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        labeled_image = self.labeled_images[idx]
        satellite_image = labeled_image.satellite_image.array[
            self.bands_indices, :, :
        ].squeeze()
        satellite_image = np.transpose(
            satellite_image, (1, 2, 0)
        )
        label = labeled_image.label

        sample = {"image": satellite_image, "label": label}
        if self.transforms:
            sample = self.transforms(image=satellite_image, mask=label)
        satellite_image = sample["image"]
        label = sample["mask"]

        return satellite_image, label

    def __len__(self):
        return len(self.labeled_images)


class SatelliteDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning Data Module.
    """

    def __init__(
        self,
        train_data: Union[List[SegmentationLabeledSatelliteImage], List[DetectionLabeledSatelliteImage]],
        test_data: Union[List[SegmentationLabeledSatelliteImage], List[DetectionLabeledSatelliteImage]],
        transforms_preprocessing: Optional[Compose] = None,
        transforms_augmentation: Optional[Compose] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        validation_prop: float = 0.2,
        bands_indices: Optional[List[int]] = None
    ):
        """
        Data Module constructor.

        Args:
            train_data (List): List of training (and validation) instances
            test_data (List): List of test instances
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

        self.data = train_data
        self.test_data = test_data
        self.transforms_preprocessing = transforms_preprocessing
        self.transforms_augmentation = transforms_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_prop = validation_prop
        self.bands_indices = bands_indices

        self.setup()

    def setup(self, stage: str = None) -> None:
        """
        Start training, validation and test datasets.

        Args:
            stage (Optional[str]): Used to separate setup logic
                for trainer.fit and trainer.test.
        """
        n_samples = len(self.data)
        random.shuffle(self.data)
        train_slice = slice(0, int(n_samples * (1 - self.validation_prop)))
        val_slice = slice(int(n_samples * (1 - self.validation_prop)), n_samples)

        self.dataset_train = SatelliteDataset(
            self.data[train_slice],
            transforms=self.transforms_augmentation,
            bands_indices=self.bands_indices
        )
        self.dataset_val = SatelliteDataset(
            self.data[val_slice],
            transforms=self.transforms_preprocessing,
            bands_indices=self.bands_indices
        )
        self.dataset_test = SatelliteDataset(
            self.test_data,
            transforms=self.transforms_preprocessing,
            bands_indices=self.bands_indices
        )

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
