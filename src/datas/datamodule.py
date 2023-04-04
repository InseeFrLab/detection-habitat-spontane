import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Union, Optional, List
from albumentations import Compose
import albumentations as album

from labeled_satellite_image import (
    DetectionLabeledSatelliteImage,
    SegmentationLabeledSatelliteImage,
)


class SegmentationDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning Data Module.
    """

    def __init__(
        self,
        train_data: List[str],
        valid_data: List[str],
        test_data:  List[str],
        transforms_preprocessing: Optional[Compose] = None,
        transforms_augmentation: Optional[Compose] = None,
        batch_size: int = 20,
        num_workers: int = 4
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

        self.train_data = train_data # list containing image1 path, imaghe 2 paths and label paths
        self.valid_data = valid_data
        self.test_data = test_data
        self.transforms_preprocessing = transforms_preprocessing
        self.transforms_augmentation = transforms_augmentation
        #self.transforms_preprocessing = False
        #self.transforms_augmentation = False
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.setup()

    def setup(self, stage: str = None) -> None:
        """
        Start training, validation and test datasets.

        Args:
            stage (Optional[str]): Used to separate setup logic
                for trainer.fit and trainer.test.
        """
        n_samples = len(self.train_data[0])
        
        self.dataset_train = ChangeDetectionS2LookingDataset(
            self.train_data[0],
            self.train_data[1],
            self.train_data[2],
            transforms=self.transforms_augmentation,
        )
            
            
         
        self.dataset_val = ChangeDetectionS2LookingDataset(
            self.valid_data[0],
            self.valid_data[1],
            self.valid_data[2],
            transforms=self.transforms_preprocessing
        )
        
        self.dataset_test = ChangeDetectionS2LookingDataset(
            self.test_data[0],
            self.test_data[1],
            self.test_data[2],
            transforms=self.transforms_preprocessing
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

class SatelliteDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning Data Module.
    """

    def __init__(self,
        train_data: Union[
            List[SegmentationLabeledSatelliteImage],
            List[DetectionLabeledSatelliteImage],
        ],
        test_data: Union[
            List[SegmentationLabeledSatelliteImage],
            List[DetectionLabeledSatelliteImage],
        ],
        transforms_preprocessing: Optional[Compose] = None,
        transforms_augmentation: Optional[Compose] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        validation_prop: float = 0.2,
        bands_indices: Optional[List[int]] = None,
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
        val_slice = slice(
            int(n_samples * (1 - self.validation_prop)), n_samples
        )

        self.dataset_train = SatelliteDataset(
            self.data[train_slice],
            transforms=self.transforms_augmentation,
            bands_indices=self.bands_indices,
        )
        self.dataset_val = SatelliteDataset(
            self.data[val_slice],
            transforms=self.transforms_preprocessing,
            bands_indices=self.bands_indices,
        )
        self.dataset_test = SatelliteDataset(
            self.test_data,
            transforms=self.transforms_preprocessing,
            bands_indices=self.bands_indices,
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
