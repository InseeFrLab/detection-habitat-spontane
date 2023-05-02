"""
All the __getitem__ functions will return a triplet
image, label, meta_data, with meta_data containing
paths to the non-transformed images or other necessary
information
"""
from typing import List, Optional, Union

import numpy as np
import torch
from albumentations import Compose
from torch.utils.data import Dataset

from classes.data.labeled_satellite_image import (
    DetectionLabeledSatelliteImage,
    SegmentationLabeledSatelliteImage,
)
from classes.data.satellite_image import SatelliteImage


class SatelliteDataset(Dataset):
    """
    Custom Dataset class.
    """

    def __init__(
        self,
        labeled_images: Union[
            List[SegmentationLabeledSatelliteImage],
            List[DetectionLabeledSatelliteImage],
        ],
        transforms: Optional[Compose] = None,
        bands_indices: Optional[List[int]] = None,
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
        mask = labeled_image.label

        sample = {"image": satellite_image, "mask": mask}
        if self.transforms:
            satellite_image = np.transpose(satellite_image, (1, 2, 0))
            sample = self.transforms(image=satellite_image, mask=mask)

        satellite_image = sample["image"]
        mask = sample["mask"]

        return satellite_image, mask

    def __len__(self):
        return len(self.labeled_images)


class PleiadeDataset(Dataset):
    """
    Custom Dataset class.
    """

    def __init__(
        self,
        list_paths_images: List,
        list_paths_labels: List,
        transforms: Optional[Compose] = None,
    ):
        """
        Constructor.

        Args:
            list_paths_images (List): list of path of the images
            list_paths_labels (List): list of paths containing the labels
            transforms (Compose) : list of transforms
        """
        self.list_paths_images = list_paths_images
        self.list_paths_labels = list_paths_labels
        self.transforms = transforms

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pathim = self.list_paths_images[idx]
        pathlabel = self.list_paths_labels[idx]

        img = SatelliteImage.from_raster(
            file_path=pathim, dep=None, date=None, n_bands=3
        ).array

        img = np.transpose(img.astype(float), [1, 2, 0])
        label = torch.tensor(np.load(pathlabel))

        if self.transforms:
            sample = self.transforms(image=img, label=label)
            img = sample["image"]
            label = sample["label"]
        else:
            img = torch.tensor(img.astype(float))
            img = img.permute([2, 0, 1])
            label = torch.tensor(label)

        img = img.type(torch.float)
        label = label.type(torch.LongTensor)
        metadata = {"pathimage": pathim, "pathlabel": pathlabel}
        return img, label, metadata

    def __len__(self):
        return len(self.list_paths_images)
