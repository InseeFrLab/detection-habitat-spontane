"""
All the __getitem__ functions will return a triplet
image, label, meta_data, with meta_data containing
paths to the non-transformed images or other necessary
information
"""
from typing import List, Optional

import numpy as np
import torch
from albumentations import Compose
from torch.utils.data import Dataset

from classes.data.satellite_image import SatelliteImage


class SegmentationDataset(Dataset):
    """
    Dataset class for segmentation.
    """

    def __init__(
        self,
        list_paths_images: List,
        list_paths_labels: List,
        n_bands: int,
        transforms: Optional[Compose] = None,
        percent_keep: int = 1,
    ):
        """
        Constructor.

        Args:
            list_paths_images (List): list of path of the images
            list_paths_labels (List): list of paths containing the labels
            transforms (Compose) : list of transforms
            percent_keep (Float) : percentage of images kept for training
        """
        self.n_bands = n_bands
        self.transforms = transforms
        self.percent_keep = percent_keep
        n_keep = int(percent_keep * len(list_paths_images))
        self.list_paths_images = list_paths_images[:n_keep]
        self.list_paths_labels = list_paths_labels[:n_keep]


class PleiadeDataset(SegmentationDataset):
    """
    Custom Dataset class.
    """

    def __init__(
        self,
        list_paths_images: List,
        list_paths_labels: List,
        n_bands: int,
        transforms: Optional[Compose] = None,
        percent_keep: int = 1,
    ):
        """
        Constructor.
        """
        super(PleiadeDataset, self).__init__(
            list_paths_images, list_paths_labels, n_bands, transforms, percent_keep
        )

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
            file_path=pathim, dep=None, date=None, n_bands=self.n_bands
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

        img = img.type(torch.float)
        label = label.type(torch.LongTensor)
        metadata = {"pathimage": pathim, "pathlabel": pathlabel}
        return img, label, metadata

    def __len__(self):
        return len(self.list_paths_images)


class SentinelDataset(SegmentationDataset):
    """
    Custom Sentinel2 Dataset class.
    """

    def __init__(
        self,
        list_paths_images: List,
        list_paths_labels: List,
        n_bands: int,
        transforms: Optional[Compose] = None,
        percent_keep: int = 1,
    ):
        """
        Constructor.
        """
        super(SentinelDataset, self).__init__(
            list_paths_images, list_paths_labels, n_bands, transforms, percent_keep
        )

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
            file_path=pathim, dep=None, date=None, n_bands=self.n_bands
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
        dic = {"pathimage": pathim, "pathlabel": pathlabel}
        return img, label, dic

    def __len__(self):
        return len(self.list_paths_images)
