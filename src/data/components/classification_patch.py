
from typing import List, Optional, Union

import numpy as np
import torch
from albumentations import Compose
from torch.utils.data import Dataset
import csv

from classes.data.labeled_satellite_image import (
    DetectionLabeledSatelliteImage,
    SegmentationLabeledSatelliteImage,
)
from classes.data.satellite_image import SatelliteImage


class PatchClassification(Dataset):
    """
    Custom Dataset class.
    """

    def __init__(
        self,
        list_paths_images: List,
        list_labels: str,
        n_bands: int,
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
        self.list_labels = list_labels
        self.transforms = transforms
        self.n_bands = n_bands

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #pathim = "../train_data-SENTINEL2-976-2022/506990_8564970_70_000.jp2"
        #pathlabel = "../train_data-SENTINEL2-976-2022/506990_8564970_70_000.npy"

        pathim = self.list_paths_images[idx]
        label = int(self.list_labels[idx])
        
        img = SatelliteImage.from_raster(
            file_path=pathim, dep=None, date=None, n_bands=self.n_bands
        ).array

        img = np.transpose(img.astype(float), [1, 2, 0])
        label = torch.tensor(label)
        
        if self.transforms:
            sample = self.transforms(image=img)
            img = sample["image"]
        else:
            img = torch.tensor(img.astype(float))
            img = img.permute([2, 0, 1])
            #label = torch.tensor(label)

        img = img.type(torch.float)
        label = label.type(torch.float)
        metadata = {"pathimage": pathim, "class": label}
        return img, label, metadata

    def __len__(self):
        return len(self.list_paths_images)
