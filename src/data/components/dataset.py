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


class SegmentationPleiadeDataset(SegmentationDataset):
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
        super(SegmentationPleiadeDataset, self).__init__(
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


class SegmentationSentinelDataset(SegmentationDataset):
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
        super(SegmentationSentinelDataset, self).__init__(
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


class ObjectDetectionDataset(Dataset):
    """
    Dataset class for object detection.
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


class ObjectDetectionPleiadeDataset(ObjectDetectionDataset):
    """
    Object Detection Dataset class.

    The `_getitem_` method returns a dictionary with the following entries :
    - image: an image of size (H, W)
    - target: a dict containing the following fields
        - boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes
        in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
        - labels (Int64Tensor[N]): the class label for each ground-truth box
    """

    def __init__(
        self,
        list_paths_images: List[str],
        list_paths_labels: List[str],
        n_bands: int,
        transforms: Optional[Compose] = None,
        percent_keep: int = 1,
    ):
        """
        Constructor.
        """
        super(ObjectDetectionPleiadeDataset, self).__init__(
            list_paths_images, list_paths_labels, n_bands, transforms, percent_keep
        )

    def __getitem__(self, idx):
        """
        Get Dataset element with index `idx`.

        Args:
            idx: Index.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pathim = self.list_paths_images[idx]
        pathlabel = self.list_paths_labels[idx]

        img = SatelliteImage.from_raster(
            file_path=pathim, dep=None, date=None, n_bands=self.n_bands
        ).array

        img = np.transpose(img.astype(float), [1, 2, 0])

        # Getting boxes
        boxes = torch.tensor(np.load(pathlabel), dtype=torch.int64)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.int64)

        if self.transforms:
            sample = self.transforms(
                image=img, bboxes=boxes, class_labels=["building"] * len(boxes)
            )
            img = sample["image"]
            boxes = sample["bboxes"]
        else:
            img = torch.tensor(img.astype(float))
            img = img.permute([2, 0, 1])

        if len(boxes) > 0:
            boxes = torch.stack([torch.tensor(item) for item in boxes])
        else:
            boxes = torch.zeros((0, 4), dtype=torch.int64)

        img = img.type(torch.float)
        metadata = {"pathimage": pathim, "pathlabel": pathlabel}
        return img, boxes, metadata

    def __len__(self):
        return len(self.list_paths_images)
