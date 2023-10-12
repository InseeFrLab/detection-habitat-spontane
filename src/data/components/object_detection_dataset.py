"""
Object detection Dataset class.
"""
from typing import List, Optional

import numpy as np
import torch
from albumentations import Compose
from torch.utils.data import Dataset

from classes.data.satellite_image import SatelliteImage


class ObjectDetectionDataset(Dataset):
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
        transforms: Optional[Compose] = None,
    ):
        """
        Constructor.

        Args:
            list_paths_images (List[str]): list of path of the images
            list_paths_labels (List[str]): list of paths containing the labels
            transforms (Compose) : list of transforms
        """
        self.list_paths_images = list_paths_images
        self.list_paths_labels = list_paths_labels
        self.transforms = transforms

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

        img = SatelliteImage.from_raster(file_path=pathim, dep=None, date=None, n_bands=3).array

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
