import random
from typing import Compose, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.satellite_image import SatelliteImage


class end_to_end_cd_dataset(Dataset):
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
            list_paths_labels (List): list of paths
             containing the list_paths_labelss
            transforms (Compose) : list of transforms
        """
        self.list_paths_images = list_paths_images
        self.list_paths_labels = list_paths_labels

        random.seed(1234)

        random_order = random.sample(
            range(len(list_paths_images)), len(list_paths_images)
        )

        self.list_paths_images1 = np.array(list_paths_images)[random_order]
        self.list_paths_labels1 = np.array(list_paths_labels)[random_order]

        d1, f1 = self.list_paths_images1[-1:], self.list_paths_images1[:-1]
        d2, f2 = self.list_paths_labels1[-1:], self.list_paths_labels1[:-1]

        self.list_paths_images2 = np.concatenate((d1, f1))
        self.list_paths_labels2 = np.concatenate((d2, f2))

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

        pathim1 = self.list_paths_images1[idx]
        pathim2 = self.list_paths_images2[idx]

        pathlabel1 = self.list_paths_labels1[idx]
        pathlabel2 = self.list_paths_labels2[idx]

        img1 = SatelliteImage.from_raster(
            file_path=pathim1, dep=None, date=None, n_bands=3
        ).array

        img2 = SatelliteImage.from_raster(
            file_path=pathim2, dep=None, date=None, n_bands=3
        ).array

        img1 = np.transpose(img1.astype(float), [1, 2, 0])
        img2 = np.transpose(img2.astype(float), [1, 2, 0])

        label1 = torch.tensor(np.load(pathlabel1))
        label2 = torch.tensor(np.load(pathlabel2))

        # label = np.where(np.logical_xor(label1,label2), 1, 0)

        if self.transforms:
            sample1 = self.transforms(image=img1, label=label1)
            img1 = sample1["image"]
            label1 = sample1["label"]

            sample2 = self.transforms(image=img2, label=label2)
            img2 = sample2["image"]
            label2 = sample2["label"]

        else:
            img1 = torch.tensor(img1.astype(float))
            img1 = img1.permute([2, 0, 1])

            img2 = torch.tensor(img2.astype(float))
            img2 = img2.permute([2, 0, 1])

            label1 = torch.tensor(label1)
            label2 = torch.tensor(label2)

        label = torch.where(
            torch.logical_xor(label1, label2), torch.tensor(1), torch.tensor(0)
        )

        img1 = img1.type(torch.float)
        img2 = img2.type(torch.float)

        img_double = torch.concatenate((img1, img2))

        label = label.type(torch.LongTensor)

        dic = {
            "pathimage1": pathim1,
            "pathimage2": pathim2,
            "pathlabel1": pathlabel1,
            "pathlabel2": pathlabel2,
        }

        return img_double, label, dic

    def __len__(self):
        return len(self.list_paths_images)
