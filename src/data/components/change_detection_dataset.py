"""
All the __getitem__ functions will return a triplet
image, label, meta_data, with meta_data containing
paths to the non-transformed images or other necessary
information
"""

import random
from typing import List, Optional

import numpy as np
import torch
from albumentations import Compose
from torch.utils.data import Dataset

from classes.data.change_detection_triplet \
    import ChangedetectionTripletS2Looking
from classes.data.satellite_image import SatelliteImage


class ChangeDetectionDataset(Dataset):

    def __init__(
        self,
        list_paths_images_1: List,
        list_paths_images_2: List,
        list_paths_labels,
        n_bands=3,
        transforms: Optional[Compose] = None,
    ):
        """
        Constructor.

        Args:
            list_paths_images_1 (List): list of path of the images
            list_paths_images_2 (List): list of path of the images
            list_paths_labels (List): list of paths
             containing the list_paths_labelss
            transforms (Compose) : list of transforms
        """
        self.list_paths_images_1 = list_paths_images_1
        self.list_paths_images_2 = list_paths_images_2
        self.list_paths_labels = list_paths_labels
        self.n_bands = n_bands
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

        pathlabel = self.list_paths_labels[idx]

        img1 = SatelliteImage.from_raster(
            file_path=pathim1, dep=None, date=None, n_bands=self.n_bands
        ).array

        img2 = SatelliteImage.from_raster(
            file_path=pathim2, dep=None, date=None, n_bands=self.n_bands
        ).array

        img1 = np.transpose(img1.astype(float), [1, 2, 0])
        img2 = np.transpose(img2.astype(float), [1, 2, 0])
        label = torch.tensor(np.load(pathlabel))   
        
        if self.transforms: # transfo séparée ne marche que pour les transfos non aléatoires
            sample1 = self.transforms(image=img1, label=label)
            img1 = sample1["image"]
            sample2 = self.transforms(image=img2, label=label)
            img2 = sample2["image"]
            label = sample2["label"]

        img1 = img1.type(torch.float)
        img2 = img2.type(torch.float)

        img_double = torch.concatenate((img1, img2))

        label = label.type(torch.LongTensor)

        meta_data = {
            "pathimage1": pathim1,
            "pathimage2": pathim2,
            "pathlabel": pathlabel,
        }

        return img_double, label, meta_data

    def __len__(self):
        return len(self.list_paths_images)


class ChangeIsEverywhereDataset(Dataset):
    """
    From the article Change is EveryWhere doi : 2108.07002v2
    This is a custom dataset class for a change detection task.

    The dataset takes in a list of image / label paths as input
    the dataset creates 2 random permutation of the original images list,

    for each new couple image1, image2 created
    the change detection label is built with an
    XOR operation based on label 1 and label 2.
    """

    def __init__(
        self,
        list_paths_images: List,
        list_paths_labels: List,
        n_bands=3,
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
        self.n_bands = n_bands

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
            file_path=pathim1, dep=None, date=None, n_bands=self.n_bands
        ).array

        img2 = SatelliteImage.from_raster(
            file_path=pathim2, dep=None, date=None, n_bands=self.n_bands
        ).array

        img1 = np.transpose(img1.astype(float), [1, 2, 0])
        img2 = np.transpose(img2.astype(float), [1, 2, 0])

        label1 = torch.tensor(np.load(pathlabel1))
        label2 = torch.tensor(np.load(pathlabel2))

        # ici on transforme les images et les labels , peut être qu'il ne faut
        # pas faire de transformation trop violente pour que les masques
        # de type change detection aient du sens

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

        meta_data = {
            "pathimage1": pathim1,
            "pathimage2": pathim2,
            "pathlabel1": pathlabel1,
            "pathlabel2": pathlabel2,
        }

        return img_double, label, meta_data

    def __len__(self):
        return len(self.list_paths_images)


class ChangeDetectionS2LookingDataset(Dataset):
    """
    Custom Dataset class from the paper S2Looking :
    https://arxiv.org/abs/2107.09244

    This change detection dataset class takes
    three lists of file paths as input:
    - paths to the "before" state image files,
    - paths to the "after" state image files,
    - paths to the labeled difference files
    (segmentation masks difference between the two images)

    """

    def __init__(
        self,
        list_paths_image1: List,
        list_paths_image2: List,
        list_paths_labels: List,
        transforms: Optional[Compose] = None,
    ):
        """
        Constructor.
        Args:
            list_paths_image1: paths of the before state pictures
            list_paths_image2: paths containing  the "after" state pictures
            list_paths_labels: paths containing the labeled differences
        """
        self.list_paths_image1 = list_paths_image1
        self.list_paths_image2 = list_paths_image2
        self.list_paths_labels = list_paths_labels
        self.transforms = transforms

    def __getitem__(self, idx):
        """_summary_

        returns a data sample
        specified by the given index.

        It first gets the paths of the "before" and "after" images
        and the labeled difference file for the current index.

        Then it creates a "ChangedetectionTripletS2Looking"
        object with these three paths,randomly crops the triplet
        to size 256x256, and extracts the label mask from it.

        The while loop repeats this process for up to 15 times if
        the label mask is all zeros to ensure that
        at least one sample with a change is obtained.

        Args:
            idx (_type_): _description_
        Returns:
            _type_: _description_
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pathim1 = self.list_paths_image1[idx]
        pathim2 = self.list_paths_image2[idx]
        pathlabel = self.list_paths_labels[idx]

        label = 0
        compteur = 0
        while np.max(label) == 0 and compteur < 15:
            cdtriplet = ChangedetectionTripletS2Looking(
                pathim1, pathim2, pathlabel
            )
            cdtriplet.random_crop(256)
            label = np.array(cdtriplet.label)
            label[label != 0] = 1
            compteur += 1

        img1 = np.array(cdtriplet.image1)
        img2 = np.array(cdtriplet.image2)

        if self.transforms:
            sample = self.transforms(image=img1, image2=img2, mask=label)
            img1 = sample["image"]
            img2 = sample["image2"]
            label = sample["mask"]
        else:
            img1 = torch.tensor(np.transpose(img1, (2, 0, 1)))
            img2 = torch.tensor(np.transpose(img2, (2, 0, 1)))

        img_double = torch.concatenate([img1, img2], axis=0).squeeze()

        img_double = img_double.type(torch.float)

        label = torch.tensor(label)
        label = label.type(torch.LongTensor)

        meta_data = {
            "pathim1": pathim1,
            "pathim2": pathim2,
            "pathlabel": pathlabel,
        }

        return (
            img_double,
            label,
            meta_data,
        )

    def __len__(self):
        return len(self.list_paths_image1)
