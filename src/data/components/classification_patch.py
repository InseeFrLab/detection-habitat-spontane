
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


class Patch_Classification(Dataset):
    """
    Custom Dataset class.
    """

    def __init__(
        self,
        list_paths_images: List,
        output_dir: str,
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
        self.list_paths_labels = list_paths_labels
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
        liste_masks = []
        
        with open(self.output_dir, 'r') as csvfile:
            reader = csv.reader(csvfile)
            
            # Ignorer l'en-tête du fichier CSV s'il y en a un
            next(reader)
            
            # Parcourir les lignes du fichier CSV et extraire la deuxième colonne
            for row in reader:
                mask = row[1]  # Index 1 correspond à la deuxième colonne (index 0 pour la première)
                liste_masks.append(mask)

        pathim = self.list_paths_images[idx]
        label = liste_masks[idx]
        
        img = SatelliteImage.from_raster(
            file_path=pathim, dep=None, date=None, n_bands=self.n_bands
        ).array

        img = np.transpose(img.astype(float), [1, 2, 0])
        label = torch.tensor(label)
        
        if self.transforms:
            sample = self.transforms(image=img, label=label)
            img = sample["image"]
            label = sample["label"]
        else:
            img = torch.tensor(img.astype(float))
            img = img.permute([2, 0, 1])
            #label = torch.tensor(label)

        img = img.type(torch.float)
        label = label.type(torch.LongTensor)
        metadata = {"pathimage": pathim, "pathlabel": pathlabel}
        return img, label, metadata

    def __len__(self):
        return len(self.list_paths_images)
