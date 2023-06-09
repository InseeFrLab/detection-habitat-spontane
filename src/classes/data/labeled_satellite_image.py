"""
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from classes.data.satellite_image import SatelliteImage
from utils.utils import get_indices_from_tile_length


class SegmentationLabeledSatelliteImage:
    """ """

    def __init__(
        self,
        satellite_image: SatelliteImage,
        label: np.array,
        source: Literal["RIL", "BDTOPO"],
        labeling_date: datetime,
    ):
        """
        Constructor.

        Args:
            satellite_image (SatelliteImage): Satellite Image.
            label (np.array): Segmentation mask.
            source (Literal["RIL", "BDTOPO"]): Labeling source.
            labeling_date (datetime): Date of labeling data.
        """
        self.satellite_image = satellite_image
        self.label = label
        self.source = source
        self.labeling_date = labeling_date

    def split(self, tile_length: int) -> List[SegmentationLabeledSatelliteImage]:
        """
        Split the SegmentationLabeledSatelliteImage into tiles of
        dimension (`tile_length` x `tile_length`).

        Args:
            tile_length (int): Dimension of tiles

        Returns:
            List[SegmentationLabeledSatelliteImage]: _description_
        """
        # 1) on split la liste de satellite image avec la fonction déjà codée
        list_sat = self.satellite_image.split(tile_length=tile_length)

        # 2) on split le masque

        m = self.satellite_image.array.shape[1]
        n = self.satellite_image.array.shape[2]

        indices = get_indices_from_tile_length(m, n, tile_length)
        splitted_labels = [
            self.label[rows[0] : rows[1], cols[0] : cols[1]] for rows, cols in indices
        ]

        list_labelled_images = [
            SegmentationLabeledSatelliteImage(im, label, self.source, self.labeling_date)
            for im, label in zip(list_sat, splitted_labels)
        ]

        return list_labelled_images

    def plot(self, bands_indices: List, alpha=0.3):
        """
        Plot a subset of bands from a satellite image and its
        corresponding labels as an image.

        Args:
        bands_indices (List): List of indices of bands to plot from the
        satellite image. The indices should be integers between 0 and
        the number of bands - 1.
        alpha (float, optional): The transparency of the label image when
        overlaid on the satellite image. A value of 0 means fully transparent
        and a value of 1 means fully opaque. The default value is 0.3.

        """

        if not self.satellite_image.normalized:
            self.satellite_image.normalize()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(np.transpose(self.satellite_image.array, (1, 2, 0))[:, :, bands_indices])
        ax.imshow(self.label, alpha=alpha)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Dimension of image {self.satellite_image.array.shape[1:]}")

        return plt.gcf()

    def plot_label_next_to_image(self, bands_indices):
        """
        Plot a subset of bands from a satellite image and its
        corresponding labels as an image next to the original image

        Args:
        bands_indices (List): List of indices of bands to plot from
        the satellite image. The indices should be integers between
         0 and the number of bands - 1.
        """

        if self.satellite_image.normalized is False:
            self.satellite_image.normalize

        show_mask = np.zeros((*self.label.shape, 3))
        show_mask[self.label == 1, :] = [255, 255, 255]
        show_mask = show_mask.astype(np.uint8)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
        ax1.imshow(np.transpose(self.satellite_image.array, (1, 2, 0))[:, :, bands_indices])
        ax1.axis("off")
        ax2.imshow(show_mask)
        plt.show()


class DetectionLabeledSatelliteImage:
    """ """

    def __init__(
        self,
        satellite_image: SatelliteImage,
        label: List[Tuple[int]],
        source: Literal["RIL", "BDTOPO"],
        labeling_date: datetime,
    ):
        """
        Constructor.

        Args:
            satellite_image (SatelliteImage): Satellite image.
            label (List[Tuple[int]]): Detection label.
            source (Literal["RIL", "BDTOPO"]): Labeling source.
            labeling_date (datetime): Date of labeling data.
        """
        self.satellite_image = satellite_image
        self.label = label
        self.source = source
        self.labeling_date = labeling_date

    def split(self, nfolds: int) -> List[DetectionLabeledSatelliteImage]:
        """
        Split the DetectionLabeledSatelliteImage into `nfolds` folds.

        Args:
            nfolds (int): _description_

        Returns:
            List[DetectionLabeledSatelliteImage]: _description_
        """
        raise NotImplementedError()

    def plot(self, bands_indices: List):
        """
        Plot a subset of bands from a satellite image and its
        corresponding labels as an image.

        Args:
        bands_indices (List): List of indices of bands to plot from the
        satellite image. The indices should be integers between 0 and
        the number of bands - 1.
        """
        image = self.satellite_image.array.copy()
        # Normalisation ?

        image = Image.fromarray(
            np.transpose(image.astype(np.uint8), (1, 2, 0))[:, :, bands_indices], mode="RGB"
        )
        # Drawing bounding boxes
        for x, y, xx, yy in self.label:
            c1 = (int(x.item()), int(y.item()))
            c2 = (int(xx.item()), int(yy.item()))
            draw = ImageDraw.Draw(image)
            draw.rectangle((c1, c2))

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image)

        return plt.gcf()
