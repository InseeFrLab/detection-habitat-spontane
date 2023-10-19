"""
"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from classes.data.satellite_image import SatelliteImage
from utils.utils import get_indices_from_tile_length


class ChangedetectionTripletS2Looking:
    """ """

    def __init__(
        self,
        pathimage1: str,
        pathimage2: str,
        pathlabel: str,
    ):
        """
        Constructor.

        Args:
            pathimage1 (str): Path to the first image.
            pathimage2 (str): Path to the second image.
            pathlabel (str): Path to the label image.

        """
        self.image1 = Image.open(pathimage1)
        self.image2 = Image.open(pathimage2)
        self.label = Image.open(pathlabel)

    def plot(self):
        fig, axes = plt.subplots(ncols=3, figsize=(10, 20))

        # Plot each image on a separate subplot
        axes[0].imshow(self.image1)
        axes[1].imshow(self.image2)
        axes[2].imshow(self.label)

        # Remove the axis labels and ticks
        for ax in axes:
            ax.set_axis_off()

        # Show the plot
        plt.show()

    def random_crop(self, tile_size):
        width = self.image1.width
        height = self.image2.height

        num_subparts_x = width // tile_size
        num_subparts_y = height // tile_size

        # sélection aléatoire d'une aprtie de l'image pour le dataset
        i = np.random.randint(num_subparts_x)
        j = np.random.randint(num_subparts_y)

        left = j * tile_size
        right = (j + 1) * tile_size
        top = i * tile_size
        bottom = (i + 1) * tile_size

        self.image1 = self.image1.crop((left, top, right, bottom))
        self.image2 = self.image2.crop((left, top, right, bottom))
        self.label = self.label.crop((left, top, right, bottom))


class ChangeDetectionTriplet:
    """ """

    def __init__(
        self,
        satellite_image1: SatelliteImage,
        satellite_image2: SatelliteImage,
        label: np.array,  # shall contain the difference mask
    ):
        """
        Constructor.

        Args:
            satellite_image1 (SatelliteImage): Satellite Image.
            satellite_image2 (SatelliteImage): Satellite Image.
            label (np.array): Building change segmentation mask.

        """
        self.satellite_image1 = satellite_image1
        self.satellite_image2 = satellite_image2
        self.label = label

    def split(self, tile_length: int) -> List:
        """
        Split the SegmentationLabeledSatelliteImage into tiles of
        dimension (`tile_length` x `tile_length`).

        Args:
            tile_length (int): Dimension of tiles

        Returns:
            List[ChangeDetectionTriplet]: _description_
        """
        # 1) on split la liste de satellite image avec la fonction déjà codée
        list_sat1 = self.satellite_image1.split(tile_length=tile_length)
        list_sat2 = self.satellite_image2.split(tile_length=tile_length)

        # 2) on split le label
        if tile_length % 2:
            raise ValueError("Tile length has to be an even number.")

        m = self.satellite_image1.array.shape[1]
        n = self.satellite_image1.array.shape[2]

        indices = get_indices_from_tile_length(m, n, tile_length)
        splitted_labels = [
            self.label[rows[0] : rows[1], cols[0] : cols[1]] for rows, cols in indices
        ]

        list_cd_triplet = [
            ChangeDetectionTriplet(im1, im2, label, self.source, self.labeling_date)
            for im1, im2, label in zip(list_sat1, list_sat2, splitted_labels)
        ]

        return list_cd_triplet

    def plot(self, bands_indices: List, alpha=0.3):
        """
        Plot a subset of bands from a change detection satellite image and its
        corresponding labels as an image.

        Args:
            bands_indices (List): List of indices of bands to plot from \
            the satellite image.The indices should be integers \
            between 0 and the number of bands - 1.
            alpha (float, optional): The transparency of the label image when \
            overlaid on the satellite image. The default value is 0.3.

        """

        if not self.satellite_image1.normalized:
            self.satellite_image1.normalize()

        if not self.satellite_image2.normalized:
            self.satellite_image2.normalize()

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 20))
        ax1.imshow(np.transpose(self.satellite_image1.array, (1, 2, 0))[:, :, bands_indices])
        ax2.imshow(np.transpose(self.satellite_image2.array, (1, 2, 0))[:, :, bands_indices])
        ax3.imshow(self.label, alpha=alpha)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Dimension of image {self.satellite_image.array.shape[1:]}")
        plt.show()
