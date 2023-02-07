"""
"""
from __future__ import annotations

from typing import List, Optional
from datetime import date
import numpy as np
import rasterio
import rasterio.plot as rp
from utils import *
import matplotlib.pyplot as plt


class SatelliteImage:
    """ """

    def __init__(
        self,
        array: np.array,
        crs: str,
        bounds,
        transform,
        n_bands: int,
        filename: str,
        date: Optional[date] = None,
        normalized: bool = False,
    ):
        """
        Constructor.

        Args:
            array (np.array): _description_
            crs (str): _description_
            bounds (): _description_
            transform (): _description_
            n_bands (int): Number of bands.
            date (Optional[date], optional): _description_. Defaults to None.
            normalized (bool): _description_. Defaults to False.
        """
        self.array = array
        self.crs = crs
        self.bounds = bounds
        self.transform = transform
        self.n_bands = n_bands
        self.filename = filename
        self.date = date
        self.normalized = normalized

    def split(self, tile_length: int) -> List[SatelliteImage]:
        """
        Split the SatelliteImage into `nfolds` folds.

        Args:
            tile_length (int): Dimension of tiles

        Returns:
            List[SatelliteImage]: _description_
        """
        if tile_length % 2:
            raise ValueError("Tile length has to be an even number.")

        m = self.array.shape[1]
        n = self.array.shape[2]

        indices = get_indices_from_tile_length(m, n, tile_length)

        splitted_images = [
            SatelliteImage(
                self.array[:, rows[0] : rows[1], cols[0] : cols[1]],
                self.crs,
                get_bounds_for_tiles(self.transform, rows, cols),
                get_transform_for_tiles(self.transform, rows[0], cols[0]),
                self.n_bands,
                self.filename,
                self.date,
                self.normalized,
            )
            for rows, cols in indices
        ]
        return splitted_images

        raise NotImplementedError()

    def to_tensor(self) -> torch.Tensor:
        """
        Return SatelliteImage array as a torch.Tensor.

        Returns:
            torch.Tensor: _description_
        """
        raise NotImplementedError()

    def normalize(self, quantile: float = 0.97):
        """
        Normalize array values.

        Args:
            params (Dict): _description_
        """
        if self.normalized:
            raise ValueError("This SatelliteImage is already normalized.")
        if quantile < 0.5 or quantile > 1:
            raise ValueError(
                "Value of the `quantile` parameter must be set between 0.5 and 1."
            )

        normalized_bands = [
            rp.adjust_band(
                np.clip(
                    self.array[i, :, :], 0, np.quantile(self.array[i, :, :], quantile)
                )
            )
            for i in range(self.n_bands)
        ]
        self.array = np.stack(normalized_bands)
        self.normalized = True

    def plot(self, bands_indices: List):
        """Plot a subset of bands from a 3D array as an image.

        Args:
            bands_indices (List): List of indices of bands to plot.
                The indices should be integers between 0 and the number of bands - 1.
        """

        if not self.normalized:
            self.normalize()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(np.transpose(self.array, (1, 2, 0))[:, :, bands_indices])
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Dimension of image {self.array.shape[1:]}")
        plt.show()

    @staticmethod
    def from_raster(
        file_path: str, date: Optional[date] = None, n_bands: int = 4
    ) -> SatelliteImage:
        """
        Factory method to create a Satellite image from a raster file.

        Args:
            file_path (str): _description_
            date (Optional[date], optional): _description_. Defaults to None.
            n_bands (int): Number of bands.

        Returns:
            SatelliteImage: _description_
        """
        with rasterio.open(file_path) as raster:
            array = raster.read(
                [i for i in range(1, n_bands + 1)],
                out_shape=(n_bands, raster.height, raster.width),
            )
            crs = raster.crs
            bounds = raster.bounds
            transform = raster.transform
            normalized = False

        return SatelliteImage(
            array, crs, bounds, transform, n_bands, file_path, date, normalized
        )

    @staticmethod
    def filter_images(
        image: SatelliteImage, black_value_threshold=100, black_area_threshold=0.5
    ) -> SatelliteImage:
        """Filter images based on black pixels and their proportion.

        This function takes in a satellite image and converts it to grayscale,
        and then filters it based on the number of black pixels and their proportion.
        The image is considered black if the pixel value is less than the specified threshold (black_value_threshold).
        The image is only returned if the proportion of black pixels is less than the specified threshold (black_area_threshold).

        Args:
            image (SatelliteImage): The input satellite image.
            black_value_threshold (int, optional): The threshold value for considering a pixel as black.
            black_area_threshold (float, optional): The threshold for the proportion of black pixels.

        Returns:
            SatelliteImage: The filtered satellite image. If the proportion of black pixels is greater than the threshold, returns None.
        """
        gray_image = (
            0.2989 * image.array[0] + 0.5870 * image.array[1] + 0.1140 * image.array[2]
        )
        nb_black_pixels = np.sum(gray_image < black_value_threshold)

        if (nb_black_pixels / (gray_image.shape[0] ** 2)) < black_area_threshold:
            return image
