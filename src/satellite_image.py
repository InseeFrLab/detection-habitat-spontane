"""
"""
from __future__ import annotations

import torch
from typing import List, Optional, Literal
from datetime import date
import numpy as np
import rasterio
import rasterio.plot as rp
from utils import (
    get_indices_from_tile_length,
    get_bounds_for_tiles,
    get_transform_for_tiles,
)
import matplotlib.pyplot as plt
import os


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
        dep: Literal["971", "972", "973", "974", "976", "977", "978"],
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
        self.dep = dep
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
                array=self.array[:, rows[0]: rows[1], cols[0]: cols[1]],
                crs=self.crs,
                bounds=get_bounds_for_tiles(self.transform, rows, cols),
                transform=get_transform_for_tiles(
                    self.transform, rows[0], cols[0]
                ),
                n_bands=self.n_bands,
                filename=self.filename,
                dep=self.dep,
                date=self.date,
                normalized=self.normalized,
            )
            for rows, cols in indices
        ]
        return splitted_images

    def to_tensor(
        self,
        bands_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Return SatelliteImage array as a torch.Tensor.

        Args:
            bands_indices (List): List of indices of bands to plot.
                The indices should be integers between 0 and the
                number of bands - 1.

        Returns:
            torch.Tensor: _description_
        """
        if not bands_indices:
            return torch.from_numpy(self.array)
        else:
            return torch.from_numpy(self.array[bands_indices, :, :])

    def normalize(self, quantile: float = 0.97):
        """
        Normalize array values.

        Args:
            params (Dict): _description_
        """
        if self.normalized:
            # TODO: clean up
            print("Warning: this SatelliteImage is already normalized.")
            return
        if quantile < 0.5 or quantile > 1:
            raise ValueError(
                "Value of the `quantile` parameter must be between 0.5 and 1."
            )

        normalized_bands = [
            rp.adjust_band(
                np.clip(
                    self.array[i, :, :],
                    0,
                    np.quantile(self.array[i, :, :], quantile),
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
                The indices should be integers between 0 and the
                number of bands - 1.
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
        file_path: str,
        dep: Literal["971", "972", "973", "974", "976", "977", "978"],
        date: Optional[date] = None,
        n_bands: int = 4,
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
            array,
            crs,
            bounds,
            transform,
            n_bands,
            os.path.basename(file_path),
            dep,
            date,
            normalized,
        )
