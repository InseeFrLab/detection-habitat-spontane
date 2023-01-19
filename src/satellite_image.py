"""
"""
from __future__ import annotations

from typing import List, Optional
from datetime import date
import numpy as np
import rasterio
import rasterio.plot as rp
import torch
from utils import *


class SatelliteImage:
    """ """

    def __init__(
        self,
        array: np.array,
        crs: str,
        bounds,
        transform,
        n_bands: int,
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

        m = self.array.shape[0]
        n = self.array.shape[1]

        indices = get_indices_from_tile_length(m, n, tile_length)

        splitted_images = [
            SatelliteImage(
                self.array[:, rows[0] : rows[1], cols[0] : cols[1]],
                self.crs,
                get_bounds_for_tiles(self.transform, rows, col),
                get_transform_for_tiles(self.transform, rows[0], col[0]),
                self.n_bands,
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

        normalized_bands = (
            rp.adjust_band(
                np.clip(
                    self.array[:, :, i], 0, np.quantile(self.array[:, :, i], quantile)
                )
            )
            for i in range(self.n_bands)
        )
        self.array = np.dstack(normalized_bands)
        self.normalized = True

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

        return SatelliteImage(array, crs, bounds, transform, n_bands, date, normalized)
