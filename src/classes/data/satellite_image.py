"""
"""
from __future__ import annotations

import os
from datetime import date
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.plot as rp
import torch
from osgeo import gdal

from utils.utils import (
    get_bounds_for_tiles,
    get_bounds_for_tiles2,
    get_indices_from_tile_length,
    get_transform_for_tiles,
)


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
        Split the SatelliteImage into folds of size tile_length

        Args:
            tile_length (int): Dimension of tiles

        Returns:
            List[SatelliteImage]: _description_
        """
        # if tile_length % 2:
        #     raise ValueError("Tile length has to be an even number.")

        m = self.array.shape[1]
        n = self.array.shape[2]

        indices = get_indices_from_tile_length(m, n, tile_length)

        splitted_images = [
            SatelliteImage(
                array=self.array[:, rows[0] : rows[1], cols[0] : cols[1]],
                crs=self.crs,
                bounds=get_bounds_for_tiles(self.transform, rows, cols),
                transform=get_transform_for_tiles(self.transform, rows[0], cols[0]),
                n_bands=self.n_bands,
                filename=self.filename,  # a adapter avec bb
                dep=self.dep,
                date=self.date,
                normalized=self.normalized,
            )
            for rows, cols in indices
        ]

        return splitted_images

    def split2(self, tile_length: int) -> List[SatelliteImage]:
        """
        Split the SatelliteImage into folds of size tile_length

        Args:
            tile_length (int): Dimension of tiles

        Returns:
            List[SatelliteImage]: _description_
        """
        original_array = self.array.copy()
        __, m, n = original_array.shape
        sub_arrays = []
        rows = []
        cols = []

        for i in range(0, m, tile_length):
            for j in range(0, n, tile_length):
                sub_array = original_array[
                    :, i : i + tile_length, j : j + tile_length
                ]
                sub_arrays.append(sub_array)
                rows.append(i)
                cols.append(j)

        splitted_images = [
            SatelliteImage(
                array=sub_array,
                crs=self.crs,
                bounds=get_bounds_for_tiles2(self.transform, row, col, tile_length),
                transform=get_transform_for_tiles(self.transform, row, col),
                n_bands=self.n_bands,
                filename=self.filename,  # a adapter avec bb
                dep=self.dep,
                date=self.date,
                normalized=self.normalized,
            )
            for sub_array, row, col in zip(sub_arrays, rows, cols)
        ]

        return splitted_images

    def to_tensor(self, bands_indices: Optional[List[int]] = None) -> torch.Tensor:
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

    def copy(self):
        copy_image = SatelliteImage(
            array=self.array.copy(),
            crs=self.crs,
            bounds=self.bounds,
            transform=self.transform,
            n_bands=self.n_bands,
            filename=self.filename,
            dep=self.dep,
            date=self.date,
            normalized=self.normalized,
        )

        return copy_image

    def plot(self, bands_indices: List):
        """Plot a subset of bands from a 3D array as an image.

        Args:
            bands_indices (List): List of indices of bands to plot.
                The indices should be integers between 0 and the
                number of bands - 1.
        """
        copy_image = self.copy()

        if not copy_image.normalized:
            copy_image.normalize()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(np.transpose(copy_image.array, (1, 2, 0))[:, :, bands_indices])
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Dimension of image {copy_image.array.shape[1:]}")
        plt.show()

        return plt.gcf()

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

    def to_raster(
        self,
        directory_name: str,
        file_name: str,
        format: str = "jp2",
        proj=None,
    ) -> None:
        """
        calls a function to save a SatelliteImage Object into a raster file\
            according to the raster type desired (.tif or .jp2).

        Args:
            directory_name: a string representing the name of the directory \
            where the output file should be saved.
            file_name: a string representing the name of the output file.
            format: a string representing the raster type desired.
            proj: the projection to assign to the raser.
        """

        if format == "jp2":
            to_raster_jp2(self, directory_name, file_name)
        elif format == "tif":
            to_raster_tif(self, directory_name, file_name, proj)
        else:
            raise ValueError('`format` must be either "jp2" or "tif"')


def to_raster_jp2(self, directory_name: str, file_name: str):
    """
    save a SatelliteImage Object into a raster file (.tif)

    Args:
        directory_name: a string representing the name of the directory \
        where the output file should be saved.
        file_name: a string representing the name of the output file.
    """

    data = self.array
    crs = self.crs
    transform = self.transform
    n_bands = self.n_bands

    metadata = {
        "dtype": "uint16",
        "count": n_bands,
        "width": data.shape[2],
        "height": data.shape[1],
        "crs": crs,
        "transform": transform,
        "driver": "JP2OpenJPEG",
        "compress": "jp2k",
        "interleave": "pixel",
    }

    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    file = directory_name + "/" + file_name
    with rasterio.open(file, "w", **metadata) as dst:
        dst.write(data, indexes=np.arange(n_bands) + 1)


def to_raster_tif(self, directory_name: str, filename: str, proj):
    """
    save a SatelliteImage Object into a raster file (.tif)

    Args:
        directory_name: a string representing the name of the directory \
        where the output file should be saved.
        file_name: a string representing the name of the output file.
        proj: the projection to assign to the raser.
    """

    transf = self.transform

    array = self.array

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        filename + ".tif",
        array.shape[2],
        array.shape[1],
        array.shape[0],
        gdal.GDT_Float64,
    )
    out_ds.SetGeoTransform(
        [transf[2], transf[0], transf[1], transf[5], transf[3], transf[4]]
    )
    out_ds.SetProjection(proj)

    for j in range(array.shape[0]):
        out_ds.GetRasterBand(j + 1).WriteArray(array[j, :, :])

    out_ds = None
