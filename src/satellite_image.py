"""
"""
from __future__ import annotations

from typing import Tuple, List, Optional, Dict
from datetime import date
import numpy as np
import rasterio
import torch


class SatelliteImage:
    """ """

    def __init__(
        self,
        array: np.array,
        crs: str,
        bounds,
        transform,
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
            date (Optional[date], optional): _description_. Defaults to None.
            normalized (bool): _description_. Defaults to False.
        """
        raise NotImplementedError()

    def split(self, nfolds: int) -> List[SatelliteImage]:
        """
        Split the SatelliteImage into `nfolds` folds.

        Args:
            nfolds (int): _description_

        Returns:
            List[SatelliteImage]: _description_
        """
        raise NotImplementedError()

    def to_tensor(self) -> torch.Tensor:
        """
        Return SatelliteImage array as a torch.Tensor.

        Returns:
            torch.Tensor: _description_
        """
        raise NotImplementedError()

    def normalize(self, **params: Dict):
        """
        Normalize array values.

        Args:
            params (Dict): _description_
        """

    @staticmethod
    def from_raster(
        file_path: str, s3: bool = True, date: Optional[date] = None
    ) -> SatelliteImage:
        """
        Factory method to create a Satellite image from a raster file.

        Args:
            file_path (str): _description_
            s3 (bool, optional): _description_. Defaults to True.
            date (Optional[date], optional): _description_. Defaults to None.


        Returns:
            SatelliteImage: _description_
        """
        with rasterio.open(file_path) as raster:
            oviews = raster.overviews(1)  # list of overviews from biggest to smallest
            print(oviews)
            oview = 1  # let's look at the smallest thumbnail

            # NOTE this is using a 'decimated read' (http://rasterio.readthedocs.io/en/latest/topics/resampling.html)
            B1 = raster.read(
                1,
                out_shape=(1, int(raster.height // oview), int(raster.width // oview)),
            )
            B2 = raster.read(
                2,
                out_shape=(1, int(raster.height // oview), int(raster.width // oview)),
            )
            B3 = raster.read(
                3,
                out_shape=(1, int(raster.height // oview), int(raster.width // oview)),
            )
            B4 = raster.read(
                4,
                out_shape=(1, int(raster.height // oview), int(raster.width // oview)),
            )

            crs = raster.crs
            bounds = raster.bounds
            transform = raster.transform
            normalized = False

        array = np.dstack((B1, B2, B3, B4))
        return SatelliteImage(array, crs, bounds, transform, date, normalized)
