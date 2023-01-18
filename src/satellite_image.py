"""
"""
from __future__ import annotations

from typing import Tuple, List, Optional, Dict
from datetime import date
import numpy as np
import torch


class SatelliteImage:
    """ """

    def __init__(
        self,
        array: np.array,
        crs: str,
        resolution: int,
        coordinates: Tuple[float, int],
        date: Optional[date] = None,
        normalized: bool = False,
    ):
        """
        Constructor.

        Args:
            array (np.array): _description_
            crs (str): _description_
            resolution (int): _description_
            coordinates (Tuple[float, int]): _description_
            date (Optional[date], optional): _description_. Defaults to None.
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
        raise NotImplementedError()

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
        raise NotImplementedError()
