"""
"""
from __future__ import annotations
from typing import List, Literal
from satellite_image import SatelliteImage
import numpy as np


class SegmentationLabeledSatelliteImage:
    """ """

    def __init__(
        self,
        satellite_image: SatelliteImage,
        mask: np.array,
        source: Literal["RIL", "BDTOPO"],
    ):
        """_summary_

        Args:
            satellite_image (SatelliteImage): _description_
            mask (np.array): _description_
            source (Literal["RIL", "BDTOPO"]): _description_
        """

    def split(self, nfolds: int) -> List[SegmentationLabeledSatelliteImage]:
        """
        Split the SegmentationLabeledSatelliteImage into `nfolds` folds.

        Args:
            nfolds (int): _description_

        Returns:
            List[SegmentationLabeledSatelliteImage]: _description_
        """
        raise NotImplementedError()


class DetectionLabeledSatelliteImage:
    """ """

    def __init__(
        self,
        satellite_image: SatelliteImage,
        mask: List,
        source: Literal["RIL", "BDTOPO"],
    ):
        """

        Args:
            satellite_image (SatelliteImage): _description_
            mask (List): _description_
            source (Literal["RIL", "BDTOPO"]): _description_
        """

    def split(self, nfolds: int) -> List[DetectionLabeledSatelliteImage]:
        """
        Split the DetectionLabeledSatelliteImage into `nfolds` folds.

        Args:
            nfolds (int): _description_

        Returns:
            List[DetectionLabeledSatelliteImage]: _description_
        """
        raise NotImplementedError()
