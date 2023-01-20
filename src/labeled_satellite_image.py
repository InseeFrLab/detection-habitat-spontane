"""
"""
from __future__ import annotations
from typing import List, Literal, Tuple
from satellite_image import SatelliteImage
import numpy as np
from datetime import datetime


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

    def split(self, nfolds: int) -> List[DetectionLabeledSatelliteImage]:
        """
        Split the DetectionLabeledSatelliteImage into `nfolds` folds.

        Args:
            nfolds (int): _description_

        Returns:
            List[DetectionLabeledSatelliteImage]: _description_
        """
        raise NotImplementedError()
