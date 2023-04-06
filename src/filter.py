"""
"""
import os
from typing import Literal, Union

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from labeled_satellite_image import (
    DetectionLabeledSatelliteImage,
    SegmentationLabeledSatelliteImage,
)
from satellite_image import SatelliteImage
from utils.utils import get_environment, get_file_system


def is_too_black(
    image: SatelliteImage, black_value_threshold=100, black_area_threshold=0.5
) -> bool:
    """
    Determine if a satellite image is too black
    based on pixel values and black area proportion.

    This function converts a satellite image to grayscale and
    filters it based on the number of black pixels and their proportion.
    A pixel is considered black if its value is less than the specified
    threshold (black_value_threshold).

    The image is considered too black if the proportion of black pixels
    is greater than or equal to the specified threshold (black_area_threshold).

    Args:
        image (SatelliteImage): The input satellite image.
        black_value_threshold (int, optional): The threshold value
            for considering a pixel as black. Default is 100.
        black_area_threshold (float, optional): The threshold for
            the proportion of black pixels. Default is 0.5.

    Returns:
        bool: True if the proportion of black pixels is greater than or equal
            to the threshold, False otherwise.
    """
    gray_image = (
        0.2989 * image.array[0]
        + 0.5870 * image.array[1]
        + 0.1140 * image.array[2]
    )
    nb_black_pixels = np.sum(gray_image < black_value_threshold)

    if (nb_black_pixels / (gray_image.shape[0] ** 2)) >= black_area_threshold:
        return True
    else:
        return False


class RILFilter:
    """
    Filter for RIL-labeled images.
    """

    def __init__(
        self,
        dep: Literal["971", "972", "973", "974", "976", "977", "978"],
        delta_threshold: int,
        area_pct_threshold: float,
    ):
        """
        Constructor.

        Args:
            dep (Literal): Departement.
            delta_threshold (int): Max number of days between label date and
                image date.
            area_pct_threshold (float): Min percentage area that should be
                included in the RIL rotation group.
        """
        self.dep = dep
        self.delta_threshold = delta_threshold
        self.area_pct_threshold = area_pct_threshold

        environment = get_environment()
        fs = get_file_system()

        old_gr_path = os.path.join(
            environment["bucket"], environment["sources"]["old_gr"][self.dep]
        )
        new_gr_path = os.path.join(
            environment["bucket"], environment["sources"]["new_gr"][self.dep]
        )
        with fs.open(old_gr_path, "r") as f:
            self.old_gr_geometries = gpd.read_file(f)
        with fs.open(new_gr_path, "r") as f:
            self.new_gr_geometries = gpd.read_file(f)

    def validate(
        self,
        labeled_image: Union[
            SegmentationLabeledSatelliteImage, DetectionLabeledSatelliteImage
        ],
    ):
        """
        Return True if labeled image passes all controls.

        Args:
            labeled_image (Union[SegmentationLabeledSatelliteImage,
                DetectionLabeledSatelliteImage]): Labeled image.
        """
        if not self.validate_labeling_date(labeled_image):
            return False
        return self.validate_rotation_group(labeled_image)

    def validate_labeling_date(
        self,
        labeled_image: Union[
            SegmentationLabeledSatelliteImage, DetectionLabeledSatelliteImage
        ],
    ):
        """
        Return True if labeled image passes date controls.

        Args:
            labeled_image (Union[SegmentationLabeledSatelliteImage,
                DetectionLabeledSatelliteImage]): Labeled image.
        """
        labeling_date = labeled_image.labeling_date
        image_date = labeled_image.satellite_image.date

        # Filter image if labeling date is too far from image date
        delta = labeling_date - image_date
        delta = abs(delta.days)
        if delta > self.delta_threshold:
            return False
        return True

    def validate_rotation_group(
        self,
        labeled_image: Union[
            SegmentationLabeledSatelliteImage, DetectionLabeledSatelliteImage
        ],
    ):
        """
        Return True if labeled image passes rotation group controls.

        Args:
            labeled_image (Union[SegmentationLabeledSatelliteImage,
                DetectionLabeledSatelliteImage]): Labeled image.
        """
        # Determine rotation group of labeling year
        labeling_year = labeled_image.labeling_date.year
        gr = str(((labeling_year - 2018) % 5) + 1)

        # Rotation group geometry
        if labeling_year > 2020:
            gr_geometries = self.new_gr_geometries
        else:
            gr_geometries = self.old_gr_geometries
        gr_geometry = gr_geometries[gr_geometries.gr == gr].geometry

        # Intersection between rotation group geometry and
        # satellite image geometry
        geom = box(*labeled_image.satellite_image.bounds)
        intersection_area = float(gr_geometry.intersection(geom).area)

        # Validate if intersection area is large enough
        if intersection_area / geom.area < self.area_pct_threshold:
            return False
        return True
