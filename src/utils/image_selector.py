"""
"""
from typing import List

from labeled_satellite_image import (
    DetectionLabeledSatelliteImage,
    SegmentationLabeledSatelliteImage,
)


class ImageSelector:
    """ """

    def __init__(self):
        """ """

    def select_detection_labeled_images(
        self, image_list: List[DetectionLabeledSatelliteImage]
    ) -> List[DetectionLabeledSatelliteImage]:
        """

        Args:
            image_list (List[DetectionLabeledSatelliteImage]): _description_

        Returns:
            List[DetectionLabeledSatelliteImage]: _description_
        """
        raise NotImplementedError()

    def select_segmentation_labeled_images(
        self, image_list: List[SegmentationLabeledSatelliteImage]
    ) -> List[SegmentationLabeledSatelliteImage]:
        """

        Args:
            image_list (List[SegmentationLabeledSatelliteImage]): _description_

        Returns:
            List[SegmentationLabeledSatelliteImage]: _description_
        """
        raise NotImplementedError()
