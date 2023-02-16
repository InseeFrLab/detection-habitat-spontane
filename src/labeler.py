"""
Labeler classes.
"""
import numpy as np
from typing import List, Tuple, Literal
from utils import load_ril, load_bdtopo
from satellite_image import SatelliteImage
from rasterio.features import rasterize, shapes
from abc import ABC, abstractmethod
from datetime import datetime
from shapely import Polygon
import geopandas as gpd


class Labeler(ABC):
    """
    Labeler abstract base class.
    """

    def __init__(
        self,
        labeling_date: datetime,
        dep: Literal["971", "972", "973", "974", "976", "977", "978"]
    ):
        """
        Constructor.

        Args:
            labeling_date (datetime): Labeling date.
            dep (Literal): Departement.
        """
        self.labeling_date = labeling_date
        self.dep = dep

    @abstractmethod
    def create_segmentation_label(self, satellite_image: SatelliteImage) -> np.array:
        """
        Create a segmentation label (mask) for a SatelliteImage.

        Args:
            satellite_image (SatelliteImage): Satellite image.

        Returns:
            np.array: Segmentation mask.
        """
        raise NotImplementedError()

    def create_detection_label(
        self, satellite_image: SatelliteImage
    ) -> List[Tuple[int]]:
        """
        Create an object detection label for a SatelliteImage.

        Args:
            satellite_image (SatelliteImage): Satellite image.

        Returns:
            List[Tuple[int]]: Object detection label.
        """
        image_height = satellite_image.array.shape[1]
        image_width = satellite_image.array.shape[2]
        segmentation_mask = self.create_segmentation_label(satellite_image)

        polygon_list = []
        for shape in list(shapes(segmentation_mask)):
            polygon = Polygon(shape[0]["coordinates"][0])
            if polygon.area > 0.85 * image_height * image_width:
                continue
            polygon_list.append(polygon)

        g = gpd.GeoSeries(polygon_list)
        clipped_g = gpd.clip(g, (0, 0, image_height, image_width))

        return [polygon.bounds for polygon in clipped_g]


class RILLabeler(Labeler):
    """
    RIL Labeler class.
    """

    def __init__(
        self,
        labeling_date: datetime,
        dep: Literal["971", "972", "973", "974", "976", "977", "978"],
        buffer_size: int = 6,
        cap_style: int = 3
    ):
        """
        Constructor.

        Args:
            labeling_date (datetime): Date of labeling data.
            dep (Literal): Departement.
            buffer_size (int): Buffer size for RIL points.
            cap_style (int): Buffer style. 1 for round buffers,
                2 for flat buffers and 3 for square buffers.
        """
        super(RILLabeler, self).__init__(labeling_date, dep)
        self.labeling_data = load_ril(
            millesime=str(self.labeling_date.year),
            dep=self.dep)

        self.buffer_size = buffer_size
        self.cap_style = cap_style

    def create_segmentation_label(self, satellite_image: SatelliteImage) -> np.array:
        """
        Create a segmentation label (mask) from RIL data for a SatelliteImage.

        Args:
            satellite_image (SatelliteImage): Satellite image.

        Returns:
            np.array: Segmentation mask.
        """
        if self.labeling_data.crs != satellite_image.crs:
            self.labeling_data.geometry = self.labeling_data.geometry.to_crs(
                satellite_image.crs
            )

        # Filtering geometries from RIL
        xmin, ymin, xmax, ymax = satellite_image.bounds
        patch = self.labeling_data.cx[xmin:xmax, ymin:ymax].copy()

        patch.geometry = patch.geometry.buffer(
            self.buffer_size, cap_style=self.cap_style
        )

        if patch.empty:
            rasterized = np.zeros(satellite_image.array.shape[1:])
        else:
            rasterized = rasterize(
                patch.geometry,
                out_shape=satellite_image.array.shape[1:],
                fill=0,
                out=None,
                transform=satellite_image.transform,
                all_touched=True,
                default_value=1,
                dtype=None,
            )

        return rasterized


class BDTOPOLabeler(Labeler):
    """ """

    def __init__(self, labeling_date: datetime):
        """
        Constructor.

        Args:
            labeling_date (datetime): Date of labeling data.
        """
        super(BDTOPOLabeler, self).__init__(labeling_date)
        self.labeling_data = load_bdtopo(self.labeling_date)

    def create_segmentation_label(self, satellite_image: SatelliteImage) -> np.array:
        """
        Create a segmentation label (mask) from BDTOPO data for a SatelliteImage.

        Args:
            satellite_image (SatelliteImage): Satellite image.

        Returns:
            np.array: Segmentation mask.
        """
        if self.labeling_data.crs != satellite_image.crs:
            self.labeling_data.geometry = self.labeling_data.geometry.to_crs(
                satellite_image.crs
            )

        # Filtering geometries from BDTOPO
        xmin, ymin, xmax, ymax = satellite_image.bounds
        patch = self.labeling_data.cx[xmin:xmax, ymin:ymax].copy()

        if patch.empty:
            rasterized = np.zeros(satellite_image.array.shape[1:])
        else:
            rasterized = rasterize(
                patch.geometry,
                out_shape=satellite_image.array.shape[1:],
                fill=0,
                out=None,
                transform=satellite_image.transform,
                all_touched=True,
                default_value=1,
                dtype=None,
            )

        return rasterized
