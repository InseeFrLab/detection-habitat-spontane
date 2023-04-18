"""
"""
import os
from typing import Literal, Union

from scipy.ndimage import label
from tqdm import tqdm
from shapely.geometry import Polygon
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from rasterio.features import rasterize, shapes

from labeled_satellite_image import (
    DetectionLabeledSatelliteImage,
    SegmentationLabeledSatelliteImage,
)
from satellite_image import SatelliteImage
from utils import get_environment, get_file_system


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

def mask_cloud(image: SatelliteImage, threshold: int, min_size: int) -> np.ndarray:
    """
    Detects clouds in a SatelliteImage using a threshold-based approach (grayscale threshold and pixel cluster size threshold) and returns a binary mask of the detected clouds.

    Args:
        image (SatelliteImage): The input satellite image to process.
        threshold (int): The threshold value to use for detecting clouds on the image transformed into grayscale. A pixel is considered part of a cloud if its value is greater than this threshold.
        min_size (int): The minimum size (in pixels) of a cloud region to be considered valid.

    Returns:
        np.ndarray: A binary mask of the detected clouds in the input image.

    Example:
        >>> filename_1 = '../data/PLEIADES/2020/MAYOTTE/ORT_2020052526656219_0508_8599_U38S_8Bits.jp2' 
        >>> date_1 = date.fromisoformat('2020-01-01')
        >>> image = SatelliteImage.from_raster(
                                    filename_1,
                                    date = date_1,
                                    n_bands = 3,
                                    dep = "976"
                                )
        >>> cloud_mask = mask_cloud(image, 250, 20000)
    """


    image = image.array.copy()

    image = image[[0, 1, 2], :, :]

    image = (image*255).astype(np.uint8)

    image = image.transpose(1, 2, 0)

    # Convertir l'image RGB en niveau de gris
    grayscale = np.mean(image, axis=2)

    # Trouver les amas de pixels blancs correspondant à 5% ou plus de l'image
    #threshold = 250  # seuil pour considérer un pixel comme "blanc"

    labeled, num_features = label(grayscale > threshold)

    #min_size = 20000  # taille minimale de l'amas (2000*2000 = 4 000 000 pixels et on souhaite détecter des nuages qui occupent + de 0,5% de l'image)
    mask = labeled.copy()

    if num_features >= 1:
        for i in tqdm(range(1, num_features + 1)): #Affichage de la barre de progression
            if np.sum(mask == i) < min_size:
                mask[mask == i] = 0
            else:
                mask[mask == i] = 1

    # Retourner le masque nuage
    return mask

def mask_full_cloud(image : SatelliteImage, threshold_center : int, threshold_full : int, min_size : int) -> np.ndarray:
    """
    Masks out clouds in a SatelliteImage using two thresholds for cloud coverage, and returns the resulting cloud-free mask as a rasterized GeoDataFrame.

    Parameters:
    -----------
    image : SatelliteImage
        An instance of the SatelliteImage class representing the input image to be processed.
    threshold_center : int
        An integer representing the threshold for center coverage of clouds in the image. Pixels with a cloud coverage value higher than this threshold are classified as cloud-covered.
    threshold_full : int
        An integer representing the threshold for full coverage of clouds in the image. Pixels with a cloud coverage value higher than this threshold are classified as fully covered by clouds.
    min_size : int
        An integer representing the minimum size (in pixels) of a cloud region that will be retained in the output mask.

    Returns:
    --------
    rasterized : np.ndarray
        A 2D numpy array representing the rasterized version of the cloud mask. Pixels with a value of 1 are classified as cloud-free, while pixels with a value of 0 are classified as cloud-covered.
    """

    nuage_center = mask_cloud(image, threshold_center, min_size)
    nuage_full = mask_cloud(image, threshold_full, min_size)
 
    image_height = image.array.shape[1]
    image_width = image.array.shape[2]

    polygon_list_center = []
    for shape in list(shapes(nuage_center)):
        polygon = Polygon(shape[0]["coordinates"][0])
        if polygon.area > 0.85 * image_height * image_width:
            continue
        polygon_list_center.append(polygon)

    # Créer un GeoSeries à partir de la liste de polygons
    g_center = gpd.GeoDataFrame(geometry=polygon_list_center)

    polygon_list_full = []
    for shape in list(shapes(nuage_full)):
        polygon = Polygon(shape[0]["coordinates"][0])
        if polygon.area > 0.85 * image_height * image_width:
            continue
        polygon_list_full.append(polygon)

    #Créer un GeoSeries à partir de la liste de polygons
    g_full = gpd.GeoDataFrame(geometry=polygon_list_full)
   
    result = gpd.sjoin(g_full, g_center, how="inner", predicate="intersects")

    rasterized = rasterize(
                    result.geometry,
                    out_shape=image.array.shape[1:],
                    fill=0,
                    out=None,
                    transform=image.transform,
                    all_touched=True,
                    default_value=1,
                    dtype=None,
                    )

    return rasterized


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
