"""
"""
import os
from typing import Literal, Union

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize, shapes
from scipy.ndimage import label
from shapely.geometry import Polygon, box
from tqdm import tqdm

from classes.data.labeled_satellite_image import (
    DetectionLabeledSatelliteImage,
    SegmentationLabeledSatelliteImage,
)
from classes.data.satellite_image import SatelliteImage
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


def is_too_black2(image: SatelliteImage, black_area=0.5) -> bool:
    """
    Determines if an image has too many black pixels.
    This function works on RGB images in the format (C x H x W)
    encoded in float.

    Parameters:
    -----------
    image (SatelliteImage) :
        A SatelliteImage object representing the image to analyze.

    black_area (float, optional) :
        A float representing the maximum percentage of black pixels allowed in
        the image. The default value is 0.5, which means that if more than 50%
        of the image is black, the function will return True.

    Returns:
    --------
    bool:
        True if the image has too many black pixels, False otherwise.

    Example:
        >>> filename = '../data/PLEIADES/2020/MAYOTTE/
        ORT_2020052526656219_0508_8599_U38S_8Bits.jp2'
        >>> date = date.fromisoformat('2020-01-01')
        >>> image = SatelliteImage.from_raster(
                filename,
                date=date,
                n_bands=3,
                dep="976"
            )
        >>> is_too_black2(image)
        False
    """

    # Extract the array from the image to get the pixel values
    img = image.array.copy()
    img = img[[0, 1, 2], :, :]
    img = img.astype(np.uint8)
    img = img.transpose(1, 2, 0)

    # Find all black pixels
    black_pixels = np.where(
        (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
    )

    nb_black_pixels = len(black_pixels[0])

    if (nb_black_pixels / (img.shape[0] ** 2)) >= black_area:
        return True
    else:
        return False


def has_cloud(
    image: SatelliteImage,
    threshold: float = 0.98,
    min_size: int = 50000,
) -> bool:
    """
    Determines if an image contains cloud(s) or not.

    Parameters:
    -----------
    image (SatelliteImage):
        A SatelliteImage object representing the image to analyze.

    threshold (int, optional):
        An integer representing the threshold for coverage of the center of
        clouds in the image. Pixels with a cloud coverage value higher than
        this threshold are classified as covered by clouds.
        Defaults to 0.98 (white pixels).
    min_size (int, optional):
        An integer representing the minimum size (in pixels) of a cloud
        region that will be retained in the output mask. Defaults to 50,000
        (2,000*2,000 = 4,000,000 pixels and we want to detect clouds that
        occupy > 1.25% of the image).


    Returns:
    --------
    bool
        True if the image contains cloud(s), False otherwise.

    Example:
        >>> filename_1 = '../data/PLEIADES/2020/MAYOTTE/
        ORT_2020052526656219_0508_8599_U38S_8Bits.jp2'
        >>> date_1 = date.fromisoformat('2020-01-01')
        >>> image_1 = SatelliteImage.from_raster(
                                    filename_1,
                                    date = date_1,
                                    n_bands = 3,
                                    dep = "976"
                                )
        >>> has_cloud(image_1)
        True
    """
    
    copy_image = image.copy()
    
    if not copy_image.normalized:
        copy_image.normalize()

    image = copy_image.array
    image = image[[0, 1, 2], :, :]
    image = (image*np.max(image)).astype(np.float64)
    image = image.transpose(1, 2, 0)

    # Convert the RGB image to grayscale
    grayscale = np.mean(image, axis=2)

    # Find clusters of white pixels that correspond to 5% or more of the image
    labeled, num_features = label(grayscale > threshold)

    if num_features >= 1:
        region_sizes = np.bincount(labeled.flat)[1:]
        max_size_feature = max(region_sizes)

        # Check if the size of the largest region of white pixels
        # exceeds the minimum size threshold
        if max_size_feature >= min_size:
            return True

    return False


def mask_cloud(
    image: SatelliteImage, threshold: float = 0.98, min_size: int = 50000
) -> np.ndarray:
    """
    Detects clouds in a SatelliteImage using a threshold-based approach
    (grayscale threshold and pixel cluster size threshold) and
    returns a binary mask of the detected clouds.
    This function works on RGB images in the format (C x H x W)
    encoded in float.

    Args:
        image (SatelliteImage):
            The input satellite image to process.
        threshold (int):
            The threshold value to use for detecting clouds on the image
            transformed into grayscale. A pixel is considered part of a
            cloud if its value is greater than this threshold.
            Default to 0.98.
        min_size (int):
            The minimum size (in pixels) of a cloud region to be
            considered valid.
            Default to 50000.

    Returns:
        mask (np.ndarray):
            A binary mask of the detected clouds in the input image.

    Example:
        >>> filename_1 = '../data/PLEIADES/2020/MAYOTTE/
        ORT_2020052526656219_0508_8599_U38S_8Bits.jp2'
        >>> date_1 = date.fromisoformat('2020-01-01')
        >>> image_1 = SatelliteImage.from_raster(
                                    filename_1,
                                    date = date_1,
                                    n_bands = 3,
                                    dep = "976"
                                )
        >>> mask = mask_cloud(image_1)
        >>> fig, ax = plt.subplots(figsize=(10, 10))
        >>> ax.imshow(np.transpose(image_1.array, (1, 2, 0))[:,:,:3])
        >>> ax.imshow(mask, alpha=0.3)
    """
    copy_image = image.copy()
    
    if not copy_image.normalized:
        copy_image.normalize()
    
    image = copy_image.array
    image = image[[0, 1, 2], :, :]
    image = (image*np.max(image)).astype(np.float64)
    image = image.transpose(1, 2, 0)

    # Convert the RGB image to grayscale
    grayscale = np.mean(image, axis=2)

    # Find clusters of white pixels that correspond to 5% or more of the image
    labeled, num_features = label(grayscale > threshold)

    region_sizes = np.bincount(labeled.flat)

    # Trier les labels de région en fonction de leur taille décroissante
    sorted_labels = np.argsort(-region_sizes)

    # Minimum size of the cluster
    mask = np.zeros_like(labeled)

    if num_features >= 1:
        #for i in tqdm(range(1, num_features + 1)):  # Display the progress bar
        for i in range(1, num_features + 1):
            if region_sizes[sorted_labels[i]] >= min_size:
                mask[labeled == sorted_labels[i]] = 1
            else:
                break

    # Return the cloud mask
    return mask


def mask_full_cloud(
    image: SatelliteImage,
    threshold_center: float = 0.98,
    threshold_full: float = 0.7,
    min_size: int = 50000,
) -> np.ndarray:
    """
    Masks out clouds in a SatelliteImage using two thresholds for cloud
    coverage, and returns the resulting cloud mask as a numpy array.

    Parameters:
    -----------
    image (SatelliteImage):
        An instance of the SatelliteImage class representing the input image
        to be processed.
    threshold_center (int, optional):
        An integer representing the threshold for coverage of the center of
        clouds in the image. Pixels with a cloud coverage value higher than
        this threshold are classified as cloud-covered.
        Defaults to 0.98 (white pixels).
    threshold_full (int, optional):
        An integer representing the threshold for coverage of the full clouds
        in the image. Pixels with a cloud coverage value higher than this
        threshold are classified as covered by clouds.
        Defaults to 0.7 (light grey pixels).
    min_size (int, optional):
        An integer representing the minimum size (in pixels) of a cloud region
        that will be retained in the output mask.
        Defaults to 50,000 (2,000*2,000 = 4,000,000 pixels and we want to
        detect clouds that occupy > 1.25% of the image).

    Returns:
    --------
    rasterized (np.ndarray):
        A numpy array representing the rasterized version of the cloud mask.
        Pixels with a value of 1 are classified as cloud-free, while pixels
        with a value of 0 are classified as cloud-covered.

    Example:
        >>> filename_1 = '../data/PLEIADES/2020/MAYOTTE/
        ORT_2020052526656219_0508_8599_U38S_8Bits.jp2'
        >>> date_1 = date.fromisoformat('2020-01-01')
        >>> image_1 = SatelliteImage.from_raster(
                                    filename_1,
                                    date = date_1,
                                    n_bands = 3,
                                    dep = "976"
                                )
        >>> mask_full = mask_full_cloud(image_1)
        >>> fig, ax = plt.subplots(figsize=(10, 10))
        >>> ax.imshow(np.transpose(image_1.array, (1, 2, 0))[:,:,:3])
        >>> ax.imshow(mask_full, alpha=0.3)
    """
    # Mask out clouds from the image using different thresholds
    cloud_center = mask_cloud(image, threshold_center, min_size)
    cloud_full = mask_cloud(image, threshold_full, min_size)

    nchannel, height, width = image.array.shape

    # Create a list of polygons from the masked center clouds in order
    # to obtain a GeoDataFrame from it
    polygon_list_center = []
    for shape in list(shapes(cloud_center)):
        polygon = Polygon(shape[0]["coordinates"][0])
        if polygon.area > 0.85 * height * width:
            continue
        polygon_list_center.append(polygon)

    g_center = gpd.GeoDataFrame(geometry=polygon_list_center)

    # Same but from the masked full clouds
    polygon_list_full = []
    for shape in list(shapes(cloud_full)):
        polygon = Polygon(shape[0]["coordinates"][0])
        if polygon.area > 0.85 * height * width:
            continue
        polygon_list_full.append(polygon)

    g_full = gpd.GeoDataFrame(geometry=polygon_list_full)

    # Spatial join on the GeoDataFrames for the masked full clouds
    # and the masked center clouds
    result = gpd.sjoin(g_full, g_center, how="inner", predicate="intersects")

    # Remove any duplicate geometries
    result = result.drop_duplicates(subset="geometry")

    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(np.transpose(image.array, (1, 2, 0))[:,:,:3])
    # result.plot(color = "orange", ax=ax)

    # Rasterize the geometries into a numpy array
    if result.empty:
        rasterized = np.zeros(image.array.shape[1:])
    else:
        rasterized = rasterize(
            result.geometry,
            out_shape=image.array.shape[1:],
            fill=0,
            out=None,
            all_touched=True,
            default_value=1,
            dtype=None,
        )

    return rasterized


def patch_nocloud(
    image: SatelliteImage,
    mask_full_cloud: np.ndarray,
    size_patch: int,
) -> list[SatelliteImage]:
    """
    Splits a SatelliteImage into patches and returns a list of patches
    that do not contain clouds.

    Args:
        image (SatelliteImage):
            An instance of the SatelliteImage class representing
            the original SatelliteImage.
        mask_full_cloud (np.ndarray):
            An array representing a cloud mask, where 1 indicates a pixel
            is covered by clouds and 0 indicates it is not.
        size_patch (int):
            The requested patch size.

    Returns:
    --------
        list[SatelliteImage]:
            A list of SatelliteImage instances representing the patches
            that do not contain clouds.

    Example:
        >>> filename_1 = '../data/PLEIADES/2020/MAYOTTE/
        ORT_2020052526656219_0508_8599_U38S_8Bits.jp2'
        >>> date_1 = date.fromisoformat('2020-01-01')
        >>> image_1 = SatelliteImage.from_raster(
                                    filename_1,
                                    date = date_1,
                                    n_bands = 3,
                                    dep = "976"
                                )
        >>> mask_full = mask_full_cloud(image_1)
        >>> list_nocloud = patch_nocloud(image_1, mask_full, size_patch = 250)
    """

    # Create an RGB mask from the input cloud mask
    height, width = mask_full_cloud.shape
    mask_full_rgb = np.zeros((height, width, 3), dtype=float)
    mask_full_rgb[:, :, 0] = mask_full_cloud
    mask_full_rgb[:, :, 1] = mask_full_cloud
    mask_full_rgb[:, :, 2] = mask_full_cloud

    # Transpose the array to match the SatelliteImage format
    mask_full_rgb = mask_full_rgb.transpose(2, 0, 1)

    # Create a SatelliteImage object from the mask with the same metadata
    # as the input image
    image_cloud = SatelliteImage(
        array=mask_full_rgb,
        crs=image.crs,
        bounds=image.bounds,
        transform=image.transform,
        n_bands=image.n_bands,
        filename=image.filename,
        dep=image.dep,
        date=image.date,
        normalized=image.normalized,
    )

    # Split the image and cloud mask into patches
    list_images_cloud = image_cloud.split(size_patch)
    list_images = image.split(size_patch)
    list_patch_nocloud = []

    # Loop through each patch
    for i, mini_image in enumerate(list_images_cloud):
        mask = mini_image.array
        # Check if the patch doesn't contain any clouds
        if len(np.where(mask == 1)[0]) == 0:
            list_patch_nocloud.append(list_images[i])

    return list_patch_nocloud


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
