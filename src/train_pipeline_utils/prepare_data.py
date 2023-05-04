import os

import numpy as np
import rasterio
from tqdm import tqdm

from classes.data.satellite_image import SatelliteImage
from classes.labelers.labeler import Labeler
from utils.filter import (
    has_cloud, is_too_black2, mask_full_cloud, patch_nocloud
)


def write_splitted_images_masks(
    file_path: str,
    output_directory_name: str,
    labeler: Labeler,
    tile_size: int,
    n_bands: int,
    dep: str,
):
    """
    write the couple images mask into a specific folder

    Args:
        file_path: a string representing the path to the directory containing \
        the input image files.
        output_directory_name: a string representing the name of the output \
        directory where the split images and masks should be saved.
        labeler: a Labeler object representing the labeler \
        used to create segmentation labels.
        tile_size: an integer representing the size of the tiles \
        to split the input image into.
        n_bands: an integer representing the number of \
        bands in the input image.
        dep: a string representing the department of the input image, \
        or None if not applicable.

    Returns:
        str: The name of the output directory.
    """

    output_images_path = output_directory_name + "/images"
    output_masks_path = output_directory_name + "/labels"

    if os.path.exists(output_images_path):
        print("fichiers déjà écrits")
        return output_directory_name

    if not os.path.exists(output_masks_path):
        os.makedirs(output_masks_path)

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    list_name = os.listdir(file_path)
    list_path = [file_path + "/" + name for name in list_name]

    for path, file_name in zip(list_path, tqdm(list_name)):  # tqdm ici
        big_satellite_image = SatelliteImage.from_raster(
            file_path=path, dep=None, date=None, n_bands=3
        )

        boolean = has_cloud(big_satellite_image)

        if boolean:
            mask_full = mask_full_cloud(big_satellite_image)
            list_patch_filtered = patch_nocloud(
                big_satellite_image, mask_full, size_patch=250
            )
            list_satellite_image = [
                patch
                for patch in list_patch_filtered
                if not is_too_black2(patch)
            ]
        else:
            list_patch_filtered = big_satellite_image.split(250)
            list_satellite_image = [
                patch
                for patch in list_patch_filtered
                if not is_too_black2(patch)
            ]

        for i, satellite_image in enumerate(list_satellite_image):
            mask = labeler.create_segmentation_label(satellite_image)
            file_name_i = file_name.split(".")[0] + "_" + str(i)
            if np.sum(mask) == 0:  # je dégage les masques vides j'écris pas
                continue
            try:
                satellite_image.to_raster(
                    output_images_path, file_name_i + ".jp2"
                )
                np.save(output_masks_path + "/" + file_name_i + ".npy", mask)
            except rasterio._err.CPLE_AppDefinedError:
                print("erreur ecriture " + file_name_i)
                continue

    dir = str(len(os.listdir(output_directory_name + "/images")))
    print(dir + " couples images masques retenus")

    return output_directory_name
