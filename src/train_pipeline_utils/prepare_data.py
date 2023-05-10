import os

import numpy as np
import rasterio

from classes.data.satellite_image import SatelliteImage
from classes.labelers.labeler import Labeler
from utils.filter import has_cloud, is_too_black2, mask_full_cloud, patch_nocloud


def check_labelled_images(output_directory_name):
    """
    checks that there is not already a directory with images and their mask.
    if it doesn't exist, it is created.

    Args:
        output_directory_name: a string representing the path to the directory \
        that may already contain data and masks.

    Returns:
        boolean: True if the directory exists and is not empty. \
            False if the directory doesn't exist or is empty.
    """

    print("Entre dans la fonction check_labelled_images")
    output_images_path = output_directory_name + "/images"
    output_masks_path = output_directory_name + "/labels"
    if (os.path.exists(output_masks_path)) and (
        len(os.listdir(output_masks_path)) != 0
    ):
        print("The directory already exists and is not empty.")
        return True
    elif (os.path.exists(output_masks_path)) and (
        len(os.listdir(output_masks_path)) == 0
    ):
        print("The directory exists but is empty.")
        return False
    else:
        print("The directory doesn't exist and is going to be created.")
        os.makedirs(output_images_path)
        os.makedirs(output_masks_path)
        print("Directory created")
        return False


# def split_images(file_path, n_bands):
#     """
#     splits the images if they are not already at the expected size.

#     Args:
#         file path: a string representing the path to the directory \
#         that contains the data to be splitted.
#         n_bands: an integer representing the number of \
#         bands in the input images.

#     Returns:
#         list[SatelliteImage] : the list containing the splitted data.
#     """

#     # print("Entre dans la fonction split_images")
#     list_name = os.listdir(file_path)
#     list_path = [file_path + "/" + name for name in list_name]

#     list_images = [
#         SatelliteImage.from_raster(
#             file_path=path, dep=None, date=None, n_bands=n_bands
#         )
#         for path in list_path
#     ]

#     if list_images[0].array.shape[1] != 250:
#         list_splitted_images = [image.split(250) for image in list_images]
#         list_splitted_images = sum(list_splitted_images, [])
#     else:
#         list_splitted_images = list_images

#     # print("Nombre d'images splitées : ", len(list_splitted_images))
#     return list_splitted_images


def filter_images(src, list_images):
    """
    calls the appropriate function according to the data type.

    Args:
        src : the string that specifies the data type.
        list_images : the list containing the splitted data to be filtered.

    Returns:
        function : a call to the appopriate filter function according to\
            the data type.
    """

    # print("Entre dans la fonction filter_images")
    if src == "PLEIADES":
        return filter_images_pleiades(list_images)
    elif src == "SENTINEL2":
        return filter_images_sentinel2(list_images)


def filter_images_pleiades(list_images):
    """
    filters the Pleiades images that are too dark and/or contain clouds.

    Args:
        list_images : the list containing the splitted data to be filtered.

    Returns:
        list[SatelliteImage] : the list containing the splitted and filtered data.
    """

    # print("Entre dans la fonction filter_images_pleiades")
    list_filtered_splitted_images = []

    for splitted_image in list_images:
        if not has_cloud(splitted_image):
            if not is_too_black2(splitted_image):
                list_filtered_splitted_images.append(splitted_image)

    # print(
    #     "Nombre d'images splitées et filtrées (nuages et sombres) : ",
    #     len(list_filtered_splitted_images),
    # )
    return list_filtered_splitted_images


def filter_images_sentinel2(list_images):
    """
    filters the Sentinel2 images.

    Args:
        list_images : the list containing the splitted data to be filtered.

    Returns:
        list[SatelliteImage] : the list containing the splitted and\
            filtered data.
    """

    # print("Entre dans la fonction filter_images_sentinel2")
    return list_images


def label_images(list_images, labeler):
    """
    labels the images according to type of labeler desired.

    Args:
        list_images : the list containing the splitted and filtered data \
            to be labeled.
        labeler : a Labeler object representing the labeler \
            used to create segmentation labels.

    Returns:
        list[SatelliteImage] : the list containing the splitted and \
            filtered data with a not-empty mask and the associated masks.
    """

    # print("Entre dans la fonction label_images")
    list_masks = []
    list_filtered_splitted_labeled_images = []

    for satellite_image in list_images:
        mask = labeler.create_segmentation_label(satellite_image)
        if np.sum(mask) != 0:
            list_filtered_splitted_labeled_images.append(satellite_image)
            list_masks.append(mask)

    # print(len(list_filtered_splitted_labeled_images), len(list_masks))
    return list_filtered_splitted_labeled_images, list_masks


def save_images_and_masks(list_images, list_masks, output_directory_name):
    """
    write the couple images/masks into a specific folder.

    Args:
        list_images : the list containing the splitted and filtered data \
            to be saved.
        list_masks : the list containing the masks to be saved.
        a string representing the name of the output \
            directory where the split images and their masks should be saved.

    Returns:
        str: The name of the output directory.
    """

    # print("Entre dans la fonction save_images_and_masks")
    output_images_path = output_directory_name + "/images"
    output_masks_path = output_directory_name + "/labels"
    i = 0
    for image, mask in zip(list_images, list_masks):

        bb = image.bounds
        filename = str(int(bb[0])) + "_" + str(int(bb[1])) + "_" + str(i)
        i = i + 1
        try:
            image.to_raster(output_images_path, filename + ".jp2")
            np.save(output_masks_path + "/" + filename + ".npy", mask)
        except rasterio._err.CPLE_AppDefinedError:
            print("Writing error")
            continue

    # nb = len(os.listdir(output_directory_name + "/images"))
    # print(str(nb) + " couples images/masques retenus")
    return None

