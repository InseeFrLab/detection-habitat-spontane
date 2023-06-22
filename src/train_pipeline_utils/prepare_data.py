import os

import numpy as np
import rasterio
import random

from utils.filter import is_too_black, is_too_water


def check_labelled_images(output_directory_name):
    """
    checks that there is not already a directory with images and their mask.
    if it doesn't exist, it is created.

    Args:
        output_directory_name: a string representing the path to \
            the directory that may already contain data and masks.

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
        os.makedirs(output_images_path)
        os.makedirs(output_masks_path)
        print("Directory created")
        return False


def filter_images(src, list_images,list_array_cloud = None):
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
        return filter_images_pleiades(list_images, list_array_cloud)
    else:
        return filter_images_sentinel(list_images)


def filter_images_pleiades(list_images, list_array_cloud):
    """
    filters the Pleiades images that are too dark and/or contain clouds.

    Args:
        list_images : the list containing the splitted data to be filtered.

    Returns:
        list[SatelliteImage] : the list containing the splitted \
            and filtered data.
    """
    # print("Entre dans la fonction filter_images_pleiades")
    list_filtered_splitted_images = []

    if list_array_cloud:
        for splitted_image, array_cloud in zip(list_images, list_array_cloud):
            if not is_too_black2(splitted_image):
                prop_cloud = np.sum(array_cloud)/(array_cloud.shape[0])**2
                if not prop_cloud > 0:
                    list_filtered_splitted_images.append(splitted_image)
    else:
        for splitted_image in list_images:
            if not is_too_black2(splitted_image):
                list_filtered_splitted_images.append(splitted_image)

    return list_filtered_splitted_images


def filter_images_sentinel(list_images):
    """
    filters the Sentinel images.

    Args:
        list_images : the list containing the splitted data to be filtered.

    Returns:
        list[SatelliteImage] : the list containing the splitted and\
            filtered data.
    """

    # print("Entre dans la fonction filter_images_sentinel")
    list_filtered_splitted_images = []
    for splitted_image in list_images:
        if is_too_water(splitted_image, 75):
            list_filtered_splitted_images.append(splitted_image)

    return list_filtered_splitted_images


def label_images(list_images, labeler, prop):
    """
    labels the images according to type of labeler desired.

    Args:
        list_images : the list containing the splitted and filtered data \
            to be labeled.
        labeler : a Labeler object representing the labeler \
            used to create segmentation labels.
        prop : the proportion of the images without label (0 for no empty data,
            1 for an equal proportion between empty and no empty data, etc)

    Returns:
        list[SatelliteImage] : the list containing the splitted and \
            filtered data with a not-empty mask and the associated masks.
    """

    # print("Entre dans la fonction label_images")
    list_filtered_splitted_labeled_images = []
    list_masks = []
    list_filtered_splitted_labeled_images_empty = []
    list_masks_empty = []

    # Store labeled images in two lists depending on whether they 
    # contain buildings
    for satellite_image in list_images:
        mask = labeler.create_segmentation_label(satellite_image)
        if np.sum(mask) != 0:
            list_filtered_splitted_labeled_images.append(satellite_image)
            list_masks.append(mask)
        else:
            list_filtered_splitted_labeled_images_empty.append(satellite_image)
            list_masks_empty.append(mask)

    # Getting images with buildings and without according
    # to a certain proportion
    length_labelled = len(list_filtered_splitted_labeled_images)
    lenght_unlabelled = prop * length_labelled
    if lenght_unlabelled < len(list_filtered_splitted_labeled_images_empty):
        list_to_add = random.sample(
            list_filtered_splitted_labeled_images_empty,
            lenght_unlabelled)
        for i in list_to_add:
            idx = list_filtered_splitted_labeled_images_empty.index(i)
            list_filtered_splitted_labeled_images.append(i)
            list_masks.append(list_masks_empty[idx])
    else:
        list_filtered_splitted_labeled_images.extend(
            list_filtered_splitted_labeled_images_empty
        )
        list_masks.extend(list_masks_empty)

    # print(
    #     "Nombre d'images labelisées : ",
    #     len(list_filtered_splitted_labeled_images),
    #     ", Nombre de masques : ",
    #     len(list_masks),
    # )
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

    for image, mask in zip(list_images, list_masks):
        # image = list_images[0]
        # bb = image.bounds

        # filename = str(bb[0]) + "_" + str(bb[1]) + "_" \
        #   + "{:03d}".format(i)
        filename = image.filename.split(".")[0]

        try:
            image.to_raster(output_images_path, filename + ".jp2", "jp2", None)
            np.save(output_masks_path + "/" + filename + ".npy", mask)
        except rasterio._err.CPLE_AppDefinedError:
            # except:
            print("Writing error", image.filename)
            continue

    return None
