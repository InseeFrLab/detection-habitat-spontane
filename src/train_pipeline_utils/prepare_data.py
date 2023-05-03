import os

import numpy as np
import rasterio
from tqdm import tqdm

from classes.data.satellite_image import SatelliteImage
from classes.labelers.labeler import Labeler
from utils.filter import (
    has_cloud,
    is_too_black2,
    mask_full_cloud,
    patch_nocloud
)


def check_labelled_images(
    output_directory_name
):
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

    print('Entre dans la fonction check_labelled_images')
    output_images_path = output_directory_name + "/images"
    output_masks_path = output_directory_name + "/labels"
    if ((os.path.exists(output_masks_path)) and (len(os.listdir(output_masks_path)) != 0)):
        print('Option 1')
        return True
    elif ((os.path.exists(output_masks_path)) and (len(os.listdir(output_masks_path)) == 0)):
        print('Option 2')
        return False
    else:
        print('Option 3')
        os.makedirs(output_images_path)
        os.makedirs(output_masks_path)
        return False


def split_images(
    file_path,
    n_bands
):
    """
    splits the images if they are not already at the expected size.

    Args:
        file path: a string representing the path to the directory \
        that contains the data to be splitted.
        n_bands: an integer representing the number of \
        bands in the input images.
   
    Returns:
        list[SatelliteImage] : the list containing the splitted data.
    """

    print('Entre dans la fonction split_images')
    list_name = os.listdir(file_path)
    list_path = [file_path + "/" + name for name in list_name]
    
    satellite_image_test = SatelliteImage.from_raster(
        file_path=list_path[0], dep=None, date=None, n_bands=n_bands
    )
    
    if satellite_image_test.array.shape[1] != 250:
        list_splitted_images = []
        for path, file_name in zip(list_path, tqdm(list_name)):  # tqdm ici
            big_satellite_image = SatelliteImage.from_raster(
                file_path=path, dep=None, date=None, n_bands=n_bands
            )
            splitted_images = big_satellite_image.split(250)
            for im in splitted_images:
                list_splitted_images.append(im)
    else:
        list_splitted_images = list_path
    print("Nombre d'images splitées : ", len(list_splitted_images))
    return list_splitted_images


def filter_images(
    list_images
):
    """
    filters the images that are too dark and/or contain clouds.

    Args:
        list_images : the list containing the splitted data to be filtered.
   
    Returns:
        list[SatelliteImage] : the list containing the splitted and filtered data.
    """

    print('Entre dans la fonction filter_images')
    list_filtered_splitted_images = []

    for splitted_image in tqdm(list_images):
        # has_cloud = has_cloud(splitted_image)

        if not has_cloud(splitted_image):
            if not is_too_black2(splitted_image):
                list_filtered_splitted_images.append(splitted_image)
    print("Nombre d'images splitées et filtrées (nuages et sombres) : ", len(list_filtered_splitted_images))
    return list_filtered_splitted_images


def label_images(
    list_images,
    labeler
):
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

    print('Entre dans la fonction label_images')
    list_masks = []
    list_filtered_splitted_labeled_images = []

    for satellite_image in tqdm(list_images):
        mask = labeler.create_segmentation_label(satellite_image)
        if np.sum(mask) != 0:
            list_filtered_splitted_labeled_images.append(satellite_image)
            list_masks.append(mask)
    print(len(list_filtered_splitted_labeled_images), len(list_masks))
    return list_filtered_splitted_labeled_images, list_masks


def save_images_and_masks(
    list_images,
    list_masks,
    output_directory_name
):
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

    print('Entre dans la fonction save_images_and_masks')

    output_images_path = output_directory_name + "/images"
    output_masks_path = output_directory_name + "/labels"

    i = 0
    for image, mask in zip(tqdm(list_images), list_masks):
        i += 1
        filename = str(i)
        print(image.filename, filename)
        try:
            image.to_raster(
                output_images_path, filename + ".jp2"
            )
            np.save(output_masks_path + "/" + filename + ".npy", mask)
            print("ok")
        except :
            print("erreur ecriture")
            continue
    
    print(str(len(os.listdir(output_directory_name + "/images"))) + " couples images/masques retenus")
    return os.listdir(output_images_path), os.listdir(output_masks_path)
    # return output_directory_name


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
            mask_full = mask_full_cloud(big_satellite_image) #Retourne un array avec 1 si pas nuage et 0 si nuage
            list_patch_filtered = patch_nocloud(
                big_satellite_image, mask_full, nb_patch=250
            ) #Retourne une liste des patch de taille de 250 de l'image qui ne contiennent pas de nuage
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
