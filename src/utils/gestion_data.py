import os
from typing import Type

import numpy as np
import s3fs
from torch.utils.data import Dataset
from tqdm import tqdm

from models.components.segmentation_models import DeepLabv3Module

from utils.labeler import Labeler
from utils.satellite_image import SatelliteImage
from utils.utils import get_environment, get_root_path, update_storage_access
from utils.filter import (
    is_too_black2, has_cloud, mask_full_cloud, patch_nocloud,
)


def load_pleiade_data(year: int, territory: str):
    """
    Load Pleiades satellite data for a given year and territory.

    This function downloads satellite data from an S3 bucket, \
    updates storage access, and saves the data locally. \
    The downloaded data is specific to the given year and territory.

    Args:
        year (int): Year of the satellite data.
        territory (str): Territory for which the satellite \
        data is being loaded.

    Returns:
        str: The local path where the data is downloaded.
    """

    update_storage_access()
    root_path = get_root_path()
    environment = get_environment()
    dep = territory.lower()

    bucket = environment["bucket"]
    path_s3 = environment["sources"]["PLEIADES"][year][dep]
    path_local = os.path.join(
        root_path, environment["local-path"]["PLEIADES"][year][dep]
    )

    if os.path.exists(path_local):
        return path_local

    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
    )
    print("download " + territory + " " + str(year) + " in " + path_local)
    fs.download(
        rpath=f"{bucket}/{path_s3}", lpath=f"{path_local}", recursive=True
    )

    return path_local


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
    """

    output_images_path = output_directory_name + "/images"
    output_masks_path = output_directory_name + "/labels"

    if os.path.exists(output_images_path):
        print("fichiers déjà écrits")
        return None

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
                big_satellite_image, mask_full, nb_patch=250
            )
            list_satellite_image = [patch for patch in list_patch_filtered if
                                    not is_too_black2(patch)]
        else:
            list_patch_filtered = big_satellite_image.split(250)
            list_satellite_image = [patch for patch in list_patch_filtered if
                                    not is_too_black2(patch)]

        for i, satellite_image in enumerate(list_satellite_image):
            mask = labeler.create_segmentation_label(satellite_image)
            file_name_i = file_name.split(".")[0] + "_" + str(i)
            if np.sum(mask) == 0:  # je dégage les masques vides j'écris pas
                continue
            satellite_image.to_raster(output_images_path, file_name_i + ".jp2")
            np.save(
                output_masks_path + "/" + file_name_i + ".npy", mask
            )  # save
    dir = str(len(os.listdir(output_directory_name + "/images")))
    print(dir + " couples images masques retenus")


def build_dataset_train(
    year: int,
    territory: str,
    dep: str,
    labeler: Labeler,
    tile_size: int,
    n_bands: int,
    train_directory_name: str,
    dataset_class: Type[Dataset],
):
    local_path = load_pleiade_data(year, territory)
    write_splitted_images_masks(
        local_path, train_directory_name, labeler, tile_size, n_bands, dep
    )

    labels = os.listdir(train_directory_name + "/labels")
    images = os.listdir(train_directory_name + "/images")
    list_path_labels = np.sort(
        [train_directory_name + "/labels/" + name for name in labels]
    )
    list_path_images = np.sort(
        [train_directory_name + "/images/" + name for name in images]
    )

    dataset = dataset_class(list_path_images, list_path_labels)

    return dataset


def build_dataset_test(
    filepath: str,
    n_bands: int,
    tile_size: int,
    labeler: Labeler,
    dataset_class: Type[Dataset],
):
    """
    Build a dataset for testing with Pleiades satellite images. \
    This dataset test is based on one image of 2000x2000 \
    and split into patchs of size tile_size
    En attente de la liste d'images annotées réaliséees par Raya !

    Args:
        filepath (str): Filepath of the input raster image.
        n_bands (int): Number of bands in the input image.
        tile_size (int): patchs size
        labeler: a Labeler object representing \
        the labeler used to create segmentation labels.

    Returns:
        PleiadeDataset: Dataset containing the test images and labels.

    filepath_examples :
        "../data/PLEIADES/2020/MAYOTTE/ORT_2020052526670967_0519_8586_U38S_8Bits.jp2"
        "../data/PLEIADES/2022/MARTINIQUE/ORT_2022_0691_1638_U20N_8Bits.jp2"
        "../data/PLEIADES/2020/MAYOTTE/ORT_2020052526670967_0524_8590_U38S_8Bits.jp2
        "../data/PLEIADES/2020/MAYOTTE/ORT_2020052526670967_0524_8590_U38S_8Bits.jp2"
    """

    satellite_image = SatelliteImage.from_raster(
        file_path=filepath, dep=None, date=None, n_bands=n_bands
    )

    list_test = satellite_image.split(tile_size)
    output_images_path = "../test_data/images/"
    output_masks_path = "../test_data/masks/"

    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    if not os.path.exists(output_masks_path):
        os.makedirs(output_masks_path)

    for i, simg in enumerate(list_test):
        file_name = simg.filename.split(".")[0] + "_" + str(i)
        simg.to_raster(output_images_path, file_name + ".jp2")
        mask = labeler.create_segmentation_label(simg)
        np.save(output_masks_path + file_name + ".npy", mask)

    labels = os.listdir(output_masks_path)
    images = os.listdir(output_images_path)

    list_path_images_test = np.sort(
        [output_images_path + filename for filename in images]
    )

    list_path_labels_test = np.sort(
        [output_masks_path + filename for filename in labels]
    )

    dataset_test = dataset_class(list_path_images_test, list_path_labels_test)

    return dataset_test


def instantiate_module(module_type, nchannel=3):
    """
    Instantiate a module based on the provided module type.

    Args:
        module_type (str): Type of module to instantiate.

    Returns:
        object: Instance of the specified module.
    """
    module_dict = {"deeplabv3": DeepLabv3Module}

    if module_type not in module_dict:
        raise ValueError("Invalid module type")

    if module_type == "deeplabv3":
        return module_dict[module_type](nchannel)
    else:
        return module_dict[module_type]()
