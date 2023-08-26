import gc
import json
import os
import random
import sys
from datetime import datetime
import csv
import shutil
from osgeo import gdal

import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from rasterio.errors import RasterioIOError
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from yaml.loader import SafeLoader

from classes.data.labeled_satellite_image import (  # noqa: E501
    SegmentationLabeledSatelliteImage,
)
from classes.data.satellite_image import SatelliteImage

from classes.optim.optimizer import generate_optimization_elements
from dico_config import (
    labeler_dict,
    dataset_dict,
    loss_dict,
    module_dict,
    task_to_evaluation,
    task_to_lightningmodule,
)
from train_pipeline_utils.download_data import (
    load_2satellites_data,
    load_donnees_test,
    load_satellite_data,
    load_s2_looking
)
from train_pipeline_utils.handle_dataset import (
    generate_transform_pleiades,
    generate_transform_sentinel,
    select_indices_to_balance,
    select_indices_to_split_dataset,
)
from train_pipeline_utils.prepare_data import (
    check_labelled_images,
    filter_images,
    label_images,
    save_images_and_masks,
    extract_proportional_subset,
    filter_images_by_path,
    prepare_data_per_doss,
)

from utils.utils import remove_dot_file, split_array, update_storage_access, list_sorted_filenames

# with open("../config.yml") as f:
#     config = yaml.load(f, Loader=SafeLoader)


def download_data(config):
    """
    Downloads data based on the given configuration.

    Args:
        config: a dictionary representing the
        configuration information for data download.

    Returns:
        A list of output directories for each downloaded dataset.
    """

    print("Entre dans la fonction download_data")
    config_data = config["donnees"]
    list_output_dir = []
    list_masks_cloud_dir = []

    years = config_data["year"]
    deps = config_data["dep"]
    src = config_data["source train"]

    if src == "S2Looking":
        list_output_dir = load_s2_looking()
        test_dir = []
    
    else:
        for year, dep in zip(years, deps):
            # year, dep = years[0], deps[0]
            if src == "PLEIADES":
                cloud_dir = load_satellite_data(year, dep, "NUAGESPLEIADES")
                list_masks_cloud_dir.append(cloud_dir)
                output_dir = load_satellite_data(year, dep, src)
            elif src == "SENTINEL1-2" or src == "SENTINEL1-2-RVB":
                output_dir = load_2satellites_data(year, dep, src)
            else:
                output_dir = load_satellite_data(year, dep, src)
            list_output_dir.append(output_dir)

        print("chargement des données test")
        test_dir = load_donnees_test(
            type=config["donnees"]["task"], src=config["donnees"]["source train"]
        )

    return list_output_dir, list_masks_cloud_dir, test_dir


def prepare_data_s2(config, output_dir):
    """
    Preprocesses and splits the raw input images
    into tiles and corresponding masks,
    and saves them in the specified output directories.

    Args:
        config: A dictionary representing the configuration settings.
        list_data_dir: A list of strings representing the paths
        to the directories containing the raw input image files.

    Returns:
        A list of strings representing the paths to
        the output directories containing the
        preprocessed tile and mask image files.
    """

    print("Entre dans la fonction prepare_data")
    tile_size = config["donnees"]["tile size"]

    dir_train = output_dir + "/" + "/train/"
    dir_val = output_dir + "/" + "/val/"
    dir_test = output_dir + "/" + "/test/"

    print("Preparation des données train")
    type_data = "train"
    output_dir_train = prepare_data_per_doss(dir_train, tile_size, type_data)

    print("Preparation des données val")
    type_data = "valid"
    output_dir_valid = prepare_data_per_doss(dir_val, tile_size, type_data)

    print("Preparation des données test")
    type_data = "test"
    output_dir_test = prepare_data_per_doss(dir_test, tile_size, type_data)

    return [output_dir_train, output_dir_valid, output_dir_test]

def prepare_train_data_else(config, list_data_dir, list_masks_cloud_dir):
    """
    Preprocesses and splits the raw input images
    into tiles and corresponding masks,
    and saves them in the specified output directories.

    Args:
        config: A dictionary representing the configuration settings.
        list_data_dir: A list of strings representing the paths
        to the directories containing the raw input image files.

    Returns:
        A list of strings representing the paths to
        the output directories containing the
        preprocessed tile and mask image files.
    """

    print("Entre dans la fonction prepare_data")
    config_data = config["donnees"]

    years = config_data["year"]
    deps = config_data["dep"]
    src = config_data["source train"]
    type_labeler = config_data["type labeler"]
    config_task = config_data["task"]
    tile_size = config_data["tile size"]
    max_size_ones = config_data["max size ones"]
    mask_rgb_threshold = config_data["mask rgb threshold"]

    # Define the total of ones we want in each dossier
    max_echant_ones = max_size_ones/len(years)

    list_output_dir = []

    for i, (year, dep) in enumerate(zip(years, deps)):
        # i, year , dep = 0,years[0],deps[0]
        output_dir = (
            "../train_data"
            + "-"
            + config_task
            + "-"
            + src
            + "-"
            + type_labeler
            + "-"
            + dep
            + "-"
            + str(year)
            + "-"
            + str(tile_size)
            + "/"
        )

        if not check_labelled_images(output_dir):
            date = datetime.strptime(str(year) + "0101", "%Y%m%d")
            Labeler = labeler_dict[type_labeler]
            if type_labeler == "RIL":
                buffer_size = config_data["buffer size"]
                labeler = Labeler(date, dep=dep, buffer_size=buffer_size)
            else:
                labeler = Labeler(date, dep=dep)

            list_name_cloud = []
            if src == "PLEIADES":
                cloud_dir = list_masks_cloud_dir[i]
                list_name_cloud = [
                    path.split("/")[-1].split(".")[0]
                    for path in os.listdir(cloud_dir)
                ]

            dir = list_data_dir[i]
            list_path = [dir + "/" + filename for filename in remove_dot_file(os.listdir(dir))]

            random.shuffle(list_path)

            full_balancing_dict = {}
            cpt_ones = 0

            for path in tqdm(list_path):
                if cpt_ones >= max_echant_ones:
                    break

                try:
                    si = SatelliteImage.from_raster(
                        file_path=path,
                        dep=dep,
                        date=None,
                        n_bands=config_data["n bands entry"],
                    )
                except RasterioIOError:
                    print("Erreur de lecture du fichier " + path)
                    continue

                filename = path.split("/")[-1].split(".")[0]
                list_splitted_mask_cloud = None

                if filename in list_name_cloud:
                    mask_full_cloud = np.load(cloud_dir + "/" + filename + ".npy")
                    list_splitted_mask_cloud = split_array(
                        mask_full_cloud, config_data["tile size"]
                    )

                list_splitted_images = si.split(config_data["tile size"])

                list_filtered_splitted_images = filter_images(
                    config_data["source train"],
                    list_splitted_images,
                    list_splitted_mask_cloud,
                )

                labels, balancing_dict = label_images(
                    list_filtered_splitted_images, labeler, config_task, mask_rgb_threshold[i]
                )

                nb_ones = sum(1 for value in balancing_dict.values() if value == 1)
                cpt_ones = cpt_ones + nb_ones

                save_images_and_masks(
                    list_filtered_splitted_images,
                    labels,
                    output_dir,
                    direc=dir,
                    task=config_task,
                )

                for k, v in balancing_dict.items():
                    full_balancing_dict[k] = v

            with open(output_dir + "/balancing_dict.json", "w") as fp:
                json.dump(full_balancing_dict, fp)

        list_output_dir.append(output_dir)
        nb = len(os.listdir(output_dir + "/images"))
        print(str(nb) + " couples images/masques retenus")

    return list_output_dir


def prepare_train_data(config, list_data_dir, list_masks_cloud_dir):
    """
    Preprocesses and splits the raw input images
    into tiles and corresponding masks,
    and saves them in the specified output directories.

    Args:
        config: A dictionary representing the configuration settings.
        list_data_dir: A list of strings representing the paths
        to the directories containing the raw input image files.

    Returns:
        A list of strings representing the paths to
        the output directories containing the
        preprocessed tile and mask image files.
    """
    src = config["donnees"]["source train"]

    if src == "S2Looking":
        list_output_dir = prepare_data_s2(config, list_data_dir)
    
    else:
        list_output_dir = prepare_train_data_else(config, list_data_dir, list_masks_cloud_dir)
    
    return list_output_dir


def prepare_test_data(config, test_dir):
    print("Entre dans la fonction prepare_test_data")

    config_data = config["donnees"]

    n_bands = config_data["n bands"]
    src = config_data["source train"]
    type_labeler = config_data["type labeler"]
    config_task = config_data["task"]
    tile_size = config_data["tile size"]

    if src != "S2Looking":

        output_test = (
            "../test_data"
            + "-"
            + config_task
            + "-"
            + src
            + "-"
            + type_labeler
            + "-"
            + str(tile_size)
            + "/"
        )

        output_labels_path = output_test + "/labels"

        if not os.path.exists(output_labels_path):
            os.makedirs(output_labels_path)
        else:
            return output_test

        labels_path = test_dir + "/masks"
        list_name_label = os.listdir(labels_path)
        list_name_label = np.sort(remove_dot_file(list_name_label))
        list_labels_path = [labels_path + "/" + name for name in list_name_label]

        if config_task == "segmentation":
            images_path = test_dir + "/images"
            list_name_image = os.listdir(images_path)
            list_name_image = np.sort(remove_dot_file(list_name_image))
            list_images_path = [images_path + "/" + name for name in list_name_image]
            output_images_path = output_test + "/images"
            if not os.path.exists(output_images_path):
                os.makedirs(output_images_path)

            if len(list_labels_path) < len(list_images_path):
                diff = len(list_images_path) - len(list_labels_path)
                file_to_duplicate = list_labels_path[0]
                for i in range(diff):
                    new_file = file_to_duplicate[0:-4]+f"_{i}.npy"
                    shutil.copyfile(file_to_duplicate, new_file)
                    list_labels_path.append(list_labels_path[0][0:-4]+f"_{i}.npy")

            for image_path, label_path, name in zip(
                list_images_path,
                list_labels_path,
                list_name_image
            ):

                si = SatelliteImage.from_raster(
                    file_path=image_path, dep=None, date=None, n_bands=n_bands
                )
                mask = np.load(label_path)

                lsi = SegmentationLabeledSatelliteImage(si, mask, "", "")
                list_lsi = lsi.split(tile_size)

                for i, lsi in enumerate(list_lsi):
                    if np.isnan(lsi.satellite_image.array).any():
                        continue
                    file_name_i = name.split(".")[0] + "_" + "{:04d}".format(i)
                    in_ds = gdal.Open(image_path)
                    proj = in_ds.GetProjection()
                    lsi.satellite_image.to_raster(
                        output_images_path, file_name_i, "tif", proj
                        )
                    # lsi.satellite_image.to_raster(
                    #     output_images_path, file_name_i + ".jp2"
                    #     )
                    np.save(output_labels_path + "/" + file_name_i + ".npy", lsi.label)

        elif config_task == "classification":

            images_path = test_dir + "/images"
            list_name_image = os.listdir(images_path)
            list_name_image = np.sort(remove_dot_file(list_name_image))
            list_images_path = [images_path + "/" + name for name in list_name_image]
            output_images_path = output_test + "/images"

            for image_path, label_path, name in zip(
                list_images_path,
                list_labels_path,
                list_name_image
            ):

                si = SatelliteImage.from_raster(
                    file_path=image_path, dep=None, date=None, n_bands=n_bands
                )
                mask = np.load(label_path)

                lsi = SegmentationLabeledSatelliteImage(si, mask, "", "")
                list_lsi = lsi.split(tile_size)
                csv_file_path = output_labels_path + "/" + "fichierlabeler.csv"

                for i, lsi in enumerate(list_lsi):
                    file_name_i = name.split(".")[0] + "_" + "{:04d}".format(i)
                    
                    if config["donnees"]["source train"] == "PLEIADES":
                        lsi.satellite_image.to_raster(
                            output_images_path, file_name_i + ".jp2"
                            )
                    else:
                        lsi.satellite_image.to_raster(
                            output_images_path, file_name_i, "tif", proj
                            )

                    if np.sum(lsi.label) != 0:
                        label = 1
                    else:
                        label = 0

                    # Create the csv file if it does not exist
                    if not os.path.isfile(csv_file_path):
                        with open(csv_file_path, "w", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(["Path_image", "Classification"])
                            writer.writerow([file_name_i, label])

                    # Open it if it exists
                    else:
                        with open(csv_file_path, "a", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([file_name_i, label])
        else:
            images_path_1 = test_dir + "/images_1"
            list_name_image_1 = os.listdir(images_path_1)
            list_name_image_1 = np.sort(remove_dot_file(list_name_image_1))
            list_images_path_1 = [images_path_1 + "/" + name for name in list_name_image_1]
            output_images_path_1 = output_test + "/images_1"

            images_path_2 = test_dir + "/images_2"
            list_name_image_2 = os.listdir(images_path_2)
            list_name_image_2 = np.sort(remove_dot_file(list_name_image_2))
            list_images_path_2 = [images_path_2 + "/" + name for name in list_name_image_2]
            output_images_path_2 = output_test + "/images_2"

            for image_path_1, image_path_2, label_path, name in zip(
                list_images_path_1,
                list_images_path_2,
                list_labels_path,
                list_name_image_1
            ):

                si1 = SatelliteImage.from_raster(
                    file_path=image_path_1, dep=None, date=None, n_bands=n_bands
                )
                si2 = SatelliteImage.from_raster(
                    file_path=image_path_2, dep=None, date=None, n_bands=n_bands
                )
                mask = np.load(label_path)

                lsi1 = SegmentationLabeledSatelliteImage(si1, mask, "", "")
                lsi2 = SegmentationLabeledSatelliteImage(si2, mask, "", "")

                list_lsi1 = lsi1.split(tile_size)
                list_lsi2 = lsi2.split(tile_size)

                for i, (lsi1, lsi2) in enumerate(zip(list_lsi1, list_lsi2)):
                    file_name_i = name.split(".")[0] + "_" + "{:04d}".format(i)

                    lsi1.satellite_image.to_raster(
                        output_images_path_1, file_name_i + ".jp2"
                        )
                    lsi2.satellite_image.to_raster(
                        output_images_path_2, file_name_i + ".jp2"
                        )
                    np.save(output_labels_path + "/" + file_name_i + ".npy", lsi1.label)

        return output_test
    
    else:
        return None


def instantiate_dataset(config, list_images, list_labels, list_images_2 = None, test=False):
    """
    Instantiates the appropriate dataset object
    based on the configuration settings.

    Args:
        config: A dictionary representing the configuration settings.
        list_path_images: A list of strings representing
        the paths to the preprocessed tile image files.
        list_path_labels: A list of strings representing
        the paths to the corresponding preprocessed mask image files.

    Returns:
        A dataset object of the specified type.
    """
    if not test:
        dataset_type = config["donnees"]["dataset"]
    else:
        dataset_type = config["donnees"]["dataset-test"]

    # inqtanciation du dataset complet
    if dataset_type not in dataset_dict:
        raise ValueError("Invalid dataset type")
    else:
        dataset_select = dataset_dict[dataset_type]

        if list_images_2 is None:
            full_dataset = dataset_select(
                list_images, list_labels, config["donnees"]["n bands"]
            )
        else:
            full_dataset = dataset_select(
                list_images, list_images_2, list_labels, config["donnees"]["n bands"]
            )

    return full_dataset


def instantiate_dataloader_s2(config, list_output_dir):
    """
    Instantiates and returns the data loaders for
    training, validation, and testing datasets.

    Args:
    - config (dict): A dictionary containing the configuration parameters
    for data loading and processing.
    - list_output_dir (list): A list of strings containing the paths to
    the directories that contain the training data.

    Returns:
    - train_dataloader (torch.utils.data.DataLoader):
    The data loader for the training dataset.
    - valid_dataloader (torch.utils.data.DataLoader):
    The data loader for the validation dataset.
    - test_dataloader (torch.utils.data.DataLoader):
    The data loader for the testing dataset.

    The function first generates the paths for the image and label data
    based on the data source (Sentinel, PLEIADES) vs pre-annotated datasets.
    It then instantiates the required dataset class
    (using the `intantiate_dataset` function) and splits the full dataset
    into training and validation datasets based on the validation proportion
    specified in the configuration parameters.

    Next, the appropriate transformations are applied to the training
    and validation datasets using the `generate_transform` function.

    Finally, the data loaders for the training and validation datasets
    are created using the `DataLoader` class from the PyTorch library,
    and the data loader for the testing dataset is set to `None`.
    """
    # génération des paths en fonction du type de Données
    # (Sentinel, PLEIADES) VS Dataset préannotés

    print("Entre dans la fonction instantiate_dataloader")

    output_dir_train, output_dir_valid, output_dir_test = list_output_dir

    train_list_images1 = list_sorted_filenames(output_dir_train + "Image1/")
    train_list_images2 = list_sorted_filenames(output_dir_train + "Image2/")
    train_list_labels = list_sorted_filenames(output_dir_train + "label/")

    val_list_images1 = list_sorted_filenames(output_dir_valid + "Image1/")
    val_list_images2 = list_sorted_filenames(output_dir_valid + "Image2/")
    val_list_labels = list_sorted_filenames(output_dir_valid + "label/")

    test_list_images1 = list_sorted_filenames(output_dir_test + "Image1/")
    test_list_images2 = list_sorted_filenames(output_dir_test + "Image2/")
    test_list_labels = list_sorted_filenames(output_dir_test + "label/")

    # Retrieving the desired Dataset class
    train_dataset = instantiate_dataset(
        config, train_list_images1, train_list_labels,  train_list_images2
    )

    valid_dataset = instantiate_dataset(
        config, val_list_images1, val_list_labels, val_list_images2
    )

    test_dataset = instantiate_dataset(
        config, test_list_images1, test_list_labels, test_list_images2
    )

    tile_size = config["donnees"]["tile size"]

    t_aug, t_preproc = generate_transform_pleiades(
        tile_size,
        True,
    )

    train_dataset.transforms = t_aug
    valid_dataset.transforms = t_preproc
    test_dataset.transforms = t_preproc

    # Creation of the dataloaders
    shuffle_bool = [True, False, False]
    batch_size = config["optim"]["batch size"]
    batch_size_test = config["optim"]["batch size test"]

    train_dataloader, valid_dataloader, test_dataloader = [
        DataLoader(
            ds, batch_size=size, shuffle=boolean, num_workers=0, drop_last=True
        )
        for ds, boolean, size in zip([train_dataset, valid_dataset, test_dataset], shuffle_bool, [batch_size, batch_size, batch_size_test])
    ]
    return train_dataloader, valid_dataloader, test_dataloader


def instantiate_dataloader_else(config, list_output_dir, output_test):
    """
    Instantiates and returns the data loaders for
    training, validation, and testing datasets.

    Args:
    - config (dict): A dictionary containing the configuration parameters
    for data loading and processing.
    - list_output_dir (list): A list of strings containing the paths to
    the directories that contain the training data.

    Returns:
    - train_dataloader (torch.utils.data.DataLoader):
    The data loader for the training dataset.
    - valid_dataloader (torch.utils.data.DataLoader):
    The data loader for the validation dataset.
    - test_dataloader (torch.utils.data.DataLoader):
    The data loader for the testing dataset.

    The function first generates the paths for the image and label data
    based on the data source (Sentinel, PLEIADES) vs pre-annotated datasets.
    It then instantiates the required dataset class
    (using the `intantiate_dataset` function) and splits the full dataset
    into training and validation datasets based on the validation proportion
    specified in the configuration parameters.

    Next, the appropriate transformations are applied to the training
    and validation datasets using the `generate_transform` function.

    Finally, the data loaders for the training and validation datasets
    are created using the `DataLoader` class from the PyTorch library,
    and the data loader for the testing dataset is set to `None`.
    """
    # génération des paths en fonction du type de Données
    # (Sentinel, PLEIADES) VS Dataset préannotés

    print("Entre dans la fonction instantiate_dataloader")
    config_task = config["donnees"]["task"]
    prop = config["donnees"]["prop"]
    if config["donnees"]["source train"] in [
        "PLEIADES",
        "SENTINEL2",
        "SENTINEL1-2",
        "SENTINEL2-RVB",
        "SENTINEL1-2-RVB"
    ]:
        list_labels = []
        list_images = []
        full_balancing_dict = {}
        for directory in list_output_dir:
            # directory = list_output_dir[0]
            labels = os.listdir(directory + "/labels")
            images = os.listdir(directory + "/images")
            labels = remove_dot_file(labels)

            if config_task != "classification":
                with open(directory + "/balancing_dict.json") as json_file:
                    balancing_dict = json.load(json_file)

                list_labels = np.concatenate(
                    (
                        list_labels,
                        np.sort([directory + "/labels/" + name for name in labels]),
                    )
                )

                for k, v in balancing_dict.items():
                    full_balancing_dict[k] = v

            if config_task == "classification":
                list_labels_dir = []
                csv_labeler = directory + "/labels/" + labels[0]

                # Balancing data
                extract_proportional_subset(
                    input_file=csv_labeler,
                    prop=prop,
                )

                filter_images_by_path(
                    csv_file=csv_labeler,
                    image_folder=directory + "/images",
                )

                # Load the initial CSV file
                df = pd.read_csv(csv_labeler)

                list_labels_dir = df[["Path_image", "Classification"]].values.tolist()

                list_labels_dir = sorted(list_labels_dir, key=lambda x: x[0])
                list_labels_dir = np.array(
                    [sous_liste[1] for sous_liste in list_labels_dir]
                )

                list_labels = np.concatenate((list_labels, list_labels_dir))

            list_images = np.concatenate(
                (
                    list_images,
                    np.sort([directory + "/images/" + name for name in images]),
                )
            )

    if config_task == "segmentation":
        unbalanced_images = list_images.copy()
        unbalanced_labels = list_labels.copy()
        indices_to_balance = select_indices_to_balance(
            list_images, full_balancing_dict, prop=prop
        )
        list_images = unbalanced_images[indices_to_balance]
        list_labels = unbalanced_labels[indices_to_balance]

    train_idx, val_idx = select_indices_to_split_dataset(
        config_task,
        config["optim"]["val prop"],
        list_labels,
        full_balancing_dict
    )

    # Retrieving the desired Dataset class
    train_dataset = instantiate_dataset(
        config, list_images[train_idx], list_labels[train_idx]
    )

    valid_dataset = instantiate_dataset(
        config, list_images[val_idx], list_labels[val_idx]
    )

    # Applying the respective transforms
    augmentation = config["donnees"]["augmentation"]
    tile_size = config["donnees"]["tile size"]

    if config["donnees"]["source train"] == "PLEIADES":
        t_aug, t_preproc = generate_transform_pleiades(tile_size, augmentation)
    else:
        t_aug, t_preproc = generate_transform_sentinel(
            tile_size,
            augmentation,
        )

    train_dataset.transforms = t_aug
    valid_dataset.transforms = t_preproc

    # Creation of the dataloaders
    batch_size = config["optim"]["batch size"]

    # Balancing batchs
    # if config_task == "classification":
    #     class_counts_train = np.bincount(train_dataset.list_labels.astype(np.int64))
    #     class_counts_valid = np.bincount(valid_dataset.list_labels.astype(np.int64))

    #     class_weights_train = class_counts_train/np.sum(class_counts_train)
    #     train_sampler = WeightedRandomSampler(weights=class_weights_train, num_samples=len(train_dataset), replacement=True)

    #     class_weights_valid = class_counts_valid/np.sum(class_counts_valid)
    #     valid_sampler = WeightedRandomSampler(weights=class_weights_valid, num_samples=len(valid_dataset), replacement=True)

    #     shuffle_bool = [False, False]
    # else:
    #     train_sampler, valid_sampler = None, None
    #     shuffle_bool = [True, False]

    shuffle_bool = [True, False]

    train_dataloader, valid_dataloader = [
        DataLoader(
            ds, batch_size=batch_size, shuffle=boolean, num_workers=0, drop_last=True
        )
        for ds, boolean in zip([train_dataset, valid_dataset], shuffle_bool)
    ]

    # Gestion datset test
    # output_test = "../donnees-test/"
    # output_test_task = output_test + config["donnees"]["task"]
    # output_images_path = output_test_task + "/images/"
    # output_labels_path = output_test_task + "/masks/"

    output_labels_path = output_test + "/labels/"
    list_name_label_test = os.listdir(output_labels_path)
    list_name_label_test = remove_dot_file(list_name_label_test)
    list_path_labels_test = np.sort([output_labels_path + name_label for name_label in list_name_label_test])

    if config_task != "change-detection":
        output_images_path = output_test + "/images/"
        list_name_image_test = os.listdir(output_images_path)
        list_path_images_test = np.sort([output_images_path + name_image for name_image in list_name_image_test])

        if config_task == "classification":
            csv_labeler = list_path_labels_test[0]

            # Load the initial CSV file
            df = pd.read_csv(csv_labeler)

            list_labels_dir = df[["Path_image", "Classification"]].values.tolist()

            list_labels_dir = sorted(list_labels_dir, key=lambda x: x[0])
            list_path_labels_test = list(np.array(
                [sous_liste[1] for sous_liste in list_labels_dir]
            ))

        dataset_test = instantiate_dataset(
            config, list_path_images_test, list_path_labels_test, test=True
        )
        dataset_test.transforms = t_preproc

    else:

        output_images_path_1 = output_test + "/images_1/"
        list_name_image_1 = os.listdir(output_images_path_1)
        list_path_images_1 = np.sort([output_images_path_1 + name_image for name_image in list_name_image_1])

        output_images_path_2 = output_test + "/images_2/"
        list_name_image_2 = os.listdir(output_images_path_2)
        list_path_images_2 = np.sort([output_images_path_2 + name_image for name_image in list_name_image_2])

        dataset_test = instantiate_dataset(
                config, list_path_images_1, list_path_labels_test, list_images_2 = list_path_images_2, test = True
            )
        dataset_test.transforms = t_preproc

    batch_size_test = config["optim"]["batch size test"]
    test_dataloader = DataLoader(
        dataset_test,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=0,
    )

    return train_dataloader, valid_dataloader, test_dataloader


def instantiate_dataloader(config, list_output_dir, output_test):
    """
    Instantiates and returns the data loaders for
    training, validation, and testing datasets.

    Args:
    - config (dict): A dictionary containing the configuration parameters
    for data loading and processing.
    - list_output_dir (list): A list of strings containing the paths to
    the directories that contain the training data.

    Returns:
    - train_dataloader (torch.utils.data.DataLoader):
    The data loader for the training dataset.
    - valid_dataloader (torch.utils.data.DataLoader):
    The data loader for the validation dataset.
    - test_dataloader (torch.utils.data.DataLoader):
    The data loader for the testing dataset.

    The function first generates the paths for the image and label data
    based on the data source (Sentinel, PLEIADES) vs pre-annotated datasets.
    It then instantiates the required dataset class
    (using the `intantiate_dataset` function) and splits the full dataset
    into training and validation datasets based on the validation proportion
    specified in the configuration parameters.

    Next, the appropriate transformations are applied to the training
    and validation datasets using the `generate_transform` function.

    Finally, the data loaders for the training and validation datasets
    are created using the `DataLoader` class from the PyTorch library,
    and the data loader for the testing dataset is set to `None`.
    """
    # génération des paths en fonction du type de Données
    # (Sentinel, PLEIADES) VS Dataset préannotés

    src = config["donnees"]["source train"]

    if src == "S2Looking":
        train_dataloader, valid_dataloader, test_dataloader = instantiate_dataloader_s2(config, list_output_dir)
    else:
        train_dataloader, valid_dataloader, test_dataloader = instantiate_dataloader_else(config, list_output_dir, output_test)

    return train_dataloader, valid_dataloader, test_dataloader

def instantiate_model(config):
    """
    Instantiate a module based on the provided module type.

    Args:
        module_type (str): Type of module to instantiate.

    Returns:
        object: Instance of the specified module.
    """
    print("Entre dans la fonction instantiate_model")
    module_type = config["optim"]["module"]
    nchannel = config["donnees"]["n channels train"]

    if module_type not in module_dict:
        raise ValueError("Invalid module type")

    if module_type in ["deeplabv3", "resnet50"]:
        return module_dict[module_type](nchannel)
    else:
        return module_dict[module_type]()


def instantiate_loss(config):
    """
    intantiates an optimizer object with the parameters
    specified in the configuration file.

    Args:
        model: A PyTorch model object.
        config: A dictionary object containing the configuration parameters.

    Returns:
        An optimizer object from the `torch.optim` module.
    """

    print("Entre dans la fonction instantiate_loss")
    loss_type = config["optim"]["loss"]

    if loss_type not in loss_dict:
        raise ValueError("Invalid loss type")
    else:
        return loss_dict[loss_type]()


def instantiate_lightning_module(config):
    """
    Create a PyTorch Lightning module for segmentation
    with the given model and optimization configuration.

    Args:
        config (dict): Dictionary containing the configuration
        parameters for optimization.
        model: The PyTorch model to use for segmentation.

    Returns:
        A PyTorch Lightning module for segmentation.
    """
    print("Entre dans la fonction instantiate_lighting_module")
    list_params = generate_optimization_elements(config)
    task_type = config["donnees"]["task"]

    if task_type not in task_to_lightningmodule:
        raise ValueError("Invalid task type")
    else:
        LightningModule = task_to_lightningmodule[task_type]

    lightning_module = LightningModule(
        model=instantiate_model(config),
        loss=instantiate_loss(config),
        optimizer=list_params[0],
        optimizer_params=list_params[1],
        scheduler=list_params[2],
        scheduler_params=list_params[3],
        scheduler_interval=list_params[4],
    )

    return lightning_module

def instantiate_trainer(config, lightning_module):
    """
    Create a PyTorch Lightning module for segmentation with
    the given model and optimization configuration.

    Args:
        config (dict): Dictionary containing the configuration
        parameters for optimization.
        model: The PyTorch model to use for segmentation.

    Returns:
        trainer: return a trainer object
    """
    # def callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss", save_top_k=1, save_last=True, mode="min"
    )

    early_stop_callback = EarlyStopping(
        monitor="validation_loss", mode="min", patience=5
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    if config["donnees"]["task"] == "segmentation":
        checkpoint_callback_IOU = ModelCheckpoint(
            monitor="validation_IOU", save_top_k=1, save_last=True, mode="max"
        )
        list_callbacks = [
            lr_monitor,
            checkpoint_callback,
            early_stop_callback,
            checkpoint_callback_IOU,
        ]

    if config["donnees"]["task"] == "classification":
        list_callbacks = [lr_monitor, checkpoint_callback, early_stop_callback]

    if config["donnees"]["task"] == "change-detection":
        checkpoint_callback_IOU = ModelCheckpoint(
                monitor="validation_IOU", save_top_k=1, save_last=True, mode="max"
                )
        list_callbacks = [lr_monitor, checkpoint_callback, early_stop_callback, checkpoint_callback_IOU]

    strategy = "auto"

    trainer = pl.Trainer(
        callbacks=list_callbacks,
        max_epochs=config["optim"]["max epochs"],
        num_sanity_val_steps=2,
        strategy=strategy,
        log_every_n_steps=2,
        accumulate_grad_batches=config["optim"]["accumulate batch"],
    )

    return trainer


def run_pipeline(remote_server_uri, experiment_name, run_name):
    """
    Runs the segmentation pipeline u
    sing the configuration specified in `config.yml`
    and the provided MLFlow parameters.
    Args:
        None
    Returns:
        None
    """
    # Open the file and load the file
    with open("../config.yml") as f:
        config = yaml.load(f, Loader=SafeLoader)

    tile_size = config["donnees"]["tile size"]
    batch_size_test = config["optim"]["batch size test"]
    task_type = config["donnees"]["task"]
    source_data = config["donnees"]["source train"]
    src_task = source_data + task_type

    list_data_dir, list_masks_cloud_dir, test_dir = download_data(config)

    list_output_dir = prepare_train_data(config, list_data_dir, list_masks_cloud_dir)
    output_test = prepare_test_data(config, test_dir)

    train_dl, valid_dl, test_dl = instantiate_dataloader(config, list_output_dir, output_test)

    torch.cuda.empty_cache()
    gc.collect()

    # train_dl.dataset[0][0].shape
    light_module = instantiate_lightning_module(config)
    trainer = instantiate_trainer(config, light_module)

    torch.cuda.empty_cache()
    gc.collect()

    remote_server_uri = "https://projet-slums-detection-128833.user.lab.sspcloud.fr"
    # experiment_name = "classification"
    # run_name = "mergemain"

    # model = mlflow.pytorch.load_model('/home/onyxia/work/detection-habitat-spontane/src/old_model/model/')

    if config["mlflow"]:
        update_storage_access()
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://minio.lab.sspcloud.fr"
        mlflow.end_run()
        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment(experiment_name)
        # mlflow.pytorch.autolog()

        with mlflow.start_run(run_name=run_name):
            mlflow.autolog()
            mlflow.log_artifact(
                "../config.yml",
                artifact_path="config.yml"
            )
            trainer.fit(light_module, train_dl, valid_dl)

            light_module = light_module.load_from_checkpoint(
                loss=instantiate_loss(config),
                #checkpoint_path=trainer.checkpoint_callback.best_model_path, #je créé un module qui charge
                checkpoint_path='epoch=14-step=13545.ckpt',
                model=light_module.model,
                # model=model,
                optimizer=light_module.optimizer,
                optimizer_params=light_module.optimizer_params,
                scheduler=light_module.scheduler,
                scheduler_params=light_module.scheduler_params,
                scheduler_interval=light_module.scheduler_interval
            )
            torch.cuda.empty_cache()
            gc.collect()

            model = light_module.model

            if src_task not in task_to_evaluation:
                raise ValueError("Invalid task type")
            else:
                evaluer_modele_sur_jeu_de_test = task_to_evaluation[src_task]
            
            # from classes.changes.mask_changes import csv_prob
            # csv_prob(
            #         test_dl,
            #         model,
            #         tile_size,
            #         batch_size_test,
            #         config["donnees"]["n bands"],
            #         False
            #     )

            evaluer_modele_sur_jeu_de_test(
                    test_dl,
                    model,
                    tile_size,
                    batch_size_test,
                    config["donnees"]["n bands"],
                    False
                    #config["mlflow"]
                )

            # if task_type == "classification":
            #     model_uri = mlflow.get_artifact_uri("model")
            #     print(model_uri)

            #     mlflow.evaluate(
            #         model_uri,
            #         test_dl,
            #         targets="labels",
            #         model_type="classifier",
            #         evaluators=["default"]
            #     )

    else:
        trainer.fit(light_module, train_dl, valid_dl)

        light_module = light_module.load_from_checkpoint(
            loss=instantiate_loss(config),
            checkpoint_path=trainer.checkpoint_callback.best_model_path,  # je créé un module qui charge
            # checkpoint_path='epoch=15-step=16048.ckpt',
            model=light_module.model,
            optimizer=light_module.optimizer,
            optimizer_params=light_module.optimizer_params,
            scheduler=light_module.scheduler,
            scheduler_params=light_module.scheduler_params,
            scheduler_interval=light_module.scheduler_interval
        )

        torch.cuda.empty_cache()
        gc.collect()

        model = light_module.model

        from classes.optim.evaluation_model import predicted_labels_classification_pleiade, variation_threshold_classification_pleiades
        variation_threshold_classification_pleiades(
            test_dl, model, tile_size, batch_size_test, n_bands=3, use_mlflow=False
        )
        predicted_labels_classification_pleiade(
            test_dl,
            model,
            tile_size,
            batch_size_test,
            config["donnees"]["n bands"],
            False,
        )

        # from classes.optim.evaluation_model import metrics_classification_pleiade2, evaluer_modele_sur_jeu_de_test_classification_pleiade2
        # trshld = metrics_classification_pleiade2(
        #     test_dl,
        #     model,
        #     tile_size,
        #     batch_size_test,
        #     config["donnees"]["n bands"],
        #     False,
        # )
        # print(trshld)
        # evaluer_modele_sur_jeu_de_test_classification_pleiade2(
        #     test_dl,
        #     model,
        #     tile_size,
        #     batch_size_test,
        #     config["donnees"]["n bands"],
        #     False,
        # )

        if src_task not in task_to_evaluation:
            raise ValueError("Invalid task type")
        else:
            evaluer_modele_sur_jeu_de_test = task_to_evaluation[src_task]

        evaluer_modele_sur_jeu_de_test(
            test_dl,
            model,
            tile_size,
            batch_size_test,
            config["donnees"]["n bands"],
            config["mlflow"],
        )


if __name__ == "__main__":
    # MLFlow params
    remote_server_uri = sys.argv[1]
    experiment_name = sys.argv[2]
    run_name = sys.argv[3]
    run_pipeline(remote_server_uri, experiment_name, run_name)


# nohup python run_training_pipeline.py
# https://projet-slums-detection-128833.user.lab.sspcloud.fr
# classification 2022_2018_971 > out.txt &
# https://www.howtogeek.com/804823/nohup-command-linux/
# TO DO :
# test routine sur S2Looking dataset
# import os

# list_data_dir = ["../data/PLEIADES/2022/MARTINIQUE/"]
# def delete_files_in_dir(dir_path,length_delete):
#    # Get a list of all the files in the directory
#  files = os.listdir(dir_path)[:length_delete]

#  for file in files:
#        file_path = os.path.join(dir_path, file)
#        if os.path.isfile(file_path):
#            os.remove(file_path)
# delete_files_in_dir(list_data_dir[0], 600)
