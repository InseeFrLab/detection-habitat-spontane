import csv
import gc
import math
import os
import re
import sys
import time
from datetime import date, datetime

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pytorch_lightning as pl
import s3fs
import torch
import torch.nn as nn
import yaml
from PIL import Image as im
from pyproj import Transformer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from rasterio.errors import RasterioIOError
from scipy.ndimage import label
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from yaml.loader import SafeLoader

import train_pipeline_utils.handle_dataset as hd
from classes.data.labeled_satellite_image import SegmentationLabeledSatelliteImage
from classes.data.satellite_image import SatelliteImage
from classes.labelers.labeler import BDTOPOLabeler, RILLabeler
from classes.optim.evaluation_model import (
    evaluer_modele_sur_jeu_de_test_classification_pleiade,
    evaluer_modele_sur_jeu_de_test_segmentation_pleiade,
)
from classes.optim.losses import CrossEntropySelfmade, SoftIoULoss
from classes.optim.optimizer import generate_optimization_elements
from data.components.classification_patch import PatchClassification
from data.components.dataset import PleiadeDataset
from models.classification_module import ClassificationModule
from models.components.classification_models import ResNet50Module
from models.components.segmentation_models import DeepLabv3Module
from models.segmentation_module import SegmentationModule
from train_pipeline_utils.download_data import load_donnees_test, load_satellite_data
from train_pipeline_utils.handle_dataset import (
    generate_transform,
    select_indices_to_split_dataset,
)
from train_pipeline_utils.prepare_data import (
    check_labelled_images,
    filter_images,
    label_images,
    save_images_and_masks,
)
from utils.utils import remove_dot_file, split_array, update_storage_access

# with open("../config.yml") as f:
#     config = yaml.load(f, Loader=SafeLoader)

# def download_data(config):
#     """
#     Downloads data based on the given configuration.

#     Args:
#         config: a dictionary representing the
#         configuration information for data download.

#     Returns:
#         A list of output directories for each downloaded dataset.
#     """

#     print("Entre dans la fonction download_data")
#     config_data = config["data"]
#     list_output_dir = []
#     list_masks_cloud_dir = []

#     years = config_data["year"]
#     deps = config_data["dep"]
#     src = config_data["source_train"]

#     for year, dep in zip(years, deps):
#         # year, dep = years[0], deps[0]
#         if src == "PLEIADES":
#             cloud_dir = load_satellite_data(year, dep, "NUAGESPLEIADES")
#             list_masks_cloud_dir.append(cloud_dir)

#         output_dir = load_satellite_data(year, dep, src)
#         list_output_dir.append(output_dir)

#     print("chargement des données test")
#     test_dir = load_donnees_test(type=config["data"]["task"])

#     return list_output_dir, list_masks_cloud_dir, test_dir


# def prepare_train_data(config, list_data_dir, list_masks_cloud_dir):
#     """
#     Preprocesses and splits the raw input images
#     into tiles and corresponding masks,
#     and saves them in the specified output directories.

#     Args:
#         config: A dictionary representing the configuration settings.
#         list_data_dir: A list of strings representing the paths
#         to the directories containing the raw input image files.

#     Returns:
#         A list of strings representing the paths to
#         the output directories containing the
#         preprocessed tile and mask image files.
#     """

#     print("Entre dans la fonction prepare_data")
#     config_data = config["data"]

#     years = config_data["year"]
#     deps = config_data["dep"]
#     src = config_data["source_train"]
#     labeler = config_data["type_labeler"]
#     config_task = config_data["task"]

#     cpt_tot =0
#     cpt_ones =0

#     for i, (year, dep) in enumerate(zip(years, deps)):
#         # i, year , dep = 0,years[0],deps[0]

#         date = datetime.strptime(str(year) + "0101", "%Y%m%d")

#         if labeler == "RIL":
#             buffer_size = config_data["buffer_size"]
#             labeler = RILLabeler(date, dep=dep, buffer_size=buffer_size)
#         elif labeler == "BDTOPO":
#             labeler = BDTOPOLabeler(date, dep=dep)

#         list_name_cloud = []
#         if src == "PLEIADES":
#             cloud_dir = list_masks_cloud_dir[i]
#             list_name_cloud = [path.split("/")[-1].split(".")[0] for path in os.listdir(cloud_dir)]

#         dir = list_data_dir[i]
#         list_path = [dir + "/" + filename for filename in os.listdir(dir)]

#         for path in tqdm(list_path):
#             # path = list_path[0]
#             # path  = dir + "/"+ "ORT_2022_0691_1641_U20N_8Bits.jp2"
#             try:
#                 si = SatelliteImage.from_raster(
#                     file_path=path,
#                     dep=dep,
#                     date=date,
#                     n_bands=config_data["n_bands"],
#                 )

#             except RasterioIOError:
#                 print("Erreur de lecture du fichier " + path)
#                 continue

#             else:
#                 list_si_filtered, __ = label_images([si], labeler)

#                 if list_si_filtered:
#                     cpt_tot += 1
#                     cpt_ones += 1
#                 else:
#                     cpt_tot += 1
#     return cpt_ones, cpt_tot


#     # Open the file and load the file
# with open("../config.yml") as f:
#     config = yaml.load(f, Loader=SafeLoader)

# list_data_dir, list_masks_cloud_dir, test_dir = download_data(config)

# cpt_ones, cpt_tot = prepare_train_data(config, list_data_dir, list_masks_cloud_dir)
# print(cpt_ones, cpt_tot)


def instantiate_dataloader(
    list_output_dir=["train_data-classification-PLEIADES-RIL-972-2022"],
):
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

    config_task = "classification"
    list_labels = []
    list_images = []

    for dir in list_output_dir:
        labels = os.listdir(dir + "/labels")
        if labels[0][0] == ".":
            del labels[0]

        if config_task == "segmentation":
            list_labels = np.concatenate(
                (list_labels, np.sort([dir + "/labels/" + name for name in labels]))
            )

        if config_task == "classification":
            list_labels_dir = []
            with open(dir + "/labels/" + labels[0], "r") as csvfile:
                reader = csv.reader(csvfile)

                # Ignorer l'en-tête du fichier CSV s'il y en a un
                next(reader)

                # Parcourir les lignes du fichier CSV et extraire la deuxième colonne
                for row in reader:
                    image_path = row[0]
                    mask = row[
                        1
                    ]  # Index 1 correspond à la deuxième colonne (index 0 pour la première)
                    list_labels_dir.append([image_path, mask])

            list_labels_dir = sorted(list_labels_dir, key=lambda x: x[0])
            list_labels_dir = np.array(
                [sous_liste[1] for sous_liste in list_labels_dir]
            )

            list_labels = np.concatenate((list_labels, list_labels_dir))
            print(list_labels)

        # Même opération peu importe la tâche
        images = os.listdir(dir + "/images")

        list_images = np.concatenate(
            (list_images, np.sort([dir + "/images/" + name for name in images]))
        )
    return list_images, list_labels


# list_images_echant = list_images[100:200]
# list_labels_echant = list_labels[100:200]


def plot_list_path_images_labels(
    list_filepaths_images, list_filepaths_labels, tile_size=50
):
    size = int(math.sqrt(len(list_filepaths_images)))
    bands_indices = [0, 1, 2]

    list_images = []
    list_labels = []

    for filepath in list_filepaths_images:
        image = SatelliteImage.from_raster(
            filepath, date=None, n_bands=len(bands_indices), dep=None
        )
        image.normalize()
        list_images.append(image)

    for label in list_filepaths_labels:
        if label == "0":
            mask = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)

        elif label == "1":
            mask = np.full((tile_size, tile_size, 3), 0, dtype=np.uint8)

        list_labels.append(mask)

    # mat_list_labels = np.transpose(np.array(list_labels).reshape(size,size))

    # Create a figure and axes
    fig, axs = plt.subplots(nrows=size, ncols=2 * size, figsize=(20, 10))

    # Iterate over the grid of masks and plot them
    for i in range(size):
        for j in range(size):
            axs[i, j].imshow(list_images[i * size + j].array.transpose(1, 2, 0))

    for i in range(size):
        for j in range(size):
            axs[i, j + size].imshow(list_labels[i * size + j], cmap="gray")

    # Remove any unused axes
    for i in range(size):
        for j in range(2 * size):
            axs[i, j].set_axis_off()

    # Show the plot
    plt.show()
    plt.gcf()
    plt.savefig("test.png")
