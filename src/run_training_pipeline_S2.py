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
from torch.nn import CrossEntropyLoss

from models.components.segmentation_models import DeepLabv3Module

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
)
from utils.utils import remove_dot_file, split_array, update_storage_access

from data.components.change_detection_dataset import ChangeDetectionS2LookingDataset 
from classes.data.change_detection_triplet import ChangedetectionTripletS2Looking

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
    output_dir = load_s2_looking()

    return output_dir

def list_sorted_filenames(dir_path):
    filenames = os.listdir(dir_path)
    sorted_filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))

    list_images = [dir_path + filename for filename in sorted_filenames]

    return list_images

def prepare_data_per_doss(directory_path, tile_size, type_data):

    output_dir = (
            "../"+ type_data +"_data-S2Looking-"
            + str(tile_size)
            + "/"
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir +"Image1/")
        os.makedirs(output_dir +"Image2/")
        os.makedirs(output_dir +"label/")

    else:
        return output_dir

    list_images1 = list_sorted_filenames(directory_path + "Image1/")
    list_images2 = list_sorted_filenames(directory_path + "Image2/")
    list_labels = list_sorted_filenames(directory_path + "label/")

    for img1, img2, lab in tqdm(zip(list_images1, list_images2, list_labels), total = len(list_images1)):
        triplet = ChangedetectionTripletS2Looking(img1, img2, lab)
        list_split_im1, list_split_im2, list_split_lab = triplet.split(tile_size)

        for small_img1, small_img2, small_lab, i in zip(list_split_im1, list_split_im2, list_split_lab, range(tile_size)):
            if np.sum(np.asarray(small_lab))>0 and small_img1: # filtre les images sans changement
                path = img1.split("/")[-1].split(".")[0] + "_" + "{:03d}".format(i)+".png"
                small_img1.save(output_dir + "Image1/" + path)
                small_img2.save(output_dir + "Image2/" + path)
                small_lab.save(output_dir + "label/" + path)

    return output_dir


def prepare_data(config, output_dir):
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

    return output_dir_train, output_dir_valid, output_dir_test


def instantiate_dataset(list_images, list_images_2, list_labels):
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

    full_dataset = ChangeDetectionS2LookingDataset(
        list_images, list_images_2, list_labels
    )

    return full_dataset


def instantiate_dataloader(config, output_dir_train, output_dir_valid, output_dir_test):
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
        train_list_images1, train_list_images2, train_list_labels
    )

    valid_dataset = instantiate_dataset(
        val_list_images1, val_list_images2, val_list_labels
    )

    test_dataset = instantiate_dataset(
        test_list_images1, test_list_images2, test_list_labels
    )

    tile_size = config["donnees"]["tile size"]
    batch_size = config["optim"]["batch size"]

    t_aug, t_preproc = generate_transform_pleiades(
        tile_size,
        True,
    )

    train_dataset.transforms = t_aug
    valid_dataset.transforms = t_preproc
    test_dataset.transforms = t_preproc

    # Creation of the dataloaders
    shuffle_bool = [True, False, False]

    train_dataloader, valid_dataloader, test_dataloader = [
        DataLoader(
            ds, batch_size=8, shuffle=boolean, num_workers=0, drop_last=True
        )
        for ds, boolean in zip([train_dataset, valid_dataset, test_dataset], shuffle_bool)
    ]

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
    return DeepLabv3Module(6)


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
    print(loss_type)
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

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    tile_size = config["donnees"]["tile size"]
    batch_size_test = config["optim"]["batch size test"]
    task_type = config["donnees"]["task"]
    source_data = config["donnees"]["source train"]
    src_task = source_data + task_type

    output_dir = download_data(config)
    output_dir_train, output_dir_valid, output_dir_test = prepare_data(config, output_dir)

    train_dl, valid_dl, test_dl = instantiate_dataloader(config, output_dir_train, output_dir_valid, output_dir_test)

    # train_dl.dataset[0][0].shape
    light_module = instantiate_lightning_module(config)
    trainer = instantiate_trainer(config, light_module)

    torch.cuda.empty_cache()
    gc.collect()

    remote_server_uri = "https://projet-slums-detection-128833.user.lab.sspcloud.fr"
    # experiment_name = "classification"
    # run_name = "mergemain"

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
                checkpoint_path=trainer.checkpoint_callback.best_model_path, #je créé un module qui charge
                # checkpoint_path='',
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
            # checkpoint_path='',
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

if __name__ == "__main__":
    # MLFlow params
    remote_server_uri = sys.argv[1]
    experiment_name = sys.argv[2]
    run_name = sys.argv[3]
    run_pipeline(remote_server_uri, experiment_name, run_name)


# nohup python run_training_pipeline_S2.py
# https://projet-slums-detection-128833.user.lab.sspcloud.fr
# change_detection essai_s2 > out3.txt &
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
