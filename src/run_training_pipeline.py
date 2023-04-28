import os
import sys
from datetime import datetime

import numpy as np
import yaml
from yaml.loader import SafeLoader

from classes.labelers.labeler import RILLabeler
from models.components.segmentation_models import DeepLabv3Module
from train_pipeline_utils.download_data import load_pleiade_data
from train_pipeline_utils.prepare_data import write_splitted_images_masks
from classes.optim.handle_dataset import (
    split_dataset,
    generate_transform,
    instanciate_dataset
    )
from torch.utils.data import DataLoader
from classes.optim.optimizer import generate_optimization_elements
from models.segmentation_module import SegmentationModule

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
import pytorch_lightning as pl
import gc
import mlflow
import torch


def download_data(config):
    config_data = config["donnees"]
    list_output_dir = []

    if config_data["source train"] == "PLEIADE":
        years = config_data["year"]
        deps = config_data["dep"]

        for year, dep in zip(years, deps):
            output_dir = load_pleiade_data(year, dep)
            list_output_dir.append(output_dir)

    return list_output_dir


def prepare_data(config, list_data_dir):
    # load labeler
    config_data = config["donnees"]

    years = config_data["year"]
    deps = config_data["dep"]

    for i, (year, dep) in enumerate(zip(years, deps)):
        print(i)
        print(year)
        print(dep)
        if config_data["type labeler"] == "RIL":
            buffer_size = config_data["buffer size"]
            date = datetime.strptime(str(year) + "0101", "%Y%m%d")

            labeler = RILLabeler(date, dep=dep, buffer_size=buffer_size)

        write_splitted_images_masks(
            list_data_dir[i],
            "train_data",
            labeler,
            config_data["tile size"],
            config_data["n channels train"],
            dep,
        )


def instanciate_dataloader(config):

    # génération des paths en fonction du type de Données
    # (Sentinel, PLEIADES) VS Dataset préannotés
    if config["donnees"]["source train"] in ["PLEIADE", "SENTINEL2"]:
        dir = "train_data"
        labels = os.listdir(dir + "/labels")
        images = os.listdir(dir + "/images")

        list_path_labels = np.sort(
            [dir + "/labels/" + name for name in labels]
        )
        list_path_images = np.sort(
            [dir + "/images/" + name for name in images]
        )

    # récupération de la classe de Dataset souhaitée
    full_dataset = instanciate_dataset(
        config,
        list_path_images,
        list_path_labels
        )
    train_dataset, valid_dataset = split_dataset(
        full_dataset,
        config["optim"]["val prop"]
        )

    # on applique les transforms respectives 
    augmentation = config["donnees"]["augmentation"]
    tile_size = config["donnees"]["tile size"]
    t_aug, t_preproc = generate_transform(tile_size, augmentation)
    train_dataset.transforms = t_aug
    valid_dataset.transforms = t_preproc

    # création des dataloader
    batch_size = config["optim"]["batch size"]

    train_dataloader, valid_dataloader = [
        DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=boolean,
                num_workers=2,
            )
        for ds, boolean in zip(
            [train_dataset, valid_dataset],
            [True, False]
            )
    ]

    # instancier le dataset de test
    # TO DO dépendra du nombre dimages dans le data set de test
    # et de la tile_size
    # test_batch_size = None
    # dataset_test = None
    test_dataloader = None
    return train_dataloader, valid_dataloader, test_dataloader


def instantiate_model(config):
    """
    Instantiate a module based on the provided module type.

    Args:
        module_type (str): Type of module to instantiate.

    Returns:
        object: Instance of the specified module.
    """
    module_type = config["optim"]["module"]
    module_dict = {"deeplabv3": DeepLabv3Module}
    nchannel = config["donnees"]["n channels train"]

    if module_type not in module_dict:
        raise ValueError("Invalid module type")

    if module_type == "deeplabv3":
        return module_dict[module_type](nchannel)
    else:
        return module_dict[module_type]()


def instantiate_lightning_module(config, model):

    list_params = generate_optimization_elements(config)

    lightning_module = SegmentationModule(
        model=model,
        optimizer=list_params[0],
        optimizer_params=list_params[1],
        scheduler=list_params[2],
        scheduler_params=list_params[3],
        scheduler_interval=list_params[4],
    )
    return lightning_module


def instantiate_trainer(config, lightning_module):

    # def callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss", save_top_k=1, save_last=True, mode="max"
    )
    early_stop_callback = EarlyStopping(
        monitor="validation_loss", mode="max", patience=3
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    list_callbacks = [lr_monitor, checkpoint_callback, early_stop_callback]

    strategy = "auto"

    trainer = pl.Trainer(
            callbacks=list_callbacks,
            max_epochs=config["optim"]["max epochs"],
            num_sanity_val_steps=2,
            strategy=strategy,
            log_every_n_steps=2,
        )

    return trainer


def run_pipeline():
    # Open the file and load the file
    with open("../config.yml") as f:
        config = yaml.load(f, Loader=SafeLoader)

    list_output_dir = download_data(config)
    prepare_data(config, list_output_dir)

    model = instantiate_model(config)
    train_dl, valid_dl, test_dl = instanciate_dataloader(config)

    light_module = instantiate_lightning_module(config, model)
    trainer = instantiate_trainer(config, light_module)
    
    # trainer.test(lightning_module,test_dataloader) TO DO
    torch.cuda.empty_cache()
    gc.collect()

    if config["mlflow"]:
        mlflow.end_run()
        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.pytorch.autolog()

        with mlflow.start_run(run_name=run_name):
            trainer.fit(light_module, train_dl, valid_dl)
            # trainer.test(light_module, test_dl)


if __name__ == "__main__":
    # MLFlow params
    remote_server_uri = sys.argv[1]
    experiment_name = sys.argv[2]
    run_name = sys.argv[3]

    run_pipeline(
        remote_server_uri,
        experiment_name,
        run_name)

# remote_server_uri =
# "https://projet-slums-detection-561009.user.lab.sspcloud.fr"
# experiment_name = "segmentation"
# run_name = "testraya"
