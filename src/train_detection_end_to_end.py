"""
Script to train a DL model.
"""
import gc
import os
import sys
from datetime import datetime

import albumentations as album
import mlflow
import pytorch_lightning as pl
import torch
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from datas.components.change_detection_dataset import end_to_end_cd_dataset
from datas.datamodule import DataModule
from models.segmentation_module import SegmentationModule
from utils.gestion_data import (
    build_dataset_test,
    build_dataset_train,
    instantiate_module,
    load_pleiade_data,
)
from utils.labeler import BDTOPOLabeler, RILLabeler
from utils.utils import update_storage_access


def main(remote_server_uri, experiment_name, run_name):
    """
    Main function.
    """

    config = {
        "tile size": 250,
        "source train": "PLEIADE",
        "type labeler": "RIL",  # None if source train != PLEIADE
        "buffer size": 10,  # None if BDTOPO
        "year": 2022,
        "territory": "martinique",
        "dep": "972",
        "n bands": 3,
        "n channels train": 6,
    }

    config_train = {
        "lr": 0.0001,
        "momentum": 0.9,
        "module": "deeplabv3",
        "batch size": 2,
        "max epochs": 100,
    }

    # params
    n_channel_train = config["n channels train"]

    tile_size = config["tile size"]
    n_bands = config["n bands"]
    dep = config["dep"]
    territory = config["territory"]
    year = config["year"]
    buffer_size = config["buffer size"]
    source_train = config["source train"]
    type_labeler = config["type labeler"]

    module = config_train["module"]
    batch_size = config_train["batch size"]

    train_directory_name = "../splitted_data"

    update_storage_access()
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://minio.lab.sspcloud.fr"

    # DL des données du territoire dont on se sert pour l'entraînement
    # On peut faire une liste de couples années/territoire également
    if source_train == "PLEIADE":
        # Plus tard décliner avec change detection etc..
        if type_labeler == "RIL":
            date = datetime.strptime(
                str(year).split("-")[-1] + "0101", "%Y%m%d"
            )
            labeler = RILLabeler(date, dep=dep, buffer_size=buffer_size)

        if type_labeler == "BDTOPO":
            date = datetime.strptime(
                str(year).split("-")[-1] + "0101", "%Y%m%d"
            )
            labeler = BDTOPOLabeler(date, dep=dep)

        dataset_train = build_dataset_train(
            year,
            territory,
            dep,
            labeler,
            tile_size,
            n_bands,
            train_directory_name,
            end_to_end_cd_dataset,
        )

        load_pleiade_data(2022, "martinique")
        dir_may = "../data/PLEIADES/2022/MARTINIQUE/"
        test_file = dir_may + "ORT_2022_0691_1638_U20N_8Bits.jp2"

        dataset_test = build_dataset_test(
            test_file, 3, tile_size, labeler, end_to_end_cd_dataset
        )
        image_size = (tile_size, tile_size)

    transforms_augmentation = album.Compose(
        [
            album.Resize(300, 300, always_apply=True),
            album.RandomResizedCrop(
                *image_size, scale=(0.7, 1.0), ratio=(0.7, 1)
            ),
            album.HorizontalFlip(),
            album.VerticalFlip(),
            album.Normalize(),
            ToTensorV2(),
        ]
    )

    transforms_preprocessing = album.Compose(
        [
            album.Resize(*image_size, always_apply=True),
            album.Normalize(),
            ToTensorV2(),
        ]
    )

    # Instanciation modèle et paramètres d'entraînement
    optimizer = torch.optim.SGD
    optimizer_params = {
        "lr": config_train["lr"],
        "momentum": config_train["momentum"],
    }
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {}
    scheduler_interval = "epoch"

    model = instantiate_module(module, n_channel_train)

    batch_size = 2
    data_module = DataModule(
        dataset=dataset_train,
        transforms_augmentation=transforms_augmentation,
        transforms_preprocessing=transforms_preprocessing,
        num_workers=1,
        batch_size=batch_size,
        dataset_test=dataset_test,
    )

    lightning_module = SegmentationModule(
        model=model,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
        scheduler_interval=scheduler_interval,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss", save_top_k=1, save_last=True, mode="max"
    )

    early_stop_callback = EarlyStopping(
        monitor="validation_loss", mode="max", patience=3
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    strategy = "auto"
    list_callbacks = [lr_monitor, checkpoint_callback, early_stop_callback]

    # !pip install mlflow
    mlflow.end_run()

    # remote_server_uri =
    # "https://projet-slums-detection-561009.user.lab.sspcloud.fr"
    # experiment_name = "segmentation"
    # run_name = "testraya"

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog()

    torch.cuda.empty_cache()
    gc.collect()

    with mlflow.start_run(run_name=run_name):
        trainer = pl.Trainer(
            callbacks=list_callbacks,
            max_epochs=config_train["max epochs"],
            num_sanity_val_steps=2,
            strategy=strategy,
            log_every_n_steps=2,
        )

        trainer.fit(lightning_module, datamodule=data_module)

        print("test")
        trainer.test(lightning_module, datamodule=data_module)


if __name__ == "__main__":
    # MLFlow params
    remote_server_uri = sys.argv[1]
    experiment_name = sys.argv[2]
    run_name = sys.argv[3]

    main(remote_server_uri, experiment_name, run_name)

# python train_segmentation.py https://projet-slums-detection-
# 561009.user.lab.sspcloud.fr segmentation testonvscodepostmerge


# python train_segementation
# https://projet-slums-detection-561009.user.lab.sspcloud.fr
#  segmentation testRaya

remote_server_uri = "https://projet-slums-detection-561009.user.lab.sspcloud.fr"
experiment_name = "segmentation"
run_name = "testraya_1"
main(remote_server_uri, experiment_name, run_name)
