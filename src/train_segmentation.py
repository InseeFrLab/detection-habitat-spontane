import sys
sys.path.append('../src')
from datetime import datetime
import numpy as np
import os
import gc

from utils.gestion_data import load_pleiade_data, write_splitted_images_masks, build_dataset_train, build_dataset_test, instantiate_module
from utils.utils import update_storage_access
from utils.labeler import RILLabeler
from datas.components.dataset import PleiadeDataset

import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from models.segmentation_module import SegmentationModule
from datas.datamodule import DataModule

import mlflow


def main(remote_server_uri, experiment_name, run_name):
    """
    Main function.
    """

    config = { 
        "tile size" : 250,
        "source train": "PLEIADE",
        "type labeler" : "RIL", # None if source train != PLEIADE
        "buffer size" : 10, # None if BDTOPO
        "year"  : 2022,
        "territory" : "martinique",
        "dep" : "972",
        "n bands" : 3
    }

    config_train = { 
        "lr": 0.0001,
        "momentum": 0.9,
        "module" : "deeplabv3",
        "batch size" : 9,
        "max epochs" : 100
    }

    # params 
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
    #%env MLFLOW_S3_ENDPOINT_URL=https://minio.lab.sspcloud.fr
    # DL des données du teritoire dont on se sert pour l'entraînement

    if source_train == "PLEIADE":

        # Plus tard décliner avec change detection etc..
        if type_labeler == "RIL":
            date = datetime.strptime(str(year)+"0101",'%Y%m%d')
            labeler = RILLabeler(date, dep = dep, buffer_size = buffer_size)     

        dataset_train =  build_dataset_train(year,territory,dep,labeler,tile_size,n_bands,train_directory_name,PleiadeDataset)
        load_pleiade_data(2020,"mayotte")

        dataset_test = build_dataset_test("../data/PLEIADES/2020/MAYOTTE/ORT_2020052526670967_0519_8586_U38S_8Bits.jp2",3,250,labeler,PleiadeDataset)
        image_size = (tile_size,tile_size)

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

    ## Instanciation modèle et paramètres d'entraînement
    optimizer = torch.optim.SGD
    optimizer_params = {"lr": config_train["lr"], "momentum":  config_train["momentum"]}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {}
    scheduler_interval = "epoch"

    model = instantiate_module(module)

    data_module = DataModule(
        dataset= dataset_train,
        transforms_augmentation=transforms_augmentation,
        transforms_preprocessing=transforms_preprocessing,
        num_workers=1, 
        batch_size= batch_size,
        dataset_test = dataset_test
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
        monitor="validation_IOU", save_top_k=1, save_last=True, mode="max"
    )

    early_stop_callback = EarlyStopping(
        monitor="validation_IOU", mode="max", patience=3
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    strategy ="auto"
    list_callbacks = [lr_monitor, checkpoint_callback, early_stop_callback]

    #!pip install mlflow
    mlflow.end_run()

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog()

    torch.cuda.empty_cache()
    gc.collect()

    with mlflow.start_run(run_name=run_name):

        trainer = pl.Trainer(
        callbacks= list_callbacks,
        max_epochs=config_train["max epochs"],
        num_sanity_val_steps=2,
        strategy=strategy,
        log_every_n_steps=2
        )

        trainer.fit(lightning_module, datamodule=data_module)

        print("test")
        trainer.test(lightning_module , datamodule= data_module)
    
    
if __name__ == "__main__":
    # MLFlow params
    remote_server_uri = sys.argv[1]
    experiment_name = sys.argv[2]
    run_name = sys.argv[3]
    
    main(remote_server_uri, experiment_name, run_name)
    
#python train_segmentation.py https://projet-slums-detection-386760.user.lab.sspcloud.fr segmentation bibi