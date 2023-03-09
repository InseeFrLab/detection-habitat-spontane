"""
Script to train a DL model.
"""
import sys
import os
import re
from datetime import datetime
from tqdm import tqdm
import mlflow
import torch
from satellite_image import SatelliteImage
from labeled_satellite_image import SegmentationLabeledSatelliteImage
from deeplabv3 import DeepLabv3Module
from dataset import SatelliteDataModule
from utils import get_environment
from filter import is_too_black
from labeler import BDTOPOLabeler
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import pytorch_lightning as pl
import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2


def main(remote_server_uri, experiment_name, run_name):
    """
    Main function.
    """
    # Loading images
    environment = get_environment()

    path_local_pleiades_data = environment["local-path"]["PLEIADES"]
    images_paths = [f"{path_local_pleiades_data}/16bits/ORT_2022072050325085_U22N/" + p for p in os.listdir(f"{path_local_pleiades_data}/16bits/ORT_2022072050325085_U22N/")]
    date = datetime.strptime(re.search(r'ORT_(\d{8})', images_paths[0]).group(1), '%Y%m%d')
    list_images = [
        SatelliteImage.from_raster(
            filename,
            dep="973",
            date=date,
            n_bands=4
        ) for filename in tqdm(images_paths)
    ]

    image_size = (250, 250)
    splitted_list_images = [
        im for sublist in tqdm(list_images) for im in sublist.split(image_size[0]) if not is_too_black(im)
    ]

    # Labeling
    labeler_BDTOPO = BDTOPOLabeler(date, dep="973")
    list_labeled_images = [
        SegmentationLabeledSatelliteImage(
            sat_im,
            labeler_BDTOPO.create_segmentation_label(sat_im),
            "BDTOPO",
            date
        ) for sat_im in tqdm(splitted_list_images[:30])
    ]

    # DataModule definition
    # Some additional normalization is done here
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
    data_module = SatelliteDataModule(
        train_data=list_labeled_images[:20],
        test_data=list_labeled_images[20:],
        transforms_augmentation=transforms_augmentation,
        transforms_preprocessing=transforms_preprocessing,
        num_workers=56,
        batch_size=16,
        bands_indices=[0, 1, 2]
    )

    # Training
    optimizer = torch.optim.SGD
    optimizer_params = {
        "lr": 0.0001,
        "momentum": 0.9
    }
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {}
    scheduler_interval = "epoch"

    model = DeepLabv3Module(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
            scheduler_interval=scheduler_interval
        )

    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss", save_top_k=1, save_last=True, mode="min"
    )
    early_stop_callback = EarlyStopping(
        monitor="validation_loss", mode="min", patience=3
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog()
    with mlflow.start_run(run_name=run_name):
        trainer = pl.Trainer(
            callbacks=[lr_monitor, checkpoint_callback, early_stop_callback],
            max_epochs=2,
            gpus=0,
            num_sanity_val_steps=2,
            strategy=None
        )
        trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    # MLFlow params
    remote_server_uri = sys.argv[1]
    experiment_name = sys.argv[2]
    run_name = sys.argv[3]

    main(remote_server_uri, experiment_name, run_name)
