import os
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import s3fs
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes.data.labeled_satellite_image import SegmentationLabeledSatelliteImage
from classes.data.satellite_image import SatelliteImage
from classes.labelers.labeler import BDTOPOLabeler
from data.components.dataset import SegmentationSentinelDataset
from train_pipeline_utils.handle_dataset import generate_transform_sentinel


def main():
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # Open the config file and load it
    model_uri = "models:/obj-detection/1"
    config_uri = "s3://projet-slums-detection/mlflow-artifacts/1/221d8f96d352402e88954af3d9d3c94e/artifacts/config.yml/config.yml"
    config = mlflow.artifacts.download_artifacts(config_uri, dst_path="run_config/")

    with open(config, "r") as stream:
        run_config = yaml.safe_load(stream)

    # Bands
    n_bands = run_config["donnees"]["n bands"]
    tile_size = run_config["donnees"]["tile size"]

    # Labeler
    labeler = BDTOPOLabeler(labeling_date=date(2022, 1, 1), dep="971", task="segmentation")

    # Load test data
    s3_path = "projet-slums-detection/Donnees/SENTINEL/SENTINEL2/SAINT_MARTIN/TUILES_2022/"
    local_path = "raw_test_data/"
    dataloader_path = "test_data/"
    dataloader_images_dir = os.path.join(dataloader_path, "images/")
    dataloader_labels_dir = os.path.join(dataloader_path, "labels/")
    if not os.path.exists(dataloader_path):
        os.mkdir(dataloader_path)
    if not os.path.exists(dataloader_images_dir):
        os.mkdir(dataloader_images_dir)
    if not os.path.exists(dataloader_labels_dir):
        os.mkdir(dataloader_labels_dir)
    if not os.path.exists("results/"):
        os.mkdir("results/")

    if os.path.exists(local_path):
        pass
    else:
        fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"})
        fs.download(rpath=s3_path, lpath=local_path, recursive=True)

    # Prepare test data
    for root, dirs, files in os.walk(local_path):
        for filename in files:
            si = SatelliteImage.from_raster(
                file_path=os.path.join(local_path, filename),
                dep=None,
                date=None,
                n_bands=n_bands,
            )
            tiles = si.split(tile_size)

            # On loop sur toutes les images et labels divisés pour les sauvegarder
            for i, tile in tqdm(enumerate(tiles)):
                im_path = f"{Path(filename).stem}_{i}.jp2"
                tile.to_raster(dataloader_images_dir, im_path)
                label = labeler.create_segmentation_label(tile)
                label_path = f"{dataloader_labels_dir}/{Path(filename).stem}_{i}.npy"
                np.save(label_path, label)

        image_paths = os.listdir(dataloader_images_dir)
        image_paths = np.sort([os.path.join(dataloader_images_dir, path) for path in image_paths])
        label_paths = os.listdir(dataloader_labels_dir)
        label_paths = np.sort([os.path.join(dataloader_labels_dir, path) for path in label_paths])

        dataset_test = SegmentationSentinelDataset(image_paths, label_paths, n_bands=n_bands)
        t_aug, t_preproc = generate_transform_sentinel(
            "SENTINEL2",
            "2022",
            "971",
            tile_size,
            False,
            "segmentation",
        )
        dataset_test.transforms = t_preproc

        test_dataloader = DataLoader(
            dataset_test,
            batch_size=2,
            shuffle=False,
            num_workers=5,
            drop_last=True,
        )

    # Load model
    model_uri = "runs:/221d8f96d352402e88954af3d9d3c94e/model"
    model = mlflow.pytorch.load_model(model_uri=model_uri, map_location=torch.device("cpu"))

    with torch.no_grad():
        # Evaluation
        model.eval()
        for batch in iter(test_dataloader):
            images, label, dic = batch

            # Inference
            model = model.to(device)
            images = images.to(device)
            output_model = model(images)

            for i in range(len(output_model)):
                prediction = output_model[i].argmax(0)
                pthimg = dic["pathimage"][i]
                si = SatelliteImage.from_raster(
                    file_path=pthimg, dep=None, date=None, n_bands=n_bands
                )

                # Plot pred
                labeled_si = SegmentationLabeledSatelliteImage(
                    satellite_image=si,
                    label=prediction,
                    source="",
                    labeling_date="",
                )

                fig1 = labeled_si.plot(bands_indices=[3, 2, 1])
                plot_file = f"results/{Path(pthimg).stem}.png"
                fig1.savefig(plot_file)

                # Plot GT
                gt_labeled_si = SegmentationLabeledSatelliteImage(
                    satellite_image=si,
                    label=label[i].numpy(),
                    source="",
                    labeling_date="",
                )

                fig2 = gt_labeled_si.plot(bands_indices=[3, 2, 1])
                plot_file = f"results/{Path(pthimg).stem}_gt.png"
                fig2.savefig(plot_file)


if __name__ == "__main__":
    main()
