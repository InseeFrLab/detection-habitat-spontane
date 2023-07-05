import gc
import json
import os
import random
import sys
from datetime import datetime

import mlflow
import numpy as np
import torch
from rasterio.errors import RasterioIOError
from tqdm import tqdm

from classes.data.labeled_satellite_image import (  # noqa: E501
    SegmentationLabeledSatelliteImage,
)
from classes.data.satellite_image import SatelliteImage
from classes.labelers.labeler import BDTOPOLabeler, RILLabeler
from configurators.configurator import Configurator
from dico_config import task_to_evaluation
from instantiators.instantiator import Instantiator
from train_pipeline_utils.download_data import (
    load_2satellites_data,
    load_donnees_test,
    load_satellite_data,
)
from train_pipeline_utils.prepare_data import (
    check_labelled_images,
    filter_images,
    label_images,
    save_images_and_masks,
)
from utils.utils import (
    get_root_path,
    remove_dot_file,
    split_array,
    update_storage_access,
)


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
    list_output_dir = []

    if config.source_train == "PLEIADES":
        list_masks_cloud_dir = []
        for year, dep in zip(config.year, config.dep):
            cloud_dir = load_satellite_data(year, dep, "NUAGESPLEIADES")
            output_dir = load_satellite_data(year, dep, config.source_train)
            list_masks_cloud_dir.append(cloud_dir)
            list_output_dir.append(output_dir)

    elif config.source_train == "SENTINEL1-2":
        for year, dep in zip(config.year, config.dep):
            output_dir = load_2satellites_data(year, dep, config.source_train)
            list_output_dir.append(output_dir)

    elif config.source_train == "SENTINEL2":
        for year, dep in zip(config.year, config.dep):
            output_dir = load_satellite_data(year, dep, config.source_train)
            list_output_dir.append(output_dir)

    print("Chargement des données test")
    test_dir = load_donnees_test(type=config.task, src=config.source_train)

    return list_output_dir, list_masks_cloud_dir, test_dir


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

    print("Entre dans la fonction prepare_data")

    list_output_dir = []

    for i, (year, dep) in enumerate(zip(config.year, config.dep)):
        # i, year , dep = 0,config.year[0],config.dep[0]
        output_dir = f"../train_data2-{config.task}-{config.type_labeler}-{dep}-{str(year)}/"

        date = datetime.strptime(f"{str(year)}0101", "%Y%m%d")

        if config.type_labeler == "RIL":
            buffer_size = config.buffer_size
            labeler = RILLabeler(date, dep=dep, buffer_size=buffer_size)
        elif config.type_labeler == "BDTOPO":
            labeler = BDTOPOLabeler(date, dep=dep)

        if not check_labelled_images(output_dir):
            list_name_cloud = []
            if config.source_train == "PLEIADES":
                cloud_dir = list_masks_cloud_dir[i]
                list_name_cloud = [
                    path.split("/")[-1].split(".")[0] for path in os.listdir(cloud_dir)
                ]

            list_path = [f"{list_data_dir[i]}/{filename}" for filename in os.listdir(dir)]

            full_balancing_dict = {}
            for path in tqdm(list_path):
                try:
                    si = SatelliteImage.from_raster(
                        file_path=path,
                        dep=dep,
                        date=date,
                        n_bands=config.n_bands,
                    )
                except RasterioIOError:
                    print(f"Erreur de lecture du fichier {path}")
                    continue

                mask = labeler.create_segmentation_label(si)
                proba = random.randint(1, 10)

                if (np.sum(mask) == 0 and proba == 10) or np.sum(mask) != 0:
                    filename = path.split("/")[-1].split(".")[0]
                    list_splitted_mask_cloud = None

                    if filename in list_name_cloud:
                        mask_full_cloud = np.load(f"{cloud_dir}/{filename}.npy")
                        list_splitted_mask_cloud = split_array(mask_full_cloud, config.tile_size)

                    list_splitted_images = si.split(config.tile_size)

                    list_filtered_splitted_images = filter_images(
                        config.source_train,
                        list_splitted_images,
                        list_splitted_mask_cloud,
                    )

                    labels, balancing_dict = label_images(
                        list_filtered_splitted_images, labeler, task=config.task
                    )

                    save_images_and_masks(
                        list_filtered_splitted_images,
                        labels,
                        output_dir,
                        task=config.task,
                    )

                    for k, v in balancing_dict.items():
                        full_balancing_dict[k] = v

                elif np.sum(mask) == 0 and proba != 10:
                    continue

            with open(f"{output_dir}/balancing_dict.json", "w") as fp:
                json.dump(full_balancing_dict, fp)

        list_output_dir.append(output_dir)
        nb = len(os.listdir(f"{output_dir}/images"))
        print(f"{str(nb)} couples images/masques retenus")

    return list_output_dir


def prepare_test_data(config, test_dir):
    print("Entre dans la fonction prepare_test_data")

    output_test = "../test-data"
    output_labels_path = f"{output_test}/labels"

    if not os.path.exists(output_labels_path):
        os.makedirs(output_labels_path)
    else:
        return None

    labels_path = f"{test_dir}/masks"
    list_name_label = os.listdir(labels_path)
    list_name_label = np.sort(remove_dot_file(list_name_label))
    list_labels_path = [f"{labels_path}/{name}" for name in list_name_label]

    if config.task != "change-detection":
        images_path = f"{test_dir}/images"
        list_name_image = os.listdir(images_path)
        list_name_image = np.sort(remove_dot_file(list_name_image))
        list_images_path = [f"{images_path}/{name}" for name in list_name_image]
        output_images_path = f"{output_test}/images"

        for image_path, label_path, name in zip(
            list_images_path, list_labels_path, list_name_image
        ):
            si = SatelliteImage.from_raster(
                file_path=image_path, dep=None, date=None, n_bands=config.n_bands
            )
            mask = np.load(label_path)

            lsi = SegmentationLabeledSatelliteImage(si, mask, "", "")
            list_lsi = lsi.split(config.tile_size)

            for i, lsi in enumerate(list_lsi):
                file_name_i = f"{name.split('.')[0]}_{i:03d}"

                lsi.satellite_image.to_raster(output_images_path, f"{file_name_i}.jp2")
                np.save(f"{output_labels_path}/{file_name_i}.npy", lsi.label)
    else:
        images_path_1 = f"{test_dir}/images_1"
        list_name_image_1 = os.listdir(images_path_1)
        list_name_image_1 = np.sort(remove_dot_file(list_name_image_1))
        list_images_path_1 = [f"{images_path_1}/{name}" for name in list_name_image_1]
        output_images_path_1 = f"{output_test}/images_1"

        images_path_2 = f"{test_dir}/images_2"
        list_name_image_2 = os.listdir(images_path_2)
        list_name_image_2 = np.sort(remove_dot_file(list_name_image_2))
        list_images_path_2 = [f"{images_path_2}/{name}" for name in list_name_image_2]
        output_images_path_2 = f"{output_test}/images_2"

        for image_path_1, image_path_2, label_path, name in zip(
            list_images_path_1,
            list_images_path_2,
            list_labels_path,
            list_name_image_1,
        ):
            si1 = SatelliteImage.from_raster(
                file_path=image_path_1, dep=None, date=None, n_bands=config.n_bands
            )
            si2 = SatelliteImage.from_raster(
                file_path=image_path_2, dep=None, date=None, n_bands=config.n_bands
            )
            mask = np.load(label_path)

            lsi1 = SegmentationLabeledSatelliteImage(si1, mask, "", "")
            lsi2 = SegmentationLabeledSatelliteImage(si2, mask, "", "")

            list_lsi1 = lsi1.split(config.tile_size)
            list_lsi2 = lsi2.split(config.tile_size)

            for i, (lsi1, lsi2) in enumerate(zip(list_lsi1, list_lsi2)):
                file_name_i = f"{name.split('.')[0]}_{i:03d}"

                lsi1.satellite_image.to_raster(output_images_path_1, f"{file_name_i}.jp2")
                lsi2.satellite_image.to_raster(output_images_path_2, f"{file_name_i}.jp2")
                np.save(f"{output_labels_path}/{file_name_i}.npy", lsi1.label)


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
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # Open the file and load the file
    configurator = Configurator(get_root_path() / "config.yml")

    # TODO :  Download data devrait rien retourner donc à améliorer
    list_data_dir, list_masks_cloud_dir, test_dir = download_data(configurator)

    list_output_dir = prepare_train_data(configurator, list_data_dir, list_masks_cloud_dir)

    prepare_test_data(configurator, test_dir)

    instantiator = Instantiator(configurator)

    train_dl, valid_dl, test_dl = instantiator.dataloader(list_output_dir)
    trainer, light_module = instantiator.trainer()

    torch.cuda.empty_cache()
    gc.collect()

    # remote_server_uri = "https://projet-slums-detection-128833.user.lab.sspcloud.fr"
    # experiment_name = "classification"
    # run_name = "essai35"

    if configurator.mlflow:
        update_storage_access()
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://minio.lab.sspcloud.fr"
        mlflow.end_run()
        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment(experiment_name)
        # mlflow.pytorch.autolog()

        with mlflow.start_run(run_name=run_name):
            mlflow.autolog()
            mlflow.log_artifact(get_root_path() / "config.yml", artifact_path="config.yml")
            trainer.fit(light_module, train_dl, valid_dl)

            if configurator.source_train == "PLEIADES":
                light_module_checkpoint = light_module.load_from_checkpoint(
                    loss=instantiator.loss(),
                    checkpoint_path=trainer.checkpoint_callback.best_model_path,
                    model=light_module.model,
                    optimizer=light_module.optimizer,
                    optimizer_params=light_module.optimizer_params,
                    scheduler=light_module.scheduler,
                    scheduler_params=light_module.scheduler_params,
                    scheduler_interval=light_module.scheduler_interval,
                )

                model = light_module_checkpoint.model
                try:
                    print(model.device)
                except Exception:
                    pass

                if configurator.task not in task_to_evaluation:
                    raise ValueError("Invalid task type")
                else:
                    evaluer_modele_sur_jeu_de_test = task_to_evaluation[configurator.task]

                evaluer_modele_sur_jeu_de_test(
                    test_dl,
                    model,
                    configurator.tile_size,
                    configurator.batch_size_test,
                    configurator.n_bands,
                    configurator.mlflow,
                    device,
                )

    else:
        trainer.fit(light_module, train_dl, valid_dl)

        light_module_checkpoint = light_module.load_from_checkpoint(
            loss=instantiator.loss(),
            checkpoint_path=trainer.checkpoint_callback.best_model_path,
            model=light_module.model,
            optimizer=light_module.optimizer,
            optimizer_params=light_module.optimizer_params,
            scheduler=light_module.scheduler,
            scheduler_params=light_module.scheduler_params,
            scheduler_interval=light_module.scheduler_interval,
        )
        model = light_module_checkpoint.model

        if configurator.task not in task_to_evaluation:
            raise ValueError("Invalid task type")
        else:
            evaluer_modele_sur_jeu_de_test = task_to_evaluation[configurator.task]

        evaluer_modele_sur_jeu_de_test(
            test_dl,
            model,
            configurator.tile_size,
            configurator.batch_size_test,
            configurator.n_bands,
            configurator.mlflow,
        )


if __name__ == "__main__":
    # MLFlow params
    remote_server_uri = sys.argv[1]
    experiment_name = sys.argv[2]
    run_name = sys.argv[3]
    run_pipeline(remote_server_uri, experiment_name, run_name)


# nohup python run_training_pipeline.py
# https://projet-slums-detection-128833.user.lab.sspcloud.fr
# classification test_classifpleiade_branchsentinel2 > out.txt &
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
