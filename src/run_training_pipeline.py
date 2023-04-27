import os
from datetime import datetime

import numpy as np
import yaml
from yaml.loader import SafeLoader

from classes.labelers.labeler import RILLabeler
from data.components.dataset import PleiadeDataset
from models.components.segmentation_models import DeepLabv3Module
from train_pipeline_utils.download_data import load_pleiade_data
from train_pipeline_utils.prepare_data import write_splitted_images_masks


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


def instanciate_dataset(config):
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

    dataset_dict = {"PLEIADE": PleiadeDataset}
    dataset_type = config["donnees"]["source train"]

    if dataset_type not in dataset_dict:
        raise ValueError("Invalid datset type")
    else:
        return dataset_dict[dataset_type](list_path_images, list_path_labels)


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


def prepare_DataLoader(list_path):
    return None


def run_pipeline():
    # Open the file and load the file
    with open("../config.yml") as f:
        config = yaml.load(f, Loader=SafeLoader)

    list_output_dir = download_data(config)
    prepare_data(config, list_output_dir)

    dataset = instanciate_dataset(config)
    model = instantiate_model(config)

    return dataset, model
