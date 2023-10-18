"""
Configurator class.
"""
import os

import torch
import yaml


class Configurator:
    """
    Configurator class.
    """

    def __init__(self, config_path: str, environment_path: str) -> None:
        """
        Constructor for the Configurator class.
        """
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        with open(environment_path) as f:
            env = yaml.load(f, Loader=yaml.SafeLoader)

        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        dep_dict = {
            "971": "GUADELOUPE",
            "972": "MARTINIQUE",
            "973": "GUYANE",
            "974": "REUNION",
            "976": "MAYOTTE",
        }
        self.millesime = config["data"]["millesime"]
        self.type_labeler = config["data"]["type_labeler"]
        self.buffer_size = config["data"]["buffer_size"]
        self.dataset = config["data"]["dataset"]
        self.dataset_test = config["data"]["dataset_test"]
        self.n_bands = config["data"]["n_bands"]
        self.percent_keep = config["data"]["percent_keep"]

        self.task = config["data"]["task"]
        self.prop = config["data"]["prop"]
        self.source_train = config["data"]["source_train"]
        self.augmentation = config["data"]["augmentation"]
        self.tile_size = config["data"]["tile_size"]
        self.n_channels_train = config["data"]["n_channels_train"]
        self.num_workers = int(os.cpu_count() * 0.1)
        self.src_task = f"{self.source_train}{self.task}"

        self.loss = config["optim"]["loss"]
        self.lr = config["optim"]["lr"]
        self.momentum = config["optim"]["momentum"]
        self.batch_size = config["optim"]["batch_size"]
        self.batch_size_test = config["optim"]["batch_size_test"]
        self.max_epochs = config["optim"]["max_epochs"]
        self.module = config["optim"]["module"]
        self.val_prop = config["optim"]["val_prop"]
        self.accumulate_batch = config["optim"]["accumulate_batch"]
        self.checkpoints = config["optim"]["monitoring"]["checkpoints"]
        self.earlystop = config["optim"]["monitoring"]["earlystop"]

        self.src_to_download = (
            ["SENTINEL1", "SENTINEL2"] if "SENTINEL1" in self.source_train else [self.source_train]
        )
        self.path_local = [
            env["local-path"][src][mlsm["year"]][mlsm["dep"]]
            for mlsm in self.millesime
            for src in self.src_to_download
        ]
        self.path_s3 = [
            f"{env['bucket']}/{env['sources'][src][mlsm['year']][mlsm['dep']]}"
            for mlsm in self.millesime
            for src in self.src_to_download
        ]
        self.path_local_test = [env["local-path"]["TEST"][self.source_train][self.task]]
        self.path_s3_test = [
            f"{env['bucket']}/{env['sources']['TEST'][self.source_train][self.task]}"
        ]
        self.path_local_cloud = self.get_cloud_local_path(env)
        self.path_s3_cloud = self.get_cloud_s3_path(env)

        self.path_prepro_data = [
            f"data/preprocessed/{self.task}/{self.source_train}/{self.type_labeler}"
            f"/{millesime['year']}/{dep_dict[millesime['dep']]}"
            for millesime in self.millesime
        ]

        self.path_prepro_test_data = [f"data/preprocessed/{self.task}/{self.source_train}/test"]
        self.path_eval_test_data = [f"data/evaluation/{self.task}/{self.source_train}"]
        self.device = device

    def get_cloud_local_path(self, env: dict):
        if self.source_train == "PLEIADES":
            path = [
                env["local-path"]["NUAGESPLEIADES"][mlsm["year"]][mlsm["dep"]]
                for mlsm in self.millesime
                for src in self.src_to_download
            ]
            return path
        return []

    def get_cloud_s3_path(self, env: dict):
        if self.source_train == "PLEIADES":
            path = [
                f"{env['bucket']}/{env['sources']['NUAGESPLEIADES'][mlsm['year']][mlsm['dep']]}"
                for mlsm in self.millesime
                for src in self.src_to_download
            ]
            return path
        return []
