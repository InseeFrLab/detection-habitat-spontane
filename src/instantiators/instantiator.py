"""
Instantiator class.
"""
import json
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader

from classes.optim.optimizer import generate_optimization_elements
from configurators.configurator import Configurator
from dico_config import dataset_dict, loss_dict, module_dict, task_to_lightningmodule
from train_pipeline_utils.handle_dataset import (
    generate_transform_pleiades,
    generate_transform_sentinel,
    select_indices_to_balance,
    select_indices_to_split_dataset,
)


class Instantiator:
    """
    Instantiator class.
    """

    def __init__(self, config: Configurator) -> None:
        """
        Constructor for the Instantiator class.
        """
        self.config = config

    def dataset(self, list_images, list_labels, list_images_2=None, test=False):
        """
        Instantiates the appropriate dataset object
        based on the configuration settings.

        Args:
            list_path_images: A list of strings representing
            the paths to the preprocessed tile image files.
            list_path_labels: A list of strings representing
            the paths to the corresponding preprocessed mask image files.

        Returns:
            A dataset object of the specified type.
        """
        if not test:
            dataset_type = self.config.dataset
        else:
            dataset_type = self.config.dataset_test

        # instanciation du dataset complet
        if dataset_type not in dataset_dict:
            raise ValueError("Invalid dataset type")
        else:
            dataset_select = dataset_dict[dataset_type]

            if list_images_2 is None:
                full_dataset = dataset_select(list_images, list_labels, self.config.n_bands)
            else:
                full_dataset = dataset_select(
                    list_images, list_images_2, list_labels, self.config.n_bands
                )

        return full_dataset

    def dataloader(self, list_output_dir):
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
        if self.config.source_train in [
            "PLEIADES",
            "SENTINEL2",
            "SENTINEL1-2",
        ]:
            list_labels = []
            list_images = []
            full_balancing_dict = {}
            for directory in list_output_dir:
                # dir = list_output_dir[0]
                labels = os.listdir(f"{directory}/labels")
                images = os.listdir(f"{directory}/images")
                if labels[0][0] == ".":
                    del labels[0]

                if self.config.task != "classification":
                    with open(f"{directory}/balancing_dict.json") as json_file:
                        balancing_dict = json.load(json_file)

                    list_labels = np.concatenate(
                        (
                            list_labels,
                            np.sort([f"{directory}/labels/{name}" for name in labels]),
                        )
                    )

                    for k, v in balancing_dict.items():
                        full_balancing_dict[k] = v

                if self.config.task == "classification":
                    list_labels_dir = []

                    # Load the initial CSV file
                    df = pd.read_csv(f"{directory}/labels/{labels[0]}")

                    list_labels_dir = df[["Path_image", "Classification"]].values.tolist()

                    list_labels_dir = sorted(list_labels_dir, key=lambda x: x[0])
                    list_labels_dir = np.array([sous_liste[1] for sous_liste in list_labels_dir])

                    list_labels = np.concatenate((list_labels, list_labels_dir))

                list_images = np.concatenate(
                    (
                        list_images,
                        np.sort([f"{directory}/images/{name}" for name in images]),
                    )
                )

        if self.config.task == "segmentation":
            unbalanced_images = list_images.copy()
            unbalanced_labels = list_labels.copy()
            indices_to_balance = select_indices_to_balance(
                list_images, full_balancing_dict, prop=self.config.prop
            )
            list_images = unbalanced_images[indices_to_balance]
            list_labels = unbalanced_labels[indices_to_balance]

        train_idx, val_idx = select_indices_to_split_dataset(
            self.config.task, self.config.val_prop, list_labels
        )

        # Retrieving the desired Dataset class
        train_dataset = self.dataset(list_images[train_idx], list_labels[train_idx])

        valid_dataset = self.dataset(list_images[val_idx], list_labels[val_idx])

        # Applying the respective transforms

        if self.config.source_train == "PLEIADES":
            t_aug, t_preproc = generate_transform_pleiades(
                self.config.tile_size,
                self.config.augmentation,
                self.config.task,
            )
        else:
            t_aug, t_preproc = generate_transform_sentinel(
                self.config.source_train,
                self.config.year[0],
                self.config.dep[0],
                self.config.tile_size,
                self.config.augmentation,
                self.config.task,
            )

        train_dataset.transforms = t_aug
        valid_dataset.transforms = t_preproc

        # Creation of the dataloaders
        batch_size = self.config.batch_size

        train_dataloader, valid_dataloader = [
            DataLoader(ds, batch_size=batch_size, shuffle=boolean, num_workers=0, drop_last=True)
            for ds, boolean in zip([train_dataset, valid_dataset], [True, False])
        ]

        output_test = "../test-data"
        output_labels_path = f"{output_test}/labels/"
        list_name_label_test = os.listdir(output_labels_path)
        list_path_labels_test = np.sort(
            [f"{output_labels_path}{name_label}" for name_label in list_name_label_test]
        )

        if self.config.task != "change-detection":
            output_images_path = f"{output_test}/images/"
            list_name_image_test = os.listdir(output_images_path)
            list_path_images_test = np.sort(
                [f"{output_images_path}{name_image}" for name_image in list_name_image_test]
            )

            dataset_test = self.dataset(list_path_images_test, list_path_labels_test, test=True)
            dataset_test.transforms = t_preproc
        else:
            output_images_path_1 = f"{output_test}/images_1/"
            list_name_image_1 = os.listdir(output_images_path_1)
            list_path_images_1 = np.sort(
                [f"{output_images_path_1}{name_image}" for name_image in list_name_image_1]
            )

            output_images_path_2 = f"{output_test}/images_2/"
            list_name_image_2 = os.listdir(output_images_path_2)
            list_path_images_2 = np.sort(
                [f"{output_images_path_2}{name_image}" for name_image in list_name_image_2]
            )

            dataset_test = self.dataset(
                list_path_images_1,
                list_path_labels_test,
                list_images_2=list_path_images_2,
                test=True,
            )
            dataset_test.transforms = t_preproc

        batch_size_test = self.config.batch_size_test
        test_dataloader = DataLoader(
            dataset_test,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=0,
        )

        return train_dataloader, valid_dataloader, test_dataloader

    def model(self):
        """
        Instantiate a module based on the provided module type.

        Args:
            module_type (str): Type of module to instantiate.

        Returns:
            object: Instance of the specified module.
        """
        print("Entre dans la fonction instantiate_model")

        if self.config.module not in module_dict:
            raise ValueError("Invalid module type")

        if self.config.module == "deeplabv3":
            return module_dict[self.config.module](self.config.n_channels_train)
        else:
            return module_dict[self.config.module]()

    def loss(self):
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

        if self.config.loss not in loss_dict:
            raise ValueError("Invalid loss type")
        else:
            return loss_dict[self.config.loss]()

    def lightning_module(self):
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
        # TODO : gérer la fonction generate_optimization_elements avec la config
        list_params = generate_optimization_elements(self.config)

        if self.config.task not in task_to_lightningmodule:
            raise ValueError("Invalid task type")
        else:
            LightningModule = task_to_lightningmodule[self.config.task]

        lightning_module = LightningModule(
            model=self.model(),
            loss=self.loss(),
            optimizer=list_params[0],
            optimizer_params=list_params[1],
            scheduler=list_params[2],
            scheduler_params=list_params[3],
            scheduler_interval=list_params[4],
        )

        return lightning_module

    def trainer(self):
        """
        Create a PyTorch Lightning module for segmentation with
        the given model and optimization configuration.

        Args:
            parameters for optimization.
            model: The PyTorch model to use for segmentation.

        Returns:
            trainer: return a trainer object
        """
        # def callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor=self.config.monitor, save_top_k=1, save_last=True, mode=self.config.mode
        )

        early_stop_callback = EarlyStopping(
            monitor=self.config.monitor, mode=self.config.mode, patience=self.config.patience
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        if self.config.task == "segmentation":
            checkpoint_callback_IOU = ModelCheckpoint(
                monitor=self.config.monitor, save_top_k=1, save_last=True, mode=self.config.mode
            )
            list_callbacks = [
                lr_monitor,
                checkpoint_callback,
                early_stop_callback,
                checkpoint_callback_IOU,
            ]

        if self.config.task == "classification":
            list_callbacks = [lr_monitor, checkpoint_callback, early_stop_callback]

        if self.config.task == "change-detection":
            checkpoint_callback_IOU = ModelCheckpoint(
                monitor=self.config.monitor, save_top_k=1, save_last=True, mode=self.config.mode
            )
            list_callbacks = [
                lr_monitor,
                checkpoint_callback,
                early_stop_callback,
                checkpoint_callback_IOU,
            ]

        strategy = "auto"

        trainer = pl.Trainer(
            callbacks=list_callbacks,
            max_epochs=self.config.max_epochs,
            num_sanity_val_steps=2,
            strategy=strategy,
            log_every_n_steps=2,
            accumulate_grad_batches=self.config.accumulate_batch,
        )

        return trainer
