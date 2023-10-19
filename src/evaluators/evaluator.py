"""
Evaluator class.
"""

import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch

from classes.data.labeled_satellite_image import SegmentationLabeledSatelliteImage
from classes.data.satellite_image import SatelliteImage
from configurators.configurator import Configurator
from utils.plot_utils import (
    plot_list_labeled_sat_images,
    plot_list_segmentation_labeled_satellite_image,
)


class Evaluator:
    """
    Evaluator class.
    """

    def __init__(self, config: Configurator) -> None:
        """
        Constructor for the Evaluator class.
        """
        self.config = config
        self.task_to_evaluation = {
            "PLEIADESsegmentation": self.evaluate_segmentation_pleiades,
            "PLEIADESclassification": self.evaluate_classification_pleiades,
            "SENTINEL1-2segmentation": self.evaluate_segmentation_sentinel,
            "SENTINEL2segmentation": self.evaluate_segmentation_sentinel,
            "change-detection": self.evaluate_changes_detection_pleiades,
            "PLEIADESdetection": self.evaluate_detection_pleiades,
        }
        if not os.path.exists(config.path_eval_test_data[0]):
            os.makedirs(config.path_eval_test_data[0])

    def evaluate_model(self, dataloader, model):
        if self.config.src_task not in self.task_to_evaluation:
            raise ValueError("Invalid task type")
        else:
            self.task_to_evaluation[self.config.src_task](dataloader, model)

    def evaluate_segmentation_pleiades(
        self,
        test_dl,
        model,
    ):
        """
        Evaluates the model on the Pleiade test dataset for image segmentation.

        Args:
            test_dl (torch.utils.data.DataLoader): The data loader for the test
            dataset.
            model (torchvision.models): The segmentation model to evaluate.
            tile_size (int): The size of each tile in pixels.
            batch_size (int): The batch size.
            use_mlflow (bool, optional): Whether to use MLflow for logging
            artifacts. Defaults to False.

        Returns:
            None
        """

        model.eval()
        npatch = int((2000 / self.config.tile_size) ** 2)
        nbatchforfullimage = int(npatch / self.config.batch_size_test)

        if not npatch % nbatchforfullimage == 0:
            print(
                "Le nombre de patchs \
                n'est pas divisible par la taille d'un batch"
            )
            return None

        list_labeled_satellite_image = []

        for idx, batch in enumerate(test_dl):
            images, label, dic = batch

            model = model.to(self.config.device)
            images = images.to(self.config.device)

            output_model = model(images)
            mask_pred = np.array(torch.argmax(output_model, axis=1).to(self.config.device))

            for i in range(self.config.batch_size_test):
                pthimg = dic["pathimage"][i]
                si = SatelliteImage.from_raster(
                    file_path=pthimg, dep=None, date=None, n_bands=self.config.n_bands
                )
                si.normalize()

                list_labeled_satellite_image.append(
                    SegmentationLabeledSatelliteImage(
                        satellite_image=si,
                        label=mask_pred[i],
                        source="",
                        labeling_date="",
                    )
                )

            if ((idx + 1) % nbatchforfullimage) == 0:
                fig = plot_list_labeled_sat_images(list_labeled_satellite_image, [0, 1, 2])

                filename = f"{self.config.path_eval_test_data[0]}/ \
                    {os.path.splitext(os.path.basename(pthimg))[0]}"

                fig.savefig(f"{filename}.png")
                plt.close(fig)
                list_labeled_satellite_image = []

                mlflow.log_artifact(f"{filename}.png", artifact_path="plots")

            del images, label, dic

    def evaluate_segmentation_sentinel(
        self,
        test_dl,
        model,
    ):
        for idx, batch in enumerate(test_dl):
            images, label, dic = batch

            if torch.cuda.is_available():
                model = model.to(self.config.device)
                images = images.to(self.config.device)

            output_model = model(images)
            mask_pred = np.array(torch.argmax(output_model, axis=1).to(self.config.device))

            for i in range(self.config.batch_size_test):
                pthimg = dic["pathimage"][i]
                si = SatelliteImage.from_raster(
                    file_path=pthimg, dep=None, date=None, n_bands=self.config.n_bands
                )
                si.normalize()

                labeled_satellite_image = SegmentationLabeledSatelliteImage(
                    satellite_image=si,
                    label=mask_pred[i],
                    source="",
                    labeling_date="",
                )

                fig = plot_list_segmentation_labeled_satellite_image(
                    [labeled_satellite_image], [3, 2, 1]
                )

                filename = f"{self.config.path_eval_test_data[0]}/ \
                    {os.path.splitext(os.path.basename(pthimg))[0]}"
                fig.savefig(f"{filename}.png")
                plt.close(fig)

                mlflow.log_artifact(f"{filename}.png", artifact_path="plots")

    def evaluate_classification_pleiades(
        self,
        test_dl,
        model,
    ):
        """
        Evaluates the model on the Pleiade test dataset for image classification.

        Args:
            test_dl (torch.utils.data.DataLoader): The data loader for the test
            dataset.
            model (torchvision.models): The classification model to evaluate.
            tile_size (int): The size of each tile in pixels.
            batch_size (int): The batch size.
            use_mlflow (bool, optional): Whether to use MLflow for logging
            artifacts. Defaults to False.

        Returns:
            None
        """
        model.eval()
        npatch = int((2000 / self.config.tile_size) ** 2)
        nbatchforfullimage = int(npatch / self.config.batch_size_test)

        if not npatch % nbatchforfullimage == 0:
            print(
                "Le nombre de patchs \
                n'est pas divisible par la taille d'un batch"
            )
            return None

        list_labeled_satellite_image = []

        for idx, batch in enumerate(test_dl):
            images, label, dic = batch

            model = model.to(self.config.device)
            images = images.to(self.config.device)

            output_model = model(images)
            output_model = output_model.to(self.config.device)
            probability_class_1 = output_model[:, 1]

            # Set a threshold for class prediction
            threshold = 0.51

            # Make predictions based on the threshold
            predictions = torch.where(
                probability_class_1 > threshold,
                torch.tensor([1]),
                torch.tensor([0]),
            )
            predicted_classes = predictions.type(torch.float)

            for i in range(self.config.batch_size_test):
                pthimg = dic["pathimage"][i]
                si = SatelliteImage.from_raster(file_path=pthimg, dep=None, date=None, n_bands=3)
                si.normalize()

                if int(predicted_classes[i]) == 0:
                    mask_pred = np.full(
                        (self.config.tile_size, self.config.tile_size, 3), 255, dtype=np.uint8
                    )

                elif int(predicted_classes[i]) == 1:
                    mask_pred = np.full(
                        (self.config.tile_size, self.config.tile_size, 3), 0, dtype=np.uint8
                    )

                list_labeled_satellite_image.append(
                    SegmentationLabeledSatelliteImage(
                        satellite_image=si,
                        label=mask_pred,
                        source="",
                        labeling_date="",
                    )
                )

            if ((idx + 1) % nbatchforfullimage) == 0:
                fig = plot_list_labeled_sat_images(list_labeled_satellite_image, [0, 1, 2])

                filename = f"{self.config.path_eval_test_data[0]}/ \
                    {os.path.splitext(os.path.basename(pthimg))[0]}"
                fig.savefig(f"{filename}.png")
                plt.close(fig)
                list_labeled_satellite_image = []

                mlflow.log_artifact(f"{filename}.png", artifact_path="plots")

    def evaluate_changes_detection_pleiades(
        self,
        test_dl,
        model,
    ):
        """
        Evaluates the model on the Pleiade test dataset for image segmentation.

        Args:
            test_dl (torch.utils.data.DataLoader): The data loader for the test
            dataset.
            model (torchvision.models): The segmentation model to evaluate.
            tile_size (int): The size of each tile in pixels.
            batch_size (int): The batch size.
            use_mlflow (bool, optional): Whether to use MLflow for logging
            artifacts. Defaults to False.

        Returns:
            None
        """

        model.eval()
        npatch = int((2000 / self.config.tile_size) ** 2)
        nbatchforfullimage = int(npatch / self.config.batch_size_test)

        if not npatch % nbatchforfullimage == 0:
            print(
                "Le nombre de patchs \
                n'est pas divisible par la taille d'un batch"
            )
            return None

        list_labeled_satellite_image = []

        for idx, batch in enumerate(test_dl):
            images, label, dic = batch
            model = model.to(self.config.device)
            images = images.to(self.config.device)

            output_model = model(images)
            mask_pred = np.array(torch.argmax(output_model, axis=1).to(self.config.device))

            for i in range(self.config.batch_size_test):
                pthimg2 = dic["pathimage2"][i]
                si2 = SatelliteImage.from_raster(file_path=pthimg2, dep=None, date=None, n_bands=3)
                si2.normalize()

                list_labeled_satellite_image.append(
                    SegmentationLabeledSatelliteImage(
                        satellite_image=si2,
                        label=mask_pred[i],
                        source="",
                        labeling_date="",
                    )
                )

            if ((idx + 1) % nbatchforfullimage) == 0:
                fig = plot_list_segmentation_labeled_satellite_image(
                    list_labeled_satellite_image, [0, 1, 2]
                )

                filename = f"{self.config.path_eval_test_data[0]}/ \
                    {os.path.splitext(os.path.basename(pthimg2))[0]}"
                fig.savefig(f"{filename}.png")
                plt.close(fig)
                list_labeled_satellite_image = []

                mlflow.log_artifact(f"{filename}.png", artifact_path="plots")

            del images, label, dic

    def evaluate_detection_pleiades(
        self,
        test_dl,
        model,
    ):
        """
        Evaluates the model on the Pleiade test dataset for image segmentation.

        Args:
            test_dl (torch.utils.data.DataLoader): The data loader for the test
            dataset.
            model (torchvision.models): The detection model to evaluate.

        Returns:
            None
        """
        return None
