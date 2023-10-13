"""
Evaluator class.
"""

import os

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
        }

    def evaluate_model(self, dataloader, model, device):
        if self.config.src_task not in self.task_to_evaluation:
            raise ValueError("Invalid task type")
        else:
            self.task_to_evaluation[self.config.src_task](dataloader, model, device)

    def evaluate_segmentation_pleiades(
        self,
        test_dl,
        model,
        device: str = "cpu",
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
            device (str): Device.

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
            print(idx)
            images, label, dic = batch

            model = model.to(device)
            images = images.to(device)

            output_model = model(images)
            mask_pred = np.array(torch.argmax(output_model, axis=1).to("cpu"))

            for i in range(len(batch)):
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
                print("ecriture image")
                if not os.path.exists("img/"):
                    os.makedirs("img/")

                fig1 = plot_list_labeled_sat_images(list_labeled_satellite_image, [0, 1, 2])

                filename = pthimg.split("/")[-1]
                filename = filename.split(".")[0]
                filename = "_".join(filename.split("_")[0:6])
                plot_file = f"img/{filename}.png"

                fig1.savefig(plot_file)
                list_labeled_satellite_image = []

                mlflow.log_artifact(plot_file, artifact_path="plots")

            del images, label, dic

    def evaluate_segmentation_sentinel(
        self,
        test_dl,
        model,
        device: str = "cpu",
    ):
        for idx, batch in enumerate(test_dl):
            images, label, dic = batch

            if torch.cuda.is_available():
                model = model.to(device)
                images = images.to(device)

            output_model = model(images)
            mask_pred = np.array(torch.argmax(output_model, axis=1).to("cpu"))

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

                print("ecriture image")
                if not os.path.exists("img/"):
                    os.makedirs("img/")

                fig1 = plot_list_segmentation_labeled_satellite_image(
                    [labeled_satellite_image], [3, 2, 1]
                )

                filename = pthimg.split("/")[-1]
                filename = filename.split(".")[0]
                filename = "_".join(filename.split("_")[0:6])
                plot_file = f"{filename}.png"

                fig1.savefig(plot_file)

                mlflow.log_artifact(plot_file, artifact_path="plots")

    def evaluate_classification_pleiades(
        self,
        test_dl,
        model,
        device: str = "cpu",
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
            device (str): Device.

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
            print(idx)
            images, label, dic = batch

            model = model.to(device)
            images = images.to(device)

            output_model = model(images)
            output_model = output_model.to(device)
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
                print("ecriture image")
                if not os.path.exists("img/"):
                    os.makedirs("img/")

                fig1 = plot_list_labeled_sat_images(list_labeled_satellite_image, [0, 1, 2])

                filename = pthimg.split("/")[-1]
                filename = filename.split(".")[0]
                filename = "_".join(filename.split("_")[0:6])
                plot_file = f"img/{filename}.png"

                fig1.savefig(plot_file)
                list_labeled_satellite_image = []

                mlflow.log_artifact(plot_file, artifact_path="plots")

    def evaluate_changes_detection_pleiades(
        self,
        test_dl,
        model,
        device: str = "cpu",
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
            device (str): Device.

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
            # idx, batch = 0, next(iter(test_dl))
            print(idx)
            images, label, dic = batch
            model = model.to(device)
            images = images.to(device)

            output_model = model(images)
            mask_pred = np.array(torch.argmax(output_model, axis=1).to("cpu"))

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
                print("ecriture image")
                if not os.path.exists("img/"):
                    os.makedirs("img/")

                fig1 = plot_list_segmentation_labeled_satellite_image(
                    list_labeled_satellite_image, [0, 1, 2]
                )

                filename = pthimg2.split("/")[-1]
                filename = filename.split(".")[0]
                filename = "_".join(filename.split("_")[0:6])
                plot_file = f"{filename}.png"

                fig1.savefig(plot_file)
                list_labeled_satellite_image = []

                mlflow.log_artifact(plot_file, artifact_path="plots")

            del images, label, dic
