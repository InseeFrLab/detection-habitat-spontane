"""
Preprocessor class.
"""
import json
import os
import random
from datetime import datetime

import numpy as np
import s3fs
from rasterio.errors import RasterioIOError
from tqdm import tqdm

from classes.data.labeled_satellite_image import (  # noqa: E501
    SegmentationLabeledSatelliteImage,
)
from classes.data.satellite_image import SatelliteImage
from classes.labelers.labeler import BDTOPOLabeler, RILLabeler
from configurators.configurator import Configurator
from train_pipeline_utils.prepare_data import (
    check_labelled_images,
    filter_images,
    label_images,
    save_images_and_masks,
)
from utils.utils import remove_dot_file, split_array


class Preprocessor:
    """
    Preprocessor class.
    """

    def __init__(self, config: Configurator) -> None:
        """
        Constructor for the Preprocessor class.
        """
        self.path_local_test = config.path_local_test
        self.path_local = config.path_local
        self.path_local_cloud = config.path_local_cloud
        self.path_s3_test = config.path_s3_test
        self.path_s3 = config.path_s3
        self.path_s3_cloud = config.path_s3_cloud

    def download_data(self):
        """
        Downloads data based on the given configuration.

        Args:
            config: a dictionary representing the
            configuration information for data download.

        Returns:
            A list of output directories for each downloaded dataset.
        """

        print("\n*** Téléchargement des données...\n")
        fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"})

        [
            fs.download(rpath=path_s3, lpath=f"../{path_local}", recursive=True)
            for path_local, path_s3 in zip(
                self.path_local_test + self.path_local + self.path_local_cloud,
                self.path_s3_test + self.path_s3 + self.path_s3_cloud,
            )
            if not os.path.exists(path_local)
        ]
        print("\n*** Téléchargement terminé !\n")

        return None

    def prepare_train_data(self, list_data_dir, list_masks_cloud_dir):
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

        for i, (year, dep) in enumerate(zip(self.config.year, self.config.dep)):
            # i, year , dep = 0,config.year[0],config.dep[0]
            output_dir = (
                f"../train_data2-{self. config.task}-{self. config.type_labeler}-{dep}-{str(year)}/"
            )

            date = datetime.strptime(f"{str(year)}0101", "%Y%m%d")

            if self.config.type_labeler == "RIL":
                buffer_size = self.config.buffer_size
                labeler = RILLabeler(date, dep=dep, buffer_size=buffer_size)
            elif self.config.type_labeler == "BDTOPO":
                labeler = BDTOPOLabeler(date, dep=dep)

            if not check_labelled_images(output_dir):
                list_name_cloud = []
                if self.config.source_train == "PLEIADES":
                    cloud_dir = list_masks_cloud_dir[i]
                    list_name_cloud = [
                        path.split("/")[-1].split(".")[0] for path in os.listdir(cloud_dir)
                    ]

                list_path = [
                    f"{list_data_dir[i]}/{filename}" for filename in os.listdir(list_data_dir[i])
                ]

                full_balancing_dict = {}
                for path in tqdm(list_path):
                    try:
                        si = SatelliteImage.from_raster(
                            file_path=path,
                            dep=dep,
                            date=date,
                            n_bands=self.config.n_bands,
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
                            list_splitted_mask_cloud = split_array(
                                mask_full_cloud, self.config.tile_size
                            )

                        list_splitted_images = si.split(self.config.tile_size)

                        list_filtered_splitted_images = filter_images(
                            self.config.source_train,
                            list_splitted_images,
                            list_splitted_mask_cloud,
                        )

                        labels, balancing_dict = label_images(
                            list_filtered_splitted_images, labeler, task=self.config.task
                        )

                        save_images_and_masks(
                            list_filtered_splitted_images,
                            labels,
                            output_dir,
                            task=self.config.task,
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

    def prepare_test_data(self, test_dir):
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

        if self.config.task != "change-detection":
            images_path = f"{test_dir}/images"
            list_name_image = os.listdir(images_path)
            list_name_image = np.sort(remove_dot_file(list_name_image))
            list_images_path = [f"{images_path}/{name}" for name in list_name_image]
            output_images_path = f"{output_test}/images"

            for image_path, label_path, name in zip(
                list_images_path, list_labels_path, list_name_image
            ):
                si = SatelliteImage.from_raster(
                    file_path=image_path, dep=None, date=None, n_bands=self.config.n_bands
                )
                mask = np.load(label_path)

                lsi = SegmentationLabeledSatelliteImage(si, mask, "", "")
                list_lsi = lsi.split(self.config.tile_size)

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
                    file_path=image_path_1, dep=None, date=None, n_bands=self.config.n_bands
                )
                si2 = SatelliteImage.from_raster(
                    file_path=image_path_2, dep=None, date=None, n_bands=self.config.n_bands
                )
                mask = np.load(label_path)

                lsi1 = SegmentationLabeledSatelliteImage(si1, mask, "", "")
                lsi2 = SegmentationLabeledSatelliteImage(si2, mask, "", "")

                list_lsi1 = lsi1.split(self.config.tile_size)
                list_lsi2 = lsi2.split(self.config.tile_size)

                for i, (lsi1, lsi2) in enumerate(zip(list_lsi1, list_lsi2)):
                    file_name_i = f"{name.split('.')[0]}_{i:03d}"

                    lsi1.satellite_image.to_raster(output_images_path_1, f"{file_name_i}.jp2")
                    lsi2.satellite_image.to_raster(output_images_path_2, f"{file_name_i}.jp2")
                    np.save(f"{output_labels_path}/{file_name_i}.npy", lsi1.label)
