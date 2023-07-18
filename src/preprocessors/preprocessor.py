"""
Preprocessor class.
"""
import csv
import json

# import json
import os
import random
from datetime import datetime

import numpy as np
import rasterio
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
    filter_images_pleiades,
    filter_images_sentinel,
)
from utils.utils import get_path_by_millesime, remove_dot_file, split_array


class Preprocessor:
    """
    Preprocessor class.
    """

    def __init__(self, config: Configurator) -> None:
        """
        Constructor for the Preprocessor class.
        """
        self.config = config

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
                self.config.path_local_test + self.config.path_local + self.config.path_local_cloud,
                self.config.path_s3_test + self.config.path_s3 + self.config.path_s3_cloud,
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

        for i, millesime in enumerate(self.config.millesime):
            labeler = self.get_labeller(millesime)

            if not self.check_labelled_images(millesime):
                full_balancing_dict = {}

                for root, dirs, files in os.walk(f"../{self.config.path_local[i]}"):
                    for filename in tqdm(files):
                        try:
                            si = SatelliteImage.from_raster(
                                file_path=os.path.join(root, filename),
                                dep=millesime["dep"],
                                date=datetime.strptime(f"{millesime['year']}0101", "%Y%m%d"),
                                n_bands=self.config.n_bands,
                            )
                        except RasterioIOError:
                            print(f"Erreur de lecture du fichier {os.path.join(root, filename)}")
                            continue

                        mask = labeler.create_segmentation_label(si)
                        # TODO: Mettre dans config la proba
                        keep = np.random.binomial(size=1, n=1, p=0.1)[0]

                        if (np.sum(mask) == 0 and keep) or np.sum(mask) != 0:
                            balancing_dict = self.prepare_yearly_data(
                                si, labeler, filename, millesime
                            )

                            for k, v in balancing_dict.items():
                                full_balancing_dict[k] = v

                        elif np.sum(mask) == 0 and not keep:
                            continue

                with open(f"../{self.config.path_prepro_data[i]}/balancing_dict.json", "w") as fp:
                    json.dump(full_balancing_dict, fp)

                nb = len(os.listdir(f"../{self.config.path_prepro_data[i]}/images"))
                print(f"{str(nb)} couples images/masques retenus")

        return None

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

    def check_labelled_images(self, millesime):
        """
        checks that there is not already a directory with images and their mask.
        if it doesn't exist, it is created.

        Args:
            output_directory_name: a string representing the path to \
                the directory that may already contain data and masks.

        Returns:
            boolean: True if the directory exists and is not empty. \
                False if the directory doesn't exist or is empty.
        """

        print("Entre dans la fonction check_labelled_images")
        path_prepro = get_path_by_millesime(f"../{self.config.path_prepro_data}", millesime)

        if (os.path.exists(path_prepro)) and (len(os.listdir(path_prepro)) != 0):
            print("The directory already exists and is not empty.")
            return True
        elif (os.path.exists(path_prepro)) and (len(os.listdir(path_prepro)) == 0):
            print("The directory exists but is empty.")
            return False
        else:
            os.makedirs(f"{path_prepro}/images")
            os.makedirs(f"{path_prepro}/labels")
            print("Directory created")
            return False

    def filter_images(self, list_images, list_array_cloud=None):
        """
        calls the appropriate function according to the data type.

        Args:
            src : the string that specifies the data type.
            list_images : the list containing the splitted data to be filtered.

        Returns:
            function : a call to the appopriate filter function according to\
                the data type.
        """

        # print("Entre dans la fonction filter_images")
        if self.config.source_train == "PLEIADES":
            return filter_images_pleiades(list_images, list_array_cloud)
        else:
            return filter_images_sentinel(list_images)

    def label_images(self, list_images, labeler):
        """
        labels the images according to type of labeler and task desired.

        Args:
            list_images : the list containing the splitted and filtered data \
                to be labeled.
            labeler : a Labeler object representing the labeler \
                used to create segmentation labels.
            task (str): task considered.

        Returns:
            list[SatelliteImage] : the list containing the splitted and \
                filtered data with a not-empty mask and the associated masks.
            Dict: Dictionary indicating if images contain a building or not.
        """
        prop = 1
        labels = []
        balancing_dict = {}
        for i, satellite_image in enumerate(list_images):
            mask = labeler.create_segmentation_label(satellite_image)
            if self.config.source_train != "classification":
                if np.sum(mask) != 0:
                    balancing_dict[f"{satellite_image.filename.split('.')[0]}_{i:04d}"] = 1
                else:
                    balancing_dict[f"{satellite_image.filename.split('.')[0]}_{i:04d}"] = 0
                labels.append(mask)
            elif self.config.source_train == "classification":
                if np.sum(mask) != 0:
                    balancing_dict[f"{satellite_image.filename.split('.')[0]}_{i:04d}"] = 1
                    labels.append(1)
                else:
                    balancing_dict[f"{satellite_image.filename.split('.')[0]}_{i:04d}"] = 0
                    labels.append(0)

        balancing_dict_copy = balancing_dict.copy()
        nb_ones = sum(1 for value in balancing_dict_copy.values() if value == 1)
        nb_zeros = sum(1 for value in balancing_dict_copy.values() if value == 0)
        length = len(list_images)

        if nb_zeros > prop * nb_ones and nb_ones == 0:
            # TODO : VERIFY THIS
            sampled_nb_zeros = int((length * 0.01) * prop)
            zeros = [key for key, value in balancing_dict_copy.items()]
            random.shuffle(zeros)
            selected_zeros = zeros[:sampled_nb_zeros]
            balancing_dict_sampled = {
                key: value for key, value in balancing_dict_copy.items() if key in selected_zeros
            }
            indices_sampled = [
                index
                for index, key in enumerate(balancing_dict_copy)
                if key in balancing_dict_sampled
            ]
            labels = [labels[index] for index in indices_sampled]
            balancing_dict_copy = balancing_dict_sampled

        elif nb_zeros > prop * nb_ones and nb_ones > 0:
            # TODO : VERIFY THIS
            sampled_nb_zeros = int(prop * nb_ones)
            zeros = [key for key, value in balancing_dict_copy.items() if value == 0]
            random.shuffle(zeros)
            selected_zeros = zeros[:sampled_nb_zeros]
            balancing_dict_sampled = {
                key: value
                for key, value in balancing_dict_copy.items()
                if value == 1 or key in selected_zeros
            }
            indices_sampled = [
                index
                for index, key in enumerate(balancing_dict_copy)
                if key in balancing_dict_sampled
            ]
            labels = [labels[index] for index in indices_sampled]
            balancing_dict_copy = balancing_dict_sampled

        return labels, balancing_dict

    def save_images_and_masks(self, list_images, list_masks, millesime):
        """
        write the couple images/masks into a specific folder.

        Args:
            list_images : the list containing the splitted and filtered data \
                to be saved.
            list_masks : the list containing the masks to be saved.
            a string representing the name of the output \
                directory where the split images and their masks should be saved.

        Returns:
            str: The name of the output directory.
        """
        # print("Entre dans la fonction save_images_and_masks")

        path_prepro = get_path_by_millesime(f"../{self.config.path_prepro_data}", millesime)
        output_images_path = f"{path_prepro}/images"
        output_masks_path = f"{path_prepro}/labels"

        for i, (image, mask) in enumerate(zip(list_images, list_masks)):
            # TODO : Make it more readable
            filename = f"{image.filename.split('.')[0]}_{i:04d}"

            i = i + 1

            try:
                if self.config.source_train != "classification":
                    image.to_raster(output_images_path, f"{filename}.jp2", "jp2", None)
                    np.save(
                        f"{output_masks_path}/{filename}.npy",
                        mask,
                    )
                if self.config.source_train == "classification":
                    # if i in selected_indices:
                    image.to_raster(output_images_path, f"{filename}.jp2", "jp2", None)
                    csv_file_path = f"{output_masks_path}/fichierlabeler.csv"

                    # Create the csv file if it does not exist
                    if not os.path.isfile(csv_file_path):
                        with open(csv_file_path, "w", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(["Path_image", "Classification"])
                            writer.writerow([filename, mask])

                    # Open it if it exists
                    else:
                        with open(csv_file_path, "a", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([filename, mask])

            except rasterio._err.CPLE_AppDefinedError:
                # except:
                print("Writing error", image.filename)
                continue

        return None

    def get_labeller(self, millesime):
        date = datetime.strptime(f"{millesime['year']}0101", "%Y%m%d")

        labeler = None

        match self.config.type_labeler:
            case "RIL":
                labeler = RILLabeler(
                    date, dep=millesime["dep"], buffer_size=self.config.buffer_size
                )
            case "BDTOPO":
                labeler = BDTOPOLabeler(date, dep=millesime["dep"])
            case _:
                pass

        return labeler

    def prepare_yearly_data(self, satellite_image, labeler, filename, millesime):
        path_clouds = get_path_by_millesime(self.config.path_local_cloud, millesime)
        list_clouds = [
            os.path.splitext(filename)[0] for filename in os.listdir(f"../{path_clouds}")
        ]

        if os.path.splitext(filename)[0] in list_clouds:
            mask_full_cloud = np.load(f"{path_clouds}/{os.path.splitext(filename)[0]}.npy")
            list_splitted_mask_cloud = split_array(mask_full_cloud, self.config.tile_size)
        else:
            list_splitted_mask_cloud = None

        list_splitted_images = satellite_image.split(self.config.tile_size)

        list_filtered_splitted_images = self.filter_images(
            list_splitted_images,
            list_splitted_mask_cloud,
        )

        labels, balancing_dict = self.label_images(list_filtered_splitted_images, labeler)

        path_prepro = get_path_by_millesime(self.config.path_prepro_data, millesime)

        self.save_images_and_masks(list_filtered_splitted_images, labels, f"../{path_prepro}")

        return balancing_dict
