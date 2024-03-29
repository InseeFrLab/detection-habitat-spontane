"""
Preprocessor class.
"""
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path

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
from utils.utils import get_path_by_millesime, split_array


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

        print("\n*** 1- Téléchargement des données...\n")

        all_exist = all(
            os.path.exists(f"{directory}")
            for directory in self.config.path_local_test
            + self.config.path_local
            + self.config.path_local_cloud
        )

        if all_exist:
            pass
        else:
            fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"})

            [
                fs.download(rpath=path_s3, lpath=f"{path_local}", recursive=True)
                for path_local, path_s3 in zip(
                    self.config.path_local_test
                    + self.config.path_local
                    + self.config.path_local_cloud,
                    self.config.path_s3_test + self.config.path_s3 + self.config.path_s3_cloud,
                )
                if not os.path.exists(path_local)
            ]

        print("\n*** Téléchargement terminé !\n")

    def prepare_train_data(self):
        """
        Preprocesses and splits the raw input images
        into tiles and corresponding labels,
        and saves them in the specified output directories.

        Args:
            config: A dictionary representing the configuration settings.
            list_data_dir: A list of strings representing the paths
            to the directories containing the raw input image files.

        Returns:
            A list of strings representing the paths to
            the output directories containing the
            preprocessed tile and label image files.
        """

        print("\n*** 2- Préparation des données d'entrainement...\n")

        for i, millesime in enumerate(self.config.millesime):
            labeler = self.get_labeler(millesime)

            if not self.check_labelled_images(millesime):
                full_balancing_dict = {}

                for root, dirs, files in os.walk(f"{self.config.path_local[i]}"):
                    for filename in files:
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

                        label = labeler.create_label(si)
                        # TODO: Mettre dans config la proba
                        keep = np.random.binomial(size=1, n=1, p=0.1)[0]

                        if (np.sum(label) == 0 and keep) or np.sum(label) != 0:
                            balancing_dict = self.prepare_yearly_data(
                                si, labeler, filename, millesime
                            )

                            for k, v in balancing_dict.items():
                                full_balancing_dict[k] = v

                        elif np.sum(label) == 0 and not keep:
                            continue

                with open(f"{self.config.path_prepro_data[i]}/balancing_dict.json", "w") as fp:
                    json.dump(full_balancing_dict, fp)

                nb = len(os.listdir(f"{self.config.path_prepro_data[i]}/images"))
                print(f"\t** {nb} couples images/labels ont été retenus")

        print("\n*** Données d'entrainement prêtes !\n")
        return None

    def prepare_test_data(self):
        print("\n*** 3- Préparation des données de test...\n")

        # TODO: Temporaire à supprimer quand on aura des données de test pour la détection
        if self.config.task == "detection":
            print("\n*** 3- Préparation des données de test skipped...\n")
            return None

        nb_test_img = self.get_nb_test_images()
        nb_test_img_prepo = (
            len(os.listdir(f"{self.config.path_prepro_test_data[0]}/images/"))
            if os.path.exists(f"{self.config.path_prepro_test_data[0]}/images/")
            else None
        )

        if nb_test_img == nb_test_img_prepo:
            print("\n\t** Données de test déjà créées!\n")
        else:
            label_folder = f"{self.config.path_prepro_test_data[0]}/labels/"
            os.makedirs(label_folder, exist_ok=True)

            # Get extension of images
            ext = list(
                set(
                    [
                        os.path.splitext(file)[1]
                        for file in os.listdir(f"{self.config.path_local_test[0]}/images")
                        if os.path.splitext(file)[1] != ""
                    ]
                )
            )[0]

        match self.config.task:
            case "change-detection":
                # Cas change-detection : On a 2 images et 1 label
                for root, dirs, files in os.walk(f"{self.config.path_local_test[0]}/labels"):
                    for filename in files:
                        label = np.load(os.path.join(root, filename))
                        # Les fichiers d'images n'ont pas de suffix
                        filename_im = filename.replace("_0000", "")

                        # On loop sur les 2 images
                        dict_lsi = {}
                        for i in range(1, 3):
                            root_im = root.replace("/labels", f"/images_{i}")

                            si = SatelliteImage.from_raster(
                                file_path=Path(os.path.join(root_im, filename_im)).with_suffix(
                                    ".tif"
                                ),
                                dep=None,
                                date=None,
                                n_bands=self.config.n_bands,
                            )

                            dict_lsi[i] = SegmentationLabeledSatelliteImage(
                                si, label, "", ""
                            ).split(self.config.tile_size)

                        # On loop sur toutes les images et labels divisés pour les sauvegarder
                        for j in range(len(dict_lsi[1])):
                            label_path = f"{label_folder}{filename.replace('_0000', f'_{j:04d}')}"
                            np.save(label_path, dict_lsi[1].label)

                            for i in range(1, len(dict_lsi) + 1):
                                im_path = Path(
                                    label_path.replace("/labels", f"/images_{i}")
                                ).with_suffix(".jp2")
                                dict_lsi[i].satellite_image.to_raster(
                                    os.path.dirname(im_path), os.path.basename(im_path)
                                )

            case _:
                # Autres cas : On a 1 image et 1 label
                for root, dirs, files in os.walk(f"{self.config.path_local_test[0]}/labels"):
                    for filename in files:
                        if filename.startswith("."):
                            continue

                        filename_im = filename.replace("_0000", "")
                        root_im = root.replace("/labels", "/images")

                        label = np.load(os.path.join(root, filename))

                        si = SatelliteImage.from_raster(
                            file_path=Path(os.path.join(root_im, filename_im)).with_suffix(ext),
                            dep=None,
                            date=None,
                            n_bands=self.config.n_bands,
                        )

                        lsi = SegmentationLabeledSatelliteImage(si, label, "", "")
                        list_lsi = lsi.split(self.config.tile_size)

                        # On loop sur toutes les images et labels divisés pour les sauvegarder

                        for i, splitted_image in tqdm(enumerate(list_lsi)):
                            label_path = f"{label_folder}{filename.replace('_0000', f'_{i:04d}')}"
                            im_path = Path(label_path.replace("/labels", "/images")).with_suffix(
                                ".jp2"
                            )

                            splitted_image.satellite_image.to_raster(
                                os.path.dirname(im_path), os.path.basename(im_path)
                            )

                            np.save(label_path, splitted_image.label)

        print("\n*** Données de test prêtes !\n")

    def check_labelled_images(self, millesime):
        """
        checks that there is not already a directory with images and their label.
        if it doesn't exist, it is created.

        Args:
            output_directory_name: a string representing the path to \
                the directory that may already contain data and labels.

        Returns:
            boolean: True if the directory exists and is not empty. \
                False if the directory doesn't exist or is empty.
        """

        path_prepro = get_path_by_millesime(self.config.path_prepro_data, millesime)

        if (os.path.exists(f"{path_prepro}")) and (len(os.listdir(f"{path_prepro}")) != 0):
            return True
        elif (os.path.exists(f"{path_prepro}")) and (len(os.listdir(f"{path_prepro}")) == 0):
            return False
        else:
            os.makedirs(f"{path_prepro}/images")
            os.makedirs(f"{path_prepro}/labels")
            print("\t** Dossiers créés !")
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
                filtered data with a not-empty label and the associated labels.
            Dict: Dictionary indicating if images contain a building or not.
        """
        prop = 1
        labels = []
        balancing_dict = {}
        for i, satellite_image in enumerate(list_images):
            label = labeler.create_label(satellite_image)
            if self.config.source_train != "classification":
                if np.sum(label) != 0:
                    balancing_dict[f"{satellite_image.filename.split('.')[0]}_{i:04d}"] = 1
                else:
                    balancing_dict[f"{satellite_image.filename.split('.')[0]}_{i:04d}"] = 0
                labels.append(label)
            elif self.config.source_train == "classification":
                if np.sum(label) != 0:
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

    def save_images_and_labels(self, list_images, list_labels, millesime):
        """
        write the couple images/labels into a specific folder.

        Args:
            list_images : the list containing the splitted and filtered data \
                to be saved.
            list_labels : the list containing the labels to be saved.
            a string representing the name of the output \
                directory where the split images and their labels should be saved.

        Returns:
            str: The name of the output directory.
        """
        path_prepro = get_path_by_millesime(self.config.path_prepro_data, millesime)
        output_images_path = f"{path_prepro}/images"
        output_labels_path = f"{path_prepro}/labels"

        for i, (image, label) in enumerate(zip(list_images, list_labels)):
            # TODO : Make it more readable
            filename = f"{image.filename.split('.')[0]}_{i:04d}"

            i = i + 1

            try:
                if self.config.source_train != "classification":
                    image.to_raster(output_images_path, f"{filename}.jp2", "jp2", None)
                    np.save(
                        f"{output_labels_path}/{filename}.npy",
                        label,
                    )
                if self.config.source_train == "classification":
                    # if i in selected_indices:
                    image.to_raster(output_images_path, f"{filename}.jp2", "jp2", None)
                    csv_file_path = f"{output_labels_path}/fichierlabeler.csv"

                    # Create the csv file if it does not exist
                    if not os.path.isfile(csv_file_path):
                        with open(csv_file_path, "w", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(["Path_image", "Classification"])
                            writer.writerow([filename, label])

                    # Open it if it exists
                    else:
                        with open(csv_file_path, "a", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([filename, label])

            except rasterio._err.CPLE_AppDefinedError:
                # except:
                print("Writing error", image.filename)
                continue

        return None

    def get_labeler(self, millesime):
        date = datetime.strptime(f"{millesime['year']}0101", "%Y%m%d")

        labeler = None
        match self.config.type_labeler:
            case "RIL":
                labeler = RILLabeler(
                    date,
                    dep=millesime["dep"],
                    task=self.config.task,
                    buffer_size=self.config.buffer_size,
                )
            case "BDTOPO":
                labeler = BDTOPOLabeler(date, dep=millesime["dep"], task=self.config.task)
            case _:
                pass
        return labeler

    def prepare_yearly_data(self, satellite_image, labeler, filename, millesime):
        path_clouds = get_path_by_millesime(self.config.path_local_cloud, millesime)
        list_clouds = (
            [os.path.splitext(filename)[0] for filename in os.listdir(f"{path_clouds}")]
            if path_clouds
            else []
        )

        if os.path.splitext(filename)[0] in list_clouds:
            label_full_cloud = np.load(f"{path_clouds}/{os.path.splitext(filename)[0]}.npy")
            list_splitted_label_cloud = split_array(label_full_cloud, self.config.tile_size)
        else:
            list_splitted_label_cloud = None

        list_splitted_images = satellite_image.split(self.config.tile_size)

        list_filtered_splitted_images = self.filter_images(
            list_splitted_images,
            list_splitted_label_cloud,
        )

        labels, balancing_dict = self.label_images(list_filtered_splitted_images, labeler)

        self.save_images_and_labels(list_filtered_splitted_images, labels, millesime)

        return balancing_dict

    def get_nb_test_images(self):
        test_images = [
            img
            for img in os.listdir(f"{self.config.path_local_test[0]}/images")
            if not img.startswith(".")
        ]
        si = SatelliteImage.from_raster(
            file_path=f"{self.config.path_local_test[0]}/images/{test_images[0]}",
            dep=None,
            date=None,
            n_bands=self.config.n_bands,
        )
        return (si.array.shape[1] / self.config.tile_size) ** 2 * len(test_images)
