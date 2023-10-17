import csv
import os
import random
from typing import List

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from utils.filter import is_too_black, is_too_water


def check_labelled_images(output_directory_name):
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

    output_images_path = f"{output_directory_name}/images"
    output_masks_path = f"{output_directory_name}/labels"
    if (os.path.exists(output_masks_path)) and (len(os.listdir(output_masks_path)) != 0):
        print("\t** The directory already exists and is not empty.")
        return True
    elif (os.path.exists(output_masks_path)) and (len(os.listdir(output_masks_path)) == 0):
        print("\t** The directory exists but is empty.")
        return False
    else:
        os.makedirs(output_images_path)
        os.makedirs(output_masks_path)
        print("\t** Directory created !")
        return False


def filter_images(src, list_images, list_array_cloud=None):
    """
    calls the appropriate function according to the data type.

    Args:
        src : the string that specifies the data type.
        list_images : the list containing the splitted data to be filtered.

    Returns:
        function : a call to the appopriate filter function according to\
            the data type.
    """

    if src == "PLEIADES":
        return filter_images_pleiades(list_images, list_array_cloud)
    else:
        return filter_images_sentinel(list_images)


def filter_images_pleiades(list_images, list_array_cloud):
    """
    filters the Pleiades images that are too dark and/or contain clouds.

    Args:
        list_images : the list containing the splitted data to be filtered.

    Returns:
        list[SatelliteImage] : the list containing the splitted \
            and filtered data.
    """
    list_filtered_splitted_images = []

    if list_array_cloud:
        for splitted_image, array_cloud in zip(list_images, list_array_cloud):
            if not is_too_black(splitted_image):
                prop_cloud = np.sum(array_cloud) / (array_cloud.shape[0]) ** 2
                if not prop_cloud > 0:
                    list_filtered_splitted_images.append(splitted_image)
    else:
        for splitted_image in list_images:
            if not is_too_black(splitted_image):
                list_filtered_splitted_images.append(splitted_image)

    return list_filtered_splitted_images


def filter_images_sentinel(list_images):
    """
    filters the Sentinel images.

    Args:
        list_images : the list containing the splitted data to be filtered.

    Returns:
        list[SatelliteImage] : the list containing the splitted and\
            filtered data.
    """

    list_filtered_splitted_images = []
    for splitted_image in list_images:
        if not is_too_water(splitted_image, 0.95):
            list_filtered_splitted_images.append(splitted_image)

    return list_filtered_splitted_images


def label_images(list_images, labeler, task: str):
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
            filtered data with the associated labels.
        Dict: Dictionary indicating if images contain a building or not.
    """
    prop = 1
    labels = []
    balancing_dict = {}
    for i, satellite_image in enumerate(list_images):
        # label = labeler.create_label(satellite_image, task=task) 
        label = labeler.create_segmentation_label(satellite_image)
        if task in ["segmentation", "change_detection"]:
            if np.sum(label) != 0:
                balancing_dict[f"{satellite_image.filename.split('.')[0]}_{i:04d}"] = 1
            else:
                balancing_dict[f"{satellite_image.filename.split('.')[0]}_{i:04d}"] = 0
            labels.append(label)
        elif task == "classification":
            if np.sum(label) != 0:
                balancing_dict[f"{satellite_image.filename.split('.')[0]}_{i:04d}"] = 1
                labels.append(1)
            else:
                balancing_dict[f"{satellite_image.filename.split('.')[0]}_{i:04d}"] = 0
                labels.append(0)
        elif task == "detection":
            labels.append(label)
            # TODO : balance data

    balancing_dict_copy = balancing_dict.copy()
    nb_ones = sum(1 for value in balancing_dict_copy.values() if value == 1)
    nb_zeros = sum(1 for value in balancing_dict_copy.values() if value == 0)
    length = len(list_images)

    if nb_zeros > prop * nb_ones and nb_ones == 0:
        sampled_nb_zeros = int((length * 0.01) * prop)
        zeros = [key for key, value in balancing_dict_copy.items()]
        random.shuffle(zeros)
        selected_zeros = zeros[:sampled_nb_zeros]
        balancing_dict_sampled = {
            key: value for key, value in balancing_dict_copy.items() if key in selected_zeros
        }
        indices_sampled = [
            index for index, key in enumerate(balancing_dict_copy) if key in balancing_dict_sampled
        ]
        labels = [labels[index] for index in indices_sampled]
        balancing_dict_copy = balancing_dict_sampled

    elif nb_zeros > prop * nb_ones and nb_ones > 0:
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
            index for index, key in enumerate(balancing_dict_copy) if key in balancing_dict_sampled
        ]
        labels = [labels[index] for index in indices_sampled]
        balancing_dict_copy = balancing_dict_sampled

    # print(
    #     "Nombre d'images labelisées : ",
    #     len(list_filtered_splitted_labeled_images),
    #     ", Nombre de masques : ",
    #     len(list_masks),
    # )
    return labels, balancing_dict


def filter_buildingless(images: List, labels: List, task: str):
    """
    Filter a list of images and associated labels to remove
    buildingless images.

    Args:
        images : list containing images.
        labels : list of corresponding labels.
        task (str): task considered.
    """
    if task == "segmentation":
        return filter_buildingless_segmentation(images, labels)
    elif task == "detection":
        return filter_buildingless_detection(images, labels)
    else:
        raise NotImplementedError("Task must be 'segmentation'" "or 'detection'.")


def filter_buildingless_segmentation(images: List, labels: List):
    """
    Filter a list of images and associated labels to remove
    buildingless images for segmentation.

    Args:
        images : list containing images.
        labels : list of corresponding labels.
    """
    filtered_images = []
    filtered_labels = []

    for image, label in zip(images, labels):
        if np.sum(label) != 0:
            filtered_images.append(image)
            filtered_labels.append(label)

    return filtered_images, filtered_labels


def filter_buildingless_detection(images: List, labels: List):
    """
    Filter a list of images and associated labels to remove
    buildingless images for segmentation.

    Args:
        images : list containing images.
        labels : list of corresponding labels.
    """
    filtered_images = []
    filtered_labels = []

    for image, label in zip(images, labels):
        if np.sum(label) != 0:
            filtered_images.append(image)
            filtered_labels.append(label)

    return filtered_images, filtered_labels


def save_images_and_labels(list_images, list_labels, output_directory_name, task: str):
    """
    write the couple images/labels into a specific folder.

    Args:
        list_images : the list containing the splitted and filtered data \
            to be saved.
        list_masks : the list containing the masks to be saved.
        a string representing the name of the output \
            directory where the split images and their masks should be saved.
        task (str): Task considered.

    Returns:
        str: The name of the output directory.
    """
    output_images_path = f"{output_directory_name}/images"
    output_masks_path = f"{output_directory_name}/labels"

    for i, (image, mask) in enumerate(zip(list_images, list_labels)):
        filename = f"{image.filename.split('.')[0]}_{i:04d}"

        i += 1

        try:
            if task != "classification":
                image.to_raster(output_images_path, f"{filename}.jp2", "jp2", None)
                np.save(
                    f"{output_masks_path}/{filename}.npy",
                    mask,
                )
            if task == "classification":
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


def extract_proportional_subset(
    input_file="train_data-classification-PLEIADES-RIL-972-2022/labels/fichierlabeler.csv",
    output_file="train_data-classification-PLEIADES-RIL-972-2022/labels/fichierlabeler_echant.csv",
    target_column="Classification",
):
    """
    Extracts a proportional subset of samples from a CSV file based on the
    target column without loss of information on class 1.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the extracted subset to a new CSV file.
        target_column (str): Name of the target column used for extracting the
        subset.

    Returns:
        None
    """

    # Load the initial CSV file
    df = pd.read_csv(input_file)

    # Split the dataframe into two dataframes based on the target column value
    df_0 = df[df[target_column] == 0]
    df_1 = df[df[target_column] == 1]

    # Randomly sample the same number of samples from each class
    df_sample_0 = df_0.sample(len(df_1))

    # Concatenate the sample dataframes
    df_sample = pd.concat([df_sample_0, df_1])

    # Save the sample dataframe to a new CSV file
    df_sample.to_csv(output_file, index=False)


def filter_images_by_path(
    csv_file="train_data-classification-PLEIADES-RIL-972-2022/labels/fichierlabeler_echant.csv",
    image_folder="train_data-classification-PLEIADES-RIL-972-2022/images",
    path_column="Path_image",
):
    """
    Filters images in a folder based on their paths listed in a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing the image paths.
        image_folder (str): Path to the folder containing the images.
        path_column (str): Name of the column in the CSV file that contains
        the image paths.

    Returns:
        None
    """

    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Extract the "path_image" column as a list
    list_name = df[path_column].tolist()

    list_name_jp2 = [f"{name}.jp2" for name in list_name]

    # Iterate over the files in the image folder
    for filename in tqdm(os.listdir(image_folder)):
        # Check if the image path is not in the list of paths from the CSV file
        if filename not in list_name_jp2:
            image_path = os.path.join(image_folder, filename)
            # Remove the image from the folder
            os.remove(image_path)
