import csv
import os
import random

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

    print("Entre dans la fonction check_labelled_images")
    output_images_path = output_directory_name + "/images"
    output_masks_path = output_directory_name + "/labels"
    if (os.path.exists(output_masks_path)) and (
        len(os.listdir(output_masks_path)) != 0
    ):
        print("The directory already exists and is not empty.")
        return True
    elif (os.path.exists(output_masks_path)) and (
        len(os.listdir(output_masks_path)) == 0
    ):
        print("The directory exists but is empty.")
        return False
    else:
        os.makedirs(output_images_path)
        os.makedirs(output_masks_path)
        print("Directory created")
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

    # print("Entre dans la fonction filter_images")
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
    # print("Entre dans la fonction filter_images_pleiades")
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

    # print("Entre dans la fonction filter_images_sentinel")
    list_filtered_splitted_images = []
    for splitted_image in list_images:
        if not is_too_water(splitted_image, 0.95):
            list_filtered_splitted_images.append(splitted_image)

    return list_filtered_splitted_images


def label_images(list_images, labeler, task="segmentation"):
    """
    labels the images according to type of labeler desired.

    Args:
        list_images : the list containing the splitted and filtered data \
            to be labeled.
        labeler : a Labeler object representing the labeler \
            used to create segmentation labels.

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
        if task != "classification":
            if np.sum(mask) != 0:
                balancing_dict[
                    satellite_image.filename.split(".")[0] + "_" + "{:04d}".format(i)
                ] = 1
            else:
                balancing_dict[
                    satellite_image.filename.split(".")[0] + "_" + "{:04d}".format(i)
                ] = 0
            labels.append(mask)
        elif task == "classification":
            if np.sum(mask) != 0:
                balancing_dict[
                    satellite_image.filename.split(".")[0] + "_" + "{:04d}".format(i)
                ] = 1
                labels.append(1)
            else:
                balancing_dict[
                    satellite_image.filename.split(".")[0] + "_" + "{:04d}".format(i)
                ] = 0
                labels.append(0)

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
            key: value
            for key, value in balancing_dict_copy.items()
            if key in selected_zeros
        }
        indices_sampled = [
            index
            for index, key in enumerate(balancing_dict_copy)
            if key in balancing_dict_sampled
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
            index
            for index, key in enumerate(balancing_dict_copy)
            if key in balancing_dict_sampled
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


def save_images_and_masks(
    list_images, list_masks, output_directory_name, task="segmentation"
):
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
    output_images_path = output_directory_name + "/images"
    output_masks_path = output_directory_name + "/labels"

    # if task == "classification":
    #     count_ones = list_masks.count(1)
    #     count_zeros_sampled = int(count_ones*prop)
    #     indices_1 = [i for i, lab in enumerate(list_masks) if lab == 1]
    #     indices_0 = [i for i, lab in enumerate(list_masks) if lab == 0]
    #     random.shuffle(indices_0)
    #     selected_indices_0 = indices_0[:count_zeros_sampled]
    #     selected_indices = indices_1 + selected_indices_0

    for i, (image, mask) in enumerate(zip(list_images, list_masks)):
        # image = list_images[0]
        # bb = image.bounds

        # filename = str(bb[0]) + "_" + str(bb[1]) + "_" \
        #   + "{:03d}".format(i)
        filename = image.filename.split(".")[0] + "_" + "{:04d}".format(i)
        i = i + 1

        try:
            if task != "classification":
                image.to_raster(output_images_path, filename + ".jp2", "jp2", None)
                np.save(
                    output_masks_path + "/" + filename + ".npy",
                    mask,
                )
            if task == "classification":
                # if i in selected_indices:
                image.to_raster(output_images_path, filename + ".jp2", "jp2", None)
                csv_file_path = output_masks_path + "/" + "fichierlabeler.csv"

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
    input_file="train_data-classification-PLEIADES-RIL-972-2022/"
    + "labels/fichierlabeler.csv",
    output_file="train_data-classification-PLEIADES-RIL-972-2022/"
    + "labels/fichierlabeler_echant.csv",
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
    csv_file="train_data-classification-PLEIADES-RIL-972-2022/"
    + "labels/fichierlabeler_echant.csv",
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

    list_name_jp2 = [name + ".jp2" for name in list_name]

    # Iterate over the files in the image folder
    for filename in tqdm(os.listdir(image_folder)):
        # Check if the image path is not in the list of paths from the CSV file
        if filename not in list_name_jp2:
            image_path = os.path.join(image_folder, filename)
            # Remove the image from the folder
            os.remove(image_path)
