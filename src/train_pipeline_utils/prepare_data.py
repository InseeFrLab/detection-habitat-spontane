import csv
import os

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from osgeo import gdal

from utils.filter import is_too_black, is_too_water
from classes.data.satellite_image import SatelliteImage
from classes.data.change_detection_triplet import ChangedetectionTripletS2Looking


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
        return filter_images_sentinel(list_images, src)


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


def filter_images_sentinel(list_images, src):
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
        array = splitted_image.array
        if not is_too_water(splitted_image, 0.95):
            if not np.isnan(array).any():
                if src == 'SENTINEL2-RVB':
                    splitted_image.array = splitted_image.array[(3, 2, 1), :, :]
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


    # print(
    #     "Nombre d'images labelisées : ",
    #     len(list_filtered_splitted_labeled_images),
    #     ", Nombre de masques : ",
    #     len(list_masks),
    # )
    return labels, balancing_dict


def save_images_and_masks(
    list_images,
    list_masks,
    output_directory_name,
    direc=None,
    task="segmentation",
):
    """
    write the couple images/masks into a specific folder.

    Args:
        list_images : the list containing the splitted and filtered data \
            to be saved.
        list_masks : the list containing the masks to be saved. \
            a string representing the name of the output \
            directory where the split images and their masks should be saved.
        output_directory_name : the name of the directory where the images \
            are saved.
        direc : the directory containing the images for which the proj \
            is to be retrieved.

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
                if np.sum(mask) >= 0.1*image.array.shape[1]**2:
                    in_ds = gdal.Open(direc+'/'+image.filename)
                    proj = in_ds.GetProjection()

                    image.to_raster(output_images_path, filename, "tif", proj)
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
    prop=1,
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

    nb_zeros = len(df_0)
    nb_ones = len(df_1)

    difference = abs(nb_ones - nb_zeros)
    tolerance = 0.2*prop*nb_ones

    if difference > tolerance:

        if nb_zeros > nb_ones:
            # Randomly sample the same number of samples from each class
            df_sample_max = df_0.sample(prop*nb_ones)
            df_not_sample = df_1
        else:
            # Randomly sample the same number of samples from each class
            df_sample_max = df_1.sample(prop*nb_zeros)
            df_not_sample = df_0

        # Concatenate the sample dataframes
        df_sample = pd.concat([df_sample_max, df_not_sample])

        # Save the sample dataframe to a new CSV file
        os.remove(input_file)
        df_sample.to_csv(input_file, index=False)


def filter_images_by_path(
    csv_file="train_data-classification-PLEIADES-RIL-972-2022/"
    + "labels/fichierlabeler.csv",
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


def split_and_filter_s2(output_dir, list_images1, list_images2, list_labels):

    for img1, img2, lab in tqdm(zip(list_images1, list_images2, list_labels), total = len(list_images1)):
            triplet = ChangedetectionTripletS2Looking(img1, img2, lab)
            list_split_im1, list_split_im2, list_split_lab = triplet.split(tile_size)

            for small_img1, small_img2, small_lab, i in zip(list_split_im1, list_split_im2, list_split_lab, range(tile_size)):
                if np.sum(np.asarray(small_lab))>0 and small_img1: # filtre les images sans changement
                    path = img1.split("/")[-1].split(".")[0] + "_" + "{:03d}".format(i)+".png"
                    small_img1.save(output_dir + "Image1/" + path)
                    small_img2.save(output_dir + "Image2/" + path)
                    small_lab.save(output_dir + "label/" + path)
    
    return output_dir


def prepare_data_per_doss(directory_path, tile_size, type_data):

    output_dir = (
            "../"+ type_data +"_data-S2Looking-"
            + str(tile_size)
            + "/"
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir +"Image1/")
        os.makedirs(output_dir +"Image2/")
        os.makedirs(output_dir +"label/")

    else:
        return output_dir

    list_images1 = list_sorted_filenames(directory_path + "Image1/")
    list_images2 = list_sorted_filenames(directory_path + "Image2/")
    list_labels = list_sorted_filenames(directory_path + "label/")

    split_and_filter_s2(output_dir, list_images1, list_images2, list_labels)

    return output_dir
