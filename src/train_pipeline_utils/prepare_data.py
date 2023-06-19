import os
import numpy as np
import rasterio
import csv
import pandas as pd
from tqdm import tqdm

from classes.data.satellite_image import SatelliteImage
from classes.labelers.labeler import Labeler
from utils.filter import has_cloud, is_too_black2, is_too_black, mask_full_cloud, patch_nocloud

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


def filter_images(src, list_images, list_array_cloud = None):
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
        return filter_images_pleiades(list_images,list_array_cloud)
    elif src == "SENTINEL2":
        return filter_images_sentinel2(list_images)


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
                prop_cloud = np.sum(array_cloud)/(array_cloud.shape[0])**2
                if not prop_cloud > 0:
                    list_filtered_splitted_images.append(splitted_image)
    else:
        for splitted_image in list_images:
            if not is_too_black(splitted_image):
                list_filtered_splitted_images.append(splitted_image)

    return list_filtered_splitted_images


def filter_images_sentinel2(list_images):
    """
    filters the Sentinel2 images.

    Args:
        list_images : the list containing the splitted data to be filtered.

    Returns:
        list[SatelliteImage] : the list containing the splitted and\
            filtered data.
    """

    # print("Entre dans la fonction filter_images_sentinel2")
    return list_images


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
    """

    # print("Entre dans la fonction label_images")
    list_masks = []
    list_filtered_splitted_labeled_images = []

    for satellite_image in list_images:
        mask = labeler.create_segmentation_label(satellite_image)
        if task=="segmentation":
            if np.sum(mask) >0:
                list_filtered_splitted_labeled_images.append(satellite_image)
                list_masks.append(mask)

        if task == "classification":
            if np.sum(mask) >=1:
                list_masks.append(1)
            else:
                list_masks.append(0)
            list_filtered_splitted_labeled_images.append(satellite_image)
        
    # print(
    #     "Nombre d'images labelisées : ",
    #     len(list_filtered_splitted_labeled_images),
    #     ", Nombre de masques : ",
    #     len(list_masks),
    # )
    return list_filtered_splitted_labeled_images, list_masks

def save_images_and_masks(list_images, list_masks, output_directory_name, task="segmentation"):
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
    i = 0
    for image, mask in zip(list_images, list_masks):
        # image = list_images[0]
        #bb = image.bounds
        
        #filename = str(bb[0]) + "_" + str(bb[1]) + "_" \
        #   + "{:03d}".format(i)
        filename = image.filename.split(".")[0] + "_" + "{:03d}".format(i)
        i = i + 1
        try:
            image.to_raster(output_images_path, filename + ".jp2", "jp2", None)
            
            if task == "segmentation":
                np.save(
                    output_masks_path + "/" + filename + ".npy",
                    mask,
                )
            if task == "classification":
                csv_file_path = output_masks_path + "/" + 'fichierlabeler.csv'

                    # Create the csv file if it does not exist
                if not os.path.isfile(csv_file_path):
                    with open(csv_file_path, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['Path_image', 'Classification'])
                        writer.writerow([filename, mask])
        
                # Open it if it exists
                else:
                    with open(csv_file_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([filename, mask])

        except rasterio._err.CPLE_AppDefinedError:
            # except:
            print("Writing error", image.filename)
            continue

    return None


def extract_proportional_subset(input_file="train_data-classification-PLEIADES-RIL-972-2022/labels/fichierlabeler.csv", output_file="train_data-classification-PLEIADES-RIL-972-2022/labels/fichierlabeler_echant.csv", target_column="Classification"):
    # Charger le fichier CSV initial
    df = pd.read_csv(input_file)
  
    # Diviser le dataframe en deux dataframes en fonction de la valeur de la colonne cible
    df_0 = df[df[target_column] == 0]
    df_1 = df[df[target_column] == 1]
     
    # Extraire aléatoirement les échantillons de chaque classe
    df_sample_0 = df_0.sample(len(df_1))

    # Concaténer les dataframes échantillons
    df_sample = pd.concat([df_sample_0, df_1])
   
    # Enregistrer le dataframe échantillon dans un nouveau fichier CSV
    df_sample.to_csv(output_file, index=False)

def filter_images_by_path(csv_file = "train_data-classification-PLEIADES-RIL-972-2022/labels/fichierlabeler_echant.csv", image_folder="train_data-classification-PLEIADES-RIL-972-2022/images", path_column="Path_image"):
    # Charger le fichier CSV
    df = pd.read_csv(csv_file)
    
    # Extraire la colonne "path_image" sous forme de liste
    list_name = df[path_column].tolist()
    # list_image_name_to_delete = [
    #     image_folder + "/" + filename 
    #     for filename in tqdm(os.listdir(image_folder))
    #     if filename not in path_list
    #     ]
    list_name_jp2 = [name+".jp2" for name in list_name]
    
    # Parcourir les fichiers dans le dossier d'images
    for filename in tqdm(os.listdir(image_folder)):
        
        # Vérifier si le chemin de l'image n'est pas dans la liste des chemins du fichier CSV
        if filename not in list_name_jp2:
            image_path = os.path.join(image_folder, filename)
            # Supprimer l'image du dossier
            os.remove(image_path)

def copy_images_by_path(csv_file = "src/train_data-classification-PLEIADES-RIL-972-2022/labels/fichierlabeler_echant.csv", source_folder="src/train_data-classification-PLEIADES-RIL-972-2022/images", destination_folder = "src/train_data-classification-PLEIADES-RIL-972-2022/images2", path_column="Path_image"):
    # Charger le fichier CSV
    df = pd.read_csv(csv_file)
    
    # Extraire la colonne "path_image" sous forme de liste
    path_list = df[path_column].tolist()
    
    # Vérifier si le dossier de destination existe, sinon le créer
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Parcourir les fichiers dans le dossier source d'images
    for filename in tqdm(os.listdir(source_folder)):
        image_path = os.path.splitext(filename)[0]
        
        # Vérifier si le chemin de l'image n'est pas dans la liste des chemins du fichier CSV
        if image_path in path_list:
            # Copier l'image vers le dossier de destination
            shutil.copy(os.path.join(source_folder, filename), destination_folder)
