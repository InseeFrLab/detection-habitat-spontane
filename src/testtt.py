import gc
import os
import sys
from datetime import datetime
import csv
import math
import yaml
import re
import s3fs
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im
import re
from pyproj import Transformer
from datetime import date
from scipy.ndimage import label
import time
from tqdm import tqdm
import os
import math

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from rasterio.errors import RasterioIOError
from torch.utils.data import DataLoader
from tqdm import tqdm
from yaml.loader import SafeLoader

import train_pipeline_utils.handle_dataset as hd
from classes.data.satellite_image import SatelliteImage
from classes.labelers.labeler import BDTOPOLabeler, RILLabeler
from classes.optim.losses import CrossEntropySelfmade
from torch.nn import CrossEntropyLoss
from classes.optim.optimizer import generate_optimization_elements
from data.components.dataset import PleiadeDataset
from data.components.classification_patch import PatchClassification
from models.components.segmentation_models import DeepLabv3Module
from models.components.classification_models import ResNet50Module
from models.segmentation_module import SegmentationModule

from train_pipeline_utils.download_data import load_satellite_data, load_donnees_test
from train_pipeline_utils.handle_dataset import (
    generate_transform,
    select_indices_to_split_dataset
)

from classes.optim.losses import SoftIoULoss

from train_pipeline_utils.prepare_data import(
    filter_images,
    label_images,
    save_images_and_masks,
    check_labelled_images
)

import torch.nn as nn
from classes.data.satellite_image import SatelliteImage
from classes.data.labeled_satellite_image import SegmentationLabeledSatelliteImage
from utils.utils import update_storage_access, split_array, remove_dot_file
from rasterio.errors import RasterioIOError
from classes.optim.evaluation_model import (
    evaluer_modele_sur_jeu_de_test_segmentation_pleiade,
    evaluer_modele_sur_jeu_de_test_classification_pleiade
    )
from models.classification_module import ClassificationModule  


# with open("../config.yml") as f:
#     config = yaml.load(f, Loader=SafeLoader)

# def download_data(config):
#     """
#     Downloads data based on the given configuration.

#     Args:
#         config: a dictionary representing the 
#         configuration information for data download.

#     Returns:
#         A list of output directories for each downloaded dataset.
#     """

#     print("Entre dans la fonction download_data")
#     config_data = config["donnees"]
#     list_output_dir = []
#     list_masks_cloud_dir = []
    
#     years = config_data["year"]
#     deps = config_data["dep"]
#     src = config_data["source train"]

#     for year, dep in zip(years, deps):
#         # year, dep = years[0], deps[0]
#         if src == "PLEIADES":
#             cloud_dir = load_satellite_data(year, dep, "NUAGESPLEIADES")
#             list_masks_cloud_dir.append(cloud_dir)

#         output_dir = load_satellite_data(year, dep, src)
#         list_output_dir.append(output_dir)
    
#     print("chargement des données test")
#     test_dir = load_donnees_test(type=config["donnees"]["task"])

#     return list_output_dir, list_masks_cloud_dir, test_dir



# def prepare_train_data(config, list_data_dir, list_masks_cloud_dir):
#     """
#     Preprocesses and splits the raw input images 
#     into tiles and corresponding masks, 
#     and saves them in the specified output directories. 
    
#     Args:
#         config: A dictionary representing the configuration settings.
#         list_data_dir: A list of strings representing the paths
#         to the directories containing the raw input image files.
    
#     Returns:
#         A list of strings representing the paths to 
#         the output directories containing the 
#         preprocessed tile and mask image files.
#     """

#     print("Entre dans la fonction prepare_data")
#     config_data = config["donnees"]

#     years = config_data["year"]
#     deps = config_data["dep"]
#     src = config_data["source train"]
#     labeler = config_data["type labeler"]
#     config_task = config_data["task"]
    
#     cpt_tot =0
#     cpt_ones =0

#     for i, (year, dep) in enumerate(zip(years, deps)):
#         # i, year , dep = 0,years[0],deps[0]

#         date = datetime.strptime(str(year) + "0101", "%Y%m%d")

#         if labeler == "RIL":
#             buffer_size = config_data["buffer size"]
#             labeler = RILLabeler(date, dep=dep, buffer_size=buffer_size)
#         elif labeler == "BDTOPO":
#             labeler = BDTOPOLabeler(date, dep=dep)

#         list_name_cloud = []
#         if src == "PLEIADES":
#             cloud_dir = list_masks_cloud_dir[i]
#             list_name_cloud = [path.split("/")[-1].split(".")[0] for path in os.listdir(cloud_dir)]
        
#         dir = list_data_dir[i]
#         list_path = [dir + "/" + filename for filename in os.listdir(dir)]

#         for path in tqdm(list_path):
#             # path = list_path[0]
#             # path  = dir + "/"+ "ORT_2022_0691_1641_U20N_8Bits.jp2"
#             try:
#                 si = SatelliteImage.from_raster(
#                     file_path=path,
#                     dep=dep,
#                     date=date,
#                     n_bands=config_data["n bands"],
#                 )
                
#             except RasterioIOError:
#                 print("Erreur de lecture du fichier " + path)
#                 continue

#             else:
#                 list_si_filtered, __ = label_images([si], labeler)

#                 if list_si_filtered:
#                     cpt_tot += 1
#                     cpt_ones += 1
#                 else:
#                     cpt_tot += 1
#     return cpt_ones, cpt_tot


#     # Open the file and load the file
# with open("../config.yml") as f:
#     config = yaml.load(f, Loader=SafeLoader)

# list_data_dir, list_masks_cloud_dir, test_dir = download_data(config)

# cpt_ones, cpt_tot = prepare_train_data(config, list_data_dir, list_masks_cloud_dir)
# print(cpt_ones, cpt_tot)

def instantiate_dataloader(list_output_dir= ["train_data-classification-PLEIADES-RIL-972-2022"]):
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

    config_task = "classification" 
    list_labels = []
    list_images = []
        
    for dir in list_output_dir:
        labels = os.listdir(dir + "/labels") 
        if labels[0][0] == ".":
            del(labels[0])

        if config_task == "segmentation":
            list_labels = np.concatenate((
                list_labels,
                np.sort([dir + "/labels/" + name for name in labels])
            ))
                                
        if config_task == "classification":
            list_labels_dir = []
            with open(dir + "/labels/" + labels[0], 'r') as csvfile:
                reader = csv.reader(csvfile)
            
                # Ignorer l'en-tête du fichier CSV s'il y en a un
                next(reader)
                
                # Parcourir les lignes du fichier CSV et extraire la deuxième colonne
                for row in reader:
                    image_path = row[0]
                    mask = row[1]  # Index 1 correspond à la deuxième colonne (index 0 pour la première)
                    list_labels_dir.append([image_path, mask])

            list_labels_dir = sorted(list_labels_dir, key=lambda x: x[0])
            list_labels_dir = np.array([sous_liste[1] for sous_liste in list_labels_dir])

            list_labels = np.concatenate((
                    list_labels,
                    list_labels_dir
                ))
            print(list_labels)

        # Même opération peu importe la tâche 
        images = os.listdir(dir + "/images")

        list_images = np.concatenate((
            list_images,
            np.sort([dir + "/images/" + name for name in images])
        ))   
    return list_images, list_labels

#list_images_echant = list_images[100:200]
#list_labels_echant = list_labels[100:200]

def plot_list_path_images_labels(list_filepaths_images, list_filepaths_labels, tile_size = 50):
    
    size = int(math.sqrt(len(list_filepaths_images)))
    bands_indices = [0,1,2]

    list_images = []
    list_labels = []

    for filepath in list_filepaths_images:
        image = SatelliteImage.from_raster(
                filepath,
                date = None, 
                n_bands = len(bands_indices),
                dep = None
            )
        image.normalize()
        list_images.append(image)

    for label in list_filepaths_labels:
        if label == "0":
            mask = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)

        elif label == "1":
            mask = np.full((tile_size, tile_size, 3), 0, dtype=np.uint8)
        
        list_labels.append(mask)

    #mat_list_labels = np.transpose(np.array(list_labels).reshape(size,size))
   
    # Create a figure and axes
    fig, axs = plt.subplots(nrows=size, ncols=2*size, figsize=(20,10))

    # Iterate over the grid of masks and plot them
    for i in range(size):
        for j in range(size):
            axs[i, j].imshow(
                list_images[i*size + j].array.transpose(1,2,0)
            )

    for i in range(size):
        for j in range(size):
            axs[i, j+size].imshow(
                list_labels[i*size + j], cmap = "gray"
            )

    # Remove any unused axes
    for i in range(size):
        for j in range(2*size):
            axs[i, j].set_axis_off()

    # Show the plot
    plt.show()
    plt.gcf()
    plt.savefig("test.png")


import random
image = SatelliteImage.from_raster("../donnees-test/classification/images/mayotte-ORT_2017_0522_8592_U38S_8Bits.jp2", None)
list_images1 = image.split(125)

random.shuffle(list_images1)

list_bounding_box = [[im.bounds[3], im.bounds[0]] for im in list_images1]

# Utiliser zip pour combiner les trois listes
combined = zip(list_bounding_box, list_images1)

# Trier les éléments combinés en fonction de la troisième liste
sorted_combined = sorted(combined, key=lambda x: (-x[0][0], x[0][1]))

# Diviser les listes triées en fonction de l'ordre des éléments
__, list_images = zip(*sorted_combined)

size = int(math.sqrt(len(list_images)))

# Create a figure and axes
fig, axs = plt.subplots(nrows=size, ncols=size, figsize=(10, 10))

# Iterate over the grid of masks and plot them
for i in range(size):
    for j in range(size):
        axs[i, j].imshow(
            np.transpose(list_images[i * size + j].array, (1, 2, 0))[:, :, [0,1,2]]
        )
        

# Remove any unused axes
for i in range(size):
    for j in range(size):
        axs[i, j].set_axis_off()

# Show the plot
fig1 = plt.gcf()

plot_file = "img2/" + "g" + ".png"
fig1.savefig(plot_file)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Prédictions de probabilité du modèle (exemple)
y_pred_prob = model.predict_proba(X_eval)[:, 1]  # Remplacez X_eval par vos données d'évaluation

# Étiquettes de classe réelles (exemple)
y_true = y_eval  # Remplacez y_eval par vos étiquettes de classe réelles

# Calculez les taux de faux positifs et les taux de vrais positifs
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

# Calculez l'AUC-ROC
auc = roc_auc_score(y_true, y_pred_prob)

# Tracez la courbe ROC
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')  # Ligne diagonale pour le classificateur aléatoire
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
fig1 = plt.gcf()

plot_file = "img2/" + "ROC" + ".png"
fig1.savefig(plot_file)

def evaluer_modele_sur_jeu_de_test_classification_pleiade(
    test_dl, model, tile_size, batch_size, n_bands=3, use_mlflow=False
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
    npatch = int((2000 / tile_size) ** 2)
    count_patch = 0

    list_labeled_satellite_image = []

    for idx, batch in enumerate(test_dl):

        images, __, dic = batch

        model = model.to("cpu")
        images = images.to("cpu")

        output_model = model(images)
        mask_pred = np.array(torch.argmax(output_model, axis=1).to("cpu"))

        if batch_size > len(images):
            batch_size_current = len(images)

        elif batch_size <= len(images):
            batch_size_current = batch_size

        for i in range(batch_size_current):
            pthimg = dic["pathimage"][i]
            si = SatelliteImage.from_raster(
                file_path=pthimg, dep=None, date=None, n_bands=n_bands
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
            count_patch += 1

            if ((count_patch) % npatch) == 0:
                print("ecriture image")
                if not os.path.exists("img/"):
                    os.makedirs("img/")

                fig1 = plot_list_segmentation_labeled_satellite_image(
                    list_labeled_satellite_image, [0, 1, 2]
                )

                filename = pthimg.split("/")[-1]
                filename = filename.split(".")[0]
                filename = "_".join(filename.split("_")[0:6])
                # plot_file = "img/" + filename + ".png"
                plot_file = filename + ".png"

                fig1.savefig(plot_file)
                list_labeled_satellite_image = []

                if use_mlflow:
                    mlflow.log_artifact(plot_file, artifact_path="plots")

        del images, __, dic

import os
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import geopandas as gpd
import hvac
import pyarrow.parquet as pq
import rasterio
import yaml
from affine import Affine
from s3fs import S3FileSystem

from .mappings import dep_to_crs

import os

import numpy as np
import s3fs
from osgeo import gdal

from classes.data.satellite_image import SatelliteImage
from utils.utils import get_environment, get_root_path, update_storage_access
import glob
import boto3
from tqdm import tqdm


def load_only_tif(year: int, dep: str, src: str):
    """
    Load satellite data for a given year and territory \
        and a given source of satellite images.

    This function downloads satellite data from an S3 bucket, \
    updates storage access, and saves the data locally. \
    The downloaded data is specific to the given year and territory.

    Args:
        year (int): Year of the satellite data.
        territory (str): Territory for which the satellite \
        data is being loaded.
        source (str): Source of the satellite images.

    Returns:
        str: The local path where the data is downloaded.
    """
    print("Entre dans la fonction load_satellite_data")

    update_storage_access()
    root_path = get_root_path()
    environment = get_environment()

    bucket = environment["bucket"]
    path_s3 = environment["sources"][src][year][dep]
    path_local = os.path.join(root_path, environment["local-path"][src][year][dep])
    
    os.makedirs(path_local)

    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
    )

    list_sous_doss = fs.ls(bucket+ "/" + path_s3)

    for sous_doss in tqdm(list_sous_doss):
        liste_fichiers = fs.ls(sous_doss)
        
        for fichier in liste_fichiers:
            # Vérification si l'élément se termine par ".tif"
            if fichier.endswith(".tif"):
                fs.download(
                    rpath=f"{fichier}",
                    lpath=f"{path_local}",
                    recursive=True,
                )

    return path_local

load_only_tif(2020, "971", "PLEIADES")

import imageio.v2 as imageio
import numpy as np

from PIL import Image

def tif_to_jp2(path_tif, year):
    path_jp2 = path_tif + "/../" + "JP2"

    os.makedirs(path_jp2)
    date = date.fromisoformat(str(year) + '-01-01')

    for tif in tqdm(os.listdir(path_tif)):
        chemin_fichier_tiff = path_tif + "/" + tif
        name_fichier_jp2 = tif.split(".")[0] + ".jp2"
        try:
            img = SatelliteImage.from_raster(
                chemin_fichier_tiff,
                "971",
                date,
                3
            )
        except:
            print("Writing error", chemin_fichier_tiff)
            continue

        img.to_raster(
                path_jp2,
                name_fichier_jp2,
                "jp2",
                "GTiff"
            )
        
        os.remove(chemin_fichier_tiff)


from classes.data.satellite_image import SatelliteImage


from datetime import date

date = date.fromisoformat('2020-01-01')

filename ="../data/PLEIADES/2020/GUADELOUPE/ORT_2020012753796177_0638_1767_U20N_8Bits.tif"

img = SatelliteImage.from_raster(
    filename,
        "971",
        date,
        3)
fig1 = img.plot([0,1,2])
fig1.savefig("imagetest.png")


img.to_raster(
        "../data/JP",
        "ORT_2020012753796177_0638_1767_U20N_8Bits.jp2",
        "jp2",
        "GTiff"
    )

    
filename2 = "../data/PLEIADES/2020/JP2/ORT_2020012753796177_0633_1770_U20N_8Bits.jp2"

img2 = SatelliteImage.from_raster(
    filename2,
        "971",
        date,
        3)
fig2 = img2.plot([0,1,2])
fig2.savefig("imagetest2.png")

def load_bdtopo(
    millesime: Literal[
        "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"
    ],
    dep: Literal["971", "972", "973", "974", "976", "977", "978"],
) -> gpd.GeoDataFrame:
    """
    Load BDTOPO for a given datetime.

    Args:
        millesime (Literal): Year.
        dep (Literal): Departement.

    Returns:
        gpd.GeoDataFrame: BDTOPO GeoDataFrame.
    """
    root_path = get_root_path()
    environment = get_environment()

    if int(millesime) >= 2019:
        couche = "BATIMENT.shp"
    elif int(millesime) < 2019:
        couche = "BATI_INDIFFERENCIE.SHP"

    bucket = environment["bucket"]
    path_s3 = environment["sources"]["BDTOPO"][int(millesime)][dep]
    dir_path = os.path.join(
        root_path,
        environment["local-path"]["BDTOPO"][int(millesime)][dep],
    )

    if os.path.exists(dir_path):
        print(
            "Le téléchargement de cette version de la \
            BDTOPO a déjà été effectué"
        )

    else:
        os.makedirs(dir_path)

        update_storage_access()
        fs = S3FileSystem(
            client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
        )
        print("download " + dep + " " + str(millesime) + " in " + dir_path)
        if int(millesime) >= 2019:
            extensions = ["cpg", "dbf", "prj", "shp", "shx"]
        elif int(millesime) < 2019:
            extensions = ['CPG', 'DBF', 'PRJ', 'SHP', 'SHX']

        couche_split = couche.split(".")[0]
        for ext in extensions:
            fs.download(
                rpath=f"{bucket}/{path_s3}/{couche_split}.{ext}",
                lpath=f"{dir_path}",
                recursive=True,
            )

    file_path = None

    for root, dirs, files in os.walk(dir_path):
        if couche in files:
            file_path = os.path.join(root, couche)
    if not file_path:
        raise ValueError(f"No valid {couche} file found.")

    df = gpd.read_file(file_path)

    return df


import os

import numpy as np
import rasterio
from tqdm import tqdm
from rasterio.errors import RasterioIOError

import sys
sys.path.append('../src')
from classes.data.satellite_image import SatelliteImage
from classes.labelers.labeler import Labeler
from utils.filter import (
    has_cloud, is_too_black2, mask_full_cloud, patch_nocloud
)


from train_pipeline_utils.download_data import load_satellite_data
from classes.data.satellite_image import SatelliteImage

load_satellite_data(2020, "971", "PLEIADES")

def create_doss_cloud(year, dep):
    file_path = '../data/PLEIADES/' + year + "/" + dep
    output_masks_path = '../data/nuagespleiades/' + year + "/" + dep

    if os.path.exists(output_masks_path):
        print("fichiers déjà écrits")
    
    if not os.path.exists(output_masks_path):
        os.makedirs(output_masks_path)
    
    
    list_name = os.listdir(file_path)
    list_path = [file_path + "/" + name for name in list_name]
    
    
    for path, file_name in tqdm(zip(list_path, list_name), total=len(list_path), desc='Processing'):
        try:
            big_satellite_image = SatelliteImage.from_raster(
                file_path=path, dep=None, date=None, n_bands=3
            )
            
        except RasterioIOError:
            continue
    
        else:
            boolean = has_cloud(big_satellite_image)
    
            if boolean:
                mask_full = mask_full_cloud(big_satellite_image)
                file_name = file_name.split(".")[0]
                np.save(output_masks_path + "/" + file_name + ".npy", mask_full)

    return(output_masks_path)

output_directory_name = create_doss_cloud("2020", "GUADELOUPE")




import zipfile
import os

chemin_dossier_zip = "../s2looking/S2Looking.zip"
dossier_destination = "../s2looking_unzip"

# Vérifier si le dossier de destination existe, sinon le créer
if not os.path.exists(dossier_destination):
    os.makedirs(dossier_destination)

# Ouvrir le dossier ZIP
with zipfile.ZipFile(chemin_dossier_zip, 'r') as zip_ref:
    # Extraire tout le contenu du dossier ZIP vers le dossier de destination
    zip_ref.extractall(dossier_destination)

print("Dossier ZIP extrait avec succès.")
