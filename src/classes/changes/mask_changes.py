import pandas
import os
import csv
from tqdm import tqdm

import os
import csv

import mlflow
import numpy as np
import torch
import matplotlib

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay
)

from classes.data.change_detection_triplet import ChangedetectionTripletS2Looking
from classes.data.labeled_satellite_image import SegmentationLabeledSatelliteImage
from classes.data.satellite_image import SatelliteImage
from utils.plot_utils import (
    plot_list_change_pleiades_images,
    plot_list_mask_rgb_pleiades_images,
    create_segmentation_labeled_satellite_image
)


def new_csv_labeler_pred(csv_labeler):
    df = pd.read_csv(csv_labeler)
    list_labels_dir = df[["Path_image", "Classification_Pred"]].values.tolist()
    middle = int(len(list_labels_dir)/2)
    list_labels_dir_2017 = list_labels_dir[:middle]
    list_labels_dir_2020 = list_labels_dir[middle:]
    list_changes = []
    for img_2017, img_2020 in zip(list_labels_dir_2017, list_labels_dir_2020):
        if img_2017[1] != img_2020[1]:
            list_changes.append(1)
        else:
            list_changes.append(0)
    df1 = pd.DataFrame({'Path_image_2017': [x[0] for x in list_labels_dir_2017], 'Path_image_2020': [x[0] for x in list_labels_dir_2020], 'Classification_Pred_2017': [x[1] for x in list_labels_dir_2017], 'Classification_Pred_2020': [x[1] for x in list_labels_dir_2020], 'Changes': [x for x in list_changes]})
    df1.to_csv('../fichierlabelerpredicted_new.csv', index=False)


def plot_changes_predicted_test(
    csv_labeler, tile_size, n_bands=3
):  
    npatch = int((2000 / tile_size) ** 2)
    count_patch = 0

    list_img1 = []
    list_img2 = []

    df = pd.read_csv(csv_labeler)
    list_labels_dir = df[["Path_image_2017","Path_image_2020", "Classification_Pred_2017", "Classification_Pred_2020", "Changes"]].values.tolist()

    list_img_path_2017 = [x[0] for x in list_labels_dir]
    list_img_path_2020 = [x[1] for x in list_labels_dir]
    list_changes = [x[4] for x in list_labels_dir]

    for img_2017, img_2020, changes in zip(list_img_path_2017, list_img_path_2020, list_changes):
        si = SatelliteImage.from_raster(
            file_path=img_2017, dep=None, date=None, n_bands=n_bands
        )
        si2 = SatelliteImage.from_raster(
            file_path=img_2020, dep=None, date=None, n_bands=n_bands
        )

        si.normalize()
        si2.normalize()

        if changes == 0:
            mask_pred = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)

        elif changes == 1:
            mask_pred = np.full((tile_size, tile_size, 3), 0, dtype=np.uint8)

            red_color = [1.0, 0.0, 0.0]

            array_red_borders = si.array.copy()
            array_red_borders = array_red_borders.transpose(1, 2, 0)
            array_red_borders[:, :7, :] = red_color
            array_red_borders[:, -7:-1, :] = red_color
            array_red_borders[:7, :, :] = red_color
            array_red_borders[-7:-1, :, :] = red_color
            array_red_borders = array_red_borders.transpose(2, 0, 1)
            si.array = array_red_borders
            del(array_red_borders)

            array_red_borders = si2.array.copy()
            array_red_borders = array_red_borders.transpose(1, 2, 0)
            array_red_borders[:, :7, :] = red_color
            array_red_borders[:, -7:-1, :] = red_color
            array_red_borders[:7, :, :] = red_color
            array_red_borders[-7:-1, :, :] = red_color
            array_red_borders = array_red_borders.transpose(2, 0, 1)
            si2.array = array_red_borders

        list_img1.append(si)
    
        list_img2.append(si2)
        count_patch += 1

        if ((count_patch) % npatch) == 0:
            print("ecriture image")
            if not os.path.exists("img2/"):
                os.makedirs("img2/")

            fig1 = plot_list_change_pleiades_images(
                        list_img1, list_img2
                    )
            filename = img_2017.split("/")[-1]
            filename = filename.split(".")[0]
            filename = "_".join(filename.split("_")[0:6])
            plot_file = "img2/" + filename + ".png"

            fig1.savefig(plot_file)
            list_img1 = []
            list_img2 = []

            plt.close()


def mask_rgb(image, threshold = 156):
    img = image.array.copy()
    img = img[:3,:,:]
    img = img.transpose(1,2,0)

    shape = img.shape[0:2]

    grayscale = np.mean(img, axis=2)
    
    black = np.ones(shape, dtype=float)
    white = np.zeros(shape, dtype=float)

    mask = np.where(grayscale < threshold, white, black)
    
    return(mask)

def plot_two_masks_predicted_test(
    csv_labeler, tile_size, n_bands=3
):

    
    npatch = int((2000 / tile_size) ** 2)
    count_patch = 0

    list_img = []
    list_mask_1 = []
    list_mask_2 = []

    df = pd.read_csv(csv_labeler)
    list_labels_dir = df[["Path_image", "Classification_Pred"]].values.tolist()

    list_img_path = [x[0] for x in list_labels_dir]
    list_pred = [x[1] for x in list_labels_dir]

    for img, pred in zip(list_img_path, list_pred):
        si = SatelliteImage.from_raster(
            file_path=img, dep=None, date=None, n_bands=n_bands
        )

        year = img.split("/")[-1].split("-")[-1].split("_")[1][:4]

        if year == "2017":
            threshold = 95
        
        elif year == "2020":
            threshold = 100

        if pred == 0:
            mask_pred_1 = mask_rgb(si, threshold)
            mask_pred_2 = np.full((tile_size, tile_size, 3), 0, dtype=np.uint8)

        elif pred == 1:
            mask_pred_1 = mask_rgb(si, threshold)
            mask_pred_2 = mask_pred_1

        si.normalize()
        list_img.append(si)
        list_mask_1.append(mask_pred_1)
        list_mask_2.append(mask_pred_2)

        count_patch += 1

        if ((count_patch) % npatch) == 0:
            print("ecriture image")
            if not os.path.exists("img2/"):
                os.makedirs("img2/")

            fig1 = plot_list_mask_rgb_pleiades_images(
                        list_img, list_mask_1, list_mask_2
                    )
            filename = img.split("/")[-1]
            filename = filename.split(".")[0]
            filename = "_".join(filename.split("_")[0:6])
            plot_file = "img2/" + filename + ".png"

            fig1.savefig(plot_file)
            list_img = []
            list_mask_1 = []
            list_mask_2 = []

            plt.close()

from train_pipeline_utils.download_data import load_satellite_data
from classes.data.satellite_image import SatelliteImage

# load_satellite_data(2022, "972", "PLEIADES")

def create_doss_mask_inv(year, dep, threshold):
    file_path = '../data/PLEIADES/' + year + "/" + dep
    output_masks_path = '../data/mask_inv/' + year + "/" + dep

    if os.path.exists(output_masks_path):
        print("fichiers déjà écrits")
    
    if not os.path.exists(output_masks_path):
        os.makedirs(output_masks_path)
    
    
    list_name = os.listdir(file_path)
    list_path = [file_path + "/" + name for name in list_name]
    
    
    for path, file_name in tqdm(zip(list_path, list_name), total=len(list_path), desc='Processing'):
        try:
            si = SatelliteImage.from_raster(
                file_path=path, dep=None, date=None, n_bands=3
            )
            
        except RasterioIOError:
            continue
    
        else:
            mask_inv = mask_rgb(si, threshold)
            file_name = file_name.split(".")[0]
            np.save(output_masks_path + "/" + file_name + ".npy", mask_inv)

    return(output_masks_path)

# output_directory_name = create_doss_mask_inv("2022", "MARTINIQUE", 110)

import gc
import json
import os
import random
import sys
from datetime import datetime
import csv
import shutil
from osgeo import gdal

import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from rasterio.errors import RasterioIOError
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from yaml.loader import SafeLoader

from classes.data.labeled_satellite_image import (  # noqa: E501
    SegmentationLabeledSatelliteImage,
)
from classes.data.satellite_image import SatelliteImage

from classes.optim.optimizer import generate_optimization_elements
from dico_config import (
    labeler_dict,
    dataset_dict,
    loss_dict,
    module_dict,
    task_to_evaluation,
    task_to_lightningmodule,
)
from train_pipeline_utils.download_data import (
    load_2satellites_data,
    load_donnees_test,
    load_satellite_data,
    load_s2_looking
)
from train_pipeline_utils.handle_dataset import (
    generate_transform_pleiades,
    generate_transform_sentinel,
    select_indices_to_balance,
    select_indices_to_split_dataset,
)
from train_pipeline_utils.prepare_data import (
    check_labelled_images,
    filter_images,
    label_images,
    save_images_and_masks,
    extract_proportional_subset,
    filter_images_by_path,
    prepare_data_per_doss,
)

from utils.utils import remove_dot_file, split_array, update_storage_access, list_sorted_filenames


def masques_segmentation_pleiade(
    test_dl, model, tile_size, batch_size, n_bands=3, use_mlflow=False
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
    # tile_size = 250
    # batch_size  = 4
    model.eval()
    npatch = int((2000 / tile_size) ** 2)
    nbatchforfullimage = int(npatch / batch_size)

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

        model = model.to("cuda:0")
        images = images.to("cuda:0")

        output_model = model(images)
        mask_pred = np.array(torch.argmax(output_model, axis=1).to("cpu"))

        for i in range(batch_size):
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

        if ((idx + 1) % nbatchforfullimage) == 0:
            print("ecriture masque")
            if not os.path.exists("mask/"):
                os.makedirs("mask/")
            
            output_mask = create_segmentation_labeled_satellite_image(
                            list_labeled_satellite_image, [0, 1, 2]
                        )

            filename = pthimg.split("/")[-1]
            filename = filename.split(".")[0]
            filename = "_".join(filename.split("_")[0:6])
            mask_file = filename + ".npy"
            
            np.save(
                    "mask/" + mask_file + ".npy",
                    output_mask,
                    )

            list_labeled_satellite_image = []

        del images, label, dic
    
from classes.data.satellite_image import SatelliteImage
from utils.utils import *
from utils.plot_utils import *
from utils.image_utils import *
import utils.mappings as mapps
from train_pipeline_utils.download_data import load_satellite_data
from classes.data.satellite_image import SatelliteImage
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
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.features import rasterize, shapes
import math
from scipy.ndimage import label
from scipy.ndimage import uniform_filter

def lissage_mask(mask, neighborhood_size = 4, threshold = 0.5):
    mask = mask/255
    # Taille du voisinage pour le lissage
    
    # Érosion suivie de dilatation (lissage)
    eroded_array = uniform_filter(mask, size=neighborhood_size, mode='constant', origin=0)
    smoothed_array = uniform_filter(eroded_array, size=neighborhood_size, mode='constant', origin=0)
    
    # Seuillage pour obtenir un masque binaire
    binary_mask = smoothed_array >= threshold
    binary_array = (binary_mask*255).astype(np.uint8)

    return(binary_array)

def mask_diff(mask1, mask2):
    array1 = (mask1).astype(np.uint8)
    array2 = (mask2).astype(np.uint8)
    
    # Perform XOR operation on the arrays
    result = np.bitwise_xor(array1, array2)

    return(result)

def mask_diff_tt1(mask_t, mask_t1):

    mask_liss1 = lissage_mask(mask_t, neighborhood_size = 4, threshold = 0.5)
    mask_liss2 = lissage_mask(mask_t1, neighborhood_size = 4, threshold = 0.5)

    diff_mask = mask_diff(mask_liss1, mask_liss2)/255

    nchannel, height, width = (3, 2000, 2000)

    # Create a list of polygons from the masked center clouds in order
    # to obtain a GeoDataFrame from it
    polygon_list_center = []
    for shape in list(shapes(diff_mask)):
        polygon = Polygon(shape[0]["coordinates"][0])
        surface = polygon.area
        perimetre = polygon.length
        RP = perimetre/(2*np.pi)
        RS = math.sqrt(surface/np.pi)
        # if RP/RS > 2:
        #     continue
        if polygon.area > 0.85 * height * width:
            continue
        if polygon.area < 0.0007 * height * width:
            continue
        polygon_list_center.append(polygon)

    result = gpd.GeoDataFrame(geometry=polygon_list_center)
    
    
    #Rasterize the geometries into a numpy array
    if result.empty:
        rasterized = np.zeros((2000,2000))
    else:
        rasterized = rasterize(
            result.geometry,
            out_shape=(2000,2000),
            fill=0,
            out=None,
            all_touched=True,
            default_value=1,
            dtype=None,
        )

    return(rasterized)

mask_t = np.load("mask/mayotte-ORT_2020052526670967_0523_8591_U38S_8Bits.npy.npy")
mask_t1 = np.load("mask/mayotte-ORT_2017_0523_8591_U38S_8Bits.npy.npy")

rasterized = mask_diff_tt1(mask_t, mask_t1)

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 20))
# ax1.imshow(np.transpose(image_6.array, (1, 2, 0))[:, :, [0,1,2]])
# ax1.set_title("image 2017")
# ax1.axis('off')
# ax2.imshow(np.transpose(image_5.array, (1, 2, 0))[:, :, [0,1,2]])
# ax2.set_title("image 2020")
# ax2.axis('off')
# ax3.imshow(rasterized, cmap = "gray")
# ax3.set_title("masque de différence filtré")
# ax3.axis('off')
mask_liss1 = lissage_mask(mask_t, neighborhood_size = 4, threshold = 0.5)
mask_liss2 = lissage_mask(mask_t1, neighborhood_size = 4, threshold = 0.5)

diff_mask = mask_diff(mask_liss1, mask_liss2)
plt.imshow(mask_t, cmap = "gray")
fig = plt.gcf()
fig.savefig("img/essai2.png")
plt.close()
