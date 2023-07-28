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
    plot_list_mask_inversion_pleiades_images
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


def mask_inversion(image, threshold = 100):
    img = image.array.copy()
    img = img[:3,:,:]
    img = (img * 255).astype(np.uint8)
    img = img.transpose(1,2,0)

    shape = img.shape[0:2]

    grayscale = np.mean(img, axis=2)
    
    black = np.ones(shape, dtype=float)
    white = np.zeros(shape, dtype=float)

    mask = np.where(grayscale > threshold, white, black)
    
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
            mask_pred_1 = mask_inversion(si, threshold)
            mask_pred_2 = np.full((tile_size, tile_size, 3), 0, dtype=np.uint8)

        elif pred == 1:
            mask_pred_1 = mask_inversion(si, threshold)
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

            fig1 = plot_list_mask_inversion_pleiades_images(
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
            mask_inv = mask_inversion(si, threshold)
            file_name = file_name.split(".")[0]
            np.save(output_masks_path + "/" + file_name + ".npy", mask_inv)

    return(output_masks_path)

# output_directory_name = create_doss_mask_inv("2022", "MARTINIQUE", 110)
