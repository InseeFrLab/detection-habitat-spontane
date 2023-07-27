import os

import mlflow
import numpy as np
import torch
import matplotlib
from skimage.morphology import closing, disk
import random
import shutil

from classes.data.labeled_satellite_image import SegmentationLabeledSatelliteImage
from classes.data.satellite_image import SatelliteImage
from utils.plot_utils import (
    plot_list_labeled_sat_images,
    plot_list_segmentation_labeled_satellite_image,
    plot_image_mask_label,
    plot_2image_mask_difference,
)

# with open("../config.yml") as f:
#     config = yaml.load(f, Loader=SafeLoader)

# list_data_dir = download_data(config)
# list_output_dir = prepare_data(config, list_data_dir)
# #download_prepare_test(config)
# model = instantiate_model(config)
# train_dl, valid_dl, test_dl = instantiate_dataloader(
#     config, list_output_dir
# )


def evaluer_modele_sur_jeu_de_test_segmentation_pleiade(
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
            print("ecriture image")
            if not os.path.exists("img/"):
                os.makedirs("img/")

            fig1 = plot_list_labeled_sat_images(
                list_labeled_satellite_image, [0, 1, 2]
            )

            filename = pthimg.split("/")[-1]
            filename = filename.split(".")[0]
            filename = "_".join(filename.split("_")[0:6])
            plot_file = "img/" + filename + ".png"

            fig1.savefig(plot_file)
            list_labeled_satellite_image = []

            if use_mlflow:
                mlflow.log_artifact(plot_file, artifact_path="plots")

        del images, label, dic


def evaluer_modele_sur_jeu_de_test_segmentation_sentinel(
    test_dl,
    model,
    tile_size,
    batch_size,
    n_bands,
    use_mlflow=False,
):
    model.eval()
    vrais_positifs_total, faux_positifs_total, faux_negatifs_total = 0, 0, 0
    vrais_positifs_total_buffer, faux_positifs_total_buffer, faux_negatifs_total_buffer = 0, 0, 0

    for idx, batch in enumerate(test_dl):
        # idx, batch = 0, next(iter(test_dl))
        images, label, dic = batch

        if torch.cuda.is_available():
            model = model.to("cuda:0")
            images = images.to("cuda:0")

        output_model = model(images)
        mask_pred = np.array(torch.argmax(output_model, axis=1).to("cpu"))

        for i in range(batch_size):
            if len(dic["pathimage"]) != batch_size:
                continue
            try:
                vrais_positifs_image, faux_positifs_image, faux_negatifs_image = 0, 0, 0
                vrais_positifs_buffer, faux_positifs_buffer, faux_negatifs_buffer = 0, 0, 0
                pthimg = dic["pathimage"][i]
                label = np.load(dic["pathlabel"][i])
                masque = mask_pred[i]
                label_buffered = closing(label.copy(),disk(2))
                src = pthimg.split('/')[1].split('segmentation-')[1].split('-BDTOPO')[0]
                si = SatelliteImage.from_raster(
                    file_path=pthimg, dep=None, date=None, n_bands=n_bands
                )
                si.normalize()

                print("ecriture image")
                if not os.path.exists("outputs_evaluation_model/"):
                    os.makedirs("outputs_evaluation_model/")

                if src == 'SENTINEL1-2' or src == 'SENTINEL2':
                    bands_idx = 3, 2, 1
                elif src == 'SENTINEL2-RVB' or src == 'SENTINEL1-2-RVB' or src == 'PLEIADES':
                    bands_idx = 0, 1, 2

                filename = pthimg.split("/")[-1]
                filename = filename.split(".")[0]
                filename = "_".join(filename.split("_")[0:6])

                fig1 = plot_image_mask_label(
                    si,
                    label,
                    masque,
                    bands_idx,
                    label_buffered=label_buffered)
                plot_file = "outputs_evaluation_model/" + filename + ".png"
                fig1.savefig(plot_file)
                matplotlib.pyplot.close()

                if use_mlflow:
                    mlflow.log_artifact(plot_file, artifact_path="plots")

                for m in range(tile_size):
                    for n in range(tile_size):
                        if label[m, n] == 1 and masque[m, n] == 1:
                            vrais_positifs_image += 1
                        elif label[m, n] == 0 and masque[m, n] == 1:
                            faux_positifs_image += 1
                        elif label[m, n] == 1 and masque[m, n] == 0:
                            faux_negatifs_image += 1

                        if label_buffered[m, n] == 1 and masque[m, n] == 1:
                            vrais_positifs_buffer += 1
                        elif label_buffered[m, n] == 0 and masque[m, n] == 1:
                            faux_positifs_buffer += 1
                        elif label_buffered[m, n] == 1 and masque[m, n] == 0:
                            faux_negatifs_buffer += 1
                vrais_positifs_total += vrais_positifs_image*100/tile_size**2
                faux_positifs_total += faux_positifs_image*100/tile_size**2
                faux_negatifs_total += faux_negatifs_image*100/tile_size**2

                vrais_positifs_total_buffer += vrais_positifs_buffer*100/tile_size**2
                faux_positifs_total_buffer += faux_positifs_buffer*100/tile_size**2
                faux_negatifs_total_buffer += faux_negatifs_buffer*100/tile_size**2

            except IndexError:
                print('IndexError')
                pass

    rappel = vrais_positifs_total/(vrais_positifs_total+faux_negatifs_total)
    precision = vrais_positifs_total/(vrais_positifs_total+faux_positifs_total)
    fscore = (2*rappel*precision)/(rappel+precision)

    rappel_buffer = vrais_positifs_total_buffer/(vrais_positifs_total_buffer+faux_negatifs_total_buffer)
    precision_buffer = vrais_positifs_total_buffer/(vrais_positifs_total_buffer+faux_positifs_total_buffer)
    fscore_buffer = (2*rappel_buffer*precision_buffer)/(rappel_buffer+precision_buffer)

    if use_mlflow:
        mlflow.log_metric("Rappel sans buffer", rappel)
        mlflow.log_metric("Précision sans buffer", precision)
        mlflow.log_metric("F-Score sans buffer", fscore)
        mlflow.log_metric("Rappel avec buffer", rappel_buffer)
        mlflow.log_metric("Précision avec buffer", precision_buffer)
        mlflow.log_metric("F-Score avec buffer", fscore_buffer)


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
    nbatchforfullimage = int(npatch / batch_size)

    if not npatch % nbatchforfullimage == 0:
        print(
            "Le nombre de patchs \
            n'est pas divisible par la taille d'un batch"
        )
        return None

    list_labeled_satellite_image = []

    for idx, batch in enumerate(test_dl):
        print(idx)
        images, label, dic = batch

        model = model.to("cuda:0")
        images = images.to("cuda:0")

        output_model = model(images)
        output_model = output_model.to("cpu")
        probability_class_1 = output_model[:, 1]

        # Set a threshold for class prediction
        threshold = 0.51

        # Make predictions based on the threshold
        predictions = torch.where(
            probability_class_1 > threshold,
            torch.tensor([1]),
            torch.tensor([0]),
        )
        predicted_classes = predictions.type(torch.float)

        for i in range(batch_size):
            pthimg = dic["pathimage"][i]
            si = SatelliteImage.from_raster(
                file_path=pthimg, dep=None, date=None, n_bands=3
            )
            si.normalize()

            if int(predicted_classes[i]) == 0:
                mask_pred = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)

            elif int(predicted_classes[i]) == 1:
                mask_pred = np.full((tile_size, tile_size, 3), 0, dtype=np.uint8)

            list_labeled_satellite_image.append(
                SegmentationLabeledSatelliteImage(
                    satellite_image=si,
                    label=mask_pred,
                    source="",
                    labeling_date="",
                )
            )

        if ((idx + 1) % nbatchforfullimage) == 0:
            print("ecriture image")
            if not os.path.exists("img/"):
                os.makedirs("img/")

            fig1 = plot_list_labeled_sat_images(
                list_labeled_satellite_image, [0, 1, 2]
            )

            filename = pthimg.split("/")[-1]
            filename = filename.split(".")[0]
            filename = "_".join(filename.split("_")[0:6])
            plot_file = "img/" + filename + ".png"

            fig1.savefig(plot_file)
            list_labeled_satellite_image = []

            if use_mlflow:
                mlflow.log_artifact(plot_file, artifact_path="plots")


def evaluer_modele_sur_jeu_de_test_change_detection_pleiade(
    test_dl,
    model,
    tile_size,
    batch_size,
    use_mlflow=False
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
    npatch = int((2000/tile_size)**2)
    nbatchforfullimage = int(npatch/batch_size)

    if not npatch % nbatchforfullimage == 0:
        print("Le nombre de patchs \
            n'est pas divisible par la taille d'un batch")
        return None

    list_labeled_satellite_image = []

    for idx, batch in enumerate(test_dl):
        # idx, batch = 0, next(iter(test_dl))
        print(idx)
        images, label, dic = batch
        device = "cpu"
        model = model.to(device)
        images = images.to(device)

        output_model = model(images)
        mask_pred = np.array(torch.argmax(output_model, axis=1).to("cpu"))

        for i in range(batch_size):
            pthimg2 = dic["pathimage2"][i]
            si2 = SatelliteImage.from_raster(
                file_path=pthimg2,
                dep=None,
                date=None,
                n_bands=3
            )
            si2.normalize()

            list_labeled_satellite_image.append(
                SegmentationLabeledSatelliteImage(
                    satellite_image=s2,
                    label=mask_pred[i],
                    source="",
                    labeling_date=""
                )
            )

        if ((idx+1) % nbatchforfullimage) == 0:
            print("ecriture image")
            if not os.path.exists("img/"):
                os.makedirs("img/")

            fig1 = plot_list_segmentation_labeled_satellite_image(
                list_labeled_satellite_image, [0, 1, 2]
                )

            filename = pthimg2.split('/')[-1]
            filename = filename.split('.')[0]
            filename = '_'.join(filename.split('_')[0:6])
            plot_file = filename + ".png"

            fig1.savefig(plot_file)
            list_labeled_satellite_image = []

            if use_mlflow:
                mlflow.log_artifact(plot_file, artifact_path="plots")

        del images, label, dic


def calculate_IOU(output, labels):
    """
    Calculate Intersection Over Union indicator
    based on output segmentation mask of a model
    and the true segmentations mask

    Args:
        output: the output of the segmentation
        label: the true segmentation mask

    """
    preds = torch.argmax(output, axis=1)

    numIOU = torch.sum((preds * labels), axis=[1, 2])  # vaut 1 si les 2 = 1
    denomIOU = torch.sum(torch.clamp(preds + labels, max=1), axis=[1, 2])

    IOU = numIOU / denomIOU
    IOU = [1 if torch.isnan(x) else x for x in IOU]
    IOU = torch.tensor(IOU, dtype=torch.float)
    IOU = torch.mean(IOU)

    return IOU


# calculate num and denomionateur IOU
def calculate_pourcentage_loss(output, labels):
    """
    Calculate the pourcentage of wrong predicted classes
    based on output classification predictions of a model
    and the true classes.

    Args:
        output: the output of the classification
        labels: the true classes

    """
    probability_class_1 = output[:, 1]

    # Set a threshold for class prediction
    threshold = 0.51

    # Make predictions based on the threshold
    predictions = torch.where(
        probability_class_1 > threshold, torch.tensor([1]), torch.tensor([0])
    )

    predicted_classes = predictions.type(torch.float)

    misclassified_percentage = (predicted_classes != labels).float().mean()

    return misclassified_percentage


def proportion_ones(labels):
    """
    Calculate the proportion of ones in the validation dataloader.

    Args:
        labels: the true classes

    """

    # Count the number of zeros
    num_zeros = int(torch.sum(labels == 0))

    # Count the number of ones
    num_ones = int(torch.sum(labels == 1))

    prop_ones = num_ones / (num_zeros + num_ones)

    # Rounded to two digits after the decimal point
    prop_ones = round(prop_ones, 2)

    return prop_ones


def evaluer_modele_sur_jeu_de_test_change_detection_algo_sentinel(
    test_dl,
    model,
    tile_size,
    batch_size,
    n_bands,
    use_mlflow=False,
):
    print("Entre dans la fonction evaluer_modele_sur_jeu_de_test_change_detection_algo_sentinel")
    model.eval()

    path_masques = 'masques_segmentation_pour_change_detection'
    if not os.path.exists(path_masques):
        os.makedirs(path_masques)
    else:
        shutil.rmtree(path_masques)
        os.makedirs(path_masques)
    
    path_masques_difference = 'masques_difference_pour_change_detection'
    if not os.path.exists(path_masques_difference):
        os.makedirs(path_masques_difference)
    else:
        shutil.rmtree(path_masques_difference)
        os.makedirs(path_masques_difference)
    
    path_plot_difference = 'outputs_evaluation_model'
    if not os.path.exists(path_plot_difference):
        os.makedirs(path_plot_difference)
    else:
        shutil.rmtree(path_plot_difference)
        os.makedirs(path_plot_difference)
    

    list_paths_images = []

    for idx, batch in enumerate(test_dl):
        # idx, batch = 0, next(iter(test_dl))
        images, label, dic = batch

        if torch.cuda.is_available():
            model = model.to("cuda:0")
            images = images.to("cuda:0")

        output_model = model(images)
        mask_pred = np.array(torch.argmax(output_model, axis=1).to("cpu"))

        for i in range(batch_size):
            if len(dic["pathimage"]) != batch_size:
                continue
            pthimg = dic["pathimage"][i]
            masque = mask_pred[i]
            np.save(path_masques + '/' + pthimg.split('/')[-1].split('.')[0], masque)
            list_paths_images.append(pthimg)
    list_names_masques = os.listdir(path_masques)
    list_paths_masques = [path_masques + '/' + name_masque for name_masque in list_names_masques]

    i = 0
    for i in range(5):
        path_img1, path_img2 = random.sample(list_paths_images, 2)
        name_img1, name_img2 = path_img1.split('/')[-1].split('.')[0], path_img2.split('/')[-1].split('.')[0]
        
        for path_masque in list_paths_masques:
            if name_img1 in path_masque:
                path_masque1 = path_masque
            if name_img2 in path_masque:
                path_masque2 = path_masque

        masque_diff = np.zeros((tile_size, tile_size))
        masque1 = np.load(path_masque1)
        masque2 = np.load(path_masque2)
        for m in range(tile_size):
            for n in range(tile_size):
                if masque1[m, n] != masque2[m, n]:
                    masque_diff[m, n] = 1
        name_diff = name_img1 + '-' + name_img2
        np.save(path_masques_difference + '/' + name_diff, masque_diff)

        src = path_img1.split('/')[1].split('segmentation-')[1].split('-BDTOPO')[0]
        if src == 'SENTINEL1-2' or src == 'SENTINEL2':
            bands_idx = 3, 2, 1
        elif src == 'SENTINEL2-RVB' or src == 'SENTINEL1-2-RVB' or src == 'PLEIADES':
            bands_idx = 0, 1, 2

        image1 = SatelliteImage.from_raster(file_path=path_img1, dep=None, date=None, n_bands=n_bands)
        image2 = SatelliteImage.from_raster(file_path=path_img2, dep=None, date=None, n_bands=n_bands)
        image1.normalize()
        image2.normalize()

        fig=plot_2image_mask_difference(image1, image2, masque_diff)#, bands_idx)
        plot_file = path_plot_difference + '/' + name_diff + ".png"
        fig.savefig(plot_file)
        matplotlib.pyplot.close()
