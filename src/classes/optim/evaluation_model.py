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
    plot_list_labeled_sat_images,
    plot_list_segmentation_labeled_satellite_image,
    plot_list_change_detection_images,
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

    mean_IOU = test_iou_change_detection(
                        test_dl, model, tile_size, batch_size, n_bands=3, use_mlflow=False
                    )
    print(mean_IOU)
    if use_mlflow:
        mlflow.log_metric("test mean IOU", mean_IOU)

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

            fig1 = plot_list_segmentation_labeled_satellite_image(
                list_labeled_satellite_image, [0, 1, 2]
            )

            filename = pthimg.split("/")[-1]
            filename = filename.split(".")[0]
            filename = "_".join(filename.split("_")[0:6])
            plot_file = filename + ".png"

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
    for idx, batch in enumerate(test_dl):
        # idx, batch = 0, next(iter(test_dl))
        images, label, dic = batch

        if torch.cuda.is_available():
            model = model.to("cuda:0")
            images = images.to("cuda:0")

        output_model = model(images)
        mask_pred = np.array(torch.argmax(output_model, axis=1).to("cpu"))

        for i in range(batch_size):
            try:
                pthimg = dic["pathimage"][i]
                src = pthimg.split('/')[1].split('segmentation-')[1].split('-BDTOPO')[0]
                si = SatelliteImage.from_raster(
                    file_path=pthimg, dep=None, date=None, n_bands=n_bands
                )
                si.normalize()

                labeled_satellite_image = SegmentationLabeledSatelliteImage(
                    satellite_image=si,
                    label=mask_pred[i],
                    source="",
                    labeling_date="",
                )

                print("ecriture image")
                if not os.path.exists("img/"):
                    os.makedirs("img/")
                if src == 'SENTINEL1-2' or src == 'SENTINEL2':
                    fig1 = plot_list_segmentation_labeled_satellite_image(
                        [labeled_satellite_image], [3, 2, 1]
                    )
                elif src == 'SENTINEL2-RVB' or src == 'SENTINEL1-2-RVB':
                    fig1 = plot_list_segmentation_labeled_satellite_image(
                        [labeled_satellite_image], [0, 1, 2]
                    )

                filename = pthimg.split("/")[-1]
                filename = filename.split(".")[0]
                filename = "_".join(filename.split("_")[0:6])
                plot_file = "img/" + filename + ".png"

                fig1.savefig(plot_file)
                matplotlib.pyplot.close()

                if use_mlflow:
                    mlflow.log_artifact(plot_file, artifact_path="plots")
            except IndexError:
                pass

def reality_prediction_classification_pleiades(
    test_dl, model, tile_size, batch_size, n_bands=3, use_mlflow=False
):
    with torch.no_grad():
        model.eval()
        y_true = []
        y_prob = []

        for idx, batch in enumerate(test_dl):

            images, labels, __ = batch

            model = model.to("cuda:0")
            images = images.to("cuda:0")

            output_model = model(images)
            output_model = output_model.to("cpu")
            y_prob_idx = output_model[:, 1].tolist()
            y_prob.append(y_prob_idx)

            y_true.append(labels.tolist())

            del images, labels

        y_true = np.array(y_true).flatten().tolist()
        y_prob = np.array(y_prob).flatten().tolist()

        return y_true, y_prob


def variation_threshold_classification_pleiades(
    test_dl, model, tile_size, batch_size, n_bands=3, use_mlflow=False
):
    y_true, y_prob = reality_prediction_classification_pleiades(
        test_dl, model, tile_size, batch_size, n_bands, use_mlflow
        )

    thresholds = np.linspace(0, 1, num=100)
    accuracies = []
    fnr_list = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        accuracies.append(accuracy)

        confusion = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = confusion.ravel()
        fnr = fn / (fn + tp)
        fnr_list.append(fnr)

    if not os.path.exists("img/"):
        os.makedirs("img/")

    plt.plot(thresholds, accuracies)
    plt.plot(thresholds, accuracies)
    plt.xlabel('Seuil de classification')
    plt.ylabel('Précision (Accuracy)')
    plt.title('Précision en fonction du seuil de classification')
    plt.show()
    plot_file_AccuracyonThreshold = "img/" + "AccuracyonThreshold.png"
    plt.savefig(plot_file_AccuracyonThreshold)
    plt.close()

    plt.plot(thresholds, fnr_list)
    plt.xlabel('Seuil de classification')
    plt.ylabel('Taux de faux négatifs (FNR)')
    plt.title('Taux de faux négatifs en fonction du seuil de classification')
    plt.show()
    plot_file_FalseNegativeRateonThreshold = "img/" + "FalseNegativeRateonThreshold.png"
    plt.savefig(plot_file_FalseNegativeRateonThreshold)
    plt.close()

    if use_mlflow:
        mlflow.log_artifact(plot_file_AccuracyonThreshold, artifact_path="plots")
        mlflow.log_artifact(plot_file_FalseNegativeRateonThreshold, artifact_path="plots")


def ROC_confusion_matrix_classification_pleiades(
    test_dl, model, tile_size, batch_size, n_bands=3, use_mlflow=False
):

    y_true, y_prob = reality_prediction_classification_pleiades(
        test_dl, model, tile_size, batch_size, n_bands, use_mlflow
        )

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    best_threshold_idx = np.argmax(tpr - fpr)  # Index of the best threshold
    best_threshold = thresholds[best_threshold_idx]
    best_tpr = tpr[best_threshold_idx]
    best_fpr = fpr[best_threshold_idx]

    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()

    if not os.path.exists("img/"):
        os.makedirs("img/")

    plot_file_roc = "img/ROC.png"
    plt.savefig(plot_file_roc)
    plt.close()

    class_names = ["Bâti", "Non bâti"]

    predictions_best = np.where(
        y_prob > best_threshold,
        np.array([1.0]),
        np.array([0.0]),
    )
    predicted_classes_best = predictions_best.tolist()
    accuracy_best = accuracy_score(y_true, predicted_classes_best)

    predictions = np.where(
        y_prob > np.array([0.5]),
        np.array([1.0]),
        np.array([0.0]),
    )
    predicted_classes = predictions.tolist()
    accuracy = accuracy_score(y_true, predicted_classes)

    disp = ConfusionMatrixDisplay.from_predictions(y_true, predicted_classes,
                                                display_labels=class_names,
                                                cmap="Pastel1",
                                                normalize="true")
    disp.plot()
    plot_file_cm = "img/" + "confusion_matrix_05.png"
    plt.savefig(plot_file_cm)
    plt.close()

    disp_best = ConfusionMatrixDisplay.from_predictions(y_true, predicted_classes_best,
                                                        display_labels=class_names,
                                                        cmap="Pastel1",
                                                        normalize="true")
    disp_best.plot()
    plot_file_cm_best = "img/" + "confusion_matrix_best.png"
    plt.savefig(plot_file_cm_best)
    plt.close()

    if use_mlflow:
        mlflow.log_artifact(plot_file_roc, artifact_path="plots")
        mlflow.log_artifact(plot_file_cm, artifact_path="plots")
        mlflow.log_artifact(plot_file_cm_best, artifact_path="plots")
        mlflow.log_metric("test best accuracy", accuracy_best)
        mlflow.log_metric("test 0.5 accuracy", accuracy)
        mlflow.log_metric("best true positif rate", best_tpr)
        mlflow.log_metric("best false positif rate", best_fpr)
        mlflow.log_metric("best threshold", best_threshold)
        mlflow.log_metric("auc", roc_auc)

    return best_threshold


def metrics_classification_pleiade(
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
    variation_threshold_classification_pleiades(
        test_dl, model, tile_size, batch_size, n_bands, use_mlflow
        )

    best_threshold = ROC_confusion_matrix_classification_pleiades(
        test_dl, model, tile_size, batch_size, n_bands, use_mlflow
        )
    return best_threshold


# def all_thresholds(
#     test_dl, model, tile_size, batch_size, n_bands=3, use_mlflow=False
# ):
#     """
#     Evaluates the model on the Pleiade test dataset for image classification.

#     Args:
#         test_dl (torch.utils.data.DataLoader): The data loader for the test
#         dataset.
#         model (torchvision.models): The classification model to evaluate.
#         tile_size (int): The size of each tile in pixels.
#         batch_size (int): The batch size.
#         use_mlflow (bool, optional): Whether to use MLflow for logging
#         artifacts. Defaults to False.

#     Returns:
#         None
#     """
#     with torch.no_grad():
#         model.eval()
#         best_threshold = []
#         npatch = int((2000 / tile_size) ** 2)
#         count_patch = 0
#         y_pred = []
#         y_true = []

#         for idx, batch in enumerate(test_dl):

#             images, labels, __ = batch

#             model = model.to("cuda:0")
#             images = images.to("cuda:0")

#             output_model = model(images)
#             output_model = output_model.to("cpu")
#             y_pred_idx = output_model[:, 1].tolist()
#             labels2 = labels.tolist()

#             if batch_size > len(images):
#                 batch_size_current = len(images)

#             elif batch_size <= len(images):
#                 batch_size_current = batch_size

#             for i in range(batch_size_current):
#                 y_pred.append(y_pred_idx[i])

#                 y_true.append(labels2[i])
#                 count_patch += 1

#                 if ((count_patch) % npatch) == 0:
#                     y_true = np.array(y_true).flatten().tolist()
#                     y_pred = np.array(y_pred).flatten().tolist()

#                     fpr, tpr, thresholds = roc_curve(y_true, y_pred)

#                     best_threshold_idx = np.argmax(tpr - fpr)  # Index of the best threshold
#                     best_threshold.append(thresholds[best_threshold_idx])
#                     y_true = []
#                     y_pred = []

#         del images, labels

#         return best_threshold


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
    threshold = metrics_classification_pleiade(
                    test_dl, model, tile_size, batch_size, n_bands, use_mlflow
                )
    print(threshold)

    model.eval()
    npatch = int((2000 / tile_size) ** 2)
    count_patch = 0

    list_labeled_satellite_image = []

    for idx, batch in enumerate(test_dl):

        images, labels, dic = batch

        model = model.to("cuda:0")
        images = images.to("cuda:0")
        labels = labels.to("cuda:0")

        output_model = model(images)
        output_model = output_model.to("cpu")
        probability_class_1 = output_model[:, 1]
        # print(probability_class_1)

        # Set a threshold for class prediction
        # threshold = 0.90

        # Make predictions based on the threshold
        
        predictions = torch.where(
            probability_class_1 > threshold,
            torch.tensor([1]),
            torch.tensor([0]),
        )
        predicted_classes = predictions.type(torch.float)

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

            if int(predicted_classes[i]) == 0:
                mask_pred = np.full((tile_size, tile_size, 3), 0, dtype=np.uint8)

                if int(predicted_classes[i]) != int(labels[i]):
                    # Contours de l'image en rouge
                    array_red_borders = si.array.copy()
                    array_red_borders = array_red_borders.transpose(1, 2, 0)
                    red_color = [1, 0, 0]
                    array_red_borders[:, :7, :] = red_color
                    array_red_borders[:, -7:-1, :] = red_color
                    array_red_borders[:7, :, :] = red_color
                    array_red_borders[-7:-1, :, :] = red_color
                    array_red_borders = array_red_borders.transpose(2,0,1)
                    si.array = array_red_borders

            elif int(predicted_classes[i]) == 1:
                mask_pred = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)

                if int(predicted_classes[i]) != int(labels[i]):
                    # Contours de l'image en rouge
                    array_red_borders = si.array.copy()
                    array_red_borders = array_red_borders.transpose(1, 2, 0)
                    # print(array_red_borders)
                    red_color = [1, 0, 0]
                    array_red_borders[:, :7, :] = red_color
                    array_red_borders[:, -7:-1, :] = red_color
                    array_red_borders[:7, :, :] = red_color
                    array_red_borders[-7:-1, :, :] = red_color
                    array_red_borders = array_red_borders.transpose(2,0,1)
                    si.array = array_red_borders

                elif int(predicted_classes[i]) == int(labels[i]):
                    # Contours de l'image en rouge
                    array_green_borders = si.array.copy()
                    array_green_borders = array_green_borders.transpose(1, 2, 0)
                    green_color = [0, 1, 0]
                    array_green_borders[:, :7, :] = green_color
                    array_green_borders[:, -7:-1, :] = green_color
                    array_green_borders[:7, :, :] = green_color
                    array_green_borders[-7:-1, :, :] = green_color
                    array_green_borders = array_green_borders.transpose(2,0,1)
                    si.array = array_green_borders

            list_labeled_satellite_image.append(
                SegmentationLabeledSatelliteImage(
                    satellite_image=si,
                    label=mask_pred,
                    source="",
                    labeling_date="",
                )
            )
            count_patch += 1

            if ((count_patch) % npatch) == 0:
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

                plt.close()

        del images, labels, dic


def evaluer_modele_sur_jeu_de_test_classification_pleiade_mask_rgb(
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
    threshold = metrics_classification_pleiade(
                    test_dl, model, tile_size, batch_size, n_bands, use_mlflow
                )

    model.eval()
    npatch = int((2000 / tile_size) ** 2)
    count_patch = 0

    list_labeled_satellite_image = []

    for idx, batch in enumerate(test_dl):

        images, labels, dic = batch

        model = model.to("cuda:0")
        images = images.to("cuda:0")
        labels = labels.to("cuda:0")

        output_model = model(images)
        output_model = output_model.to("cpu")
        probability_class_1 = output_model[:, 1]

        # Set a threshold for class prediction
        # threshold = 0.90

        # Make predictions based on the threshold
        predictions = torch.where(
            probability_class_1 > threshold,
            torch.tensor([1]),
            torch.tensor([0]),
        )
        predicted_classes = predictions.type(torch.float)

        if batch_size > len(images):
            batch_size_current = len(images)

        elif batch_size <= len(images):
            batch_size_current = batch_size

        for i in range(batch_size_current):
            pthimg = dic["pathimage"][i]
            si = SatelliteImage.from_raster(
                file_path=pthimg, dep=None, date=None, n_bands=n_bands
            )

            if int(predicted_classes[i]) == 0:
                mask_pred = np.full((tile_size, tile_size, 3), 0, dtype=np.uint8)

            elif int(predicted_classes[i]) == 1:
                img = si.array.copy()
                img = img[:3, :, :]
                img = (img * 255).astype(np.uint8)
                img = img.transpose(1, 2, 0)

                shape = img.shape[0:2]

                grayscale = np.mean(img, axis=2)

                black = np.ones(shape, dtype=float)
                white = np.zeros(shape, dtype=float)

                # Creation of the mask : all grayscaled prixels below the threshold \
                # will be black and all the grayscaled prixels above the threshold \
                # will be white.

                mask_pred = np.where(grayscale > 100, white, black)

            si.normalize()
            list_labeled_satellite_image.append(
                SegmentationLabeledSatelliteImage(
                    satellite_image=si,
                    label=mask_pred,
                    source="",
                    labeling_date="",
                )
            )
            count_patch += 1

            if ((count_patch) % npatch) == 0:
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

                plt.close()

        del images, labels, dic


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
        device = "cuda:0"
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
                    satellite_image=si2,
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
            plot_file = "img/" + filename + ".png"

            fig1.savefig(plot_file)
            list_labeled_satellite_image = []

            if use_mlflow:
                mlflow.log_artifact(plot_file, artifact_path="plots")

        del images, label, dic


def evaluer_modele_sur_jeu_de_test_change_detection_S2(
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
    # batch_size  = 8

    model.eval()
    mean_IOU = test_iou_change_detection(
                        test_dl, model, tile_size, batch_size, n_bands=3, use_mlflow=False
                    )
    print(mean_IOU)
    if use_mlflow:
        mlflow.log_metric("test mean IOU", mean_IOU)

    npatch = int((1024 / tile_size) ** 2)
    if 1024 % tile_size!=0:
        npatch = npatch + 2*int(1024 / tile_size) + 1

    count_patch = 0

    list_img1 = []
    list_img2 = []
    list_label_true = []
    list_label_pred = []
    list_path = []

    for idx, batch in enumerate(test_dl):
        # idx, batch = 0, next(iter(test_dl))
        print(idx)
        images, label, dic = batch

        model = model.to("cuda:0")
        images = images.to("cuda:0")

        output_model = model(images)
        mask_pred = np.array(torch.argmax(output_model, axis=1).to("cpu"))

        if batch_size > len(images):
            batch_size_current = len(images)

        elif batch_size <= len(images):
            batch_size_current = batch_size

        for i in range(batch_size_current):
            pthimg1 = dic["pathim1"][i]
            pthimg2 = dic["pathim2"][i]
            pthlabel = dic["pathlabel"][i]

            triplet = ChangedetectionTripletS2Looking(pthimg1, pthimg2, pthlabel)
            list_img1.append(triplet.image1)
            list_img2.append(triplet.image2)
            list_label_true.append(triplet.label)
            list_label_pred.append(mask_pred[i])
            list_path.append(pthimg1)

            count_patch += 1

            if ((count_patch) % npatch) == 0:
                print("ecriture image")
                if not os.path.exists("img/"):
                    os.makedirs("img/")

                fig1 = plot_list_change_detection_images(
                            list_img1,
                            list_img2,
                            list_label_true,
                            list_label_pred,
                            list_path,
                        )

                filename = pthimg1.split("/")[-1]
                filename = filename.split(".")[0]
                filename = filename.split("_")[0]
                filename = filename + "_result"
                plot_file = "img/" + filename + ".png"

                fig1.savefig(plot_file)
                list_img1 = []
                list_img2 = []
                list_label_true = []
                list_label_pred = []
                list_path = []

                if use_mlflow:
                    mlflow.log_artifact(plot_file, artifact_path="plots")

                plt.close()

        del images, label, dic


def test_iou_change_detection(
    test_dl, model, tile_size, batch_size, n_bands=3, use_mlflow=False
):
    model.eval()
    list_IOU = []

    for idx, batch in enumerate(test_dl):

        images, labels, __ = batch

        model = model.to("cuda:0")
        images = images.to("cuda:0")

        output_model = model(images)
        output_model = output_model.to("cpu")

        iou = calculate_IOU(output_model, labels)
        value_iou = iou.tolist()[0]
        list_IOU.append(value_iou)

    del images, labels

    mean_IOU = np.mean(list_IOU)

    return mean_IOU


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



def predicted_labels_classification_pleiade(
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
    csv_file_path = "../fichierlabelerpredicted.csv"

    # threshold = metrics_classification_pleiade(
    #                 test_dl, model, tile_size, batch_size, n_bands, use_mlflow
    #             )
    threshold = 0.25

    for idx, batch in enumerate(test_dl):

        images, labels, dic = batch

        model = model.to("cuda:0")
        images = images.to("cuda:0")
        labels = labels.to("cuda:0")

        output_model = model(images)
        output_model = output_model.to("cpu")
        probability_class_1 = output_model[:, 1]

        # Set a threshold for class prediction
        # threshold = 0.90

        # Make predictions based on the threshold
        predictions = torch.where(
            probability_class_1 > threshold,
            torch.tensor([1]),
            torch.tensor([0]),
        )
        predicted_classes = predictions.type(torch.float)

        for i in range(batch_size):
            print(i)
            pthimg = dic["pathimage"][i]
            
            if not os.path.isfile(csv_file_path):
                    with open(csv_file_path, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["Path_image", "Classification_Pred"])
                        writer.writerow([pthimg, int(predicted_classes[i])])

            # Open it if it exists
            else:
                with open(csv_file_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([pthimg, int(predicted_classes[i])])

        del images, labels, dic


def proba_classification_pleiade(
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
    csv_file_path = "../fichierlabelerproba.csv"

    # threshold = metrics_classification_pleiade(
    #                 test_dl, model, tile_size, batch_size, n_bands, use_mlflow
    #             )

    for idx, batch in enumerate(test_dl):

        images, labels, dic = batch

        model = model.to("cuda:0")
        images = images.to("cuda:0")
        labels = labels.to("cuda:0")

        output_model = model(images)
        output_model = output_model.to("cpu")
        probability_class_1 = output_model[:, 1]

        for i in range(batch_size):
            print(i)
            pthimg = dic["pathimage"][i]
            
            if not os.path.isfile(csv_file_path):
                    with open(csv_file_path, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["Path_image", "Classification_Pred"])
                        writer.writerow([pthimg, probability_class_1[i].item()])

            # Open it if it exists
            else:
                with open(csv_file_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([pthimg, probability_class_1[i].item()])

        del images, labels, dic