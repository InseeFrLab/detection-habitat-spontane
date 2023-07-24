import os

import mlflow
import numpy as np
import torch
import matplotlib
from math import sqrt

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay
)

from classes.data.labeled_satellite_image import SegmentationLabeledSatelliteImage
from classes.data.satellite_image import SatelliteImage
from utils.plot_utils import (
    plot_list_labeled_sat_images,
    plot_list_segmentation_labeled_satellite_image,
    plot_satellite_image_and_mask,
    represent_grid_images_and_labels,
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


def evaluer_modele_sur_jeu_de_test_classification_sentinel(
    test_dl, model, tile_size, batch_size, n_bands, use_mlflow=False
):
    """
    Evaluates the model on the Pleiade test dataset for image classification.

    Args:
        test_dl (torch.utils.data.DataLoader): The dataloader for the test
        dataset.
        model (torchvision.models): The classification model to evaluate.
        tile_size (int): The size of each tile in pixels.
        batch_size (int): The batch size.
        use_mlflow (bool, optional): Whether to use MLflow for logging
        artifacts. Defaults to False.

    Returns:
        None
    """

    print("Entre dans la fonction d'évaluation")
    threshold = metrics_classification_pleiade(
        test_dl, model, tile_size, batch_size, n_bands, use_mlflow
    )

    model.eval()

    list_labels = []
    list_arrays = []
    list_names = []
    
    for idx, batch in enumerate(test_dl):

        images, labels, dic = batch

        if torch.cuda.is_available():
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
            if len(dic["pathimage"]) != batch_size_current:
                continue
            pthimg = dic["pathimage"][i]
            src = pthimg.split('/')[1].split('classification-')[1].split('-BDTOPO')[0]
            si = SatelliteImage.from_raster(
                file_path=pthimg, dep=None, date=None, n_bands=n_bands
            )
            si.normalize()

            if src == 'SENTINEL1-2' or src == 'SENTINEL2':
                bands_list = (3, 2, 1)
                bands_idx = 3, 2, 1
            elif src == 'SENTINEL2-RVB' or src == 'SENTINEL1-2-RVB' or src == 'PLEIADES':
                bands_list = (0, 1, 2)
                bands_idx = 0, 1, 2

            try:
                if int(predicted_classes[i]) == 0:
                    mask_pred = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)

                    if int(predicted_classes[i]) != int(labels[i]):
                        # Contours de l'image en rouge
                        array_red_borders = si.array.copy()
                        array_red_borders = array_red_borders.transpose(1, 2, 0)
                        red_color = [1.0, 0.0, 0.0]
                        array_red_borders[:, :2, bands_list] = red_color
                        array_red_borders[:, -2:, bands_list] = red_color
                        array_red_borders[:2, :, bands_list] = red_color
                        array_red_borders[-2:, :, bands_list] = red_color
                        array_red_borders = array_red_borders.transpose(2, 1, 0)
                        si.array = array_red_borders

                elif int(predicted_classes[i]) == 1:
                    mask_pred = np.full((tile_size, tile_size, 3), 0, dtype=np.uint8)

                    if int(predicted_classes[i]) != int(labels[i]):
                        # Contours de l'image en rouge
                        array_red_borders = si.array.copy()
                        array_red_borders = array_red_borders.transpose(1, 2, 0)
                        red_color = [1.0, 0.0, 0.0]
                        array_red_borders[:, :2, :] = red_color
                        array_red_borders[:, -2:, :] = red_color
                        array_red_borders[:2, :, :] = red_color
                        array_red_borders[-2:, :, :] = red_color
                        array_red_borders = array_red_borders.transpose(2, 1, 0)
                        si.array = array_red_borders

                    elif int(predicted_classes[i]) == int(labels[i]):
                        # Contours de l'image en rouge
                        array_green_borders = si.array.copy()
                        array_green_borders = array_green_borders.transpose(1, 2, 0)
                        green_color = [0.0, 1.0, 0.0]
                        array_green_borders[:, :2, bands_list] = green_color
                        array_green_borders[:, -2:, bands_list] = green_color
                        array_green_borders[:2, :, bands_list] = green_color
                        array_green_borders[-2:, :, bands_list] = green_color
                        array_green_borders = array_green_borders.transpose(2, 1, 0)
                        si.array = array_green_borders
                list_labels.append(mask_pred)
                list_arrays.append(np.transpose(si.array,(1,2,0))[:, :, bands_idx])
                list_names.append(si.filename)

                labeled_satellite_image = SegmentationLabeledSatelliteImage(
                        satellite_image=si,
                        label=mask_pred,
                        source="src",
                        labeling_date="",
                    )

                print("ecriture image")
                if not os.path.exists("outputs_evaluation_model/"):
                    os.makedirs("outputs_evaluation_model/")

                fig1 = plot_satellite_image_and_mask(
                    labeled_satellite_image,
                    bands_idx,
                )
                
                filename = pthimg.split("/")[-1]
                filename = filename.split(".")[0]
                filename = "_".join(filename.split("_")[0:6])
                plot_file = "outputs_evaluation_model/" + filename + ".png"

                fig1.savefig(plot_file)

                if use_mlflow:
                    mlflow.log_artifact(plot_file, artifact_path="plots")
                plt.close()
            except ValueError:
                print("ValueError sur l'image ", si.filename)

        del images, labels, dic
    size = int(sqrt(len(list_arrays)))**2
    list_arrays = list_arrays[:size]
    list_labels = list_labels[:size]

    grid = represent_grid_images_and_labels(list_arrays, list_labels, False, list_names)
    grid_name = "outputs_evaluation_model/grid.png"
    grid.savefig(grid_name)

    if use_mlflow:
        mlflow.log_artifact(grid_name, artifact_path="plots")



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
    with torch.no_grad():
        model.eval()
        y_true = []
        y_pred = []

        for idx, batch in enumerate(test_dl):

            images, labels, __ = batch

            model = model.to("cuda:0")
            images = images.to("cuda:0")

            output_model = model(images)
            output_model = output_model.to("cpu")
            y_pred_idx = output_model[:, 1].tolist()

            if ((len(labels.tolist()) != batch_size) or (len(y_pred_idx) != batch_size)):
                continue
            y_pred.append(y_pred_idx)
            y_true.append(labels.tolist())

            del images, labels

        y_true = np.array(y_true).flatten().tolist()
        y_pred = np.array(y_pred).flatten().tolist()

        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
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

        class_names = ["Bâti", "Non bâti"]

        predictions_best = np.where(
            y_pred > best_threshold,
            np.array([1.0]),
            np.array([0.0]),
        )
        predicted_classes_best = predictions_best.tolist()
        accuracy_best = accuracy_score(y_true, predicted_classes_best)

        predictions = np.where(
            y_pred > np.array([0.5], dtype=np.float64),
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
        plot_file_cm = "img/confusion_matrix.png"
        plt.savefig(plot_file_cm)

        if use_mlflow:
            mlflow.log_artifact(plot_file_roc, artifact_path="plots")
            mlflow.log_artifact(plot_file_cm, artifact_path="plots")
            mlflow.log_metric("test best accuracy", accuracy_best)
            mlflow.log_metric("test 0.5 accuracy", accuracy)
            mlflow.log_metric("best true positif rate", best_tpr)
            mlflow.log_metric("best false positif rate", best_fpr)
            mlflow.log_metric("best threshold", best_threshold)
            mlflow.log_metric("auc", roc_auc)

        return best_threshold


def metrics_classification_pleiade2(
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
    with torch.no_grad():
        model.eval()
        best_threshold = []
        npatch = int((2000 / tile_size) ** 2)
        count_patch = 0
        y_pred = []
        y_true = []

        for idx, batch in enumerate(test_dl):

            images, labels, __ = batch

            model = model.to("cuda:0")
            images = images.to("cuda:0")

            output_model = model(images)
            output_model = output_model.to("cpu")
            y_pred_idx = output_model[:, 1].tolist()
            labels2 = labels.tolist()

            if batch_size > len(images):
                batch_size_current = len(images)

            elif batch_size <= len(images):
                batch_size_current = batch_size

            for i in range(batch_size_current):
                y_pred.append(y_pred_idx[i])

                y_true.append(labels2[i])
                count_patch += 1

                if ((count_patch) % npatch) == 0:
                    y_true = np.array(y_true).flatten().tolist()
                    y_pred = np.array(y_pred).flatten().tolist()

                    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

                    best_threshold_idx = np.argmax(tpr - fpr)  # Index of the best threshold
                    best_threshold.append(thresholds[best_threshold_idx])
                    y_true = []
                    y_pred = []

        del images, labels

        return best_threshold


def metrics_classification_pleiade3(
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

        # Définir les seuils de classification à tester
        thresholds = np.linspace(0, 1, num=100)  # De 0 à 1 avec 100 valeurs

        # Initialiser la liste pour stocker les précisions
        accuracies = []

        # Calculer la précision pour chaque seuil de classification
        for threshold in thresholds:
            # Convertir les probabilités en prédictions binaires en utilisant le seuil
            y_pred = (y_prob >= threshold).astype(int)

            # Calculer la précision
            accuracy = accuracy_score(y_true, y_pred)

            # Ajouter la précision à la liste
            accuracies.append(accuracy)

        fnr_list = []

        # Calculer le taux de faux négatifs pour chaque seuil de classification
        for threshold in thresholds:
            # Convertir les probabilités en prédictions binaires en utilisant le seuil
            y_pred = (y_prob >= threshold).astype(int)

            # Calculer la matrice de confusion
            confusion = confusion_matrix(y_true, y_pred)

            # Extraire les valeurs de la matrice de confusion
            tn, fp, fn, tp = confusion.ravel()

            # Calculer le taux de faux négatifs (FNR)
            fnr = fn / (fn + tp)

            # Ajouter le taux de faux négatifs à la liste
            fnr_list.append(fnr)

        if not os.path.exists("img/"):
            os.makedirs("img/")

        plt.plot(thresholds, accuracies)
        plt.plot(thresholds, accuracies)
        plt.xlabel('Seuil de classification')
        plt.ylabel('Précision (Accuracy)')
        plt.title('Précision en fonction du seuil de classification')
        plt.show()
        plot_file = "img/AccuracyonThreshold.png"
        plt.savefig(plot_file)
        plt.close()

        plt.plot(thresholds, fnr_list)
        plt.xlabel('Seuil de classification')
        plt.ylabel('Taux de faux négatifs (FNR)')
        plt.title('Taux de faux négatifs en fonction du seuil de classification')
        plt.show()
        plot_file = "img/FalseNegativeRateonThreshold.png"
        plt.savefig(plot_file)
        plt.close()


def metrics_classification_pleiade4(
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
    bande_gradients = []

    for band in range(n_bands):
        one_bande_gradients = []
        for idx, batch in enumerate(test_dl):
            images, __, __ = batch

            model = model.to("cuda:0")
            images = images.to("cpu")
            print(images.shape)
            images = images.tolist()
            print(np.array(images).shape)
            images_oneband = []
            for image in images:
                array = np.array(image)
                print(array.shape)
                bande_1 = array.copy()
                bande_1 = bande_1[band, :, :]
                # Étendre les dimensions de la première bande pour correspondre au format [1, 250, 250]
                bande_1 = np.expand_dims(bande_1, axis=0)

                # Répéter la première bande pour créer les deux autres bandes
                bande_2 = np.repeat(bande_1, 1, axis=0)
                bande_3 = np.repeat(bande_1, 1, axis=0)

                # Concaténer les trois bandes pour avoir une image qui passe dans le modele
                array = np.concatenate((bande_1, bande_2, bande_3), axis=0)
                print(array.shape)
                images_oneband.append(array)

            images_oneband = torch.tensor(images_oneband)
            images_oneband = images_oneband.type(torch.float)
            images_oneband.requires_grad_(True)
            images_oneband = images_oneband.to("cuda:0")
            output_model = model(images_oneband)
            images_oneband = images_oneband.to("cpu")
            output_model = output_model.to("cpu")

            loss = torch.mean(output_model)
            print("gradient1")
            gradient = torch.autograd.grad(loss, images_oneband, allow_unused=True)
            one_bande_gradients.append(gradient)

        gradient_tensor = torch.stack(one_bande_gradients)

        # Calculer le gradient moyen
        gradient_moyen = torch.mean(gradient_tensor, dim=0)
        print("gradient2")
        gradient_moyen = torch.autograd.grad(gradient_moyen, gradient_tensor, allow_unused=True)
        bande_gradients.append(gradient_moyen)

        del images

    importances = np.mean(np.abs(np.array(bande_gradients)), axis=(1, 2, 3))  # Moyenne des gradients sur les dimensions (13, height, width)

    # Trier les bandes par importance décroissante
    indices_tries = np.argsort(importances)[::-1]

    # Afficher les bandes d'image par ordre d'importance
    for i, bande in enumerate(indices_tries):
        print(f"Bande {bande}: Importance {importances[indices_tries[i]]}")


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
            si.normalize()

            if int(predicted_classes[i]) == 0:
                mask_pred = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)

                if int(predicted_classes[i]) != int(labels[i]):
                    # Contours de l'image en rouge
                    array_red_borders = si.array.copy()
                    array_red_borders = array_red_borders.transpose(1, 2, 0)
                    red_color = [1.0, 0.0, 0.0]
                    array_red_borders[:, :7, :] = red_color
                    array_red_borders[:, -7:-1, :] = red_color
                    array_red_borders[:7, :, :] = red_color
                    array_red_borders[-7:-1, :, :] = red_color
                    array_red_borders = array_red_borders.transpose(2, 1, 0)
                    si.array = array_red_borders

            elif int(predicted_classes[i]) == 1:
                mask_pred = np.full((tile_size, tile_size, 3), 0, dtype=np.uint8)

                if int(predicted_classes[i]) != int(labels[i]):
                    # Contours de l'image en rouge
                    array_red_borders = si.array.copy()
                    array_red_borders = array_red_borders.transpose(1, 2, 0)
                    red_color = [1.0, 0.0, 0.0]
                    array_red_borders[:, :7, :] = red_color
                    array_red_borders[:, -7:-1, :] = red_color
                    array_red_borders[:7, :, :] = red_color
                    array_red_borders[-7:-1, :, :] = red_color
                    array_red_borders = array_red_borders.transpose(2, 1, 0)
                    si.array = array_red_borders

                elif int(predicted_classes[i]) == int(labels[i]):
                    # Contours de l'image en rouge
                    array_green_borders = si.array.copy()
                    array_green_borders = array_green_borders.transpose(1, 2, 0)
                    green_color = [0.0, 1.0, 0.0]
                    array_green_borders[:, :7, :] = green_color
                    array_green_borders[:, -7:-1, :] = green_color
                    array_green_borders[:7, :, :] = green_color
                    array_green_borders[-7:-1, :, :] = green_color
                    array_green_borders = array_green_borders.transpose(2, 1, 0)
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


def evaluer_modele_sur_jeu_de_test_classification_pleiade2(
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
