import torch
import numpy as np
from classes.data.satellite_image import SatelliteImage
from classes.data.labeled_satellite_image import \
    SegmentationLabeledSatelliteImage
from utils.plot_utils import \
    plot_list_segmentation_labeled_satellite_image
import os
import mlflow
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
    test_dl,
    model,
    tile_size,
    batch_size,
    use_mlflow=False
):  
    
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
        
        model = model.to("cuda:0")
        images = images.to("cuda:0")
        
        output_model = model(images)
        mask_pred = np.array(torch.argmax(output_model, axis=1).to("cpu"))

        for i in range(batch_size):
            pthimg = dic["pathimage"][i]
            si = SatelliteImage.from_raster(
                file_path=pthimg,
                dep=None,
                date=None,
                n_bands=3
            )
            si.normalize()

            list_labeled_satellite_image.append( 
                SegmentationLabeledSatelliteImage(
                    satellite_image=si,
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
    
            filename = pthimg.split('/')[-1]
            filename = filename.split('.')[0]
            filename = '_'.join(filename.split('_')[0:6])
            plot_file = filename + ".png"
        
            fig1.savefig(plot_file)
            list_labeled_satellite_image = []
            
            if use_mlflow:
                mlflow.log_artifact(plot_file, artifact_path="plots")
            
        del images, label, dic

def evaluer_modele_sur_jeu_de_test_classification_pleiade(
    test_dl,
    model,
    tile_size,
    batch_size,
    use_mlflow=False
):  
    
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
        
        model = model.to("cuda:0")
        images = images.to("cuda:0")
        
        output_model = model(images)
        output_model = output_model.to("cpu")
        probability_class_1 = output_model[:, 1]

        # Set a threshold for class prediction
        threshold = 0.51

        # Make predictions based on the threshold
        predictions = torch.where(probability_class_1 > threshold, torch.tensor([1]), torch.tensor([0]))
        predicted_classes = predictions.type(torch.float)

        for i in range(batch_size):
            pthimg = dic["pathimage"][i]
            si = SatelliteImage.from_raster(
                file_path=pthimg,
                dep=None,
                date=None,
                n_bands=3
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
    
            filename = pthimg.split('/')[-1]
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
    predictions = torch.where(probability_class_1 > threshold, torch.tensor([1]), torch.tensor([0]))

    predicted_classes = predictions.type(torch.float)

    misclassified_percentage = (predicted_classes != labels).float().mean()

    return misclassified_percentage


def proportion_ones(labels):
    """
    Calculate the proportion of ones in the validation dataloader.

    Args:
        labels: the true classes

    """

    # Compter le nombre de zéros
    num_zeros = int(torch.sum(labels == 0))

    # Compter le nombre de uns
    num_ones = int(torch.sum(labels == 1))

    prop_ones = num_ones/(num_zeros + num_ones)

    # Arrondi deux chiffres après la virgule
    prop_ones = round(prop_ones, 2)

    return prop_ones
