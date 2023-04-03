from typing import List
from datetime import date
import re
import os
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../src')
from satellite_image import SatelliteImage
from mappings import *

def order_list_from_bb(list_bounding_box: List, list_to_order: List):
    """Order a given List according to the X,Y coordinates
    of the list of bounding boxes taken as input

    Args:
        list_bounding_box (List): List of bouding box (4-dimensional tuples)
        list_to_order (List): List of object we want to order according
        to the coordinates of the bounding boxes
    """
    Y = np.array([bb[0] for bb in list_bounding_box])
    order_y = np.argsort(np.array(Y))
    Y = Y[order_y]

    list_to_order = [list_to_order[i] for i in order_y]
    list_bounding_box = [list_bounding_box[i] for i in order_y]

    X = np.array([bb[3] for bb in list_bounding_box])
    order = np.lexsort((Y, X))

    list_to_order = [list_to_order[i] for i in order]

    return list_to_order


def plot_list_satellite_images(list_images: List, bands_indices: List):
    """Plot a list of SatelliteImage (with a subset of bands) into a grid
    following the coordinates of the SatelliteImage.
    The list of SatelliteImages taken as input when represented
    in the correct order, has to fully cover a rectangular area.

    Args:
        list_images (List): List of SatelliteImage objects
        bands_indices (List): List of indices of bands to plot.
            The indices should be integers between 0 and the
            number of bands - 1.
    """
    list_bounding_box = np.array([im.bounds for im in list_images])

    list_images = order_list_from_bb(list_bounding_box, list_images)

    n_col = len(np.unique(np.array([bb[0] for bb in list_bounding_box])))
    n_row = len(np.unique(np.array([bb[3] for bb in list_bounding_box])))

    mat_list_images = np.transpose(np.array(list_images).reshape(n_row, n_col))

    # Create the grid of pictures and fill it
    images = np.empty((n_col, n_row), dtype=object)

    for i in range(n_col):
        for j in range(n_row):
            images[i, j] = mat_list_images[i, j].array

    images = np.flip(np.transpose(images), axis=0)

    # Create a figure and axes
    fig, axs = plt.subplots(nrows=n_row, ncols=n_col, figsize=(10, 10))

    # Iterate over the grid of images and plot them
    for i in range(n_row):
        for j in range(n_col):
            axs[i, j].imshow(
                np.transpose(images[i, j], (1, 2, 0))[:, :, bands_indices]
            )

    # Remove any unused axes
    for i in range(n_row):
        for j in range(n_col):
            axs[i, j].set_axis_off()

    # Show the plot
    plt.show()


def plot_list_segmentation_labeled_satellite_image(
    list_labeled_image: List, bands_indices: List
):
    """Plot a list of SegmentationLabeledSatelliteImage:
    (with a subset of bands) into 2 pictures, one with the satelliteImage,
    one with the labels. The list of SatelliteImages contained in the
    labeled_images taken as input, when represented in the correct order,
    has to fully cover a rectangular area.

    Args:
        list_labeled_image (List): List of SatelliteImage objects
        bands_indices (List): List of indices of bands to plot.
            The indices should be integers between 0 and the
            number of bands - 1.
    """
    tile_size = list_labeled_image[0].satellite_image.array.shape[1]
    stride = tile_size

    list_bounding_box = np.array(
        [iml.satellite_image.bounds for iml in list_labeled_image]
    )
    list_images = [iml.satellite_image for iml in list_labeled_image]
    list_labels = [iml.label for iml in list_labeled_image]

    # calcul du bon ordre relativement aux coordonnées
    list_images = order_list_from_bb(list_bounding_box, list_images)
    list_labels = order_list_from_bb(list_bounding_box, list_labels)

    n_col = len(np.unique(np.array([bb[0] for bb in list_bounding_box])))
    n_row = len(np.unique(np.array([bb[3] for bb in list_bounding_box])))

    mat_list_images = np.transpose(np.array(list_images).reshape(n_col, n_row))
    mat_list_labels = np.transpose(
        np.array(list_labels).reshape(n_col, n_row, tile_size, tile_size),
        (1, 0, 2, 3),
    )

    mat_list_images = np.flip(np.transpose(mat_list_images), axis=0)
    mat_list_labels = np.flip(np.transpose(mat_list_labels, (1, 0, 2, 3)), 0)

    # Get input image dimensions
    width = tile_size * n_col
    height = tile_size * n_row

    # Create empty output image
    output_image = np.zeros((height, width, 3))
    output_mask = np.zeros((height, width, 3))
    compteur_ligne = 0
    compteur_col = 0

    for i in range(0, height - tile_size + 1, stride):
        for j in range(0, width - tile_size + 1, stride):
            output_image[
                i : i + tile_size, j : j + tile_size, :
            ] = np.transpose(
                mat_list_images[compteur_ligne, compteur_col].array,
                (1, 2, 0),
            )[
                :, :, bands_indices
            ]

            label = mat_list_labels[compteur_ligne, compteur_col, :, :]
            show_mask = np.zeros((label.shape[0], label.shape[1], 3))
            show_mask[label == 1, :] = [255, 255, 255]
            show_mask = show_mask.astype(np.uint8)
            output_mask[i : i + tile_size, j : j + tile_size, :] = show_mask
            compteur_col += 1

        compteur_col = 0
        compteur_ligne += 1

    # Display input image, tiles, and output image as a grid
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(output_image)  # avec la normalisation pour l'affichage
    ax[0].set_title("Input Image")
    ax[0].set_axis_off()
    ax[1].imshow(output_mask)
    ax[1].set_title("Output Image")
    ax[1].set_axis_off()
    plt.show()
    

def plot_infrared_simple_mask(
    satellite_image: SatelliteImage
):
    """Plot the infrared mask based on threshold on the infrared median.

    Args:
        satellite_image (SatelliteImage): A satellite image with 4 bands.
        
    Returns:
        The simple infrared mask of the image.
    """   
    if satellite_image.n_bands < 4 :
        print("Cette image n'a pas de bande infrarouge.")
    
    else :
        #on extrait l'array de l'image pour avoir les valeurs des pixels
        img = satellite_image.array

        #multiplication par 255 et convertion en uint8 pour avoir le bon format
        img = (img * 255).astype(np.uint8)

        img = img.transpose()

        seuil = np.quantile(img[:,:,3],0.5)

        #on parcours tous les pixels et on les modifie en fonction du seuil
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                i = img[row,col,3]
                if i > seuil: #médiane
                    img[row, col] = np.array([0, 0, 0,0]) # blanc

                else : 
                    img[row, col] = np.array([255,255,255,255]) # noir

        img = img.transpose()

        #On veut le bon format
        img = (img/255).astype(np.float64)

        satellite_image.array = img

        satellite_image.plot([0,1,2])

   

def plot_infrared_patch_mask(
    satellite_image: SatelliteImage
):
    """Plot the infrared mask based on patch-by-patch thresholding on the infrared median of the patch. 250 patches.

    Args:
        satellite_image (SatelliteImage): A satellite image with 4 bands.
        
    Returns:
        All the patchs of infrared masks of the image.
    """
    if satellite_image.n_bands < 4 :
        print("Cette image n'a pas de bande infrarouge.")

    else : 
        list_images = satellite_image.split(250)

        #on parcourt chaque patch
        for mini_image in list_images:
            #on extrait l'array de l'image pour avoir les valeurs des pixels
            img = mini_image.array

            #multiplication par 255 et convertion en uint8 pour avoir le bon format
            img = (img * 255).astype(np.uint8)

            img = img.transpose()

            seuil = np.quantile(img[:,:,3],0.5)

            #on parcours tous les pixels et on les modifie en fonction du seuil
            for row in range(img.shape[0]):
                for col in range(img.shape[1]):
                    i = img[row,col,3]
                    if i > seuil: #médiane
                        img[row, col] = np.array([0, 0, 0,0]) # blanc

                    else : 
                        img[row, col] = np.array([255,255,255,255]) # noir

            img = img.transpose()

            #On veut le bon format
            img = (img/255).astype(np.float64)

            mini_image.array = img

        plot_list_satellite_images(list_images,bands_indices = [0,1,2])

    
def plot_infrared_complex_mask(
    satellite_image: SatelliteImage
):
    """Plot the infrared mask based on thresholding on infrared, green and blue to recover certain shades of infrared.

    Args:
        satellite_image (SatelliteImage): A satellite image with 4 bands.
    
    Returns:
        The complex infrared mask of the image.  
    """
    
    if satellite_image.n_bands < 4 :
        print("Cette image n'a pas de bande infrarouge.")
    
    else :
        img = satellite_image.array

        img = (img * 255).astype(np.uint8)


        img = img.transpose()

        # On va parcourir tous les pixels de l'image
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                b = img[row,col, 1]
                g = img[row,col, 2]
                r = img[row,col, 3]
                mini = min(b,g)
                maxi = max(b,g)

                if maxi-mini <= 20 : #étape 1

                    if r > 200 and mini >= 110 and r>= (20+mini): #étape 2
                        img[row, col] = [255,255,255,255] # blanc
                    elif  r>= (20+mini) and r >= 110: #étape 3
                        img[row, col] = [0,0,0,0] #noir
                    else : #étape 4
                        img[row, col] = [255,255,255,255] # blanc

                else : #étape 4
                    img[row, col] = [255,255,255,255] # blanc

        img = img.transpose()

        #On veut le bon format
        img = (img/255).astype(np.float64)

        satellite_image.array = img

        satellite_image.plot([0,1,2])


def plot_square_images(bands_indices: list, distance = 1, satellite_image : SatelliteImage = None, filepath_center_image : str = None):
    """Plot all the images surrounding the image we give. This image will be in the middle and we specify who much images we want to have around. For example, if we give a distance of 2, we will plot all the images with that are maximum 2 images far from the center image. It will return a square 5x5. The arguments can be either a SatteliteImage or the filepath of the center image.

    Args:
        bands_indice (list): the list of the band indices we want to plot.
        distance (int): the distance of the images from the center image we want.
        satellite_image (SatelliteImage): the center SatelliteImage.
        filepath_center_image (str): the filepath of the center image. 

    Returns:
        The square of the images surrounding the center image.
        
    Example:
        >>> plot_square_images([0,1,2], 1 , None ,'../data/PLEIADES/2017/MARTINIQUE/72-2017-0711-1619-U20N-0M50-RVB-E100.jp2')
    """
    
    #récupérer le path du dossier
    delimiter = ["/"]

    pattern = "|".join(delimiter)

    split_filepath_center = re.split(pattern, filepath_center_image)

    folder_path = "/".join(split_filepath_center[0:5])
    
    if satellite_image != None:
        folder_path = '../data/PLEIADES/' + str((satellite_image.date).year) + '/' + num_dep_to_name_dep[satellite_image.dep]
        filepath_center_image = folder_path + '/'+ satellite_image.filename
    
    else :
        
        #récupérer l'année et le département
        annee = split_filepath_center[3]
        departement = split_filepath_center[4]
        dep_num = name_dep_to_num_dep[departement]

        #récupérer les coordonnées left-top de l'image du centre
        delimiters = ["-", "_"]

        pattern = "|".join(delimiters)

        split_filepath_center = re.split(pattern, filepath_center_image)

        left_center = float(split_filepath_center[2])*1000
        top_center = float(split_filepath_center[3])*1000

        list_images = []
        list_images_path = []

        for filename in os.listdir(folder_path):

            #récupérer les coordonnées left-top de toutes les images
            split_filename = re.split(pattern, filename)

            left = float(split_filename[2])*1000.0
            top = float(split_filename[3])*1000.0

            if left_center - distance*1000.0 <= left <= left_center + distance*1000.0:
                if top_center - distance*1000.0 <= top <= top_center + distance*1000.0:
                    image = SatelliteImage.from_raster(
                            folder_path + '/' +filename,
                            date = date.fromisoformat(annee + '-01-01'), 
                            n_bands = len(bands_indices),
                            dep = dep_num
                        )
                    image.normalize()
                    list_images.append(image)
                    list_images_path.append(image.filename)

        plot_list_satellite_images(list_images, bands_indices)