import math
import os
import re
from datetime import date
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from classes.data.satellite_image import SatelliteImage
from utils.mappings import name_dep_to_num_dep
from utils.utils import get_environment


def order_list_from_bb(
    list_bounding_box: List,
    list_to_order: List,
):
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


def plot_list_satellite_images(
    list_images: List,
    bands_indices: List,
):
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

    return plt.gcf()


def plot_list_sat_images_square(
    list_images: List,
    bands_indices: List,
):
    size = int(math.sqrt(len(list_images)))

    # Create a figure and axes
    fig, axs = plt.subplots(nrows=size, ncols=size, figsize=(10, 10))

    # Iterate over the grid of masks and plot them
    for i in range(size):
        for j in range(size):
            axs[i, j].imshow(
                list_images[i * size + j].array.transpose(1, 2, 0)[
                    :, :, bands_indices
                ]
            )

    # Remove any unused axes
    for i in range(size):
        for j in range(size):
            axs[i, j].set_axis_off()

    # Show the plot
    return plt.gcf()


def plot_list_segmentation_labeled_satellite_image(
    list_labeled_image: List,
    bands_indices: List,
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

    # Correct order relative to the coordinates
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
            output_image[i : i + tile_size, j : j + tile_size, :] = np.transpose(
                mat_list_images[compteur_ligne, compteur_col].array,
                (1, 2, 0),
            )[:, :, bands_indices]

            label = mat_list_labels[compteur_ligne, compteur_col, :, :]
            show_mask = np.zeros((label.shape[0], label.shape[1], 3))
            show_mask[label == 1, :] = [255, 255, 255]
            output_mask[i : i + tile_size, j : j + tile_size, :] = show_mask
            compteur_col += 1

        compteur_col = 0
        compteur_ligne += 1

    # Display input image, tiles, and output image as a grid
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(output_image)  # with normalization for display
    ax[0].set_title("Input Image")
    ax[0].set_axis_off()
    ax[1].imshow(output_mask)
    ax[1].set_title("Output Image")
    ax[1].set_axis_off()

    return plt.gcf()


def plot_list_labeled_sat_images(
    list_labeled_image: List,
    bands_indices: List,
):
    list_images1 = [iml.satellite_image for iml in list_labeled_image]
    list_labels1 = [iml.label for iml in list_labeled_image]

    list_bounding_box = [[im.bounds[3], im.bounds[0]] for im in list_images1]

    # Utiliser zip pour combiner les trois listes
    combined = zip(list_bounding_box, list_images1, list_labels1)

    # Trier les éléments combinés en fonction de la troisième liste
    sorted_combined = sorted(combined, key=lambda x: (-x[0][0], x[0][1]))

    # Diviser les listes triées en fonction de l'ordre des éléments
    __, list_images, list_labels = zip(*sorted_combined)

    size = int(math.sqrt(len(list_images1)))

    # Create a figure and axes
    fig, axs = plt.subplots(nrows=size, ncols=2 * size, figsize=(20, 10))

    # Iterate over the grid of masks and plot them
    for i in range(size):
        for j in range(size):
            axs[i, j].imshow(
                list_images1[i * size + j].array.transpose(1, 2, 0)[
                    :, :, bands_indices
                ]
            )

    for i in range(size):
        for j in range(size):
            axs[i, j + size].imshow(list_labels[i * size + j], cmap="gray")

    # Remove any unused axes
    for i in range(size):
        for j in range(2 * size):
            axs[i, j].set_axis_off()

    # Show the plot
    return plt.gcf()


def plot_infrared_simple_mask(satellite_image: SatelliteImage):
    """Plot the infrared mask based on threshold on the infrared median.

    Args:
        satellite_image (SatelliteImage): A satellite image with 4 bands.

    Returns:
        The simple infrared mask of the image.
    """
    if satellite_image.n_bands < 4:
        print("This image has no infrared band.")

    else:
        # Extract the array from the image to get the pixel values
        img = satellite_image.array.copy()

        img = img.transpose(2, 1, 0)
        shape = img.shape[0:2]

        # Threshold : median of the 4th band
        threshold = np.quantile(img[:, :, 3], 0.5)  # only on the 4th band
        black = np.ones(shape, dtype=float)
        white = np.zeros(shape, dtype=float)

        # Creation of the mask : all the infrared prixels below the threshold \
        # will be black and all the infrared prixels above the threshold \
        # will be white.

        mask = np.where(img[:, :, 3] > threshold, white, black)

        # Return to the right shape
        mask = mask.transpose(1, 0)

        # Plot the mask
        plt.imshow(mask, cmap="gray")
        plt.show()


def plot_infrared_patch_mask(satellite_image: SatelliteImage):
    """Plot the infrared mask based on patch-by-patch thresholding
        on the infrared median of the patch. 250 patches.

    Args:
        satellite_image (SatelliteImage): A satellite image with 4 bands.

    Returns:
        All the patchs of infrared masks of the image.
    """
    if satellite_image.n_bands < 4:
        print("This image has no infrared band.")

    else:
        list_images = satellite_image.split(250)
        list_mask = []

        # We go through each patch
        for indice, mini_image in enumerate(list_images):
            # Extract the array from the image to get the pixel values
            img = mini_image.array.copy()

            img = img.transpose(2, 1, 0)
            shape = img.shape

            # Threshold : median of the 4th band on the patch
            threshold = np.quantile(img[:, :, 3], 0.5)  # only on the 4th band
            black = np.ones(shape[0:2], dtype=float)
            white = np.zeros(shape[0:2], dtype=float)

            # Creation of the mask : all the infrared prixels below \
            # the threshold will be \ black and all the infrared prixels \
            # above the threshold will be white.

            mask = np.where(img[:, :, 3] > threshold, white, black)

            # Return to the right shape
            mask = mask.transpose(1, 0)

            list_mask.append(mask)

        list_bounding_box = np.array([im.bounds for im in list_images])
        n_col = len(np.unique(np.array([bb[0] for bb in list_bounding_box])))
        n_row = len(np.unique(np.array([bb[3] for bb in list_bounding_box])))

        # Create the grid of pictures and fill it with masks
        masks = np.empty((n_col, n_row), dtype=object)

        for i in range(n_col):
            for j in range(n_row):
                masks[i, j] = list_mask[8 * i + j]

        # Create a figure and axes
        fig, axs = plt.subplots(nrows=n_row, ncols=n_col, figsize=(10, 10))

        # Iterate over the grid of masks and plot them
        for i in range(n_row):
            for j in range(n_col):
                axs[i, j].imshow(masks[i, j], cmap="gray")

        # Remove any unused axes
        for i in range(n_row):
            for j in range(n_col):
                axs[i, j].set_axis_off()

        # Show the plot
        plt.show()


def plot_square_images(
    bands_indices: list,
    distance: int = 1,
    satellite_image: SatelliteImage = None,
    filepath_center_image: str = None,
):
    """Plot all the images surrounding the image we give. This image will
    be in the middle and we specify who much images we want to have around.
    For example, if we give a distance of 2, we will plot all the images
    with that are maximum 2 images far from the center image.
    It will return a square 5x5.
    The arguments can be either a SatteliteImage
    or the filepath of the center image.

    Args:
        bands_indice (list): the list of the band indices we want to plot.
        distance (int): the distance of the images
        from the center image we want.
        satellite_image (SatelliteImage): the center SatelliteImage.
        filepath_center_image (str): the filepath of the center image.

    Returns:
        The square of the images surrounding the center image.

    Example:
        >>> plot_square_images(
            [0,1,2], 1 , None ,
            '../data/PLEIADES/2017/MARTINIQUE/72-2017-0711-1619-U20N-0M50-RVB-E100.jp2'
            )
    """

    environment = get_environment()

    # Get folder path
    pattern = "/"

    split_filepath_center = re.split(pattern, filepath_center_image)

    folder_path = pattern.join(split_filepath_center[0:5])

    if satellite_image is not None:
        name_dep = str(satellite_image.dep)
        year = (satellite_image.date).year
        path = environment["local-path"]["PLEIADES"][year][name_dep]

        folder_path = "../" + path
        filepath_center_image = folder_path + "/" + satellite_image.filename
    else:
        # Retrieve the year and the department
        annee = split_filepath_center[3]
        departement = split_filepath_center[4]
        dep_num = name_dep_to_num_dep[departement]

        # Retrieve the left-top coordinates of the center image
        delimiters = ["-", "_"]

        pattern = "|".join(delimiters)

        split_filepath_center = re.split(pattern, filepath_center_image)

        left_center = float(split_filepath_center[2]) * 1000
        top_center = float(split_filepath_center[3]) * 1000

        list_images = []
        list_images_path = []

        for filename in os.listdir(folder_path):
            # Retrieve left-top coordinates of all images
            split_filename = re.split(pattern, filename)

            left = float(split_filename[2]) * 1000.0
            top = float(split_filename[3]) * 1000.0

            limit_left_r = left_center + distance * 1000.0
            limit_top_r = top_center + distance * 1000.0
            limit_left_l = left_center - distance * 1000.0
            limit_top_l = top_center - distance * 1000.0

            if limit_left_l <= left <= limit_left_r:
                if limit_top_l <= top <= limit_top_r:
                    image = SatelliteImage.from_raster(
                        folder_path + "/" + filename,
                        date=date.fromisoformat(annee + "-01-01"),
                        n_bands=len(bands_indices),
                        dep=dep_num,
                    )
                    image.normalize()
                    list_images.append(image)
                    list_images_path.append(image.filename)

        plot_list_satellite_images(list_images, bands_indices)


def plot_list_images_square(folder_path, borne_inf, borne_sup):
    """
    Plot a square grid of images from a folder. You must specify a lower limit
    and an upper limit,
    to be able to display the images of the folder between these two limits.
    The difference of the
    two bounds must be a square number to obtain a list of images of square
    length.

    Args:
        folder_path (str): Path to the folder containing the images.
        borne_inf (int): Lower bound index for selecting images.
        borne_sup (int): Upper bound index for selecting images.

    Returns:
        Plot of the images.
    """

    list_filepaths = os.listdir(folder_path)[borne_inf : borne_sup + 1]
    size = int(math.sqrt(len(list_filepaths)))

    list_images = []

    for filepath in tqdm(list_filepaths):
        # Retrieve left-top coordinates of all images
        image = SatelliteImage.from_raster(
            folder_path + "/" + filepath, date=None, n_bands=3, dep=None
        )
        image.normalize()
        list_images.append(image)

    mat_list_images = np.transpose(np.array(list_images).reshape(size, size))

    # Create a figure and axes
    fig, axs = plt.subplots(nrows=size, ncols=size, figsize=(20, 20))

    # Iterate over the grid of masks and plot them
    for i in range(size):
        for j in range(size):
            axs[i, j].imshow(mat_list_images[i, j].array.transpose(1, 2, 0))

    # Remove any unused axes
    for i in range(size):
        for j in range(size):
            axs[i, j].set_axis_off()

    # Show the plot
    plt.show()


def creer_array_to_plot(pth_image):
    """
    Gives the correctly formatted arrays corresponding to an image to plot.

    Args:
        pth_image (list[str]): paths to the images to plot.

    Returns:
        the correctly formatted arrays corresponding to the entry image.
    """

    si = SatelliteImage.from_raster(pth_image, dep=None, date=None, n_bands=3)
    si.normalize()
    bands_indices = [0, 1, 2]
    array = si.array
    #normalized_array = (array.astype(np.float32) - np.min(array)) / (
     #   np.max(array) / 3 - np.min(array)
    #)
    normalized_array = array
    array_to_plot = np.transpose(normalized_array, (1, 2, 0))[:, :, bands_indices]

    return array_to_plot


def represent_grid_images_and_labels(
    list_array_to_plot,
    list_label,
    creer_array=True,
    list_names = None
):
    """
    Plot a square grid of images and their masks from their paths.

    Args:
        list_array_to_plot (list[str]): paths to the images to plot.
        list_label (list[np]): the masks to plot.

    Returns:
        Calls a function that plots the images and their masks.
    """
    N = len(list_label)
    nrow = int(math.sqrt(N))
    fig, axes = plt.subplots(nrow, 2 * nrow, figsize=(20, 10))

    # Iterate over the RGB arrays and plot them in the left grid
    for i, ax in enumerate(axes[:, :nrow].flat):
        if creer_array:
            array_to_plot = creer_array_to_plot(list_array_to_plot[i])
        else:
            array_to_plot = list_array_to_plot[i]
        x = int(i/nrow) + 1
        y = (i+1)%nrow
        if y == 0:
            y = nrow
        file = open("img/Noms_vignette", "a")
        file.write("\n" + "(" + str(x) + "," + str(y) + ")" + "   " + list_names[i])
        file.close()
        ax.imshow(array_to_plot)
        ax.axis("off")

    # Iterate over the 0-1 arrays and plot them in the right grid
    for i, ax in enumerate(axes[:, nrow:].flat):
        ax.imshow(list_label[i], cmap="binary")
        ax.axis("off")

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0)

    # Show the plot
    plt.show()
    if creer_array:
        plt.savefig("test.png")
    else:
        return plt.gcf()


def plot_list_images_and_masks_square(
    dataloader,
    size_of_grid,
):
    """
    Calls a function that plot a square grid of images and their masks\
        from their folder. You must specify the size of the grid.

    Args:
        train_dataset (dataloader): the dataloader from which\
            the dataset should be extracted.
        size_of_the_grid (int): the number of images you want on a line\
            (the grid is a square so the number on a column will be the same).

    Returns:
        Calls a function that plots the images and their masks.
    """
    dataset = dataloader.dataset

    list_array_to_plot = []
    for i in range(size_of_grid**2):
        input, label, dic = dataset[i]
        pth_image = dic["pathimage"]
        list_array_to_plot.append(pth_image)

    list_label = []

    for i in range(size_of_grid**2):
        if 'classification' in pth_image:
            size = input.shape[1]
            if np.array(dataset[i][1]) == 1:
                label = np.full((size, size, 3), 0, dtype=np.uint8)
            elif np.array(dataset[i][1]) == 0:
                label = np.full((size, size, 3), 255, dtype=np.uint8)
            list_label.append(label)
                
        elif 'segmentation' in pth_image:
            list_label.append(np.array(dataset[i][1]))

    return represent_grid_images_and_labels(list_array_to_plot, list_label, True)


def plot_satellite_image_and_mask(
    labeled_satellite_image,
    bands_indices,
):
    """
    Calls a function that plot a square grid of images and their masks\
        from their folder. You must specify the size of the grid.

    Args:
        dataloader: the dataloader from which\
            the dataset should be extracted.
        size_of_the_grid (int): the number of images you want on a line\
            (the grid is a square so the number on a column will be the same).

    Returns:
        Calls a function that plots the images and their masks.
    """
    image = labeled_satellite_image.satellite_image.array[bands_indices, :, :]
    label = labeled_satellite_image.label

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 25))
    ax1.imshow(image.transpose(1,2,0))
    ax1.set_title("image")
    ax1.axis('off')
    ax2.imshow(label)
    ax2.set_title("mask")
    ax2.axis('off')
    return plt.gcf()

    # Show the plot
    # plt.show()
    # plt.gcf()
    # plt.savefig("test.png")


def draw_change_is_everywhere_exemples(changeiseverywheredataset, n_exemples):

    """
    Draws and saves a grid of examples from the ChangeIsEverywhere dataset.

    Args:
        changeiseverywheredataset (list): The ChangeIsEverywhere dataset containing image paths and labels.
        n_examples (int): The number of examples to draw.

    Returns:
        None
    """

    triplets = [
        {
            "pth1": changeiseverywheredataset[i][2]["pathimage1"],
            "pth2": changeiseverywheredataset[i][2]["pathimage2"],
            "label": changeiseverywheredataset[i][1]
        }
        for i in range(n_exemples)
    ]

    num_triplets = len(triplets)
    num_cols = 3  # Number of columns in the subplot grid
    # Iterate over each triplet and plot the images and labels

    fig, axs = plt.subplots(
        num_triplets,
        num_cols,
        figsize=(15, 15),
        constrained_layout=True
    )

    for i, dic in enumerate(triplets):
        # i = 0
        pathimage1, pathimage2, label = dic["pth1"],  dic["pth2"], dic["label"]

        # Load the jp2 files
        image1 = SatelliteImage.from_raster(pathimage1, "972").array
        image2 = SatelliteImage.from_raster(pathimage2, "972").array

        # Plot the first image in the left subplot
        axs[i, 0].imshow(np.transpose(image1, (1, 2, 0)))
        
        # Plot the second image in the right subplot
        axs[i, 1].imshow(np.transpose(image2, (1, 2, 0)))
    
        # Add the label as the title of the bottom subplot
        axs[i, 2].imshow(label, cmap='gray')
        
        # Remove the ticks and labels in the subplots
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        axs[i, 1].set_xticks([])
        axs[i, 1].set_yticks([])
        axs[i, 2].set_xticks([])
        axs[i, 2].set_yticks([])

    # set the spacing between subplots
    plt.subplots_adjust(left=0.01)
    
    # Adjust the spacing between subplots
    # Show the plot
    plt.show()
    res = plt.gcf()
    res.savefig("triplet_train.png")


def plot_image_mask_label(
    satellite_image,
    label,
    mask,
    bands_idx,
    label_buffered=None
):
    image = satellite_image.array[bands_idx, :, :].transpose(1,2,0)

    if label_buffered is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(27, 9))
        ax1.imshow(image)
        ax1.imshow(label, alpha=0.5, cmap='OrRd')
        ax1.set_title("image et label")
        ax1.axis('off')
        ax2.imshow(image)
        ax2.imshow(mask, alpha=0.3)
        ax2.set_title("image et masque")
        ax2.axis('off')
        ax3.imshow(image)
        ax3.imshow(label_buffered, alpha=0.5, cmap='OrRd')
        ax3.set_title("image et label bufferisé")
        ax3.axis('off')
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 25))
        ax1.imshow(image)
        ax1.imshow(label, alpha=0.5, cmap='OrRd')
        ax1.set_title("image et label")
        ax1.axis('off')
        ax2.imshow(image)
        ax2.imshow(mask, alpha=0.3)
        ax2.set_title("image et masque")
        ax2.axis('off')
    return plt.gcf()
