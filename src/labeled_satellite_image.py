"""
"""
from __future__ import annotations
from typing import List, Literal, Tuple
from satellite_image import SatelliteImage
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from utils import (
    get_indices_from_tile_length
)


class SegmentationLabeledSatelliteImage:
    """ """

    def __init__(
        self,
        satellite_image: SatelliteImage,
        label: np.array,
        source: Literal["RIL", "BDTOPO"],
        labeling_date: datetime,
    ):
        """
        Constructor.

        Args:
            satellite_image (SatelliteImage): Satellite Image.
            label (np.array): Segmentation mask.
            source (Literal["RIL", "BDTOPO"]): Labeling source.
            labeling_date (datetime): Date of labeling data.
        """
        self.satellite_image = satellite_image
        self.label = label
        self.source = source
        self.labeling_date = labeling_date

    def split(self, tile_length: int) -> List[SegmentationLabeledSatelliteImage]:
        """
        Split the SegmentationLabeledSatelliteImage into tiles of dimension (`tile_length` x `tile_length`).

        Args:
            tile_length (int): Dimension of tiles

        Returns:
            List[SegmentationLabeledSatelliteImage]: _description_
        """
        # 1) on split la liste de satellite image avec la fonction déjà codée
        list_sat = self.satellite_image.split(tile_length = tile_length)

        # 2) on split le masque 
        if tile_length % 2:
            raise ValueError("Tile length has to be an even number.")

        m = self.satellite_image.array.shape[1]
        n = self.satellite_image.array.shape[2]

        indices = get_indices_from_tile_length(m, n, tile_length)
        splitted_labels = [self.label[rows[0] : rows[1], cols[0] : cols[1]] for rows, cols in indices]

        list_labelled_images = [ 
            SegmentationLabeledSatelliteImage(im,label,self.source,self.labeling_date)
            for im, label in zip(list_sat,splitted_labels)
        ]
        
        return(list_labelled_images)

        raise NotImplementedError()

    def plot(self, bands_indices: List, alpha=0.3):
        """
        Plot a subset of bands from a satellite image and its corresponding labels as an image.

        Args:
        bands_indices (List): List of indices of bands to plot from the satellite image. The indices should be integers between 0 and the number of bands - 1.
        alpha (float, optional): The transparency of the label image when overlaid on the satellite image. A value of 0 means fully transparent and a value of 1 means fully opaque. The default value is 0.3.

        """

        if not self.satellite_image.normalized:
            self.satellite_image.normalize()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(
            np.transpose(self.satellite_image.array, (1, 2, 0))[:, :, bands_indices]
        )
        ax.imshow(self.label, alpha=alpha)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Dimension of image {self.satellite_image.array.shape[1:]}")
        plt.show()
    
    def plot_label_next_to_image(self, bands_indices):
        
        """
        Plot a subset of bands from a satellite image and its corresponding labels as an image next to the original image

        Args:
        bands_indices (List): List of indices of bands to plot from the satellite image. The indices should be integers between 0 and the number of bands - 1.
        """
    
        if self.satellite_image.normalized == False :
            self.satellite_image.normalize

        show_mask = np.zeros((*self.label.shape, 3))
        show_mask[self.label == 1, :] = [255,255,255]
        show_mask = show_mask.astype(np.uint8)

        fig,(ax1,ax2) = plt.subplots(1,2, figsize = (10,10))
        ax1.imshow(
                    np.transpose(self.satellite_image.array, (1, 2, 0))[:, :, bands_indices]
                )
        ax1.axis("off")
        ax2.imshow(show_mask)
        plt.show()

    @staticmethod
    def plot_list_segmentation_labeled_satellite_image(list_labeled_image: List,bands_indices: List):
        """Plot a list of SegmentationLabeledSatelliteImage: (with a subset of bands) into 2 pictures, one with the image , one with the labels

            Args:
                list_labeled_image (List): List of SatelliteImage objects
                bands_indices (List): List of indices of bands to plot.
                    The indices should be integers between 0 and the
                    number of bands - 1.
            """
        tile_size = list_labeled_image[0].satellite_image.array_to_plot.shape[1]
        stride = tile_size

        list_bounding_box = np.array([iml.satellite_image.bounds for iml in list_labeled_image])
        list_images = np.array([iml.satellite_image for iml in list_labeled_image])
        list_labels =  [iml.label for iml in list_labeled_image]

        # calcul du bon ordre relativement aux coordonnées
        Y = np.array([bb[0] for bb in list_bounding_box])
        order_y = np.argsort(np.array(Y))
        Y = Y[order_y]

        list_images = list_images[order_y]
        list_labels = [list_labels[i] for i in order_y]
        list_bounding_box = list_bounding_box[order_y]

        X = np.array([bb[3] for bb in list_bounding_box])
        order = np.lexsort((Y,X))
        list_images = list_images[order]
        list_labels = [list_labels[i] for i in order]

        n_col = len(np.unique(np.array([bb[0] for bb in list_bounding_box])))
        n_row = len(np.unique(np.array([bb[3] for bb in list_bounding_box])))

        mat_list_images = np.transpose(list_images.reshape(n_col,n_row))
        mat_list_labels = np.transpose(np.array(list_labels).reshape(n_col,n_row,tile_size,tile_size),(1,0,2,3))

        mat_list_images =np.flip(np.transpose(mat_list_images),axis=0)
        mat_list_labels =np.flip(np.transpose(mat_list_labels,(1,0,2,3)),0)

        # Create the grid of pictures and fill it
        images = np.empty((n_row,n_col), dtype = object)
        labels = np.empty((n_row,n_col), dtype = object)

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
                output_image[i:i+tile_size, j:j+tile_size, :] = np.transpose(mat_list_images[compteur_ligne,compteur_col].array_to_plot, (1, 2, 0))[:, :, bands_indices]

                label = mat_list_labels[compteur_ligne,compteur_col,:,:]
                show_mask = np.zeros((label.shape[0],label.shape[1],3))
                show_mask[label == 1, :] = [255,255,255]
                show_mask = show_mask.astype(np.uint8)
                output_mask[i:i+tile_size, j:j+tile_size, :] = show_mask
                compteur_col += 1

            compteur_col = 0
            compteur_ligne += 1


        # Display input image, tiles, and output image as a grid
        fig, ax = plt.subplots(1, 2, figsize=(15,15))
        ax[0].imshow(output_image) # avec la normalisation pour l'affichage
        ax[0].set_title('Input Image')
        ax[1].imshow(output_mask)
        ax[1].set_title('Output Image')
        plt.show()




class DetectionLabeledSatelliteImage:
    """ """

    def __init__(
        self,
        satellite_image: SatelliteImage,
        label: List[Tuple[int]],
        source: Literal["RIL", "BDTOPO"],
        labeling_date: datetime,
    ):
        """
        Constructor.

        Args:
            satellite_image (SatelliteImage): Satellite image.
            label (List[Tuple[int]]): Detection label.
            source (Literal["RIL", "BDTOPO"]): Labeling source.
            labeling_date (datetime): Date of labeling data.
        """

    def split(self, nfolds: int) -> List[DetectionLabeledSatelliteImage]:
        """
        Split the DetectionLabeledSatelliteImage into `nfolds` folds.

        Args:
            nfolds (int): _description_

        Returns:
            List[DetectionLabeledSatelliteImage]: _description_
        """
        raise NotImplementedError()
    
    