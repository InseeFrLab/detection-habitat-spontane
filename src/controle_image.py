import matplotlib.pyplot as plt
import numpy as np
from classes.data.satellite_image import SatelliteImage
from classes.data.labeled_satellite_image import SegmentationLabeledSatelliteImage
import os


dir = "../train_data-PLEIADES-972-2022/"
i = 5
masque_path =  dir + "labels/" + np.sort(os.listdir(dir+ "labels/"))[i]
image_path =  dir + "images/" + np.sort(os.listdir(dir+ "images/"))[i]

masque_path = dir + "labels/ORT_2022_0690_1638_U20N_8Bits_000.npy"
image_path = dir + "images/ORT_2022_0690_1638_U20N_8Bits_000.jp2"

masque = np.load(masque_path)
si = SatelliteImage.from_raster(image_path, "972", n_bands =3)
SegmentationLabeledSatelliteImage(si, masque, "", "").plot_label_next_to_image([0,1,2])

res = plt.gcf()
res.savefig("controle.png") 



len(os.listdir( dir + "images/"))
len(os.listdir( dir + "labels/"))

