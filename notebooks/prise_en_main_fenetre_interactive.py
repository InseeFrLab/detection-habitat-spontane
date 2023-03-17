 pip install rasterio  -q -q -q
 pip install geopandas -q -q -q
 pip install matplotlib -q -q -q
 pip install pyarrow
 
# %%
%load_ext autoreload
%autoreload 2

# %%
import sys
sys.path.append('../src')
from satellite_image import SatelliteImage
from utils import *

import yaml
import re
import s3fs
import numpy as np
import matplotlib.pyplot as plt

# Mise en place des liens de téléchargement
update_storage_access()
environment = get_environment()
root_path = get_root_path()
bucket = environment["bucket"]
path_s3_cayenne_data = environment["sources"]["PLEIADES"]
path_local_cayenne_data = os.path.join(root_path, environment["local-path"]["PLEIADES"])
bucket = environment["bucket"]

path_s3_pleiades_data = environment["sources"]["PLEIADES"]
path_s3_bdtopo_data = environment["sources"]["BDTOPO"][2022]["guyane"]
path_local_pleiades_data = environment["local-path"]["PLEIADES"]
path_local_bdtopo_data = environment["local-path"]["BDTOPO"][2022]["guyane"]

fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"})

# %%
# DL PLEIADE
fs.download(
        rpath=f"{bucket}/{path_s3_pleiades_data}",
        lpath=f"../{path_local_pleiades_data}",
        recursive=True)

# DL BDTOPO
fs.download(
        rpath=f"{bucket}/{path_s3_bdtopo_data}",
        lpath=f"../{path_local_bdtopo_data}",
        recursive=True)

# Utilisation donnée pleiade 

# %%
filename = '../data/PLEIADES/Cayenne/16bits/ORT_2022072050325085_U22N/ORT_2022072050325085_0353_0545_U22N_16Bits.jp2'
date = datetime.strptime(re.search(r'ORT_(\d{8})', filename).group(1), '%Y%m%d')
date

image = SatelliteImage.from_raster(
        filename,
        date = date, 
        n_bands = 4,
        dep = "973"
    )
image.normalize()

list_images = image.split(250)
list_images[0].array
len(list_images)


# %%
image.plot([0,1,2])
image.plot([3,1,2])

# %% Représenter la liste des images sous forme de grille (ajouter une fonction)
SatelliteImage.plot_list_satellite_images(list_images,bands_indices = [3,1,2])

# %% Instanciation des labellers
from labeler import RILLabeler
labeler = RILLabeler(date, dep = "973", buffer_size= 10)
mask = labeler.create_segmentation_label(image) # va chercher les données et rasterise les bounding box
image.normalize
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(np.transpose(image.array_to_plot, (1, 2, 0))[:,:,:3])
ax.imshow(mask, alpha=0.3) # magnifique ! Faire une fonction  de représentation  ? # faire du découpage ? ou une fonction de sélection aléatoire d'une tuile ?


# %% Mont Baduel

filename = '../data/PLEIADES/Cayenne/16bits/ORT_2022072050325085_U22N/ORT_2022072050325085_0354_0545_U22N_16Bits.jp2'
date = datetime.strptime(re.search(r'ORT_(\d{8})', filename).group(1), '%Y%m%d')
date

image = SatelliteImage.from_raster(
        filename,
        date = date, 
        n_bands = 4,
        dep = "973"
    )

image.plot([3,1,2])
SatelliteImage.plot_list_satellite_images(image.split(250),[0,1,2])


# %%
from labeler import BDTOPOLabeler
labeler_bdtopo = BDTOPOLabeler(date, dep = "973")

mask = labeler_bdtopo.create_segmentation_label(image)
if image.normalize == False:
    image.normalize
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(np.transpose(image.array_to_plot, (1, 2, 0))[:,:,:3])
ax.imshow(mask, alpha=0.3) # magnifique ! Faire une fonction  de représentatio



# %% Le mont Baduel

filename = '../data/PLEIADES/Cayenne/16bits/ORT_2022072050325085_U22N/ORT_2022072050325085_0354_0545_U22N_16Bits.jp2'
date = datetime.strptime(re.search(r'ORT_(\d{8})', filename).group(1), '%Y%m%d')
date

image = SatelliteImage.from_raster(
        filename,
        date = date, 
        n_bands = 4,
        dep = "973"
    )

image.plot([3,1,2])

mask = labeler_bdtopo.create_segmentation_label(image)

if image.normalize == False:
    image.normalize
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(np.transpose(image.array_to_plot, (1, 2, 0))[:,:,:3])
ax.imshow(mask, alpha=0.3) # m

# %% Les labelled satelliteimage
from labeled_satellite_image import SegmentationLabeledSatelliteImage

#image.normalize()
image_labellisee = SegmentationLabeledSatelliteImage(image,label = mask, labeling_date = date, source = "BDTOPO")

# 1) plot image et masque superposés
image_labellisee.plot([0,1,2])

# 2) plot image et masque associés côte à côte
image_labellisee.plot_label_next_to_image([0,1,2])

liste_image_labelisee = image_labellisee.split(250)

# plot la liste d'images labellisées
## Ici on recolle les morceaux en partant de la liste
SegmentationLabeledSatelliteImage.plot_list_segmentation_labeled_satellite_image(liste_image_labelisee,[0,1,2])


