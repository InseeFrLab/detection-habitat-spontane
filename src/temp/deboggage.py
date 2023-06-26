import sys
sys.path.append('../src')
from classes.data.satellite_image import SatelliteImage
from utils.utils import *
from utils.plot_utils import *

import yaml
import re
import s3fs
import numpy as np
import matplotlib.pyplot as plt
#import cv2
from PIL import Image as im

from datetime import date
import re
import pyproj
import os
from tqdm import tqdm
from classes.labelers.labeler import RILLabeler
from utils.filter import is_too_black

import pytorch_lightning as pl
import torch
import gc
import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

## from src
from datas.components.dataset import PleiadeDataset, ChangeDetectionS2LookingDataset
from models.components.segmentation_models import DeepLabv3Module
from models.segmentation_module import SegmentationModule
import mlflow

root_path = get_root_path()
root_path

update_storage_access()
environment = get_environment()

root_path = get_root_path()
bucket = environment["bucket"]
path_s3_cayenne_data = environment["sources"]["PLEIADES"][2022]["976"]
path_local_cayenne_data = os.path.join(root_path, environment["local-path"]["PLEIADES"][2022]["976"])

path_s3_pleiades_data_2022_martinique = environment["sources"]["PLEIADES"][2022]["972"]
path_local_pleiades_data_2022_martinique = os.path.join(root_path,environment["local-path"]["PLEIADES"][2022]["972"])

fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"})

# path_s3_pleiades_data_2020_mayotte = environment["sources"]["PLEIADES"][2020]["mayotte"]
# path_local_pleiades_data_2020_mayotte = os.path.join(root_path,environment["local-path"]["PLEIADES"][2020]["mayotte"])
# fs.download(
#         rpath=f"{bucket}/{path_s3_pleiades_data_2020_mayotte}",
#         lpath=f"{path_local_pleiades_data_2020_mayotte}",
#         recursive=True)

# fs.download(
#         rpath=f"{bucket}/{path_s3_cayenne_data}",
#         lpath=f"{path_local_cayenne_data}",
#         recursive=True)
# fs.download(
#         rpath=f"{bucket}/{path_s3_pleiades_data_2022_martinique}",
#         lpath=f"{path_local_pleiades_data_2022_martinique}",
#         recursive=True)

def to_raster(satellite_image,directory_name,file_name):
    """
    save a SatelliteImage Object into a raster file (.tif)

    Args:
        satellite_image: a SatelliteImage object representing the input image to be saved as a raster file.
        directory_name: a string representing the name of the directory where the output file should be saved.
        file_name: a string representing the name of the output file.
    """

    data = satellite_image.array
    crs  = satellite_image.crs
    transform = satellite_image.transform
    n_bands = satellite_image.n_bands

    metadata = {
        'dtype': data.dtype,
        'count': n_bands,
        'width': data.shape[2],
        'height': data.shape[1],
        'crs': crs,
        'transform': transform
    }
    
    #print(os.path.exists(directory_name))
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    # Save the array as a raster file in jp2 format
    with rasterio.open(directory_name + "/" + file_name, 'w', **metadata) as dst:
        dst.write(data, indexes = np.arange(n_bands)+1)


def write_splitted_images_masks(file_path,output_directory_name,labeler,tile_size,n_bands, dep):
    
    """
    write the couple images mask into a specific folder

    Args:
        file_path: a string representing the path to the directory containing the input image files.
        output_directory_name: a string representing the name of the output directory where the split images and masks should be saved.
        labeler: a Labeler object representing the labeler used to create segmentation labels.
        tile_size: an integer representing the size of the tiles to split the input image into.
        n_bands: an integer representing the number of bands in the input image.
        dep: a string representing the department of the input image, or None if not applicable.
    """
    
    output_images_path  = output_directory_name + "/images"
    output_masks_path  = output_directory_name + "/labels"
    
    if not os.path.exists(output_masks_path):
        os.makedirs(output_masks_path)
        
    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)
        
    list_name = os.listdir(file_path)
    list_path = [file_path + "/" + name for name in list_name]
    
    for path, file_name in zip(list_path,tqdm(list_name)): # tqdm ici 

        big_satellite_image = SatelliteImage.from_raster(
            file_path = path,
            dep = None,
            date = None,
            n_bands= 3
        )

        list_satellite_image = big_satellite_image.split(tile_size)
        list_satellite_image = [im for im in list_satellite_image if not is_too_black(im)]
        # mettre le filtre is too black ici !!!
        compteur_j =0
        for i, satellite_image in enumerate(list_satellite_image):
                
            mask = labeler.create_segmentation_label(satellite_image) 
            if(np.sum(mask) == 0): # je dégage les masques vides j'écris pasd
                continue
            file_name_j = file_name.split(".")[0]+"_"+ "{:03d}".format(compteur_j)
            compteur_j = compteur_j + 1
            to_raster(satellite_image,output_images_path, file_name_j + ".tif")
            np.save(output_masks_path+"/"+file_name_j+".npy",mask) # save


satellite_image = SatelliteImage.from_raster(
        file_path = f"{path_local_pleiades_data_2022_martinique}"+ "/ORT_2022_0696_1630_U20N_8Bits.jp2",
        dep = None,
        date = None,
        n_bands= 3)

satellite_image.plot([0,1,2])
res = plt.gcf()
res.savefig("coucou.png")
# print(satellite_image.array.shape)
# i = 2
# directory_name = "../bibi"
# file_name = "ORT_2022072050325085_0352_0545_U22N_16Bits"+"_"+str(i)+".tif"
# to_raster(satellite_image,directory_name,file_name)


## Préparer les données
# params 
file_path = f"{path_local_pleiades_data_2022_martinique}"
tile_size = 250
n_bands = 3
dep ="972"
from datetime import datetime
date = datetime.strptime("20220101",'%Y%m%d')
print(date)
labeler = RILLabeler(date, dep = dep, buffer_size = 10) 

#from classes.labelers.labeler import BDTOPOLabeler
#labeler = BDTOPOLabeler(date, dep = dep) 

output_directory_name = "../splitted_data2"
write_splitted_images_masks(file_path,output_directory_name,labeler,tile_size,n_bands,dep)

# 1 min pour 250 -> 4min pour 1000, ça se tente un peu lionguet mais 

len(os.listdir(output_directory_name+"/labels"))
len(os.listdir(output_directory_name+"/images"))

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://minio.lab.sspcloud.fr"

# dataset test
torch.cuda.empty_cache()
gc.collect()

dir_data = "../splitted_data2"
list_path_labels =  np.sort([dir_data + "/labels/" + name for name in os.listdir(dir_data+"/labels")])# os.wlk dans d'autres cas sous des sous arbres de S2Looking
list_path_images =  np.sort([dir_data + "/images/" + name for name in os.listdir(dir_data+"/images")])
dataset = PleiadeDataset(list_path_images,list_path_labels)




# comparaison des outputs dans les dossiers de train
import os
import numpy as np
list_image_now =np.sort(os.listdir("train_data-PLEIADES-RIL-972-2022eee"+"/images"))
list_image_before =np.sort(os.listdir("../splitted_data2"+"/images"))

len(list_image_before)
len(list_image_now)

list_image_now = [name.split(".")[0] for name in list_image_now]
list_image_before = [name.split(".")[0] for name in list_image_before]


images_avant_pas_apres = [im for im in list_image_before if im not in list_image_now]
images_apres_pas_avant = [im for im in list_image_now if im not in list_image_before]


len(images_avant_pas_apres) # la diff ça devrait être les nuages ?
len(images_apres_pas_avant)

exemple = "../splitted_data2"+"/images/"+images_avant_pas_apres[20]+".tif"

from classes.data.satellite_image import SatelliteImage
satellite_image = SatelliteImage.from_raster(
        file_path = exemple,
        dep = None,
        date = None,
        n_bands= 3)

import matplotlib.pyplot as plt
satellite_image.plot([0,1,2])
res = plt.gcf()
res.savefig("exemple2.png")
len(os.listdir(output_directory_name+"/images"))

### Observation des résultats de la détection de changements
# Load the checkpoint file
