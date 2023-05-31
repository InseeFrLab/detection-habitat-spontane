import sys
sys.path.append('../src')
from utils.satellite_image import SatelliteImage
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
from utils.labeler import RILLabeler
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
from data.components.dataset import PleiadeDataset
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

path_s3_pleiades_data_2020_mayotte = environment["sources"]["PLEIADES"][2020]["976"]
path_local_pleiades_data_2020_mayotte = os.path.join(root_path,environment["local-path"]["PLEIADES"][2020]["976"])
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
        for i, satellite_image in enumerate(list_satellite_image):
                
                mask = labeler.create_segmentation_label(satellite_image) 
                file_name_i = file_name.split(".")[0]+"_"+str(i)
                if(np.sum(mask) == 0): # je dégage les masques vides j'écris pasd
                    continue
                to_raster(satellite_image,output_images_path,file_name_i + ".tif")
                np.save(output_masks_path+"/"+file_name_i+".npy",mask) # save


satellite_image = SatelliteImage.from_raster(
        file_path = f"{path_local_pleiades_data_2022_martinique}"+ "/ORT_2022_0696_1630_U20N_8Bits.jp2",
        dep = None,
        date = None,
        n_bands= 3)

print(satellite_image.array.shape)
i = 2
directory_name = "../bibi"
file_name = "ORT_2022072050325085_0352_0545_U22N_16Bits"+"_"+str(i)+".tif"
to_raster(satellite_image,directory_name,file_name)

satellite_image.plot([0,1,2])
res = plt.gcf()
res.savefig("coucou.png")


## Préparer les données
# params 
file_path = f"{path_local_pleiades_data_2022_martinique}"
tile_size = 250
n_bands = 3
dep ="972"

from datetime import datetime
from classes.labelers.labeler import RILLabeler

date = datetime.strptime("20220101",'%Y%m%d')
print(date)
labeler = RILLabeler(date, dep = dep, buffer_size = 10) 
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
dataset = PleiadeDataset(list_path_images,list_path_labels,n_bands = 3)


from train_pipeline_utils.handle_dataset import select_indices_to_split_dataset 
ind_train, ind_val = select_indices_to_split_dataset(len(os.listdir(output_directory_name+"/labels"))
, 0.2)

from torch.utils.data import DataLoader

labels_train = list_path_labels[ind_train]   
labels_val = list_path_labels[ind_val]
images_train = list_path_images[ind_train]   
images_val = list_path_images[ind_val]

train_dataloader = DataLoader(PleiadeDataset(images_train,labels_train,3),batch_size=9)
valid_dataloader = DataLoader(PleiadeDataset(images_val,labels_val,3), batch_size=9)


# liste d'images test à tester 
satellite_image = SatelliteImage.from_raster(
#file_path = "../data/PLEIADES/2020/MAYOTTE/ORT_2020052526670967_0523_8591_U38S_8Bits.jp2",
#file_path = "../data/PLEIADES/2020/MAYOTTE/ORT_2020052526670967_0524_8590_U38S_8Bits.jp2",
file_path = "../data/PLEIADES/2020/MAYOTTE/ORT_2020052526670967_0519_8586_U38S_8Bits.jp2",
#file_path =  "../data/PLEIADES/2020/MAYOTTE/ORT_2020052526670967_0524_8590_U38S_8Bits.jp2",
# "../data/PLEIADES/2022/MARTINIQUE/ORT_2022_0691_1638_U20N_8Bits.jp2",
dep = None,
date = None,
n_bands= 3) # ../data/PLEIADES/2020/MAYOTTE/ORT_2020052526670967_0524_8590_U38S_8Bits.jp2 fonctions d'affichage de Raya
#satellite_image.plot([0,1,2])
#res = plt.gcf()
#res.savefig("imagetest.png")


list_test = satellite_image.split(250)
directory_image_name = "../test_data/images/"

for i,simg in enumerate(list_test):
    file_name = simg.filename.split(".")[0] +"_"+str(i)+".tif"
    to_raster(simg,directory_image_name,file_name)

list_path_images_test = np.sort([directory_image_name + filename for filename in os.listdir(directory_image_name)])
list_path_labels_test =   np.sort(list_path_labels[:len(list_path_images_test)])# pas propre je metsd une liste de labels de même taille inutilisés

dataset_test = PleiadeDataset(list_path_images_test,list_path_labels_test,n_bands = 3) 


# entrainement

image_size = (250,250)
transforms_preprocessing = album.Compose(
        [
            album.Resize(*image_size, always_apply=True),
            album.Normalize(),
            ToTensorV2(),
        ]
)



## Instanciation modèle et paramètres d'entraînement

optimizer = torch.optim.SGD
optimizer_params = {"lr": 0.0001, "momentum": 0.9}
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
scheduler_params = {}
scheduler_interval = "epoch"

model = DeepLabv3Module()



##Instanciation des datamodule et plmodule

from torch.nn import CrossEntropyLoss

lightning_module = SegmentationModule(
    model=model,
    optimizer=optimizer,
    optimizer_params=optimizer_params,
    scheduler=scheduler,
    scheduler_params=scheduler_params,
    scheduler_interval=scheduler_interval,
    loss = CrossEntropyLoss()
)

checkpoint_callback = ModelCheckpoint(
    monitor="validation_IOU", save_top_k=1, save_last=True, mode="max"
)

early_stop_callback = EarlyStopping(
    monitor="validation_loss", mode="min", patience=3
)
lr_monitor = LearningRateMonitor(logging_interval="step")

strategy ="auto"
list_callbacks = [lr_monitor, checkpoint_callback, early_stop_callback]

mlflow.end_run()

run_name = "modele deeplabV44"
remote_server_uri = "https://projet-slums-detection-874257.user.lab.sspcloud.fr"
experiment_name = "segmentation"

mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment(experiment_name)
mlflow.pytorch.autolog()

with mlflow.start_run(run_name=run_name):

    trainer = pl.Trainer(
    callbacks= list_callbacks,
    max_epochs=50,
    num_sanity_val_steps=2,
    strategy=strategy,
    log_every_n_steps=2,
    accumulate_grad_batches = 1
    )
    trainer.fit(lightning_module, train_dataloader , valid_dataloader)

    lightning_module = SegmentationModule(
    model=model,
    loss = None,
    optimizer=optimizer,
    optimizer_params=optimizer_params,
    scheduler=scheduler,
    scheduler_params=scheduler_params,
    scheduler_interval=scheduler_interval,
    )


    lightning_module_checkpoint = lightning_module.load_from_checkpoint(
    checkpoint_path='lightning_logs/version_2/checkpoints/epoch=13-step=9926.ckpt',
    model= model,
    optimizer=optimizer,
    optimizer_params=optimizer_params,
    scheduler=scheduler,
    scheduler_params=scheduler_params,
    scheduler_interval=scheduler_interval,
    map_location= torch.device('cpu'),
    loss = None
                                            )
    
    model = lightning_module_checkpoint.model

from classes.optim.evaluation_model import evaluer_modele_sur_jeu_de_test_segmentation_pleiade

dataset_test.transforms = transforms_preprocessing
test_dl =DataLoader(dataset_test,batch_size = 4)
model = model.eval() 
evaluer_modele_sur_jeu_de_test_segmentation_pleiade(test_dl, model, 250, 4)
# TO DO, test en normalisant le data set et test sans le normaliser..
#