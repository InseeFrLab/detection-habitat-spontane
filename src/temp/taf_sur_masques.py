
import torch
import pytorch_lightning    
from models.segmentation_module import SegmentationModule
import matplotlib.pyplot as plt
from classes.data.satellite_image import *
from models.components.segmentation_models import DeepLabv3Module
import pickle

from classes.data.labeled_satellite_image import SegmentationLabeledSatelliteImage
from utils.plot_utils import plot_list_segmentation_labeled_satellite_image

# mc cp s3/projet-slums-detection/mlflow-artifacts/1/6a8f70d2eb8644b79b006b963a45e425/artifacts/restored_model_checkpoint/epoch=28-step=23113.ckpt model.ckpt

# je le télécharge en mc cp via le lien copié sur mlflow !!
optimizer = torch.optim.SGD
optimizer_params = {"lr": 0.0001, "momentum": 0.9}
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
scheduler_params = {}
scheduler_interval = "epoch"

model = DeepLabv3Module()

##Instanciation des datamodule et plmodule

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
checkpoint_path='model.ckpt',
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

# load_test 
si1 = SatelliteImage.from_raster(
    "../donnees-test/segmentation/images/ORT_2017_0523_8591_U38S_8Bits.jp2",
    dep = "976",
    n_bands = 3
    )

si1.plot([0,1,2])
res = plt.gcf()
res.savefig("imaget.png")

si2 = SatelliteImage.from_raster(
    "../donnees-test/segmentation/images/mayotte-ORT_2020052526670967_0523_8591_U38S_8Bits.jp2",
    dep = "976",
    n_bands = 3
    )
    
si2.plot([0,1,2])
res = plt.gcf()
res.savefig("imagetplus1.png")


model.eval()

lsi1 = si1.split(250)
liste_res1 = []

compteur = 0
for si1 in lsi1:
   print(compteur)
   compteur+=1
   out1 = model(torch.tensor(si1.array, dtype = torch.float).unsqueeze(0))
   liste_res1.append(out1)
   del out1

with open('res1.pkl', 'wb') as file:
    pickle.dump(liste_res1, file)
del liste_res1

# image 2
lsi2 = si2.split(250)
liste_res2 = []
compteur =0
for si2 in lsi2:
   print(compteur)
   compteur+=1
   out2 = model(torch.tensor(si2.array, dtype = torch.float).unsqueeze(0))
   liste_res2.append(out2)
   del out2

with open('res2.pkl', 'wb') as file:
    pickle.dump(liste_res2, file)
del liste_res2

# comaraison
with open('res1.pkl', 'rb') as file:
    liste_res1 = pickle.load(file)

with open('res2.pkl', 'rb') as file:
    liste_res2 = pickle.load(file)

# masques normaux
# différence entre les probas affichées
seuil = 0.001
seuil = 0.01
seuil =  0.1
seuil = 0.2
seuil = 0.25
seuil = 0.35


list_labeled_si =[]
seuil = 2000
for res1,res2,si1,si2 in zip(liste_res1,liste_res2,lsi1,lsi2):
    # res1 = liste_res1[4]
    # res2 = liste_res2[4]
    # diff_proba_quadra = (torch.softmax(res1.squeeze(0), dim= 0)[0]- torch.softmax(res2.squeeze(0), dim= 0)[0])**2
    diff_proba_quadra = (res1.squeeze(0)[0]/res1.squeeze(0)[1]- res2.squeeze(0)[0]/res2.squeeze(0)[1])**2
    
    #np.quantile(diff_proba_quadra.detach(),[0.25,0.5,0.75,1])

    # calcul masque_diff
    diff_proba_quadra = np.array(diff_proba_quadra.detach())
    mask_diff = np.zeros(diff_proba_quadra.shape)
    mask_diff[diff_proba_quadra > seuil]=1

    list_labeled_si.append(SegmentationLabeledSatelliteImage(si1,mask_diff, None, None))


    # creation d'une liste labeled satellite image et plot, puis triple plot
plot_list_segmentation_labeled_satellite_image(list_labeled_si,[0,1,2])
res = plt.gcf()
res.savefig("diff_Exemple_seuil_" + str(seuil)+".png")


## 
list_mask1 = []
for res1 in liste_res1:
    list_mask1.append(np.array(torch.argmax(res1.squeeze(0).detach(),axis =0)))

list_mask2 = []
for res2 in liste_res2:
    list_mask2.append(np.array(torch.argmax(res2.squeeze(0).detach(),axis =0)))

seuil = 0.2
list_labeled_si = []
for mask1, maks2, si1 in zip(list_mask1,list_mask2,lsi1):
    # i  = 0
    # mask1 = list_mask1[i]
    # mask2 = list_mask2[i]
    prop1 = np.sum(mask1)/(250*250)
    prop2 = np.sum(mask2)/(250*250)


    if (prop1 > seuil and prop2 < seuil) or (prop1 < seuil and prop2 > seuil):
        label = np.ones([250,250])
    else:
        label = np.zeros([250,250])
    
    list_labeled_si.append(
        SegmentationLabeledSatelliteImage(si1,label, None, None)
        )

plot_list_segmentation_labeled_satellite_image(list_labeled_si, [0,1,2])    
res = plt.gcf()
res.savefig("diff_patch_" + str(seuil)+".png")





### masque normal
# t
list_labeled_si =[]
for mask1, si1 in zip(list_mask1,lsi1):
    list_labeled_si.append(SegmentationLabeledSatelliteImage(si1,mask1, None, None))

plot_list_segmentation_labeled_satellite_image(list_labeled_si,[0,1,2])
res = plt.gcf()
res.savefig("masque_t.png")

# t + 1
list_labeled_si =[]
for mask2, si2 in zip(list_mask2,lsi2):
    list_labeled_si.append(SegmentationLabeledSatelliteImage(si2,mask2, None, None))

plot_list_segmentation_labeled_satellite_image(list_labeled_si,[0,1,2])
res = plt.gcf()
res.savefig("masque_plus1.png")


### Différence !
list_labeled_si =[]
for mask1, mask2, si2 in zip(list_mask1,list_mask2,lsi2):
    mask_diff = np.zeros(mask1.shape)
    mask_diff[mask1 != mask2] = 1
    list_labeled_si.append(SegmentationLabeledSatelliteImage(si2,mask_diff, None, None))

plot_list_segmentation_labeled_satellite_image(list_labeled_si,[0,1,2])
res = plt.gcf()
res.savefig("diff_masque.png")


## lissage de la différence ?
