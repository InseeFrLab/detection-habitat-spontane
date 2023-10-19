# Utilisation de données satellites pour détecter l'habitat spontané

[![Onyxia](https://img.shields.io/static/v1?logo=visualstudiocode&label=&message=Open%20in%20VS%20Code&labelColor=2c2c32&color=007acc&logoColor=007acc)](https://datalab.sspcloud.fr/launcher/ide/vscode-python?autoLaunch=false&onyxia.friendlyName=%C2%ABslums-detection%C2%BB&init.personalInit=%C2%ABhttps%3A%2F%2Fraw.githubusercontent.com%2FInseeFrLab%2Fdetection-bidonvilles%2Fmain%2Fsetup.sh%C2%BB&service.image.custom.enabled=true&service.image.custom.version=%C2%ABinseefrlab%2Fdetection-bidonvilles%3Av0.0.2%C2%BB&persistence.size=%C2%AB80Gi%C2%BB)
[![Build](https://img.shields.io/github/actions/workflow/status/InseeFrLab/detection-bidonvilles/build-image.yaml?label=Build
)](https://hub.docker.com/repository/docker/inseefrlab/detection-bidonvilles)


## Setup

Il faut une installation de `GDAL` :

```
sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
sudo apt-get update
sudo apt-get install gdal-bin
sudo apt-get install libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
```

Les dépendances Python à installer se trouvent dans le fichier `requirements.txt`. Pour contribuer,

```
pre-commit install
```

permet de bénéficier de plusieurs hooks.

## Pipeline d'entraînement

Le fichier `src/run_training_pipeline.py` contient le pipeline entier d'entraînement de modèles. Pour lancer un entraînement, définir le fichier `config.yaml` puis exécuter la commande

```
python src/run_training_pipeline.py <remote_server_uri> <experiment_name> <run_name>
```

où `<remote_server_uri>` correspond à l'adresse du serveur MLflow, et `<experiment_name>` et `<run_name>` correspondent au nom d'*experiment* et au nom de *run* choisis.

## Configuration

Les paramètres à spécifier dans le fichier de configuration sont :

- `data`:
  - `source_train`: Source des données satellites: "PLEIADES", "SENTINEL2" ou "SENTINEL1-2"
  - `dataset`: Nom du `Dataset` à utiliser pour l'entraînement du modèle: "PLEIADE" (`PleiadeDataset`), "CLASSIFICATION" (`PatchClassification`), "SENTINEL" (`SentinelDataset`), "CHANGEISEVERYWHERE" (`ChangeIsEverywhereDataset`), "CHANGEDETECTIONDATASET" (`ChangeDetectionDataset`)
  - `dataset_test`: Nom du `Dataset` à utiliser pour le test
  - `task`: Tâche: "segmentation", "classification", "change-detection", "detection"
  - `dep`: Département(s) utilisé(s) pour l'entraînement. A fournir sous la forme d'une liste (par exemple `["971", "972"]`).
  - `year`: Année(s) utilisée(s) pour l'entraînement. A fournir sous la forme d'une liste (par exemple `["2021", "2021"]`).
  - `type_labeler`: Type d'annotateur utilisé : "BDTOPO" ou "RIL"
  - `buffer_size`: Si le RIL est utilisé pour annoter, taille des buffers créés autour des points RIL
  - `n_channels_train`: Nombre de canaux en entrée du modèle
  - `n_bands`: Nombre de bandes des images en entrée de pipeline
  - `tile_size`: Taille des images en entrée du modèle
  - `augmentation`: Booléen, `True` si on incorpore de l'augmentation
  - `prop`: Dans l'entraînement, rapport entre le nombre d'images ne contenant pas de bâtiment et le nombre d'images en contenant au moins un
  - `num_workers`: Nombre de workers utilisés par le `DataLoader`

- `optim`:
  - `loss`: Fonction de perte à utiliser: "softiou" (`SoftIoULoss`), "crossentropy" (`CrossEntropyLoss`), "crossentropyselmade" (`CrossEntropySelfmade`), "lossbinaire" (`nn.BCELoss`)
  - `lr`: Learning rate initiale
  - `momentum`: Momentum
  - `module`: Modèle: "fasterrcnn", "deeplabv3" (`DeepLabv3Module`), "resnet50" (`ResNet50Module`)
  - `batch_size`: Batch size
  - `batch_size_test`: Batch size pour le test
  - `max_epochs`: Nombre d'epochs maximum
  - `val_prop`: Pourcentage du jeu de données d'entraînement initial utilisé comme validation
  - `accumulate_batch`: Nombre de gradients à [accumuler](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html)
  - `monitor`: Métrique à monitorer pendant l'entraînement: "validation_accuracy" par exemple
  - `mode`: Mode de monitoring: "min" par exemple
  - `patience`: Patience pour l'early stopping

- `mlflow`: Booléen, indique si MLflow doit être utilisé
