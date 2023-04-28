import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import random_split
from data.components.dataset import PleiadeDataset


def instanciate_dataset(
        config,
        list_path_images,
        list_path_labels
):

    dataset_dict = {"PLEIADE": PleiadeDataset}
    dataset_type = config["donnees"]["source train"]

    # inqtanciation du dataset comple
    if dataset_type not in dataset_dict:
        raise ValueError("Invalid dataset type")
    else:
        full_dataset = dataset_dict[dataset_type](
            list_path_images,
            list_path_labels
            )

    return full_dataset


def instanciate_dataset_test(config):
    # charger les exemples tests sur le datalab
    # change detection ou segmentation et / Sentinele 2 / PLeiade  et ho
    # récupérer tout le jeu de test
    # le splitter et le laisser dans l'ordre
    # save les noms etc et s'arranger pour reconstruire les masques totaux
    # sur grosses images
    return None


def split_dataset(dataset, prop_val):

    val_size = int(prop_val * len(dataset))
    train_size = len(dataset) - val_size

    dataset_train, dataset_val = random_split(
        dataset, [train_size, val_size]
    )

    dataset_train = dataset_train.dataset
    dataset_val = dataset_val.dataset

    return dataset_train, dataset_val


def generate_transform(tile_size, augmentation):

    image_size = (tile_size, tile_size)

    transforms_augmentation = None

    if augmentation:
        transforms_augmentation = album.Compose(
                [
                    album.Resize(300, 300, always_apply=True),
                    album.RandomResizedCrop(
                        *image_size, scale=(0.7, 1.0), ratio=(0.7, 1)
                    ),
                    album.HorizontalFlip(),
                    album.VerticalFlip(),
                    album.Normalize(),
                    ToTensorV2(),
                ]
            )

    transforms_preprocessing = album.Compose(
            [
                album.Resize(*image_size, always_apply=True),
                album.Normalize(),
                ToTensorV2(),
            ]
        )

    return transforms_augmentation, transforms_preprocessing
