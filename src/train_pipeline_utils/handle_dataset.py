import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import random_split


def instantiate_dataset_test(config):
    # charger les exemples tests sur le datalab
    # change detection ou segmentation et / Sentinele 2 / PLeiade  et ho
    # récupérer tout le jeu de test
    # le splitter et le laisser dans l'ordre
    # save les noms etc et s'arranger pour reconstruire les masques totaux
    # sur grosses images
    return None


def split_dataset(dataset, prop_val):
    """
    Splits a given dataset into training and 
    validation sets based on a given proportion.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        prop_val (float): The proportion of the dataset to use for validation,
        should be between 0 and 1.

    Returns:
        (torch.utils.data.Dataset, torch.utils.data.Dataset):
        A tuple containing the training and validation datasets.

    """
    val_size = int(prop_val * len(dataset))
    train_size = len(dataset) - val_size

    dataset_train, dataset_val = random_split(dataset, [train_size, val_size])

    dataset_train = dataset_train.dataset
    dataset_val = dataset_val.dataset

    return dataset_train, dataset_val


def generate_transform(tile_size, augmentation):
    """
    Generates PyTorch transforms for data augmentation and preprocessing.

    Args:
        tile_size (int): The size of the image tiles.
        augmentation (bool): Whether or not to include data augmentation.

    Returns:
        (albumentations.core.composition.Compose,
        albumentations.core.composition.Compose):
        A tuple containing the augmentation and preprocessing transforms.

    """
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
