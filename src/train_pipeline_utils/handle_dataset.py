import random
from typing import List, Dict
import albumentations as album
import yaml
import os
from albumentations.pytorch.transforms import ToTensorV2


def select_indices_to_split_dataset(len_dataset, prop_val):
    num_val_indices = int(prop_val * len_dataset)

    all_indices = list(range(len_dataset))
    random.shuffle(all_indices)

    # Split the shuffled list into train and validation indices
    val_indices = all_indices[:num_val_indices]
    train_indices = all_indices[num_val_indices:]

    return train_indices, val_indices


def select_indices_to_balance(
    list_path_images: List,
    balancing_dict: Dict,
    prop: float
):
    """
    Select indices to balance Dataset according to a balancing dict
    containing info on the images that have buildings or not.

    Args:
        list_path_images (List): List of image paths.
        balancing_dict (Dict): Balancing dict.
        prop (float): the proportion of the images without label
            (0 for no empty data, 1 for an equal proportion between
            empty and no empty data, etc)
    """
    idx_building = []
    idx_no_building = []
    for idx, filepath in list_path_images:
        basename = os.path.basename(filepath)
        if balancing_dict[basename] == 1:
            idx_building.append(idx)
        else:
            idx_no_building.append(idx)

    # Get images with buildings and without according
    # to a certain proportion
    length_labelled = len(idx_building)
    lenght_unlabelled = prop * length_labelled
    idx_balanced = idx_building.copy()
    if lenght_unlabelled < len(idx_no_building):
        list_to_add = random.sample(
            idx_no_building,
            lenght_unlabelled
        )
        for i in list_to_add:
            idx = idx_no_building.index(i)
            idx_balanced.append(i)
    else:
        idx_balanced.extend(
            idx_no_building
        )
    return idx_balanced


def generate_transform_pleiades(tile_size, augmentation):
    """
    Generates PyTorch transforms for data augmentation and preprocessing\
        for PLEIADES images.

    Args:
        tile_size (int): The size of the image tiles.
        augmentation (bool): Whether or not to include data augmentation.

    Returns:
        (albumentations.core.composition.Compose,
        albumentations.core.composition.Compose):
        A tuple containing the augmentation and preprocessing transforms.

    """
    image_size = (tile_size, tile_size)

    transforms_preprocessing = album.Compose(
        [
           #album.Resize(*image_size, always_apply=True),
            album.Normalize(),
            ToTensorV2(),
        ]
    )

    if augmentation:
        transforms_augmentation = album.Compose(
            [
                #album.Resize(300, 300, always_apply=True),
                album.RandomResizedCrop(
                    *image_size, scale=(0.7, 1.0), ratio=(0.7, 1)
                ),
                album.HorizontalFlip(),
                album.VerticalFlip(),
                album.Normalize(),
                ToTensorV2(),
            ]
        )
    else:
        transforms_augmentation = transforms_preprocessing

    return transforms_augmentation, transforms_preprocessing


def generate_transform_sentinel(src, year, dep, tile_size, augmentation):
    """
    Generates PyTorch transforms for data augmentation and preprocessing\
        for SENTINEL2 images.

    Args:
        tile_size (int): The size of the image tiles.
        augmentation (bool): Whether or not to include data augmentation.

    Returns:
        (albumentations.core.composition.Compose,
        albumentations.core.composition.Compose):
        A tuple containing the augmentation and preprocessing transforms.

    """
    with open("utils/normalize_sentinel.yml", "r") as stream:
        normalize_sentinel = yaml.safe_load(stream)
    mean = eval(normalize_sentinel[src]["mean"][year][dep])
    std = eval(normalize_sentinel[src]["std"][year][dep])

    image_size = (tile_size, tile_size)

    transforms_preprocessing = album.Compose(
        [
            album.Resize(*image_size, always_apply=True),
            album.Normalize(mean, std),
            ToTensorV2(),
        ]
    )

    if augmentation:
        transforms_augmentation = album.Compose(
            [
                album.Resize(300, 300, always_apply=True),
                album.RandomResizedCrop(
                    *image_size, scale=(0.7, 1.0), ratio=(0.7, 1)
                ),
                album.HorizontalFlip(),
                album.VerticalFlip(),
                album.Normalize(mean, std),
                ToTensorV2(),
            ]
        )
    else:
        transforms_augmentation = transforms_preprocessing

    return transforms_augmentation, transforms_preprocessing
