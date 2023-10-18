import os
import random
from typing import Dict, List

import albumentations as album
import yaml
from albumentations.pytorch.transforms import ToTensorV2


def select_indices_to_split_dataset(config_task, prop_val, list_labels):
    """
    Selects indices to split a dataset into training and validation sets based
    on the configuration task.

    Args:
        config_task (str): The configuration task.
        prop_val (float): The proportion of indices to allocate for the
        validation set.
        list_labels (list): The list of labels for each data point.

    Returns:
        tuple: A tuple containing two lists - train_indices and val_indices.
            train_indices (list): The selected indices for the training set.
            val_indices (list): The selected indices for the validation set.
    """
    len_dataset = len(list_labels)

    if config_task != "classification":
        num_val_indices = int(prop_val * len_dataset)

        all_indices = list(range(len_dataset))
        random.shuffle(all_indices)

        # Split the shuffled list into train and validation indices
        val_indices = all_indices[:num_val_indices]
        train_indices = all_indices[num_val_indices:]

    elif config_task == "classification":
        # Separating indices based on labels
        zero_indices = [i for i, label in enumerate(list_labels) if label == 0.0]
        one_indices = [i for i, label in enumerate(list_labels) if label == 1.0]

        # Randomly shuffle the indices
        random.shuffle(zero_indices)
        random.shuffle(one_indices)

        # Calculate the number of indices for each class in the validation set
        num_val_zeros = int(prop_val * len(zero_indices))
        num_val_ones = int(prop_val * len(one_indices))

        # Select indices for the validation set
        val_indices = zero_indices[:num_val_zeros] + one_indices[:num_val_ones]

        # Select indices for the training set
        train_indices = zero_indices[num_val_zeros:] + one_indices[num_val_ones:]

        # Randomly shuffle the training and validation indices
        random.shuffle(train_indices)
        random.shuffle(val_indices)

    return train_indices, val_indices


def select_indices_to_balance(list_path_images: List, balancing_dict: Dict, prop: float):
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
    for idx, filepath in enumerate(list_path_images):
        basename = os.path.basename(filepath).split(".")[0]
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
        list_to_add = random.sample(idx_no_building, lenght_unlabelled)
        for i in list_to_add:
            idx = idx_no_building.index(i)
            idx_balanced.append(i)
    else:
        idx_balanced.extend(idx_no_building)
    return idx_balanced


def generate_transform_pleiades(tile_size, augmentation, task):
    """
    Generates PyTorch transforms for data augmentation and preprocessing\
        for PLEIADES images.

    Args:
        tile_size (int): The size of the image tiles.
        augmentation (bool): Whether or not to include data augmentation.
        task (str): Task.

    Returns:
        (albumentations.core.composition.Compose,
        albumentations.core.composition.Compose):
        A tuple containing the augmentation and preprocessing transforms.

    """
    image_size = (tile_size, tile_size)

    bbox_params = None
    if task == "detection":
        bbox_params = album.BboxParams(format="pascal_voc", label_fields=["class_labels"])

    transforms_preprocessing = album.Compose(
        [
            # album.Resize(*image_size, always_apply=True),
            album.Normalize(),
            ToTensorV2(),
        ],
        bbox_params=bbox_params,
    )

    if augmentation:
        transforms_augmentation = album.Compose(
            [
                # album.Resize(300, 300, always_apply=True),
                album.RandomResizedCrop(*image_size, scale=(0.7, 1.0), ratio=(0.7, 1)),
                album.HorizontalFlip(),
                album.VerticalFlip(),
                album.Normalize(),
                ToTensorV2(),
            ],
            bbox_params=bbox_params,
        )
    else:
        transforms_augmentation = transforms_preprocessing

    return transforms_augmentation, transforms_preprocessing


def generate_transform_sentinel(src, year, dep, tile_size, augmentation, task):
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
    # TODO: normalization functions only when 13 bands are used,
    # change to make it work for less
    with open("./src/utils/normalize_sentinel.yml", "r") as stream:
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
                album.RandomResizedCrop(*image_size, scale=(0.7, 1.0), ratio=(0.7, 1)),
                album.HorizontalFlip(),
                album.VerticalFlip(),
                album.Normalize(mean, std),
                ToTensorV2(),
            ]
        )
    else:
        transforms_augmentation = transforms_preprocessing

    return transforms_augmentation, transforms_preprocessing


def generate_transform(tile_size, augmentation, task: str):
    """
    Generates PyTorch transforms for data augmentation and preprocessing.

    Args:
        tile_size (int): The size of the image tiles.
        augmentation (bool): Whether or not to include data augmentation.
        task (str): Task.

    Returns:
        (albumentations.core.composition.Compose,
        albumentations.core.composition.Compose):
        A tuple containing the augmentation and preprocessing transforms.

    """
    image_size = (tile_size, tile_size)

    transforms_augmentation = None

    if augmentation:
        transforms_list = [
            album.Resize(300, 300, always_apply=True),
            album.RandomResizedCrop(*image_size, scale=(0.7, 1.0), ratio=(0.7, 1)),
            album.HorizontalFlip(),
            album.VerticalFlip(),
            album.Normalize(),
            ToTensorV2(),
        ]
        if task == "detection":
            transforms_augmentation = album.Compose(
                transforms_list,
                bbox_params=album.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
            )
        else:
            transforms_augmentation = album.Compose(transforms_list)

    test_transforms_list = [
        album.Resize(*image_size, always_apply=True),
        album.Normalize(),
        ToTensorV2(),
    ]
    if task == "detection":
        transforms_preprocessing = album.Compose(
            test_transforms_list,
            bbox_params=album.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        )
    else:
        transforms_preprocessing = album.Compose(test_transforms_list)

    return transforms_augmentation, transforms_preprocessing
