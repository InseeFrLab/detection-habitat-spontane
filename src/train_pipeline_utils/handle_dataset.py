import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2
import random


def select_indices_to_split_dataset(
    config_task, prop_val, list_labels
):
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
        zero_indices = [i for i, label in enumerate(list_labels) if label == "0"]
        one_indices = [i for i, label in enumerate(list_labels) if label == "1"]

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
