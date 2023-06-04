import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2
import random


def select_indices_to_split_dataset(len_dataset, prop_val):

    num_val_indices = int(prop_val * len_dataset)
    
    all_indices = list(range(len_dataset))
    random.shuffle(all_indices)
    
    # Split the shuffled list into train and validation indices
    val_indices = all_indices[:num_val_indices]
    train_indices = all_indices[num_val_indices:]
    
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
