"""
"""
import os
from s3fs import S3FileSystem
from pathlib import Path
from affine import Affine
from typing import List, Tuple, Optional


def get_root_path() -> Path:
    """
    Return root path of project.

    Returns:
        Path: Root path.
    """
    return Path(__file__).parent.parent


def get_file_system() -> S3FileSystem:
    """
    Return the s3 file system.
    """
    return S3FileSystem(
        client_kwargs={"endpoint_url": "https://" + os.environ["AWS_S3_ENDPOINT"]},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def get_transform_for_tiles(transform: Affine, row_off: int, col_off: int) -> Affine:
    """
    Compute the transform matrix of a tile

    Args:
        transform (Affine): an affine transform matrix.
        row_off (int): _description_
        col_off (int): _description_

    Returns:
        Affine: The affine transform matrix for the given tile
    """

    x, y = transform * (col_off, row_off)
    return Affine.translation(x - transform.c, y - transform.f) * transform


def get_bounds_for_tiles(
    transform: Affine, row_indices: Tuple, col_indices: Tuple
) -> Tuple:
    """
    Given an Affine transformation, and indices for a tile's row and column, returns the bounding coordinates (left, bottom, right, top) of the tile.

    Args:
        transform: An Affine transformation
        row_indices (Tuple): A tuple containing the minimum and maximum indices for the tile's row
        col_indices (Tuple): A tuple containing the minimum and maximum indices for the tile's column

    Returns:
        Tuple: A tuple containing the bounding coordinates (left, bottom, right, top) of the tile
    """

    row_min = row_indices[0]
    row_max = row_indices[1]
    col_min = col_indices[0]
    col_max = col_indices[1]

    left, bottom = transform * (col_min, row_max)
    right, top = transform * (col_max, row_min)
    return left, bottom, right, top


def get_indices_from_tile_length(m: int, n: int, tile_length: int) -> List:
    """
    Given the dimensions of an original image and a desired tile length, this function returns a list of tuples, where each tuple contains the border indices of a tile that can be extracted from the original image.
    The function raises a ValueError if the size of the tile is larger than the size of the original image.

    Args:
        m (int): Height of the original image
        n (int): Width of the original image
        tile_length (int): Dimension of tiles

    Returns:
        List: A list of tuples, where each tuple contains the border indices of a tile that can be extracted from the original image
    """

    if (tile_length > m) | (tile_length > n):
        raise ValueError(
            "The size of the tile should be smaller than the size of the original image."
        )

    indices = [
        ((m - tile_length, m), (n - tile_length, n))
        if (x + tile_length > m) & (y + tile_length > n)
        else ((x, x + tile_length), (y, y + tile_length))
        if (x + tile_length <= m) & (y + tile_length <= n)
        else ((m - tile_length, m), (y, y + tile_length))
        if (x + tile_length > m) & (y + tile_length <= n)
        else ((x, x + tile_length), (n - tile_length, n))
        for x in range(0, m, tile_length)
        for y in range(0, n, tile_length)
    ]
    return indices
