"""
"""
import os
from s3fs import S3FileSystem
from pathlib import Path
from affine import Affine


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
    """Compute the transform matrix of a tile

    Args:
        transform (Affine): an affine transform matrix.
        row_off (int): _description_
        col_off (int): _description_

    Returns:
        Affine: The affine transform matrix for the given tile
    """
    x, y = transform * (col_off, row_off)
    return Affine.translation(x - transform.c, y - transform.f) * transform


def get_bounds_for_tiles(transform: Affine, row_idx: Tupple, col_idx: Tupple) -> Tupple:

    row_min = row_idx[0]
    row_max = row_idx[1]
    col_min = col_idx[0]
    col_max = col_idx[1]

    left, bottom = transform * (col_min, row_max)
    right, top = transform * (col_max, row_min)
    return left, bottom, right, top


def get_indices_from_tile_length(m: int, n: int, tile_length: int) -> List:
    """Return the indices of tiles

    Args:
        m (int): Height of the original image
        n (int): Width of the original image
        tile_length (int): Dimension of tiles

    Returns:
        List: Splitted SatelliteImage of dimension `tile_length`
    """
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
