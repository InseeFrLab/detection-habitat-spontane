"""
"""
import os
from s3fs import S3FileSystem
from pathlib import Path
from affine import Affine
from typing import List, Tuple, Dict
from datetime import datetime
import geopandas as gpd
import yaml
import rasterio
import hvac


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
    return rasterio.coords.BoundingBox(left, bottom, right, top)


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


def load_ril(datetime: datetime) -> gpd.GeoDataFrame:
    """
    Load RIL for a given datetime.

    Args:
        datetime (datetime): Date of labeling data.

    Returns:
        gpd.GeoDataFrame: RIL GeoDataFrame.
    """
    environment = get_environment()
    fs = get_file_system()

    # For now only one version of RIL.
    with fs.open(
        os.path.join(environment["bucket"], environment["sources"]["RIL"])
    ) as f:
        df = gpd.read_file(f)

    return df


def load_bdtopo(datetime: datetime) -> gpd.GeoDataFrame:
    """
    Load BDTOPO for a given datetime.

    Args:
        datetime (datetime): Date of labeling data.

    Returns:
        gpd.GeoDataFrame: BDTOPO GeoDataFrame.
    """
    root_path = get_root_path()
    environment = get_environment()

    dir_path = os.path.join(
        root_path,
        environment["local-path"]["BDTOPO"][2022]["guyane"],
    )

    file_path = None
    for root, dirs, files in os.walk(dir_path):
        if "BATIMENT.shp" in files:
            file_path = os.path.join(root, "BATIMENT.shp")
    if not file_path:
        raise ValueError("No valid `BATIMENT.shp` file found.")

    df = gpd.read_file(file_path)

    return df


def get_environment() -> Dict:
    """
    Get environment dictionary from `environment.yml` file.

    Returns:
        Dict: Environment dictionary.
    """
    root_path = get_root_path()
    with open(os.path.join(root_path, "environment.yml"), "r") as stream:
        environment = yaml.safe_load(stream)
    return environment


def update_storage_access():
    """
    This function updates the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables with values obtained from a HashiCorp Vault server.
    The Vault server URL, token, and secret path are taken from the VAULT_TOKEN and VAULT_MOUNT+VAULT_TOP_DIR/s3 environment variables.
    If AWS_SESSION_TOKEN is present, it will be deleted.
    """

    client = hvac.Client(
        url="https://vault.lab.sspcloud.fr", token=os.environ["VAULT_TOKEN"]
    )

    secret = os.environ["VAULT_MOUNT"] + os.environ["VAULT_TOP_DIR"] + "/s3"
    mount_point, secret_path = secret.split("/", 1)
    secret_dict = client.secrets.kv.read_secret_version(
        path=secret_path, mount_point=mount_point
    )

    os.environ["AWS_ACCESS_KEY_ID"] = secret_dict["data"]["data"]["ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_dict["data"]["data"][
        "SECRET_ACCESS_KEY"
    ]
    try:
        del os.environ["AWS_SESSION_TOKEN"]
    except KeyError:
        pass
