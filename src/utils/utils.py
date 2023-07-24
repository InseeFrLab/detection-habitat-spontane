"""
Utils.
"""
import os
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import geopandas as gpd
import hvac
import pyarrow.parquet as pq
import rasterio
import yaml
from affine import Affine
from s3fs import S3FileSystem

from .mappings import dep_to_crs


def remove_dot_file(list_name):
    """
    Removes filenames starting with a dot from the given list.

    Args:
        list_name (list): A list of filenames.

    Returns:
        list: The modified list with dot filenames removed.
    """
    for filename in list_name:
        if filename[0] == ".":
            list_name.remove(filename)

    return list_name


def split_array(array, tile_length):
    """
    Splits an array into smaller tiles of the specified length.

    Args:
        array (numpy.ndarray): The input array.
        tile_length (int): The length of each tile.

    Returns:
        list: A list of smaller tiles obtained from the input array.
    """

    m = array.shape[0]
    n = array.shape[1]

    indices = get_indices_from_tile_length(m, n, tile_length)

    list_array = [array[rows[0] : rows[1], cols[0] : cols[1]] for rows, cols in indices]

    return list_array


def get_root_path() -> Path:
    """
    Return root path of project.

    Returns:
        Path: Root path.
    """
    return Path(__file__).parent.parent.parent


def get_file_system() -> S3FileSystem:
    """
    Return the s3 file system.
    """
    return S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
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


def get_bounds_for_tiles(transform: Affine, row_indices: Tuple, col_indices: Tuple) -> Tuple:
    """
    Given an Affine transformation, and indices for a tile's row and column,
    returns the bounding coordinates (left, bottom, right, top) of the tile.

    Args:
        transform: An Affine transformation
        row_indices (Tuple): A tuple containing the minimum and maximum
            indices for the tile's row
        col_indices (Tuple): A tuple containing the minimum and maximum
            indices for the tile's column

    Returns:
        Tuple: A tuple containing the bounding coordinates
            (left, bottom, right, top) of the tile
    """

    row_min = row_indices[0]
    row_max = row_indices[1]
    col_min = col_indices[0]
    col_max = col_indices[1]

    left, bottom = transform * (col_min, row_max)
    right, top = transform * (col_max, row_min)
    return rasterio.coords.BoundingBox(left, bottom, right, top)


def get_bounds_for_tiles2(transform: Affine, row, col, tile_length) -> Tuple:
    """
    Given an Affine transformation, and indices for a tile's row and column,
    returns the bounding coordinates (left, bottom, right, top) of the tile.

    Args:
        transform: An Affine transformation
        row (int): The minimum indice for the tile's row
        col (int): The minimum indice for the tile's column
        tile_length (int): The length of the tile.

    Returns:
        Tuple: A tuple containing the bounding coordinates
            (left, bottom, right, top) of the tile
    """

    row_min = row
    row_max = row + tile_length
    col_min = col
    col_max = col + tile_length

    left, bottom = transform * (col_min, row_max)
    right, top = transform * (col_max, row_min)
    return rasterio.coords.BoundingBox(left, bottom, right, top)


def get_indices_from_tile_length(m: int, n: int, tile_length: int) -> List:
    """
    Given the dimensions of an original image and a desired tile length,
    this function returns a list of tuples, where each tuple contains the
    border indices of a tile that can be extracted from the original image.
    The function raises a ValueError if the size of the tile is larger than
    the size of the original image.

    Args:
        m (int): Height of the original image
        n (int): Width of the original image
        tile_length (int): Dimension of tiles

    Returns:
        List: A list of tuples, where each tuple contains the border indices
            of a tile that can be extracted from the original image
    """

    if (tile_length > m) | (tile_length > n):
        raise ValueError(
            "The size of the tile should be smaller" "than the size of the original image."
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


def load_ril(
    millesime: Literal["2020", "2021", "2022", "2023"],
    dep: Literal["971", "972", "973", "974", "976", "977", "978"],
) -> gpd.GeoDataFrame:
    """
    Load RIL for a given datetime.

    Args:
        millesime (Literal): Year.
        dep (Literal): Departement.

    Returns:
        gpd.GeoDataFrame: RIL GeoDataFrame.
    """
    update_storage_access()
    environment = get_environment()
    fs = get_file_system()

    dataset = pq.ParquetDataset(
        os.path.join(
            environment["bucket"],
            environment["sources"]["RIL"],
            f"dep={dep}",
            f"millesime={millesime}",
        ),
        filesystem=fs,
    )

    df = dataset.read().to_pandas()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    crs = dep_to_crs[dep]
    gdf = gdf.set_crs(f"epsg:{crs}")

    return gdf


def load_bdtopo(
    millesime: Literal["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"],
    dep: Literal["971", "972", "973", "974", "976", "977", "978"],
) -> gpd.GeoDataFrame:
    """
    Load BDTOPO for a given datetime.

    Args:
        millesime (Literal): Year.
        dep (Literal): Departement.

    Returns:
        gpd.GeoDataFrame: BDTOPO GeoDataFrame.
    """
    root_path = get_root_path()
    environment = get_environment()

    if int(millesime) >= 2019:
        couche = "BATIMENT.shp"
    elif int(millesime) < 2019:
        couche = "BATI_INDIFFERENCIE.shp"

    bucket = environment["bucket"]
    path_s3 = environment["sources"]["BDTOPO"][millesime][dep]
    dir_path = os.path.join(
        root_path,
        environment["local-path"]["BDTOPO"][millesime][dep],
    )

    if os.path.exists(dir_path):
        print("\t** Le téléchargement de cette version de la BDTOPO a déjà été effectué")

    else:
        os.makedirs(dir_path)

        update_storage_access()
        fs = S3FileSystem(client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"})
        extensions = ["cpg", "dbf", "prj", "shp", "shx"]
        couche_split = couche.split(".")[0]
        for ext in extensions:
            fs.download(
                rpath=f"{bucket}/{path_s3}/{couche_split}.{ext}",
                lpath=f"{dir_path}",
                recursive=True,
            )

    file_path = None

    for root, dirs, files in os.walk(dir_path):
        if couche in files:
            file_path = os.path.join(root, couche)
    if not file_path:
        raise ValueError(f"No valid {couche} file found.")

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
    This function updates the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
    environment variables with values obtained from a HashiCorp Vault server.
    The Vault server URL, token, and secret path are taken from the
    VAULT_TOKEN and VAULT_MOUNT+VAULT_TOP_DIR/s3 environment variables.

    If AWS_SESSION_TOKEN is present, it will be deleted.
    """

    client = hvac.Client(url="https://vault.lab.sspcloud.fr", token=os.environ["VAULT_TOKEN"])

    secret = f"{os.environ['VAULT_MOUNT']}{os.environ['VAULT_TOP_DIR']}/s3"
    mount_point, secret_path = secret.split("/", 1)
    secret_dict = client.secrets.kv.read_secret_version(path=secret_path, mount_point=mount_point)

    os.environ["AWS_ACCESS_KEY_ID"] = secret_dict["data"]["data"]["ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_dict["data"]["data"]["SECRET_ACCESS_KEY"]
    try:
        del os.environ["AWS_SESSION_TOKEN"]
    except KeyError:
        pass


def get_path_by_millesime(paths, millesime):
    dep_dict = {
        "971": "GUADELOUPE",
        "972": "MARTINIQUE",
        "973": "GUYANE",
        "974": "REUNION",
        "976": "MAYOTTE",
    }

    idx = [path.endswith(f"{millesime['year']}/{dep_dict[millesime['dep']]}") for path in paths]

    path = paths[idx.index(True)] if any(idx) and True in idx else []
    return path
