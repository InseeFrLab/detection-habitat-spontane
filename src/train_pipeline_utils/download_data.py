import os
import s3fs
from utils.utils import (
    get_environment,
    get_root_path,
    update_storage_access
)


def load_satellite_data(year: int, dep: str, src: str):
    """
    Load satellite data for a given year and territory and a given source of
    satellite images.

    This function downloads satellite data from an S3 bucket, \
    updates storage access, and saves the data locally. \
    The downloaded data is specific to the given year and territory.

    Args:
        year (int): Year of the satellite data.
        territory (str): Territory for which the satellite \
        data is being loaded.
        source (str): Source of the satellite images.

    Returns:
        str: The local path where the data is downloaded.
    """

    print("Entre dans la fonction load_satellite_data")

    update_storage_access()
    root_path = get_root_path()
    environment = get_environment()

    bucket = environment["bucket"]
    path_s3 = environment["sources"][src][year][dep]
    path_local = os.path.join(
        root_path, environment["local-path"][src][year][dep]
    )

    if os.path.exists(path_local):
        print("Le dossier existe déjà")
        return path_local

    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
    )
    print(
        "download " + src + " " + dep + " " + str(year) + " in " + path_local
    )
    fs.download(
        rpath=f"{bucket}/{path_s3}", lpath=f"{path_local}", recursive=True
    )

    return path_local


def load_donnees_test(type="segmentation"):

    """
    Load test data (images,masks) for a given task\
         (segmentation or change detection).

    This function downloads satellite data from an S3 bucket,  \
        updates storage access, and saves the data locally.
    The downloaded data is specific to the given task type.

    Args:
        type (str, optional): The type of task-data to load.\
             Defaults to "segmentation".

    Returns:
        str: The local path where the data is downloaded.
    """
    update_storage_access()
    root_path = get_root_path()
    environment = get_environment()

    bucket = environment["bucket"]
    path_s3 = environment["sources"]["TEST"][type]
    path_local = os.path.join(
        root_path, environment["local-path"]["TEST"][type]
    )

    if os.path.exists(path_local):
        print("le jeu de données test existe déjà")
        return path_local

    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
    )
    fs.download(
        rpath=f"{bucket}/{path_s3}", lpath=f"{path_local}", recursive=True
    )

    return path_local
