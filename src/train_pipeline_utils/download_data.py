import os

import numpy as np
import s3fs
from osgeo import gdal

from classes.data.satellite_image import SatelliteImage
from utils.utils import get_environment, get_root_path, update_storage_access


def load_satellite_data(year: int, dep: str, src: str):
    """
    Load satellite data for a given year and territory \
        and a given source of satellite images.

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
    path_local = os.path.join(root_path, environment["local-path"][src][year][dep])

    if os.path.exists(path_local):
        print("Le dossier existe déjà")
        return path_local

    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
    )
    print("download " + src + " " + dep + " " + str(year) + " in " + path_local)
    fs.download(rpath=f"{bucket}/{path_s3}", lpath=f"{path_local}", recursive=True)

    return path_local


def load_donnees_test(type="segmentation", src="PLEIADES"):
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
    path_s3 = environment["sources"]["TEST"][src][type]
    path_local = os.path.join(root_path, environment["local-path"]["TEST"][src][type])

    if os.path.exists(path_local):
        print("le jeu de données test existe déjà")
        return path_local

    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
    )
    fs.download(rpath=f"{bucket}/{path_s3}", lpath=f"{path_local}", recursive=True)

    return path_local


def load_2satellites_data(year: int, dep: str, src: str):
    """
    Load a concatenation of Sentinel1 data and Sentinel2 data for a given year\
        and territory and a given source of satellite images.

    This function downloads S1 and S2 data from an S3 bucket, \
        updates storage access, and saves the data locally. \
    The downloaded data is specific to the given year and territory.

    Args:
        year (int): Year of the satellite data.
        dep (str): Territory for which the satellite \
        data is being loaded.
        source (str): Source of the satellite images.

    Returns:
        str: The local path where the data is downloaded.
    """

    update_storage_access()
    root_path = get_root_path()
    environment = get_environment()

    path_local = os.path.join(root_path, environment["local-path"][src][year][dep])

    if os.path.exists(path_local):
        print("Le dossier existe déjà")
        return path_local

    src2 = 'SENTINEL' + src.split('SENTINEL1-')[-1]
    output_dir_s1 = load_satellite_data(year, dep, "SENTINEL1")
    output_dir_s2 = load_satellite_data(year, dep, src2)

    list_paths_s1 = os.listdir(output_dir_s1)
    list_paths_s2 = os.listdir(output_dir_s2)

    list_paths_s1_rac = [path[0:14] for path in list_paths_s1]
    list_paths_s2_rac = [path[0:14] for path in list_paths_s2]

    if not os.path.exists(path_local):
        os.makedirs(path_local)

    for path in list_paths_s1_rac:
        if path in list_paths_s2_rac:
            path_s1 = list_paths_s1[list_paths_s1_rac.index(path)]
            path_s2 = list_paths_s2[list_paths_s2_rac.index(path)]

            image_s1 = SatelliteImage.from_raster(
                output_dir_s1 + "/" + path_s1, dep=dep, date=year, n_bands=2
            )

            try:
                image_s2 = SatelliteImage.from_raster(
                    output_dir_s2 + "/" + path_s2, dep=dep, date=year, n_bands=13
                )
            except IndexError:
                image_s2 = SatelliteImage.from_raster(
                    output_dir_s2 + "/" + path_s2, dep=dep, date=year, n_bands=12
                )

            matrice = np.concatenate((image_s2.array, image_s1.array))
            n_bands = matrice.shape[0]

            transf = image_s1.transform
            driver = gdal.GetDriverByName("GTiff")
            out_ds = driver.Create(
                path_local + "/" + path + ".tif",
                matrice.shape[2],
                matrice.shape[1],
                matrice.shape[0],
                gdal.GDT_Float64,
            )
            out_ds.SetGeoTransform(
                [
                    transf[2],
                    transf[0],
                    transf[1],
                    transf[5],
                    transf[3],
                    transf[4],
                ]
            )
            in_ds = gdal.Open(output_dir_s1 + "/" + path_s1)
            proj = in_ds.GetProjection()
            out_ds.SetProjection(proj)

            for i in range(n_bands):
                out_ds.GetRasterBand(i + 1).WriteArray(matrice[i, :, :])

            out_ds = None

            list_paths_s2_rac.remove(path)
            list_paths_s2.remove(path_s2)

    return path_local
