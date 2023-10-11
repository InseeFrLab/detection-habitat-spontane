import os
import shutil

import ee
import geemap
import PIL
from osgeo import gdal
from tqdm import tqdm

from classes.data.satellite_image import SatelliteImage
from utils.mappings import name_dep_to_aoi, name_dep_to_crs, name_dep_to_num_dep
from utils.utils import (
    exportToMinio,
    get_environment,
    get_root_path,
    update_storage_access,
)

service_account = "slums-detection-sa@ee-insee-sentinel.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(service_account, "GCP_credentials.json")

# Initialize the library.
ee.Initialize(credentials)


def get_s1_grd(aoi, start_date, end_date):
    """
    Imports and filters S1 images according to date, place,\
        direction of the orbit and polarisation.

    Args:
        aoi: bounding box of the images to download.
        start_date: date from which the images can be downloaded.
        end_date: date after which the images can no longer be downloaded.

    Returns:
        An image collection of the most appropriated S1 image\
            according to specified filters.
    """

    sentinel1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(ee.Date(start_date), ee.Date(end_date))
        .select("VV", "VH")
    )

    return sentinel1


def export_s1(DOM, AOIs, EPSGs, start_date, end_date):
    """
    Downloads the images locally and calls a function that\
        uplaods them on MinIO.

    Args:
        DOM: name of the DOM.
        AOIs: western, southern, eastern and northern boudaries of the DOMs.
        EPSGs: EPSGs of the DOMs.
        start_date: date from which the images can be downloaded.
        end_date: date after which the images can no longer be downloaded.
    """

    AOI = ee.Geometry.BBox(**AOIs[DOM.upper()])
    AOI_DOM = AOIs[DOM.upper()]
    feature_aoi = ee.Geometry.Polygon(
        [
            [AOI_DOM["west"], AOI_DOM["north"]],
            [AOI_DOM["east"], AOI_DOM["north"]],
            [AOI_DOM["east"], AOI_DOM["south"]],
            [AOI_DOM["west"], AOI_DOM["south"]],
        ]
    )

    s1_grd = get_s1_grd(AOI, start_date, end_date)

    image = ee.Image(s1_grd.first()).clip(feature_aoi)

    if ee.Feature(image).contains(feature_aoi).getInfo():
        export_s1_grd_first(start_date, AOI, image, DOM)
    else:
        export_s1_grd_mean(start_date, AOI, ee.Image(s1_grd.mean()).clip(feature_aoi), DOM)


def upload_satelliteImages(
    lpath,
    rpath,
    dep,
    year,
    dim,
    n_bands,
    check_nbands12=False,
):
    """
    Transforms a raster in a SatelliteImage and calls a function\
        that uploads it on MinIO and deletes it locally.

    Args:
        lpath: path to the raster to transform into SatelliteImage\
            and to upload on MinIO.
        rpath: path to the MinIO repertory in which the image\
            should be uploaded.
        dep: department number of the DOM.
        dim: tiles' size.
        n_bands: number of bands of the image to upload.
        check_nbands12: boolean that, if set to True, allows to check\
            if the image to upload is indeed 12 bands.\
            Usefull in download_sentinel2_ee.py
    """

    images_paths = os.listdir(lpath)

    for i in range(len(images_paths)):
        images_paths[i] = lpath + "/" + images_paths[i]

    list_satellite_images = [
        SatelliteImage.from_raster(filename, dep=dep, n_bands=n_bands)
        for filename in tqdm(images_paths)
    ]

    splitted_list_images = [
        im for sublist in tqdm(list_satellite_images) for im in sublist.split(dim)
    ]

    for i in range(len(splitted_list_images)):
        image = splitted_list_images[i]
        bb = image.bounds
        filename = str(int(bb[0])) + "_" + str(int(bb[1])) + "_" + str(i)
        in_ds = gdal.Open(images_paths[1])
        proj = in_ds.GetProjection()

        image.to_raster("/", filename, "tif", proj)

        if check_nbands12:
            try:
                image = SatelliteImage.from_raster(
                    file_path=filename + ".tif",
                    dep=dep,
                    date=year,
                    n_bands=12,
                )
                exportToMinio(filename + ".tif", rpath)
                os.remove(filename + ".tif")

            except PIL.UnidentifiedImageError:
                print("L'image ne possède pas assez de bandes")
        else:
            exportToMinio(filename + ".tif", rpath)
            os.remove(filename + ".tif")


def export_s1_grd_first(start_date, AOI, s1_grd, DOM):
    """
    Calls a function that downloads the images locally and calls a function \
        that uplaods them on MinIO.

    Args:
        DOM: name of the DOM.
        AOIs: western, southern, eastern and northern boudaries of the DOMs.
        EPSGs: EPSGs of the DOMs.
        start_date: date from which the images can be downloaded.
        end_date: date after which the images can no longer be downloaded.
    """

    print("Entre dans la fonction s1_grd_first")

    update_storage_access()
    root_path = get_root_path()
    environment = get_environment()

    bucket = environment["bucket"]
    path_s3 = environment["sources"]["SENTINEL1"][int(start_date[0:4])][DEPs[DOM.upper()]]
    path_local = os.path.join(
        root_path,
        environment["local-path"]["SENTINEL1"][int(start_date[0:4])][DEPs[DOM.upper()]],
    )

    fishnet = geemap.fishnet(AOI, rows=4, cols=4, delta=0.5)
    geemap.download_ee_image_tiles(
        image=s1_grd,
        features=fishnet,
        out_dir=path_local,
        prefix="data_",
        crs=f"EPSG:{EPSGs[DOM.upper()]}",
        scale=10,
        num_threads=50,
    )

    upload_satelliteImages(
        path_local,
        f"{bucket}/{path_s3}",
        f"{DEPs[DOM.upper()]}",
        int(start_date[0:4]),
        250,
        1,
        False,
    )

    shutil.rmtree(path_local, ignore_errors=True)


def export_s1_grd_mean(start_date, AOI, s1_grd, DOM):
    print("Entre dans la fonction s1_grd_mean")

    update_storage_access()
    root_path = get_root_path()
    environment = get_environment()

    bucket = environment["bucket"]
    path_s3 = environment["sources"]["SENTINEL1"][int(start_date[0:4])][DEPs[DOM.upper()]]
    path_local = os.path.join(
        root_path,
        environment["local-path"]["SENTINEL1"][int(start_date[0:4])][DEPs[DOM.upper()]],
    )

    fishnet = geemap.fishnet(AOI, rows=4, cols=4, delta=0.5)
    geemap.download_ee_image_tiles(
        image=s1_grd,
        features=fishnet,
        out_dir=path_local,
        prefix="data_",
        crs=f"EPSG:{EPSGs[DOM.upper()]}",
        scale=10,
        num_threads=50,
    )

    upload_satelliteImages(
        path_local,
        f"{bucket}/{path_s3}",
        f"{DEPs[DOM.upper()]}",
        int(start_date[0:4]),
        250,
        1,
        False,
    )

    shutil.rmtree(path_local, ignore_errors=True)


if __name__ == "__main__":
    START_DATE = "2022-08-20"
    END_DATE = "2022-09-01"

    EPSGs = name_dep_to_crs
    DEPs = name_dep_to_num_dep
    AOIs = name_dep_to_aoi

    export_s1("Guadeloupe", AOIs, EPSGs, START_DATE, END_DATE)

    # export_s1(
    #     "Martinique",
    #     AOIs,
    #     EPSGs,
    #     START_DATE,
    #     END_DATE
    # )

    # export_s1(
    #     "Mayotte",
    #     AOIs,
    #     EPSGs,
    #     START_DATE,
    #     END_DATE
    # )

    # export_s1(
    #     "Reunion",
    #     AOIs,
    #     EPSGs,
    #     START_DATE,
    #     END_DATE
    # )

    # export_s1(
    #     "Guyane",
    #     AOIs,
    #     EPSGs,
    #     START_DATE,
    #     END_DATE
    # )
