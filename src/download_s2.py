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


def get_s2_sr_cld_col(aoi, start_date, end_date):
    # Import and filter S2 SR.
    s2_sr_col = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", CLOUD_FILTER))
    )

    # Import and filter s2cloudless.
    s2_cloudless_col = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    # Join the filtered s2cloudless collection to the SR collection
    # by the 'system:index' property.
    return ee.ImageCollection(
        ee.Join.saveFirst("s2cloudless").apply(
            **{
                "primary": s2_sr_col,
                "secondary": s2_cloudless_col,
                "condition": ee.Filter.equals(
                    **{
                        "leftField": "system:index",
                        "rightField": "system:index",
                    }
                ),
            }
        )
    )


def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get("s2cloudless")).select("probability")

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename("clouds")

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))


def add_shadow_bands(img):
    # Identify water pixels from the SCL band.
    not_water = img.select("SCL").neq(6)

    # Identify dark NIR pixels that are not water
    # (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = (
        img.select("B8")
        .lt(NIR_DRK_THRESH * SR_BAND_SCALE)
        .multiply(not_water)
        .rename("dark_pixels")
    )

    # Determine the direction to project cloud shadow from clouds
    # (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE")))

    # Project shadows from clouds for the distance specified
    # by the CLD_PRJ_DIST input.
    cld_proj = (
        img.select("clouds")
        .directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
        .reproject(**{"crs": img.select(0).projection(), "scale": 100})
        .select("distance")
        .mask()
        .rename("cloud_transform")
    )

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename("shadows")

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select("clouds").add(img_cloud_shadow.select("shadows")).gt(0)

    # Remove small cloud-shadow patches and
    # dilate remaining pixels by BUFFER input.

    # 20 m scale is for speed,
    # and assumes clouds don't require 10 m precision.
    is_cld_shdw = (
        is_cld_shdw.focalMin(2)
        .focalMax(BUFFER * 2 / 20)
        .reproject(**{"crs": img.select([0]).projection(), "scale": 20})
        .rename("cloudmask")
    )

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)


def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select("cloudmask").Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select("B.*").updateMask(not_cld_shdw)


def export_s2_no_cloud(
    DOM,
    AOIs,
    EPSGs,
    start_date,
    end_date,
    cloud_filter,
    cloud_prb_thresh,
    nir_drk_thresh,
    cld_prj_dist,
    buffer,
):
    """
    Downloads the images locally and calls a function that\
        uplaods them on MinIO.

    Args:
        DOM: name of the DOM.
        AOIs: western, southern, eastern and northern boudaries of the DOMs.
        EPSGs: EPSGs of the DOMs.
        start_date: date from which the images can be downloaded.
        end_date: date after which the images can no longer be downloaded.
        cloud_filter: maximum image cloud cover percent allowed\
            in image collection.
        cloud_prb_thresh: cloud probability (%); values greater than\
            are considered cloud.
        nir_drk_thresh: near-infrared reflectance; values less than\
            are considered potential cloud shadow.
        cld_prj_dist: maximum distance (km) to search for cloud shadows\
            from cloud edges.
        buffer: distance (m) to dilate the edge of cloud-identified objects.
    """

    update_storage_access()
    root_path = get_root_path()
    environment = get_environment()

    bucket = environment["bucket"]
    path_s3 = environment["sources"]["SENTINEL2"][start_date[0:4]][DEPs[DOM.upper()]]
    path_local = os.path.join(
        root_path,
        environment["local-path"]["SENTINEL2"][start_date[0:4]][DEPs[DOM.upper()]],
    )

    AOI = ee.Geometry.BBox(**AOIs[DOM.upper()])
    s2_sr_cld_col = get_s2_sr_cld_col(AOI, START_DATE, END_DATE)
    s2_sr_median = s2_sr_cld_col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask).median()

    fishnet = geemap.fishnet(AOI, rows=4, cols=4, delta=0.5)
    geemap.download_ee_image_tiles(
        s2_sr_median,
        fishnet,
        path_local,
        prefix="data_",
        crs=f"EPSG:{EPSGs[DOM.upper()]}",
        scale=10,
        num_threads=50,
    )

    upload_satelliteImages(
        path_local,
        f"{bucket}/{path_s3}",
        DEPs[DOM.upper()],
        int(start_date[0:4]),
        250,
        12,
        True,
    )

    shutil.rmtree(path_local, ignore_errors=True)


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


if __name__ == "__main__":
    EPSGs = name_dep_to_crs
    DEPs = name_dep_to_num_dep
    AOIs = name_dep_to_aoi

    START_DATE = "2022-05-01"
    END_DATE = "2022-09-01"
    CLOUD_FILTER = 60
    CLD_PRB_THRESH = 40
    NIR_DRK_THRESH = 0.15
    CLD_PRJ_DIST = 2
    BUFFER = 50

    export_s2_no_cloud(
        "Saint-Martin",
        AOIs,
        EPSGs,
        START_DATE,
        END_DATE,
        CLOUD_FILTER,
        CLD_PRB_THRESH,
        NIR_DRK_THRESH,
        CLD_PRJ_DIST,
        BUFFER,
    )

    # export_s2_no_cloud(
    #     "Martinique",
    #     AOIs,
    #     EPSGs,
    #     START_DATE,
    #     END_DATE,
    #     CLOUD_FILTER,
    #     CLD_PRB_THRESH,
    #     NIR_DRK_THRESH,
    #     CLD_PRJ_DIST,
    #     BUFFER,
    # )

    # export_s2_no_cloud(
    #     "Mayotte",
    #     AOIs,
    #     EPSGs,
    #     START_DATE,
    #     END_DATE,
    #     CLOUD_FILTER,
    #     CLD_PRB_THRESH,
    #     NIR_DRK_THRESH,
    #     CLD_PRJ_DIST,
    #     BUFFER,
    # )

    # export_s2_no_cloud(
    #     "Guyane",
    #     AOIs,
    #     EPSGs,
    #     START_DATE,
    #     END_DATE,
    #     CLOUD_FILTER,
    #     CLD_PRB_THRESH,
    #     NIR_DRK_THRESH,
    #     CLD_PRJ_DIST,
    #     BUFFER,
    # )

    # export_s2_no_cloud(
    #     "Reunion",
    #     AOIs,
    #     EPSGs,
    #     START_DATE,
    #     END_DATE,
    #     CLOUD_FILTER,
    #     CLD_PRB_THRESH,
    #     NIR_DRK_THRESH,
    #     CLD_PRJ_DIST,
    #     BUFFER,
    # )
