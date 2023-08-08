import os
import shutil

import ee
import geemap
from download_sentinel1_ee import upload_satelliteImages

import utils.mappings
from utils.utils import get_environment, get_root_path, update_storage_access

service_account = "slums-detection-sa@ee-insee-sentinel.iam.gserviceaccount.com"
credentials = ee.ServiceAccountCredentials(service_account, "GCP_credentials.json")

# Initialize the library.
ee.Initialize(credentials)


def get_s2_sr_cld_col(aoi, start_date, end_date, collection):
    # Import and filter S2 SR.
    s2_sr_col = (
        ee.ImageCollection(collection)
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
    if COLLECTION == "COPERNICUS/S2_SR_HARMONIZED":
        not_water = img.select("SCL").neq(6)

    # Identify dark NIR pixels that are not water
    # (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    if COLLECTION == "COPERNICUS/S2_SR_HARMONIZED":
        dark_pixels = (
            img.select("B8")
            .lt(NIR_DRK_THRESH * SR_BAND_SCALE)
            .multiply(not_water)
            .rename("dark_pixels")
        )
    elif COLLECTION == "COPERNICUS/S2_HARMONIZED":
        dark_pixels = (
            img.select("B8").lt(NIR_DRK_THRESH * SR_BAND_SCALE).rename("dark_pixels")
        )

    # Determine the direction to project cloud shadow from clouds
    # (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(
        ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
    )

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
    is_cld_shdw = (
        img_cloud_shadow.select("clouds")
        .add(img_cloud_shadow.select("shadows"))
        .gt(0)
    )

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
    collection,
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

    if collection == "COPERNICUS/S2_HARMONIZED":
        check_nbands12 = False
        check_nbands13 = True
        n_bands = 13
        src = "SENTINEL2-L1C"
    if collection == "COPERNICUS/S2_SR_HARMONIZED":
        check_nbands12 = True
        check_nbands13 = False
        n_bands = 12
        src = "SENTINEL2-L2A"

    bucket = environment["bucket"]
    path_s3 = environment["sources"][src][int(start_date[0:4])][DEPs[DOM.upper()]]
    path_local = os.path.join(
        root_path,
        environment["local-path"][src][int(start_date[0:4])][DEPs[DOM.upper()]],
    )

    AOI = ee.Geometry.BBox(**AOIs[DOM.upper()])
    s2_sr_cld_col = get_s2_sr_cld_col(AOI, START_DATE, END_DATE, collection)
    s2_sr_median = (
        s2_sr_cld_col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask).median()
    )

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
        n_bands,
        check_nbands12,
        check_nbands13,
    )

    shutil.rmtree(path_local, ignore_errors=True)


if __name__ == "__main__":
    EPSGs = utils.mappings.name_dep_to_crs
    DEPs = utils.mappings.name_dep_to_num_dep
    AOIs = utils.mappings.name_dep_to_aoi

    COLLECTION = "COPERNICUS/S2_HARMONIZED"
    # COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

    START_DATE = "2022-05-01"
    END_DATE = "2022-09-01"
    CLOUD_FILTER = 60
    CLD_PRB_THRESH = 40
    NIR_DRK_THRESH = 0.15
    CLD_PRJ_DIST = 2
    BUFFER = 50

    # export_s2_no_cloud(
    #     "Guadeloupe",
    #     AOIs,
    #     EPSGs,
    #     START_DATE,
    #     END_DATE,
    #     COLLECTION,
    #     CLOUD_FILTER,
    #     CLD_PRB_THRESH,
    #     NIR_DRK_THRESH,
    #     CLD_PRJ_DIST,
    #     BUFFER,
    # )

    # export_s2_no_cloud(
    #     "Martinique",
    #     AOIs,
    #     EPSGs,
    #     START_DATE,
    #     END_DATE,
    #     COLLECTION,
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
    #     COLLECTION,
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
    #     COLLECTION,
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
    #     COLLECTION,
    #     CLOUD_FILTER,
    #     CLD_PRB_THRESH,
    #     NIR_DRK_THRESH,
    #     CLD_PRJ_DIST,
    #     BUFFER,
    # )
