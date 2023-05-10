import sys
import os
sys.path.append('../src')
import shutil
import ee
import geemap
import tqdm
import s3fs
import fs
import hvac
from minio import Minio
from satellite_image import SatelliteImage
from osgeo import gdal


service_account = (
    "slums-detection-sa@ee-insee-sentinel.iam.gserviceaccount.com"
)
credentials = ee.ServiceAccountCredentials(
    service_account, "GCP_credentials.json"
)

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

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
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

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = (
        img.select("B8")
        .lt(NIR_DRK_THRESH * SR_BAND_SCALE)
        .multiply(not_water)
        .rename("dark_pixels")
    )

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(
        ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
    )

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
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

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
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
    AOI = ee.Geometry.BBox(**AOIs[DOM])
    s2_sr_cld_col = get_s2_sr_cld_col(AOI, START_DATE, END_DATE)
    s2_sr_median = (
        s2_sr_cld_col.map(add_cld_shdw_mask).map(apply_cld_shdw_mask).median()
    )

    fishnet = geemap.fishnet(AOI, rows=4, cols=4, delta=0.5)
    geemap.download_ee_image_tiles(
        s2_sr_median,
        fishnet,
        f'{DOM}_{start_date[0:4]}/',
        prefix="data_",
        crs=f"EPSG:{EPSGs[DOM]}",
        scale=10,
        num_threads=50,
    )

    upload_satelliteImages(
        f'{DOM}_{start_date[0:4]}',
        f'projet-slums-detection/Donnees/SENTINEL2/{DOM.upper()}/TUILES_{start_date[0:4]}',
        250)
    
    shutil.rmtree(f"{DOM}_{start_date[0:4]}",ignore_errors=True)


def exportToMinio(image,rpath):
    client = hvac.Client(
            url='https://vault.lab.sspcloud.fr', token=os.environ["VAULT_TOKEN"]
        )

    secret = os.environ["VAULT_MOUNT"] + os.environ["VAULT_TOP_DIR"] + "/s3"
    mount_point, secret_path = secret.split("/", 1)
    secret_dict = client.secrets.kv.read_secret_version(
        path=secret_path, mount_point=mount_point
    )

    os.environ["AWS_ACCESS_KEY_ID"] = secret_dict["data"]["data"][
        "ACCESS_KEY_ID"
    ]
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_dict["data"]["data"][
        "SECRET_ACCESS_KEY"
    ]

    try:
        del os.environ['AWS_SESSION_TOKEN']
    except KeyError:
        pass

    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    
    return fs.put(image,rpath,True)


def upload_satelliteImages(
    lpath,
    rpath,
    dim
):
    images_paths = os.listdir(lpath)

    for i in range(len(images_paths)):
        images_paths[i] = lpath+'/'+images_paths[i]

    list_satelliteImages = [
        SatelliteImage.from_raster(
            filename,
            dep = "973",
            n_bands = 12
        ) for filename in tqdm(images_paths)]

    splitted_list_images = [im for sublist in tqdm(list_satelliteImages) for im in sublist.split(dim)]

    for i in range(len(splitted_list_images)):
        image = splitted_list_images[i]

        transf = image.transform
        in_ds = gdal.Open(images_paths[1])
        proj = in_ds.GetProjection()

        array = image.array

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(f'image{i}.tif', array.shape[2], array.shape[1], array.shape[0], gdal.GDT_Float64)
        out_ds.SetGeoTransform([transf[2],transf[0],transf[1],transf[5],transf[3],transf[4]])
        out_ds.SetProjection(proj)

        for j in range(array.shape[0]):
            out_ds.GetRasterBand(j+1).WriteArray(array[j,:,:])

        out_ds = None

        exportToMinio(f'image{i}.tif',rpath)
        os.remove(f'image{i}.tif')




AOIs = {
    "Guadeloupe": {
        "west": -61.811124,
        "south": 15.828534,
        "east": -60.998518,
        "north": 16.523944,
    },
    "Martinique": {
        "west": -61.264617,
        "south": 14.378599,
        "east": -60.781573,
        "north": 14.899453,
    },
    "Mayotte": {
        "west": 45.013633,
        "south": -13.006619,
        "east": 45.308891,
        "north": -12.633022,
    },
    "Guyane": {
        "west": -52.883,
        "south": 4.148,
        "east": -51.813,
        "north": 5.426
    }
}

EPSGs = {"Guadeloupe": "4559", "Martinique": "4559", "Mayotte": "4471", "Guyane": "4235"}

START_DATE = "2021-05-01"
END_DATE = "2021-09-01"
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
