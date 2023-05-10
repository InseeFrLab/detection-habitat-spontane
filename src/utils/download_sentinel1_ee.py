import os
import shutil

import ee
import geemap
import hvac
import s3fs
from osgeo import gdal
from tqdm import tqdm

from classes.data.satellite_image import SatelliteImage

service_account = (
    "slums-detection-sa@ee-insee-sentinel.iam.gserviceaccount.com"
)
credentials = ee.ServiceAccountCredentials(
    service_account, "GCP_credentials.json"
)

# Initialize the library.
ee.Initialize(credentials)


def get_s1_grd(aoi, start_date, end_date):
    sentinel1 = ee.Image(
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(ee.Date(start_date), ee.Date(end_date))
        .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
        .first()
        .select("VV", "VH")
        .clip(aoi)
    )

    return sentinel1


def export_s1(DOM, AOIs, EPSGs, start_date, end_date):
    AOI = ee.Geometry.BBox(**AOIs[DOM])
    s1_grd = get_s1_grd(AOI, start_date, end_date)

    fishnet = geemap.fishnet(AOI, rows=4, cols=4, delta=0.5)
    geemap.download_ee_image_tiles(
        image=s1_grd,
        features=fishnet,
        out_dir=f"{DOM}_{start_date[0:4]}/",
        prefix="data_",
        crs=f"EPSG:{EPSGs[DOM]}",
        scale=10,
        num_threads=50,
    )

    upload_satelliteImages(
        f"{DOM}_{start_date[0:4]}",
        f"projet-slums-detection/Donnees/SENTINEL1/{DOM.upper()}/TUILES_2023",
        f"{DEPs[DOM]}",
        250,
    )

    shutil.rmtree(f"{DOM}_{start_date[0:4]}", ignore_errors=True)


def exportToMinio(image, rpath):
    client = hvac.Client(
        url="https://vault.lab.sspcloud.fr", token=os.environ["VAULT_TOKEN"]
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
        del os.environ["AWS_SESSION_TOKEN"]
    except KeyError:
        pass

    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    return fs.put(image, rpath, True)


def upload_satelliteImages(lpath, rpath, dep, dim):
    images_paths = os.listdir(lpath)

    for i in range(len(images_paths)):
        images_paths[i] = lpath + "/" + images_paths[i]

    list_satelliteImages = [
        SatelliteImage.from_raster(filename, dep=dep, n_bands=1)
        for filename in tqdm(images_paths)
    ]

    splitted_list_images = [
        im
        for sublist in tqdm(list_satelliteImages)
        for im in sublist.split(dim)
    ]

    for i in range(len(splitted_list_images)):
        image = splitted_list_images[i]

        transf = image.transform
        in_ds = gdal.Open(images_paths[1])
        proj = in_ds.GetProjection()

        array = image.array

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(
            f"image{i}.tif",
            array.shape[2],
            array.shape[1],
            array.shape[0],
            gdal.GDT_Float64,
        )
        out_ds.SetGeoTransform(
            [transf[2], transf[0], transf[1], transf[5], transf[3], transf[4]]
        )
        out_ds.SetProjection(proj)

        for j in range(array.shape[0]):
            out_ds.GetRasterBand(j + 1).WriteArray(array[j, :, :])

        out_ds = None

        exportToMinio(f"image{i}.tif", rpath)
        os.remove(f"image{i}.tif")


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
        "north": 5.426,
    },
}

EPSGs = {
    "Guadeloupe": "4559",
    "Martinique": "4559",
    "Mayotte": "4471",
    "Guyane": "4235",
}

DEPs = {
    "Guadeloupe": "971",
    "Martinique": "972",
    "Mayotte": "976",
    "Guyane": "973",
}

START_DATE = "2022-08-01"
END_DATE = "2022-09-01"
CLOUD_FILTER = 60
CLD_PRB_THRESH = 40
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 2
BUFFER = 50


# export_s1(
#     "Guadeloupe",
#     AOIs,
#     EPSGs,
#     START_DATE,
#     END_DATE
# )


# export_s1(
#     "Martinique",
#     AOIs,
#     EPSGs,
#     START_DATE,
#     END_DATE
# )


export_s1("Mayotte", AOIs, EPSGs, START_DATE, END_DATE)


# export_s1(
#     "Guyane",
#     AOIs,
#     EPSGs,
#     START_DATE,
#     END_DATE
# )
