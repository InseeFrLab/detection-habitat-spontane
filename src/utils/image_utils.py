import os
import re
from typing import List

from mappings import dep_to_crs, name_dep_to_num_dep, num_dep_to_name_dep
from pyproj import Transformer
from satellite_image import SatelliteImage

from utils import get_environment


def crs_to_gps_image(
    satellite_image: SatelliteImage = None,
    filepath: str = None,
) -> (float, float):
    """
    Gives the gps point of the left-top boundingbox of the image.
    These bounds are found in the filename (quicker than if we have
    to open all the images). So this method is based on the filenames
    of the pleiades images. Argument is either a SatelliteImage or a filepath.

    Args:
        satellite_image (SatelliteImage)
        filepath (str):
            The full filepath.

    Returns:
        GPS coordinate (float, float):
            Latitude and longitutude.

    Example:
        >>> filename_1 = '../data/PLEIADES/2022/MARTINIQUE/
        ORT_2022_0711_1619_U20N_8Bits.jp2'
        >>> crs_to_gps_image(None, filename_1)
        (14.827025734562506, -61.16930531772711)
    """
    environment = get_environment()

    if satellite_image is not None:
        year = (satellite_image.date).year
        dep = num_dep_to_name_dep[satellite_image.dep].lower()
        folder_path = "../" + environment["local-path"]["PLEIADES"][year][dep]
        filepath = folder_path + "/" + satellite_image.filename

    delimiters = ["-", "_"]

    pattern = "|".join(delimiters)

    split_filepath = re.split(pattern, filepath)

    x = float(split_filepath[2]) * 1000.0  # left
    y = float(split_filepath[3]) * 1000.0  # top

    pattern = "/"

    split_filepath = re.split(pattern, filepath)

    dep_num = name_dep_to_num_dep[split_filepath[4]]
    str_crs = dep_to_crs[dep_num]

    transformer = Transformer.from_crs(
        "EPSG:" + str_crs, "EPSG:4326", always_xy=True
    )
    lon, lat = transformer.transform(x, y)

    # Return GPS coordinates (latitude, longitude)
    return lat, lon


def gps_to_crs_point(
    lat: float,
    lon: float,
    crs: int,
) -> (float, float):
    """
    Gives the CRS point of a GPS point.

    Args:
        lat (float):
            Latitude
        lon (float):
            Longitude
        crs (int):
            The coordinate system of the point.

    Returns:
        CRS coordinate (float, float):

    Example:
        >>> gps_to_crs_point(14.636195717948983, -61.04095442371388, '5490')
        (711000.0000002225, 1618999.9999483444)
    """
    # Convert GPS coordinates to coordinates in destination coordinate system
    # (CRS)
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:" + str(crs), always_xy=True
    )  # in case the input CRS is of integer type
    x, y = transformer.transform(lon, lat)
    # because y=lat and x=lon, the gps coordinates are in (lat,lon)

    # Return coordinates in the specified CRS
    return x, y


def find_image_of_point(
    coordinates: List,
    folder_path: str,
    coord_gps: bool = False,
) -> str:
    """
    Gives the image in the folder which contains the point (gps or crs).
    This method is based on the filenames of the pleiades images.
    Returns a message if the image is not in the folder.

    Args:
        coordinates (List):
            [x,y] CRS coordinate or [lat, lon] gps coordinate
        folder_path (str):
            The path of the folder in which we search the image containing
            the point.
        coord_gps (boolean):
            Specifies if the coordinate is a gps coordinate or not.

    Returns:
        str:
            The path of the image containing the point.

    Examples:
        >>> find_image_of_point([713000.0, 1606000.0], '../data/PLEIADES/2022/
        MARTINIQUE')
        '../data/PLEIADES/2022/MARTINIQUE/ORT_2022_0713_1607_U20N_8Bits.jp2'

        >>> find_image_of_point([14.635338, -61.038345], '../data/PLEIADES/
        2022/MARTINIQUE', coord_gps = True)
        '../data/PLEIADES/2022/MARTINIQUE/ORT_2022_0711_1619_U20N_8Bits.jp2'
    """

    if coord_gps:
        # Retrieve the crs via the department

        pattern = "/"

        split_folder = re.split(pattern, folder_path)

        departement = split_folder[4]
        dep_num = name_dep_to_num_dep[departement]
        crs = dep_to_crs[dep_num]

        lat, lon = coordinates
        x, y = gps_to_crs_point(lat, lon, crs)

    else:
        x, y = coordinates

    # Retrieve left-top coordinates
    delimiters = ["-", "_"]

    pattern = "|".join(delimiters)

    for filename in os.listdir(folder_path):
        split_filename = re.split(pattern, filename)

        left = float(split_filename[2]) * 1000
        top = float(split_filename[3]) * 1000
        right = left + 1000.0
        bottom = top - 1000.0

        if left <= x <= right:
            if bottom <= y <= top:
                return folder_path + "/" + filename
    else:
        return "The point is not find in the folder."


def find_image_different_years(
    different_year: int,
    satellite_image: SatelliteImage = None,
    filepath: str = None,
) -> str:
    """
    Finds the image which represents the same place but in a different year.
    The arguments can be either a SatteliteImage or the filepath of an image.
    This method is based on the filenames of the pleiades images.

    Args:
        different_year (int):
            The year we are interested in.
        satellite_image (SatelliteImage):
            The SatelliteImage.
        filepath (str):
            The filepath of the image.

    Returns:
        str:
            The path of the image representing the same place but in a
            different period of time.

    Example:
        >>> filename_1 = '../data/PLEIADES/2022/MARTINIQUE/
        ORT_2022_0711_1619_U20N_8Bits.jp2'
        >>> find_image_different_years(2017, None, filename_1)
        '../data/PLEIADES/2017/MARTINIQUE/972-2017-0711-1619-U20N-0M50-RVB-E100.jp2'
    """

    environment = get_environment()

    if satellite_image is not None:
        year = (satellite_image.date).year
        dep = num_dep_to_name_dep[satellite_image.dep].lower()
        folder_path = "../" + environment["local-path"]["PLEIADES"][year][dep]
        filepath = folder_path + "/" + satellite_image.filename

    # Retrieve base department
    pattern = "/"

    split_folder = re.split(pattern, filepath)

    departement_base = split_folder[4]
    dep = departement_base.lower()
    year = different_year

    folder_path = "../" + environment["local-path"]["PLEIADES"][year][dep]

    # Retrieve left-top coordinates
    if filepath.find("_") != -1:
        pattern = "_"

    elif filepath.find("-") != -1:
        pattern = "-"

    split_filepath = re.split(pattern, filepath)

    filename = os.listdir(folder_path)[0]

    if filename.find("_") != -1:
        pattern = "_"

    elif filename.find("-") != -1:
        pattern = "-"

    split_filename = re.split(pattern, filename)

    split_filename[2] = split_filepath[2]
    split_filename[3] = split_filepath[3]

    new_filename = pattern.join(split_filename)

    if new_filename in os.listdir(folder_path):
        return folder_path + "/" + new_filename
    else:
        return (
            "There is no image of this place in the requested year "
            "in the database Pléiades."
        )
