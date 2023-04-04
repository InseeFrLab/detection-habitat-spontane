import re
from pyproj import Transformer
import os

import sys
sys.path.append('../src')
from satellite_image import SatelliteImage
from mappings import *

def crs_to_gps_image(satellite_image: SatelliteImage = None, filepath: str = None) :
    
    """
    Gives the gps point of the left-top boundingbox of the image. Argument is either a SatelliteImage or a filepath.

    Args:
        satellite_image (SatelliteImage)
        filepath (str): The full filepath 

    Returns:
        GPS coordinate : latitude and logitutude.
        
    Example:
        >>> filename_1 = '../data/PLEIADES/2022/MARTINIQUE/ORT_2022_0711_1619_U20N_8Bits.jp2'
        >>> crs_to_gps_image(None, filename_1)
        (14.827025734562506, -61.16930531772711)
    """
    
    if satellite_image != None:
        folder_path = '../data/PLEIADES/' + str((satellite_image.date).year) + '/' + num_dep_to_name_dep[satellite_image.dep]
        filepath = folder_path + '/' + satellite_image.filename

    delimiters = ["-", "_"]

    pattern = "|".join(delimiters)

    split_filepath = re.split(pattern, filepath)

    x = float(split_filepath[2])*1000 #left
    y = float(split_filepath[3])*1000 #top
    
    delimiters = ["/"]

    pattern = "|".join(delimiters)

    split_filepath = re.split(pattern, filepath)
    
    dep_num = name_dep_to_num_dep[split_filepath[4]]
    str_crs = dep_to_crs[dep_num]
    
    transformer = Transformer.from_crs(str_crs, 'EPSG:4326',always_xy=True) 
    lon, lat = transformer.transform(x,y)
    
    # Retourner les coordonnées GPS (latitude, longitude)
    return lat, lon


def gps_to_crs_point(lat : float, lon : float, crs : int or str) :
    
    """
    Give the crs point of a gps point.

    Args:
        lat (float): latitude
        lon (float): longitude
        crs (int or str)

    Returns:
        CRS coordinate.
        
    Example:
        >>> gps_to_crs_point(14.636195717948983, -61.04095442371388, '5490')
        (711000.0000002225, 1618999.9999483444)
    """
    # Convertir les coordonnées GPS en coordonnées dans le système de coordonnées de destination (CRS)
    transformer = Transformer.from_crs('EPSG:4326','EPSG:'+str(crs),always_xy=True) #au cas où le CRS en entrée est de type entier 
    x, y = transformer.transform(lon, lat) #car y=lat et x=lon, les coordonnées gps sont en (lat,lon)
    
    # Retourner les coordonnées dans le CRS spécifié
    return x, y


def find_image_of_point(coordinate : list, folder_path : str, coord_gps = False) :
    
    """
    Gives the image in the folder which contains the point (gps or crs). Return a message if the image is not in the folder.

    Args:
        coordinate (list): [x,y] crs coordinate or [lat, lon] gps coordinate
        folder_path (str): the path of the folder in which we search the image containing the point
        coord_gps (boolean): specifies if the coordinate is a gps coordinate or not

    Returns:
        The path of the image containing the point.
        
    Examples: 
        >>> find_image_of_point([713000.0, 1606000.0], '../data/PLEIADES/2022/MARTINIQUE')
        '../data/PLEIADES/2022/MARTINIQUE/ORT_2022_0713_1607_U20N_8Bits.jp2'
        
        >>> find_image_of_point([14.635338, -61.038345], '../data/PLEIADES/2022/MARTINIQUE', coord_gps = True)
        '../data/PLEIADES/2022/MARTINIQUE/ORT_2022_0711_1619_U20N_8Bits.jp2'
    """
        
    if coord_gps == True :  
        #récupérer le crs via le département
        delimiter = ["/"]

        pattern = "|".join(delimiter)

        split_folder = re.split(pattern, folder_path)

        departement = split_folder[4]
        dep_num = name_dep_to_num_dep[departement]
        crs = dep_to_crs[dep_num]
        
        lat, lon = coordinate
        x,y = gps_to_crs_point(lat,lon,crs) 
    else :
        x,y = coordinate
        
    #récupérer les coordonnées left-top
    delimiters = ["-", "_"]

    pattern = "|".join(delimiters)
        
    for filename in os.listdir(folder_path):

        split_filename = re.split(pattern, filename)

        left = float(split_filename[2])*1000
        top = float(split_filename[3])*1000
        right = left + 1000.0
        bottom = top - 1000.0

        if left <= x <= right:
            if bottom <= y <= top:
                return(folder_path + '/' +filename)
    else : 
        return("Le point n'est pas retrouvé dans ce fichier d'images")

    

def find_image_different_years(different_year : int, satellite_image : SatelliteImage = None, filepath : str = None) :
    
    """
    Find the image which represents the same place but in a different year. The arguments can be either a SatteliteImage or the filepath of the image.

    Args:
        different_year (int): the year we are interested in.
        satellite_image (SatelliteImage): the SatelliteImage.
        filepath (str): the filepath of the image. 

    Returns:
        The path of the image representing the same place but in a different period of time.
        
    Example:
        >>> filename_1 = '../data/PLEIADES/2022/MARTINIQUE/ORT_2022_0711_1619_U20N_8Bits.jp2'
        >>> find_image_different_years(2017, None, filename_1)
        '../data/PLEIADES/2017/MARTINIQUE/972-2017-0711-1619-U20N-0M50-RVB-E100.jp2'
    """
    
    if satellite_image != None:
        folder_path = '../data/PLEIADES/' + str((satellite_image.date).year) + '/' + num_dep_to_name_dep[satellite_image.dep]
        filepath = folder_path + '/'+ satellite_image.filename
    

    #récupérer le département de base
    delimiter = ["/"]

    pattern = "|".join(delimiter)

    split_folder = re.split(pattern, filepath)

    departement_base = split_folder[4]
    dep_num_base = name_dep_to_num_dep[departement_base]

    folder_path = '../data/PLEIADES/'+str(different_year)+'/'+departement_base

    #récupérer les coordonnées left-top
    if filepath.find('_') != -1 :
        delimiter = ["_"]

    elif filepath.find('-') != -1 :
        delimiter = ["-"]

    pattern = "|".join(delimiter)

    split_filepath = re.split(pattern, filepath)

    filename = os.listdir(folder_path)[0]

    if filename.find('_') != -1 :
        delimiter = ["_"]

    elif filename.find('-') != -1 :
        delimiter = ["-"]    

    pattern = "|".join(delimiter)

    split_filename = re.split(pattern, filename)

    split_filename[2] = split_filepath[2]
    split_filename[3] = split_filepath[3]
    
    new_filename = delimiter[0].join(split_filename)
    
    if new_filename in os.listdir(folder_path): 

        return(folder_path+ '/' + new_filename)
    else: 
        return("Il n'y a pas d'image de ce lieu dans l'année demandée")
