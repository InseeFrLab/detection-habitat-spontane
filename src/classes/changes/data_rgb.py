import numpy as np 
from classes.data.satellite_image import SatelliteImage


def mask_rgb(image, threshold = 156):
    img = image.array.copy()
    img = img[:3,:,:]
    img = img.transpose(1,2,0)

    shape = img.shape[0:2]

    grayscale = np.mean(img, axis=2)
    
    black = np.ones(shape, dtype=float)
    white = np.zeros(shape, dtype=float)

    mask = np.where(grayscale < threshold, white, black)
    
    return(mask)