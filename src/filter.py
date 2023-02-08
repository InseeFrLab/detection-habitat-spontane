from satellite_image import SatelliteImage
import numpy as np


def is_too_black(
    image: SatelliteImage, black_value_threshold=100, black_area_threshold=0.5
) -> bool:
    """Filter images based on black pixels and their proportion.

    This function takes in a satellite image and converts it to grayscale,
    and then filters it based on the number of black pixels and their proportion.
    The image is considered black if the pixel value is less than the specified threshold (black_value_threshold).
    The image is only returned if the proportion of black pixels is less than the specified threshold (black_area_threshold).

    Args:
        image (SatelliteImage): The input satellite image.
        black_value_threshold (int, optional): The threshold value for considering a pixel as black.
        black_area_threshold (float, optional): The threshold for the proportion of black pixels.

    Returns:
        SatelliteImage: The filtered satellite image. If the proportion of black pixels is greater than the threshold, returns None.
    """
    gray_image = (
        0.2989 * image.array[0] + 0.5870 * image.array[1] + 0.1140 * image.array[2]
    )
    nb_black_pixels = np.sum(gray_image < black_value_threshold)

    if (nb_black_pixels / (gray_image.shape[0] ** 2)) < black_area_threshold:
        return True
    else:
        return False
