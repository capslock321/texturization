import numpy as np
import skimage.color

from typing import Union


def calculate_deltaE_similarity(source: np.ndarray, target_hue: list) -> Union[float, np.ndarray]:
    """Calculate the color similarity using deltaE from two different colors.
    Requires a source image and a hue with at least 3 items.

    Args:
        source (numpy.ndarray): The image to look for the hue in.
        target_hue (list): The hue in which to look for.

    Returns:
        numpy.ndarray: A numpy array of DeltaE values.
    """
    source_lab = skimage.color.rgb2lab(np.uint8(source))
    target_hue = np.uint8(np.asarray([[target_hue]]))
    target_lab = skimage.color.rgb2lab(target_hue)
    return skimage.color.deltaE_ciede2000(source_lab, target_lab)


def calculate_3d_euclidian_distance(hue: np.ndarray, target_hue: list) -> Union[float, np.ndarray]:
    """Calculate the 3d euclidian distance of all pixels of an image given a target_hue.

    Args:
        hue (numpy.ndarray): The image to calculate the distance of.
        target_hue (list): The hue in which to look for in the image.

    Returns:
        numpy.ndarray: A numpy array of euclidian distances.
    """
    squared = np.square(hue[:, np.newaxis, :] - target_hue)
    euclidian_sum = np.sum(squared, axis=1)
    return np.sqrt(euclidian_sum)
