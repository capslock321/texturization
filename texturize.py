# -*- coding: utf-8 -*-

import math
from typing import Union

import numpy as np
from PIL import Image


def load_images(input_img: str, output_img: str) -> Union[np.ndarray, np.ndarray]:
    """Loads and converts the image to a numpy array.

    Args:
        input_img (str): The file to load.
        output_img (str): The other file to load.

    Returns:
        np.ndarray: A copy of the first file loaded.
        np.ndarray: A copy of the second file loaded.
    """
    input_img = np.asarray(Image.open(input_img))
    output_img = np.asarray(Image.open(output_img))
    return input_img.copy(), output_img.copy()


def calculate_3d_euclidian_distance(hue: list, target_hue: list) -> int:
    """Calculate the 3d euclidian distance from two diffrent colors.
    Requires an iterable with at least 3 items.

    Args:
        hue (list): The hue in which to compare.
        target_hue (list): The other hue in which to compare.

    Returns:
        int: The euclidian distance between the two hues.
    """
    x = math.pow(hue[0] - target_hue[0], 2)
    y = math.pow(hue[1] - target_hue[1], 2)
    z = math.pow(hue[2] - target_hue[2], 2)
    return math.sqrt(x + y + z)


def texturize(
        source: str,
        overlay: str,
        hue: np.ndarray = np.array([255, 255, 255]),
        threshold: int = 10,
) -> Image:
    """Texturizes a given image from another image.
    This works by replacing the pixel which does not meet the threshold given with the pixel from the overlay.

    Args:
        source (str): The given input image in which to texturize.
        overlay (str): The overlay in which to texturize the source with.
        hue (np.ndarray): The target hue in which to replace the pixel if not within a certain threshold of that hue.
        threshold (int): The threshold required in order to replace the pixel.
    Returns:
        Image: The texturized image.
    """
    source, overlay = load_images(source, overlay)
    np.resize(overlay, source.shape)
    for x in range(source.shape[0]):
        for y in range(source.shape[1]):
            euclidian_distance = calculate_3d_euclidian_distance(source[x, y], hue)
            if euclidian_distance > threshold:
                source[x, y][:3] = overlay[x, y][:3]
    return Image.fromarray(source)


if __name__ == '__main__':
    texturize("./assets/cikn.png", "./assets/orange.jpg")
