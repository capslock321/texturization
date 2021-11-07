# -*- coding: utf-8 -*-

"""MIT License

Copyright (c) 2021 capslock321

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Union

import numpy as np
from PIL import Image


def load_images(input_img: str, output_img: str) -> Union[np.ndarray, np.ndarray]:
    """Loads and converts the image to a numpy array.

    Args:
        input_img (str): The file to load.
        output_img (str): The other file to load.

    Returns:
        numpy.ndarray: A copy of the first file loaded.
        numpy.ndarray: A copy of the second file loaded.
    """
    input_img = Image.open(input_img).convert("RGB")
    output_img = Image.open(output_img).resize(input_img.size).convert("RGB")
    return np.array(input_img).copy(), np.array(output_img).copy()


def calculate_3d_euclidian_distance(hue: list, target_hue: list) -> Union[float, np.ndarray]:
    """Calculate the 3d euclidian distance from two diffrent colors.
    Requires an iterable with at least 3 items.

    Args:
        hue (list): The hue in which to compare.
        target_hue (list): The other hue in which to compare.

    Returns:
        numpy.ndarray: A numpy array of euclidian distances.
    """
    squared = np.square(hue[:, np.newaxis, :] - target_hue)
    euclidian_sum = np.sum(squared, axis=1)
    return np.sqrt(euclidian_sum)


def texturize(
    source: str,
    overlay: str,
    hue: np.ndarray = np.array([255, 255, 255]),
    threshold: int = 30,
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
    pixels = calculate_3d_euclidian_distance(source, hue) < threshold
    source[pixels] = overlay[pixels]
    return Image.fromarray(source)


if __name__ == "__main__":
    image = texturize("./assets/tweakie.png", "./assets/orange.jpg")
    image.save("./assets/output.png")
