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

from comparisons import calculate_deltaE_similarity


def get_primary_color(source: str) -> list:
    """Given an image, get the dominant color of that image.
    Resizes the image to one pixel and gets the color of that pixel.

    Args:
        source (str): The image to get the primary color of.

    Returns:
        list: The RGB of that pixel.
    """
    img = Image.fromarray(source.copy()).convert("RGB")
    img.resize((1, 1), resample=0)
    primary_color = img.getpixel((0, 0))
    return primary_color


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


def texturize(
    source: str,
    overlay: str,
    hue: list = None,
    threshold: int = 10,
) -> Image:
    """Texturizes a given image from another image.
    This works by replacing the pixel which does not meet the threshold given with the pixel from the overlay.

    Args:
        source (str): The given input image in which to texturize.
        overlay (str): The overlay in which to texturize the source with.
        hue (list): The target hue in which to replace the pixel if not within a certain threshold of that hue.
        threshold (int): The threshold required in order to replace the pixel.

    Returns:
        Image: The texturized image.
    """
    source, overlay = load_images(source, overlay)
    if hue is None:
        hue = get_primary_color(source)
    pixels = calculate_deltaE_similarity(source, hue) < threshold
    source[pixels] = overlay[pixels]
    return Image.fromarray(source)


if __name__ == "__main__":
    image = texturize(
        "./assets/tweakie.png",
        "./assets/textures/diamond.jpg",
        hue=[5, 182, 215],
        threshold=30,
    )
    image.save("./assets/out/tweak_diamond.png")
