"""
This module contains the function to extract the Sieberth et al.'s
SIEDS metric.
"""

import numpy as np

from skimage.transform import rescale
from skimage.color import rgb2hsv
from skimage.filters import laplace
from pre_processors import imread
from cv2 import blur


def sieberth_blur_metric(img_path, **kwargs):
    """
    Compute the Sieberth et al.'s Saturation Image Edge
    Difference Standard Deviation (SIEDS) metric.

    T. Sieberth, R. Wackrow, and J. H. Chandler. Automatic
    Detection of Blurred Images in UAV Image Sets. ISPRS
    Journal of Photogrammetry and Remote Sensing, 122:1–16,
    Dec. 2016.

    :param img_path: String. Path to the image file.

    :return: List of floats. Containing the SIEDS metric value.
    """

    if img_path is None:
        return [-1]

    # read the image
    img = imread(img_path)

    # downsample the image to 1/3
    img = rescale(img, 1/3., preserve_range=True, mode='constant')

    # get the saturation info
    s = rgb2hsv(img)

    # blur the original image
    b = blur(s, (3,3))

    # apply high-pass filter on both versions
    se = laplace(s, 3)
    be = laplace(b, 3)

    # compute the discrepancy map
    d = np.abs(se - be)

    # compute the SIEDS (Saturation Image Edge Difference Standard Deviation)
    sieds = d.std()

    return [sieds]