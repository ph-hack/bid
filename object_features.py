"""
This module contains functions to extract features from
the main object's bounding box.
"""

# here goes the imports
from pre_processors import imread, normalize_size, denoise
from skimage.color import rgb2grey
from skimage.transform import resize
from skimage.feature import hog
from numpy import array, zeros
from fft import fft_features
from sharpness_behavior import sharpness_behavior

import csv
import matplotlib.pyplot as plt

from unittest import TestLoader, TextTestRunner
from unittest.case import TestCase


class BoundingBox(object):
    """
    Class to represent a bounding box. It follows the
    same structure used by the Darknet object detector.
    """

    def __init__(self, x, y, w, h):

        self.x = x if type(x) is float else float(x)
        self.y = y if type(y) is float else float(y)
        self.w = w if type(w) is float else float(w)
        self.h = h if type(h) is float else float(h)

    def area(self):

        return self.w * self.h


def extract(img_path, **kwargs):
    """
    Extracts the SHB and FTV features of the bounding box areas
    of the main object of an image.

    :param img_path: String. Path to the image file.

    :return: List of numbers. The SHB and FTV feature sets
             concatenated.
    """

    if img_path is None:

        return [-1] * 9

    # Pre-processing
    img = imread(img_path)
    img = rgb2grey(img)
    img = normalize_size(img)
    img = denoise(img)

    # Read the objects bounding boxes
    boxes = read_boxes(img_path)

    if len(boxes) > 0:

        # Compute the area of each box
        b_areas = [b.area() for b in boxes]
        b_areas = array(b_areas)

        # Find the box with the biggest area
        biggest_box_idx = b_areas.argmax()

        # Get the window corresponding the that box
        window = get_window_from_box(boxes[biggest_box_idx], img)
    else:
        # the window is the full image
        box = BoundingBox(0.5, 0.5, 1., 1.)
        window = get_window_from_box(box, img)
        print(window)

    # Compute FFT features
    f1 = fft_features(img[window])

    # Compute SHB features
    f2 = sharpness_behavior(img[window])

    # Return all features
    return f1 + f2

def extract_hog(img_path, **kwargs):
    """
    Extracts the HOG features of the bounding box areas
    of the main object of an image.

    :param img_path: String. Path to the image file.

    :return: List of numbers. The HOG features extracted.
    """

    use_boxes = False if 'use_boxes' not in kwargs else kwargs['use_boxes']

    if img_path is None:

        return [-1] * 225

    # Pre-processing
    img = imread(img_path)
    img = rgb2grey(img)
    img = normalize_size(img)
    img = denoise(img)

    # Read the objects bounding boxes
    boxes = read_boxes(img_path)

    if len(boxes) > 0 and use_boxes:

        # Compute the area of each box
        b_areas = [b.area() for b in boxes]
        b_areas = array(b_areas)

        # Find the box with the biggest area
        biggest_box_idx = b_areas.argmax()

        # Get the window corresponding the that box
        window = get_window_from_box(boxes[biggest_box_idx], img)
    else:
        # the window is the full image
        box = BoundingBox(0.5, 0.5, 1., 1.)
        window = get_window_from_box(box, img)
        print(window)

    img = resize(img[window], (500, 500))

    # Compute FFT features
    f = hog(img, pixels_per_cell=(100, 100), cells_per_block=(1, 1))

    # Return all features
    return f

def read_boxes(img_path):
    """
    Reads the bounding box information of each image from a
    file with the same filename of the image file but with
    the "boxes" extension.

    :param img_path: String. The path to the image file.

    :return: List of BoundingBox objects.
    """

    boxes_path = '{}.boxes'.format(img_path)

    boxes = []

    with open(boxes_path, 'r') as boxes_file:

        reader = csv.DictReader(boxes_file, delimiter=',')

        for row in reader:

            box = BoundingBox(
                row['x'],
                row['y'],
                row['w'],
                row['h']
            )

            boxes.append(box)

    return boxes

def get_window_from_box(box, img):
    """
    Returns the window of the image that corresponds to the
    given bounding box.

    :param box: BoundingBox object.
    :param img: Numpy array representing the image.

    :return: Tuple of slices. The slices for the horizontal
             and vertical dimensions of the image.
    """

    y = img.shape[0] * box.y
    x = img.shape[1] * box.x
    h = img.shape[0] * box.h
    w = img.shape[1] * box.w

    window = (

        slice(
            max(0, int(y - h/2.)),
            min(int(y - h/2. + h), img.shape[0])
        ),
        slice(
            max(0, int(x - w/2.)),
            min(int(x - w/2. + w), img.shape[1])
        )
    )

    return window


# Unit Tests 
class ClassTests(TestCase):

    def assertEqual(self, first, second, msg=None):
        if first != second:
            e_msg = ''.join(
                ['\nExpected: ', str(second), ' Found: ', str(first)])
            print(e_msg)

        TestCase.assertEqual(self, first, second, msg)

    def test_1_reading_box(self):

        img_path = 'image.jpg'

        boxes = read_boxes(img_path)

        self.assertEqual(2, len(boxes), 'There should be 2 boxes total')

        self.assertEqual(0.647641, boxes[1].y, 'Failed retrieving y value of the second box')
        self.assertEqual(0.115317, boxes[0].h, 'Failed retrieving h value of the first box')

        self.assertAlmostEqual(boxes[0].area(), 0.004174937, delta=0.001, msg='Failed computing the area of the first box')
        self.assertAlmostEqual(boxes[1].area(), 0.214910633, delta=0.001, msg='Failed computing the area of the second box')

    def test_2_getting_window_from_box(self):

        box = BoundingBox(x=0.5, y=0.4, w= 0.3, h=0.1)
        img = zeros((500,500))

        window = get_window_from_box(box, img)

        wgt = (
            slice(175, 225),
            slice(175, 325)
        )

        self.assertEqual(window, wgt, 'The window should be correct')

    def test_3_extraction(self):

        img_path = 'image.jpg'

        features = extract(img_path)
        print(features)


if __name__ == '__main__':
    # loads and runs the Unit Tests
    suite = TestLoader().loadTestsFromTestCase(ClassTests)
    TextTestRunner(verbosity=2, ).run(suite)