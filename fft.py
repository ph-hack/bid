"""
This module contains the function to compute the FTV features
"""

# here goes the imports
from unittest import TestLoader, TextTestRunner
from unittest.case import TestCase
from numpy import *
import cv2
import skimage.io as im_io


def fft_features(img, **kwargs):
    """
    Extracts a 4-D feature vector based on the Fast Fourier Transform (FFT).
    The features are basically the mean variation on the x and y directions
    regarding the magnitude of the transformation. So it is called the
    Fourier Transform Variance (FTV) features.

    Parameters:
        :img: A gray-scale image as a numpy 2-D array

    Optional Parameters:
        :weights_function: A function that returns weights for a given index
                           set.

    Example::

        img = skimage.data.imread('image.jpg', as_grey=True)

        def weights(x):

            return x * 2

        features = fft_features(img, weights_function=weights)
    """

    if img is None:

        return [-1] * 4

    # computes the FFT complex components of the image
    dft = cv2.dft(img.astype(float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    # computes the magnitude of the complex FFT
    mag = cv2.magnitude(dft[:,:,0], dft[:,:,1])
    # passes the magnitude to the log scale
    mag = 20*log(mag + 1)

    # blurs the image with a 9 x 9 mean filter
    img = cv2.blur(img, (9,9))

    # computes the FFT magnitude of blurred version in log scale
    dft = cv2.dft(img.astype(float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    mag2 = cv2.magnitude(dft[:,:,0], dft[:,:,1])
    mag2 = 20*log(mag2 + 1)

    # gets the number of columns and rows of the image
    r, c = img.shape

    # retrieves the indexes to be used for each direction, that is, the line
    # of the lowest frequency of each direction but ignoring the extremes
    # (lowest and heights frequencies)
    range_c = slice(int(c/8), 3*int(c/8)+1)
    range_r = slice(int(r/8), 3*int(r/8)+1)

    # computes the weights for the column (x) and row (y) directions using the
    # given weight function or leaves them as None
    weights_c = None if 'weights_function' not in kwargs else \
              kwargs['weights_function'](array(range(range_c.start, range_c.stop)))

    weights_r = None if 'weights_function' not in kwargs else \
              kwargs['weights_function'](array(range(range_r.start, range_r.stop)))

    # computes the absolute difference for the magnitude of the original image
    # in the x direction
    varx = absolute(mag[0,:int(c/2)-1] - mag[0,1:int(c/2)])
    # computes the weighted average in the selected range in the x direction
    varx = average(varx[range_c], weights=weights_c)

    # same thing for the y direction
    vary = absolute(mag[:int(r/2)-1,0] - mag[1:int(r/2),0])
    vary = average(vary[range_r], weights=weights_r)

    # and same thing for the magnitude of the blurred image
    varx2 = absolute(mag2[0,:int(c/2)-1] - mag2[0,1:int(c/2)])
    varx2 = average(varx2[range_c], weights=weights_c)
    vary2 = absolute(mag2[:int(r/2)-1,0] - mag2[1:int(r/2),0])
    vary2 = average(vary2[range_r], weights=weights_r)

    # returns the weighted averages as features
    return [varx, vary, varx2, vary2]


# Unit Tests ##################################################################

class FFTTests(TestCase):

    def test_1_features(self):

        img = im_io.imread('image.jpg', as_grey=True)

        features = fft_features(img)

        print('features =', features)

        self.assertEqual(len(features), 4, 'The feature set should have 4 values')


if __name__ == '__main__':

    # loads and runs the Unit Tests
    suite = TestLoader().loadTestsFromTestCase(FFTTests)
    TextTestRunner(verbosity=2, ).run(suite)