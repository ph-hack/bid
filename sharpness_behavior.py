"""
This module contains the functions to compute the SHB features.
"""

from unittest import TestLoader, TextTestRunner
from unittest.case import TestCase
from numpy import *
import skimage.data as im_data
import skimage.transform as im_transf
import skimage.color as im_color
from cv2 import resize, blur, calcHist, \
    morphologyEx, MORPH_CLOSE, getGaussianKernel, absdiff


def sharpness_behavior(img, **kwargs):
    """
    The sharpness behavior (SHB) function. It returns the 5-D feature vector
    extracted from the given image.

    The features are:
        #. The sharpness score for the original image;
        #. The sharpness score for the blurred image;
        #. The mean gradient for the original image;
        #. A score estimating how much of the sharpness region is near the
           center of the image;
        #. The sharpness score of the original image weighted according to the
           pixel location related to the center of the image;

    Parameters:
        :img: The numpy.array object containing the image at gray scale.

    Example::

        img = skimage.data.imread('image.jpg')
        img = skimage.color.rgb2gray(img)

        f = sharpnessBehavior(img)
    """

    if img is None:

        return [-1] * 5

    # computes the blurred version of the image
    blurred = blur(img, (9, 9))

    # computes both version's gradient
    g1 = two_neighbors_gradient(img)
    g2 = two_neighbors_gradient(blurred)

    # gets the mean gradient of the blurred image
    m = mean(g2)

    # finds out where the original image is sharper than the blurred one by
    # comparing their gradients
    i1 = [x + m < y for x, y in zip(g2, g1)]
    i1 = where(i1)

    # if there is no such pixels, returns 0 for most features
    if not len(i1[0]):

        return [0, 0, mean(g1) * 255, 0, 0]

    # calculates the histogram of the gradient of the original image
    maxg1 = amax(g1) + 0.001
    ming1 = amin(g1)
    hg1 = calcHist([g1[i1].astype(float32)], [0], None, [256],
                       [ming1, maxg1])
    # computes the accumulated distribution of gradient of the original image
    cg1 = cumsum(hg1) / sum(hg1)

    # finds out a threshold in the image's gradient where its distribution is at
    # more than 97% of the "total energy"
    tg1 = where(cg1 > 0.97)[0][0]
    tg1 = (maxg1 - ming1) / 256 * tg1

    # creates a 2-D gaussian mask with the image's dimensions where it is bigger
    # in the center and its maximum value is equal to 1
    max_size = max(g1.shape[0], g2.shape[1])
    g_mask = getGaussianKernel(g1.shape[0], max_size / 5) * transpose(
        getGaussianKernel(g1.shape[1], max_size / 5))
    g_mask = g_mask / amax(g_mask)

    # uses the morphological operation closing to make the original image's
    # gradient thicker
    closed = morphologyEx(g1, MORPH_CLOSE, ones([15, 15]))

    # finds where the "closed" gradient is bigger than the previously found
    # threshold
    i3 = where(closed > tg1, ones(closed.shape), zeros(closed.shape))

    # uses the morphological operation closing to make the found region in the
    # previous step, thicker
    i3 = morphologyEx(i3, MORPH_CLOSE,
                          ones([int(max_size * 0.02), int(max_size * 0.02)]))

    # the following piece of code is comment for performance cost issues
    # and it is meant to close holes in the detected sharpened region:
    #
    # seed = copy(i3)
    # seed[1:-1, 1:-1] = i3.max()
    # mask = i3
    # i3 = im_morpho.reconstruction(seed, mask, method='erosion')

    # applies the weights from the gaussian mask to the detected sharpened
    # region of the image's gradient
    i3 = where(i3)
    if len(i3[0]):
        w = g1[i3] * 255
        w = w * g_mask[i3]
    else:
        w = array([0])

    # creates the final feature vector
    features = [mean(g1[i1]) * 255,
                mean(g2[i1]) * 255,
                mean(g1) * 255,
                sum(g_mask[i3]) / sum(g_mask),
                mean(w)]

    return features


def two_neighbors_gradient(img):
    """
     This is a simpler method to compute gradient for it only considers two
    neighbors of each pixel. The last column and row can be ignored.
     It returns a numpy.array object with the same dimensions of *img* with the
    absolute gradient approximation value.

    Parameters:
        :img: The numpy.array object containing the image at gray scale.

    Example::

        img = skimage.data.imread('image.jpg')
        img = skimage.color.rgb2gray(img)

        gd = two_neighbors_gradient(img)

        assert img.shape == gd.shape  # this is true
    """

    # creates the transformation objects to translate the image one column and
    # one row down
    transf_y = im_transf.SimilarityTransform(translation=(1,0))
    transf_x = im_transf.SimilarityTransform(translation=(0,1))

    # translates the image
    img_y = im_transf.warp(img, transf_y)
    img_x = im_transf.warp(img, transf_x)

    # computes the gradient in both directions
    gx = absdiff(img, img_x)
    gy = absdiff(img, img_y)

    # returns the maximum gradient between the two directions
    return maximum(gx, gy)


# Unit Tests ##################################################################

class SHBTests(TestCase):
    def assertEqual(self, first, second, msg=None):
        if first != second:
            e_msg = ''.join(
                ['\nExpected: ', str(second), ' Found: ', str(first)])
            print(e_msg)

        TestCase.assertEqual(self, first, second, msg)

    def test_1_gradient(self):
        img = im_data.imread('image.jpg')
        img = im_color.rgb2gray(img)

        gd = two_neighbors_gradient(img)

        blurred = blur(img, (9, 9))
        gd2 = two_neighbors_gradient(blurred)

        sh1 = mean(gd) * 256
        sh2 = mean(gd2) * 256

        self.assertGreater(sh1, sh2, 'Original sharpness should be bigger!')

    def test_2_shb(self):
        img = im_data.imread('image.jpg')
        img = im_color.rgb2gray(img)
        img = resize(img, None, fx=0.5, fy=0.5)

        features = sharpness_behavior(img)

        self.assertEqual(len(features), 5, 'The features returned should have 5'
                                           ' values')

if __name__ == '__main__':
    # loads and runs the Unit Tests
    suite = TestLoader().loadTestsFromTestCase(SHBTests)
    TextTestRunner(verbosity=2, ).run(suite)
