"""
This module contains some of the functions used as pre-processing steps.
"""

# here goes the imports
from numpy import *
from skimage.transform import rescale
import skimage.data as im_data
import skimage.color as im_color
import scipy.signal as signal
import cv2


def denoise(img, **kwargs):
    """
    Applies a denoise function in the given image *img*.

    Parameters:
        :img: The numpy.array object containing the image at gray scale.

    Optional Parameters:
        :method: A string containing the name of the denoise method to be
                 applied. Only 'wiener' and 'bilateral' are supported. The
                 *wiener* method is the default one.

    Example::

        img = skimage.data.imread('image.jpg')
        img = skimage.color.rgb2gray(img)

        img2 = denoise(img, method='wiener')
    """

    # creates a 2-level dictionary with the currently available denoise methods
    # where the 1st-level key is the method name and 2nd-level keys 'f' and 'p'
    # corresponds the function object and its parameters, respectively.
    methods = {
        'wiener': {'f': signal.wiener, 'p': [3]},
        'bilateral': {'f': cv2.bilateralFilter, 'p':[3, 0, 3]}
    }

    # if the method is specified, ...
    if 'method' in kwargs:

        # gets the name of the chosen method
        name = kwargs['method']

        # gets its function from the methods' dictionary
        f = methods[name]['f']
        # gets its function parameters from the dictionary
        p = methods[name]['p']

        # applies the denoise function with its parameters to the image
        img2 = f(img.astype(float32), *p)

    # if no method is specified, ...
    else:

        # applies the default
        img2 = signal.wiener(img, 3)

    # returns the image after a denoise method is applied
    return img2


def imread(file_path, **kwargs):
    """
    Reads the content of an image from the file. It has a list of
    plugins to use. It tries them in some order and if one doesn't
    work it tries the next one.

    :param file_path: String. Path to the image file.
    :param plugin: String. The name of the preferred plugin.

    :return: Numpy array. The image read from the file.
    """

    plugin = 'pil' if 'plugin' not in kwargs else kwargs['plugin']

    try:
        return im_data.imread(file_path, plugin=plugin)

    except Exception:

        nextPlugin = {
            'pil': 'qt',
            'qt': 'matplotlib',
            'matplotlib': 'test',
            'test': 'opencv'
        }

        if nextPlugin[plugin] == 'opencv':

            return cv2.imread(file_path)

        else:
            return imread(file_path, plugin=nextPlugin[plugin])

def noise(image, chr_i=20, ill_i=20, chr_size=11, ill_size=1, verbose=False):
    """
    Emulates the chromatic and luminance noise independently.

    Parameters:
        :chr_i: The intensity of the chromatic noise. (default = 20);
        :ill_i: The intensity of the luminance noise. (default = 20);
        :chr_size: The size of the squared blur filter for the chromatic noise.
                   It must be >= 1. (default = 11);
        :ill_size: The size of the squared blur filter for the luminance
                   noise. It must be >= 1. (default = 1);

    Example::

        image = skimage.io.imread('path/to/image.jpg')

        new_image = noise(image, chr_i=40, ill_i=30, chr_size=15, ill_size=3)
    """

    # computes the gray-scale version of the image

    img_gray = im_color.rgb2gray(image)

    if image.dtype == uint8:

        img_gray *= 255.

    # computes the inverse image
    img_gray_i = (255. - img_gray)
    # weight the inverse image based on its mean
    mean_im = mean(img_gray_i) / 255.
    img_gray_i = img_gray_i / mean_im

    # computes the actual intensity values
    chr_i = 3. / (1000./chr_i) if chr_i > 0 else 0.
    ill_i = 1. / (1000./ill_i) if ill_i > 0 else 0.

    if verbose:

        print('chroma noise var =', chr_i)
        print('illuminant noise var =', ill_i)

    # makes the inverse image as the weights for the noise masks
    weights = zeros(image.shape)
    weights[:, :, 0] = img_gray_i
    weights[:, :, 1] = img_gray_i
    weights[:, :, 2] = img_gray_i

    # computes the chromatic noise mask:
    # the random values times the intensity and weights
    chr_var = random.randn(image.shape[0], image.shape[1],
                                 image.shape[2]) * chr_i * weights
    # and applies the given amount of blurring
    chr_var = cv2.blur(chr_var, (chr_size, chr_size))

    # computes the luminance noise mask:
    # initiates it as zeros
    ill_var = zeros(image.shape)
    # fills it with the random values times the intensity and weights
    temp_ill_var = \
        random.randn(image.shape[0], image.shape[1]) * ill_i * weights[:,:,0]
    # blurs it by the given amount
    temp_ill_var = cv2.blur(temp_ill_var, (ill_size, ill_size))
    # fills the noise for the all channels to be exactly the same
    ill_var[:,:,0] = temp_ill_var
    ill_var[:,:,1] = temp_ill_var
    ill_var[:,:,2] = temp_ill_var

    # applies the noise
    noisy_img = image + ill_var + chr_var
    # makes sure the image is the range [0,255]
    noisy_img[noisy_img < 0] = 0
    noisy_img[noisy_img > 255] = 255
    # converts it back to 8-bit integer format
    noisy_img = noisy_img.astype('uint8')

    # return the new image
    return noisy_img

def motion_blur(image, filter_size=11, angle=45):
    """
    Emulates camera motion blur by applying a directional blur filter to the
    image.

    Parameters:
        :image: The input image;
        :filter_size: The size of the squared filter kernel. It must be >= 1,
                      where if it is equal to 1, no changes happen.
                      (default = 11);
        :angle: The angle of the "movement" direction, given in degrees, in
                clock-wise orientation. (default = 45);

    Example::

        image = skimage.io.imread('path/to/image.jpg')

        new_image = motion_blur(image, filter_size=21, angle=-30)
    """

    # if the size of the filter is less than 1, uses the default value and shows
    # a warning
    if filter_size <= 0:

        warnings.warn('Motion Blur: The filter_size must be >= 1! The default '
                      'value (11) is used instead!', Warning)
        filter_size = 11

    # converts the input angle from degrees to radians
    theta = pi * -angle/180.

    # initiates the filter with zeros and with the given size
    motion_filter = zeros((filter_size, filter_size))

    # fills the filter to blur in the given direction
    x1 = int(filter_size / 2 - filter_size * sin(theta))
    x2 = int(filter_size / 2 + filter_size * sin(theta))
    y1 = int(filter_size / 2 - filter_size * cos(theta))
    y2 = int(filter_size / 2 + filter_size * cos(theta))
    cv2.line(motion_filter, (x1, y1), (x2, y2), 1)
    motion_filter = motion_filter / sum(motion_filter)
    # applies the filter
    mblur = cv2.filter2D(image, -1, motion_filter)

    # returns the new image
    return mblur

def normalize_size(img, **kwargs):
    """
    Resize the image so that its biggest side will have the
    size of *size*, whose default value is 800.

    :param img: Numpy array. The image.
    :param size: Integer. The size of the biggest dimension of the
                 image.

    :return: Numpy array. The image resized.
    """

    size = 800. if 'size' not in kwargs else kwargs['size']

    factor = float(size)/max(img.shape)

    new_img = rescale(img, factor)

    return new_img