"""
This module contains the functions and classes to create the
artificial database of images.
"""

# here goes the imports
from unittest import TestLoader, TextTestRunner
from unittest.case import TestCase
from skimage.io import imread, imsave
from skimage.transform import rescale
from skimage.color import rgb2grey, rgb2hsv, hsv2rgb
from skimage.feature import blob_doh
from skimage.draw import circle
from cv2 import filter2D, getGaussianKernel
from numpy import zeros_like, array, zeros, where, histogram, minimum, random
from pre_processors import noise, motion_blur
from PIL import Image

import matplotlib.pyplot as plt
import time
import glob
import random as rd
import csv


class CombLogger(object):
    """
    Utility class to manage the storage of metadata of the images created.
    It records the metadata of the generated images in a CSV file.
    """

    def __init__(self, filepath, init_rows=0):
        """
        Creates a new logger object and also create the output CSV file
        with the header part.

        :param filepath: String. Path to the CSV file.
        :param init_rows: Integer. Initial sample ID.
        """

        self.filepath = filepath

        with open(filepath, 'w') as f:

            f.writelines([
                'combination,', 'background,', 'foreground,', 'position,', 'depths,',
                'blur_method,', 'blur_level,', 'motion_method,', 'motion_level,',
                'motion_angle,', 'noise,', 'compression,'
            ])

        self.rows = init_rows -1

    def __len__(self):

        return self.rows + 1

    def add(self, back, fore_list, specs, fore_specs):
        """
        Writes the metadata of a image into the CSV file.

        :param back: ImageLayer object. The object of the
                     background layer.
        :param fore_list: List of strings. The names of the
                          foreground images.
        :param specs: Spec object. The specifications for
                      the generated image. i.e. the metadata
                      to be recorded.
        :param fore_specs: Integer. The index of the foreground
                           specification in the *specs* object.

        :return: Integer. The ID/index of the recorded data.
        """

        with open(self.filepath, 'a') as f:

            f.writelines([
                '\n',
                str(self.rows+1), ',',
                back.name, ',',
                ';'.join(fore_list), ',',
                ';'.join([str(x) for x in specs.foreground_specs[fore_specs].positions]), ',',
                ';'.join([str(x) for x in specs.foreground_specs[fore_specs].depths]), ',',
                specs.blur_method, ',',
                str(specs.blur_level), ',',
                str(specs.motion_method), ',',
                str(specs.motion_level), ',',
                str(specs.motion_angle), ',',
                str(specs.noise_level), ',',
                str(specs.compression)
            ])

            self.rows += 1

            f.close()

        return self.rows


class ForeSpecs(object):
    """
    Class to represent the specifications of the foreground part
    when generating new images.
    """

    def __init__(self, positions, depths):
        """
        Creates a new foreground specification object.

        :param positions: List of integers. The numbers corresponding
                          to the horizontal position of the foreground
                          objects.
        :param depths: List of integers. Relative z-position of the
                       different foreground objects. The value of 0
                       corresponds to the closest object to the
                       camera.
        """

        self.positions = positions
        self.depths = depths

    def __len__(self):
        """
        Returns the number of foreground objects specified by
        this object.
        """

        return len(self.positions)

    def __iter__(self):
        """
        Returns a tuple containing a horizontal and relative
        z-position of a foreground specification to each
        iteration using the "in" operator.
        """

        for p, d in zip(self.positions, self.depths):

            yield p, d

    @property
    def min_depth(self):
        """
        Returns the smallest z-position. i.e. the closest
        foreground specification to the camera.
        """

        return min(self.depths)

    @property
    def number(self):
        """
        Alias to the len() call.
        """

        return len(self)


class Spec(object):
    """
    Class to represent the specification of artificial transformations and the types of
    foreground positions to be applied to each background images.
    """

    def __init__(self, foreground_specs, blur_method='none',
                 blur_level=0, motion_method='none', motion_level=0, motion_angle=45, noise_level=0, compression=0, scale=1.):
        """
        Creates a new combination specification object.

        :param foreground_specs: List of ForeSpecs objects.
        :param blur_method: String. Name of the focus blur method. Possible values are
                            "none", "hyperfocal", "macro" and "whole".
        :param blur_level: Integer. The higher, the stronger is the blurring effect.
        :param motion_method: String. Name of the motion blur method. Possible values
                              are "none", "camera_motion" and "object_motion".
        :param motion_level: Integer. The higher, the stronger the blurring effect is.
        :param motion_angle: Float. The angle of the motion blur in degrees.
        :param noise_level: Integer. The higher, the stronger the noise is.
        :param compression: Float. The percentage of compression loss. Possible values
                            ranges from 0 to 100.
        :param scale: Float. Rescaling factor for the size of the images.
        """

        self.foreground_specs = foreground_specs
        self.blur_method = blur_method
        self.blur_level = blur_level
        self.motion_method = motion_method
        self.motion_level = motion_level
        self.motion_angle = motion_angle
        self.noise_level = noise_level
        self.compression = compression
        self.scale = scale


class SpecList(object):
    """
    Class to represent a collection of different specifications for the
    combinations and transformations to be made.
    """

    def __init__(self):
        """
        Creates a new specification collection object.
        """

        self.specs = []

    def __iter__(self):
        """
        When iterated with the "in" operator, it returns a Spec object
        at each iteration.
        """

        for s in self.specs:

            yield s

    def add(self, spec):
        """
        Adds a Spec object in the collection.

        :param spec: Spec object.
        """

        self.specs.append(spec)

        return self

    def n_combinations(self, n_background=1):
        """
        Computes the total number of images that will be generated
        given a number of background images to be used.

        :param n_background: Integer. The number of background images.

        :return: Integer. The calculated number of images that will
                 be generated.
        """

        n = 0

        for s in self:

            n += len(s.foreground_specs)

        return n_background * n

    def __len__(self):
        """
        Returns the number of Spec objects in this collection.
        """

        return len(self.specs)


class ImageLayer(object):
    """
    Class to represent a layer of image, both background and
    foreground. It contains the image file annotations: the
    ones defined in Section 3.1.1 of the dissertation.
    """

    FOREGROUND = 1
    BACKGROUND = 0

    def __init__(self, filepath, filetype):
        """
        Creates a new image layer object. The annotation metadata
        are automatically extracted from the filename.

        :param filepath: String. The path to the image file.
        :param filetype: String. The image file's extension. e.g. "jpg".
        """

        try:
            self.filepath = filepath
            self.name, self.base, self.env, self.light, self.spots = filepath.replace(filetype, '').split('-')
            self.name = self.name.split('/')[-1]
            self.base = float(self.base)
            self.spots = not 'no' in self.spots
            self.z = ImageLayer.FOREGROUND if 'foreground' in filepath else ImageLayer.BACKGROUND

        except Exception as e:

            print('Error ', e)
            print('on file ', filepath, filetype)
            raise Exception('Error creating the image layer!')

    def is_foreground(self):
        """
        Returns True if this image layer is a foreground layer and
        False, otherwise.
        """

        return self.z >= ImageLayer.FOREGROUND

    def is_background(self):
        """
        Returns True if this image layer is a background layer and
        False, otherwise.
        """

        return self.z == ImageLayer.BACKGROUND

    def matches(self, other):
        """
        Checks if this layer can be combined with another one. Returns
        True if they can and false, otherwise.

        :param other: ImageLayer object.
        :return: Boolean.
        """

        return (self.env == other.env or self.env == 'both' or other.env == 'both') \
               and self.light == other.light

    def __str__(self):
        """
        Returns a string representation of this image layer object.
        """

        return 'name = {}, base = {}\nenv = {}, light = {}, spot lights = {}'.format(
            self.name, self.base, self.env, self.light, self.spots
        )


class PickerState(object):
    """
    Utility class that helps to select matchable pairs of
    background and foreground images.
    """

    def __init__(self, layers):
        """
        Creates a new picker state object.

        :param layers: List of ImageLayer objects.
        """

        self.inner = {
            'bright': {
                'in': [],
                'out': []
            },
            'dark': {
                'in': [],
                'out': []
            }
        }

        for i, l in enumerate(layers):

            if l.env == 'both' and l.light == 'bright':

                self.inner['bright']['in'].append(i)
                self.inner['bright']['out'].append(i)

            elif l.env == 'in' and l.light == 'bright':

                self.inner['bright']['in'].append(i)

            elif l.env == 'out' and l.light == 'bright':

                self.inner['bright']['out'].append(i)

            elif l.env == 'both' and l.light == 'dark':

                self.inner['dark']['in'].append(i)
                self.inner['dark']['out'].append(i)

            elif l.env == 'in' and l.light == 'dark':

                self.inner['dark']['in'].append(i)

            elif l.env == 'out' and l.light == 'dark':

                self.inner['dark']['out'].append(i)


class LayerPicker(object):
    """
    Class to manage the matching pair selection of background and
    foreground images. A object of this class will also work as
    a container for all image for a given layer, that is, either
    for the background or foreground layer.
    """

    def __init__(self, folder, filetype):
        """
        Creates a new LayerPicker object given a folder containing
        the images for one of the layers (background or foreground).

        :param folder: String. The path to the folder containing the
                       images.
        :param filetype: String. The file extension for the image
                         files. e.g. ".jpg".
        """

        self.layers = []
        self.folder = folder
        self.filetype = filetype

        samples = glob.glob('{}*{}'.format(folder, filetype))

        for s in samples:

            self.layers.append(ImageLayer(s, filetype))

        self.initial_state = PickerState(self.layers)
        self.current_state = PickerState(self.layers)

        rd.seed(5)

    def reset(self):
        """
        Resets the state of the matching selector so that the same
        combinations can be made again.
        """

        self.current_state = PickerState(self.layers)

    def pop_match(self, other):
        """
        Returns a layer object for the image that can be combined
        with 'other'.

        :param other: ImageLayer object. The image layer object to
                      which one want to find the match for.
        :return: ImageLayer object. The image that can be combined
                 with 'other'.
        """

        rd.seed(time.clock())
        picked = rd.randint(0, len(self.current_state.inner[other.light][other.env]) - 1)
        return self.layers[self.current_state.inner[other.light][other.env].pop(picked)]

    def __iter__(self):
        """
        In the iteration with the operator "in", a object
        from the list of ImageLayer will be returned.
        """

        for l in self.layers:

            yield l

    def __len__(self):
        """
        Returns the number of ImageLayer objects contained.
        """

        return len(self.layers)

    def print_inventory(self):
        """
        Prints the number of images for each lighting and
        environment pair.
        """

        print('Inventory for ', self.folder)

        for light in self.current_state.inner:
            for env in self.current_state.inner[light]:

                print('light: {}, env: {} = {}'.format(
                    light, env, len(self.current_state.inner[light][env])
                ))


def compute_pos_and_scale(fimg, bimg, fore_base, back_base, depth, pos, min_height_factor=0.25):
    """
    Computes the position of a foreground image inside a background image and
    its scale, given a depth and pos value.

    :param fimg: Numpy array. The foreground image.
    :param bimg: Numpy array. The background image.
    :param fore_base: Not currently used, just ignore it.
    :param back_base: Not currently used, just ignore it.
    :param depth: Integer. The relative depth value of the foreground object
                  from the foreground specifications.
    :param pos: Integer. The horizontal position of the foreground object
                from the foreground specifications.
    :param min_height_factor: Float. The factor to determine the minimum
                              height of the foreground image and therefore,
                              its scale factor.

    :return: Tuple. The position given by 2 coordinates (a list) and the
             scale (float).
    """

    scale = (min_height_factor * bimg.shape[0])/(fimg.shape[0] * 0.9)

    scale_step = 3*scale/10.
    max_depth = 10
    scale += (max_depth - depth) * scale_step

    new_shape = array(fimg.shape[:2]) * scale

    position = [
        bimg.shape[0] - 0.9 * new_shape[0],
        bimg.shape[1]/2. - new_shape[1]/2.
    ]

    step = (bimg.shape[1] - new_shape[1])/4.

    position[1] += step * pos

    position = [round(x) for x in position]

    return position, scale

def generate_composition(back_layer, fore_layers, dest_folder, specs, logger, verbose=0):
    """
    Generates all compositions possible with the given background image and saves them
    in the specified folder.

    :param back_layer: ImageLayer object. The background image.
    :param fore_layers: LayerPicker object. The foreground images.
    :param dest_folder: String. The path to the folder where the resulting images
                        should be saved.
    :param specs: SpecList object. All specifications to generate each composition.
    :param logger: CombLogger object. Object to control metadata recording.
    :param verbose: Integer. The higher the more info are printed/shown.
    """

    if not isinstance(back_layer, ImageLayer):

        raise Exception('The back_layer must be an object of the class ImageLayer!')

    if not isinstance(fore_layers, LayerPicker):

        raise Exception('The fore_layers must be an object of the class LayerPicker!')

    if not isinstance(specs, SpecList):

        raise Exception('The specs must be an object of the class SpecList!')

    if verbose > 0:

        print('background = ', back_layer.name)

    for j, s in enumerate(specs):

        bimg = imread(back_layer.filepath)

        if len(bimg.shape) == 1:

            if len(bimg[0].shape) == 3:

                bimg = bimg[0]
            else:
                raise Exception('The image looks corrupted or it is in grayscale!')

        bimg = rescale(bimg, s.scale)
        bimg *= 255.

        for i, fs in enumerate(s.foreground_specs):

            if verbose > 0:

                print('spec {}, foreground set {} with {} foregrounds <{} %>'.format(j, i, fs.number, (j+1)*100./len(specs) + (100./len(specs))/len(s.foreground_specs) * i))

            comb = bimg.copy()
            fore_names = []
            class_str = 'sharp'

            # apply focus/motion blur in the background as specified
            if s.blur_method == 'macro':

                if back_layer.spots:

                    if verbose > 1:
                        print('blurring on background with spot lights')

                    # comb /= 255.
                    comb = blur_with_spot_light(comb, s.blur_level, verbose)
                    # comb *= 255.

                else:
                    if verbose > 1:
                        print('blurring on background')

                    comb = blur_img(comb, s.blur_level)

            fore_pos = []

            for pos, depth in fs:

                try:
                    f = fore_layers.pop_match(back_layer)

                except ValueError:

                    if verbose > 0:
                        print('Foreground layers RESET')

                    fore_layers.reset()
                    f = fore_layers.pop_match(back_layer)

                fore_names.append(f.name)

                if verbose > 1:

                    print(f.name)

                fimg = imread(f.filepath)

                if verbose > 1:
                    print('computing position and scale for foreground')

                # compute position and scale
                position, scale = compute_pos_and_scale(fimg, bimg, f.base, back_layer.base, depth, pos)
                # apply scale
                fimg = rescale(fimg, scale)
                fimg[:,:,:3] *= 255.

                fimg_mask, back_window, _ = foreground_mask(bimg, fimg, position)

                fore_pos.append(back_window)

                # apply focus/motion blur in the foreground as specified
                if s.blur_method == 'hyperfocal':

                    if verbose > 1:
                        print('blurring the foreground from front to back')

                    fimg_mask = blur_img(fimg_mask, max(0, s.blur_level - int((depth - fs.min_depth)/(10. - fs.min_depth) * s.blur_level)))
                    class_str = 'blur'

                elif s.blur_method == 'macro':

                    if verbose > 1:
                        print('blurring the foreground from back to front')

                    fimg_mask = blur_img(fimg_mask, max(0, depth - fs.min_depth - 2), fimg.shape)

                if s.motion_method == 'object_motion':

                    if verbose > 1:
                        print('applying motion blur in the foreground')

                    fimg_mask = motion_blur(fimg_mask, filter_size=get_motion_level(s.motion_level, fimg.shape), angle=s.motion_angle)
                    class_str = 'blur'

                if verbose > 1:
                    print('combining background and foreground')

                # combine both
                comb, _, _ = combine(comb, fimg_mask, (0,0))

            # apply focus/motion to both if specified
            if s.blur_method == 'whole':

                if back_layer.spots:

                    if verbose > 1:
                        print('blurring the whole image with spot lights')

                    comb = blur_with_spot_light(comb, s.blur_level, verbose)

                else:

                    if verbose > 1:
                        print('blurring the whole image')

                    comb = blur_img(comb, s.blur_level)

                class_str = 'blur'

            if s.motion_method == 'camera_motion':

                if verbose > 1:
                    print('applying motion blur to the whole image')

                comb = motion_blur(comb, filter_size=get_motion_level(s.motion_level, fimg.shape), angle=s.motion_angle)
                class_str = 'blur'

            if verbose > 1:
                print('applying noise')

            # apply noise
            comb = noise(comb, **get_noise_params(s.noise_level))

            img_idx = logger.add(back_layer, fore_names, s, i)

            # save image
            img_name = '{}combination_{}_{}.jpg'.format(dest_folder, str(img_idx), class_str)

            Image.fromarray(comb).save(img_name, format='JPEG', quality=100-s.compression)

            # save boxes
            save_boxes(fore_pos, comb.shape, img_name)


def generate_dataset(background_folder, foreground_folder, dest_folder, specs, log_file='dataset_inventory.csv', log_init_rows=0, verbose=2):
    """
    Generates all compositions specified in 'specs' with all background and foreground
    images, saving the resulting images in the given folder ('dest_folder').

    :param background_folder: String. Path to the folder containing all background image files.
    :param foreground_folder: String. Path to the folder containing all foreground image files.
    :param dest_folder: String. Path to the folder where the resulting images should be saved.
    :param specs: SpecList object. The specifications for the image compositions.
    :param log_file: String. Path to the file where the metadata should be saved.
    :param log_init_rows: Integer. Index of the row in the metadata file from which data should
                          be added in.
    :param verbose: Integer. The higher the more information are printed/shown.
    """

    background_layers = LayerPicker(background_folder, '.jpg')
    foreground_layers = LayerPicker(foreground_folder, '.png')

    logger = CombLogger(log_file, log_init_rows)

    for i, back in enumerate(background_layers):

        try:
            generate_composition(back, foreground_layers, dest_folder, specs, logger, verbose)

            foreground_layers.reset()

            print('>>>>>>> Status: {}%'.format((i+1)*100./len(background_layers)))

        except Exception as e:

            print(str(e))
            print('on line', e.__traceback__.tb_lineno)
            print(e.__traceback__.tb_frame.f_code)
            print('\nError on background {}'.format(str(back)))
            raise e

def get_motion_level(level, sh):
    """
    Maps the motion level value from the specification object to
    the actual usable parameter, i.e. the kernel size.

    :param level: Integer. Motion level value from the specification.
    :param sh: Array-like object. 2D shape of the image.

    :return: Integer. Kernel size.
    """

    kernel_min = max(3, 0.003 * min(sh[:2]))
    kernel_max = 60

    return int((kernel_max - kernel_min) * (level - 1.)/9. + kernel_min)

def get_noise_params(level):
    """
    Maps the noise level value from the specification to the
    actual usable parameters, i.e. noise intensities and kernel
    sizes.

    :param level: Integer. The noise level value from the specification.

    :return: Dictionary. The values for the "chr_i", "chr_size", "ill_i",
             "ill_size" parameters for the "noise" function.
    """

    chr_i_min = 5
    chr_i_max = 100
    chr_size_min = 7
    chr_size_max = 21
    ill_i_min = 5
    ill_i_max = 100
    ill_size_min = 1
    ill_size_max = 2

    level = (level - 1)/9.

    return dict(
        chr_i = int((chr_i_max - chr_i_min) * level + chr_i_min),
        chr_size = int((chr_size_max - chr_size_min) * level + chr_size_min),
        ill_i = int((ill_i_max - ill_i_min) * level + ill_i_min),
        ill_size = int((ill_size_max - ill_size_min) * level + ill_size_min)
    )

def detect_spot_lights(img, verbose=0):
    """
    Detects in the image the regions of light sources/high reflectance.
    So that we can simulate their "special behavior" when out of focus.

    :param img: Numpy array. The image.
    :param verbose: Integer. The higher the more information are
                    printed/shown.

    :return: Tuple. The detected blobs (Numpy array) and their
             RGB color (Numpy array).
    """

    if len(img.shape) == 3:

        if verbose > 1:
            print('\t\tconverting to gray scale')

        img_gray = rgb2grey(img)
    else:
        img_gray = img.copy()

    factor = 1.
    img_temp = img.copy()

    if max(img.shape) > 1000:

        factor = 1000./max(img.shape)
        img_gray = rescale(img_gray, factor)
        img_temp = rescale(img_temp, factor)

        if verbose > 1:
            print('\t\tresizing image | previous size: {}, new size: {}'.format(img.shape, img_gray.shape))

    img_gray[img_gray > 255] = 255.
    img_gray[img_gray < 0] = 0.

    if verbose > 1:
        print('\t\tlooking for blobs')
        print('\t\t', im_stats(img_gray))

    blobs = blob_doh(img_gray.astype('uint8'), max_sigma=50, num_sigma=100, threshold=0.01)

    if verbose > 1:
        print('\t\t{} blobs found'.format(len(blobs)))

    sizes = []
    brightness = []
    colors = []

    if verbose > 1:
        print('\t\tgetting the blobs colors and sizes')

    if verbose > 2:
        print('temp img stats =', im_stats(img_temp))
        plt.imshow(img_temp, interpolation='none')
        plt.show()

    for blob in blobs:
        y, x, r = blob
        window = imwindow(img_gray, y, x, (r+1, r+1))[0:2]
        brightness.append(img_gray[window].mean())
        colorwindow = img_temp[window[0], window[1],:]
        colorwindowselection = where(colorwindow.sum(2) <= 0.8*255*3)

        colors.append(colorwindow[colorwindowselection[0], colorwindowselection[1],:].mean(0))

        if verbose > 3:
            print('\t\t\tx = {}, y = {}, r = {}'.format(x, y, r))
            print('\t\t\twindow =', window)
            print('\t\t\tcolorwindow shape =', colorwindow.shape)
            print('\t\t\tcolor selection =', colorwindowselection,)
            print('\t\t\tcolor =', colors[-1])
            print('\t\t\tcolorwindow sums =', colorwindow.sum(2))
            plt.imshow(colorwindow/255., interpolation='none')
            plt.title('colorwindow')
            plt.show()

        sizes.append(r)

    blobs /= factor

    colors = array(colors)

    if verbose > 2:
        _, ax = plt.subplots(1,2)
        ax[0].hist(sizes, bins=100)
        ax[0].set_title('sizes')
        ax[1].hist(brightness, bins=100)
        ax[1].set_title('brightness')
        plt.show()

    if verbose > 1:
        print('\t\tcomputing the histograms')

    gh, gx = histogram(sizes, bins=100)

    h, x = histogram(brightness, bins=100)

    hmin = x[80]

    if verbose > 2:
        plt.imshow(img/255., interpolation='none')

        for blob in blobs:
            y, x, r = blob
            plt.scatter(x,y, color='b', lw=r)

    spots = where(brightness >= hmin)

    if verbose > 2:
        print('\t\t', im_stats(img))

        for blob in blobs[spots[0]]:
            y, x, r = blob
            plt.scatter(x,y, color='r', lw=r)

        plt.show()

    return blobs[spots[0]], colors[spots[0]]

def draw_spot_light(img, blobs, level, colors=None, verbose=0):
    """
    Draws the out of focus "sources of light" in the given image
    based on the given 'blobs' location.

    :param img: Numpy array. The image.
    :param blobs: Numpy array. The blobs coordinates.
    :param level: Integer. The intensity level of the blur.
    :param colors: Numpy array. The RGB color of each blob.
    :param verbose: Integer. The higher the more information are
                    printed/shown.

    :return: Numpy array. The image with the out of focus blobs drawn.
    """

    if len(img.shape) == 3:

        new_img = draw_spot_light_colorful(img, blobs, level, colors, verbose)

    else:
        new_img = draw_spot_light_gray(img, blobs, level)

    return new_img

def draw_spot_light_gray(img, blobs, size=0.2):
    """
    Draws the out of focus "sources of light" in the given image
    based on the given 'blobs' location. Works only with
    gray-scale images.

    :param img: Numpy array. The image.
    :param blobs: Numpy array. The blobs coordinates.
    :param size: Float. Radius of the circles to be drawn.

    :return: Numpy array. The image with the out of focus blobs drawn.
    """

    if len(img.shape) == 3:

        img = rgb2grey(img)

    alpha_img = zeros_like(img)
    temp_img = zeros_like(img)

    for blob in blobs:

        y, x, r = blob
        rr, cc = circle(y, x, size)

        temp_img[rr, cc] += 0.5 * r
        alpha_img[rr, cc] = 1.

    kernel = getGaussianKernel(2, 1) * getGaussianKernel(2, 1).T

    _, ax = plt.subplots(1,3)

    ax[0].imshow(temp_img.copy())

    temp_img = filter2D(temp_img, -1, kernel)
    alpha_img = filter2D(alpha_img, -1, kernel)
    ax[1].imshow(temp_img)

    ax[2].imshow(alpha_img)

    plt.show()

    new_img = img.copy()

    new_img[temp_img > 0] = 0.2 * (temp_img[temp_img > 0] - temp_img[temp_img > 0].min())/(temp_img[temp_img > 0].max() - temp_img[temp_img > 0].min()) + 0.8
    new_img[temp_img > 0] = (1 - alpha_img[temp_img > 0]) * img[temp_img > 0] + alpha_img[temp_img > 0] * new_img[temp_img > 0]

    return new_img

def draw_spot_light_colorful(img, blobs, level=0.2, colors=None, verbose=0):
    """
    Draws the out of focus "sources of light" in the given image
    based on the given 'blobs' location. Works only with colorful
    images, i.e. 3D arrays.

    :param img: Numpy array. The image.
    :param blobs: Numpy array. The blobs coordinates.
    :param level: Integer. The intensity level of the blur.
    :param colors: Numpy array. The RGB color of each blob.
    :param verbose: Integer. The higher the more information are
                    printed/shown.

    :return: Numpy array. The image with the out of focus blobs drawn.
    """

    if len(img.shape) == 1:

        blob_img = img.copy()
        img = zeros((img.shape[0], img.shape[1], 3))
        img[:,:,0] = blob_img.copy()
        img[:,:,1] = blob_img.copy()
        img[:,:,2] = blob_img.copy()

    blob_img = zeros_like(img)
    alpha_img = zeros_like(img)
    selection = zeros_like(img)

    rs = [b[2] for b in blobs]
    rmax = max(rs)
    rmin = min(rs)

    size = get_kernel_size(level, img.shape)

    for i, blob in enumerate(blobs):

        y, x, r = blob
        rr, cc = circle(y, x, size * 1.5)

        rr = safe_list_range(rr, (0, selection.shape[0]-1))
        cc = safe_list_range(cc, (0, selection.shape[1]-1))

        selection[rr, cc, :] = 1.

        if colors is not None:

            hsv = rgb2hsv(colors[i].reshape((1,1,3)))

            if rmax == rmin:
                hsv[0,0,2] = 0.9
            else:
                hsv[0,0,2] = 0.6 * (r - rmin)/(rmax - rmin) + 0.4

            hsv[0,0,1] *= 1.6
            rgb = hsv2rgb(hsv)[0,0,:]

            if verbose > 1:
                print('hsv =', hsv, ', rgb =', rgb)

            blob_img[rr, cc, :] = minimum(1., blob_img[rr, cc, :] + rgb)
        else:
            blob_img[rr, cc, :] += 0.5 * r

        alpha_img[rr, cc, 0] = 1.

    img /= 255.

    if verbose > 1:
        print('img = ', im_stats(img))
        print('blob img = ', im_stats(blob_img))

    if verbose > 2:
        _, ax = plt.subplots(1,2)
        ax[0].imshow(img, interpolation='none')
        ax[0].set_title('img')
        ax[1].imshow(blob_img, interpolation='none')
        ax[1].set_title('blob img')

        plt.show()

    ksize = 1.
    kernel = getGaussianKernel(int(ksize*3+1), ksize)
    kernel = kernel * kernel.T

    selection = filter2D(selection, -1, kernel)
    blob_img[blob_img == 0] = blob_img[selection > 0].max()
    blob_img = filter2D(blob_img, -1, kernel)
    noise_mask = random.randn(*blob_img.shape[:2])
    noise_mask = filter2D(noise_mask, -1, kernel) * 0.1
    blob_img[:,:,0] += noise_mask
    blob_img[:,:,1] += noise_mask
    blob_img[:,:,2] += noise_mask
    blob_img[blob_img > 1] = 1.
    blob_img[blob_img < 0] = 0.

    alpha_img[:,:,0] = filter2D(alpha_img[:,:,0], -1, kernel)
    alpha_img[:,:,1] = alpha_img[:,:,0]
    alpha_img[:,:,2] = alpha_img[:,:,0]

    new_img = img.copy()

    new_img[selection > 0] = 0.3 * (blob_img[selection > 0] - blob_img[selection > 0].min())/(blob_img[selection > 0].max() - blob_img[selection > 0].min()) + 0.7

    new_img[selection > 0] = (1. - alpha_img[selection > 0]) * img[selection > 0] + alpha_img[selection > 0] * new_img[selection > 0]

    return new_img * 255.

def get_kernel_size(level, shapes):
    """
    Maps the blur level value from the specifications to the actual
    usable parameter, i.e. the kernel size.

    :param level: Integer. The blur level value from the specification.
    :param shapes: Array-like object. The 2D shape of the image.

    :return: Float. The kernel size.
    """

    factor = level * 0.0015 + 0.003

    ksize = factor * min(shapes[:2])

    return ksize

def blur_img(img, level, shape=None, verbose=0):
    """
    Blurs an image simulating the out of focus blur,
    that is omnidirectional.

    :param img: Numpy array. The image.
    :param level: Integer. The blur level value from the specification.
    :param shape: Array-like object. The relative shape. If left None,
                  the shape of 'img' is used.
    :param verbose: Integer. The higher the more information are
                    printed/shown.

    :return: Numpy array. The blurred image.
    """

    if level == 0:

        return img

    if shape is None:

        shape = img.shape

    ksize = get_kernel_size(level, shape)
    kernel = getGaussianKernel(int(ksize*3) + 1, ksize)
    kernel = kernel * kernel.T

    img = filter2D(img, -1, kernel)

    if verbose == 1:

        print('bluring -------------\nlevel = {}\nksize = {}\nkernel shape = {}'.format(level, ksize, kernel.shape))

    return img

def blur_with_spot_light(img, level, verbose=0):
    """
    Blurs an image with regions with source lights or high reflectance,
    simulating the out of focus blur.

    :param img: Numpy array. The image.
    :param level: Integer. The blur level value from the specification.
    :param verbose: Integer. The higher the more information are
                    printed/shown.

    :return: Numpy array. The blurred image.
    """

    if verbose > 1:
        print('\tdetecting spot lights')

    blobs, colors = detect_spot_lights(img, verbose)

    start = time.clock()

    img = blur_img(img, level, verbose=verbose)

    end = time.clock()

    if len(blobs) >= 1:
        img = draw_spot_light(img, blobs, level, colors, verbose)

    if verbose > 2:
        print('\tdetection time =', end - start)

    return img

def combine(back, fore, pos):
    """
    Draws the foreground image in the background image in the given
    position. The foreground image needs to contain the alpha channel.

    :param back: Numpy array. 3D array with the background image.
    :param fore: Numpy array. 3D array with the foreground image.
    :param pos: Array-like object. The 2D position of the upper-
                left corner of the foreground image inside the
                background image.

    :return: Tuple. The resulting image and the slices for the
             background and the foreground image.
    """

    alpha = zeros((fore.shape[0], fore.shape[1], 3))
    alpha[:,:,0] = fore[:,:,3]
    alpha[:,:,1] = fore[:,:,3]
    alpha[:,:,2] = fore[:,:,3]

    back_indexes = (
        slice(
            max(0, pos[0]),
            min(back.shape[0], pos[0]+fore.shape[0])
        ),
        slice(
            max(0, pos[1]),
            min(back.shape[1], pos[1]+fore.shape[1])
        ),
        slice(0,3)
    )

    fore_indexes = (
        slice(
            0 - min(0, pos[0]),
            fore.shape[0] + min(0, (back.shape[0] - pos[0]) - fore.shape[0])
        ),
        slice(
            0 - min(0, pos[1]),
            fore.shape[1] + min(0, (back.shape[1] - pos[1]) - fore.shape[1])
        ),
        slice(0,3)
    )

    comb = back[:,:,:]

    comb[back_indexes] = (1 - alpha[fore_indexes]) * back[back_indexes]
    comb[back_indexes] += alpha[fore_indexes] * fore[fore_indexes]

    return comb, back_indexes, fore_indexes

def foreground_mask(back, fore, pos):
    """
    Creates an image with the shape of the background image, but
    containing only the foreground image information in the
    given position 'pos'.

    :param back: Numpy array. The background image.
    :param fore: Numpy array. The foreground image.
    :param pos: Array-like object. The 2D position of the upper-
                left corner of the foreground image inside the
                background image.

    :return: Tuple. The resulting image and the slices for the
             background and foreground images.
    """

    comb = zeros((back.shape[0], back.shape[1], 4))

    pos[0] = int(pos[0])
    pos[1] = int(pos[1])

    back_indexes = (
        slice(
            max(0, pos[0]),
            min(back.shape[0], pos[0]+fore.shape[0])
        ),
        slice(
            max(0, pos[1]),
            min(back.shape[1], pos[1]+fore.shape[1])
        ),
        slice(0,4)
    )

    fore_indexes = (
        slice(
            0 - min(0, pos[0]),
            fore.shape[0] + min(0, (back.shape[0] - pos[0]) - fore.shape[0])
        ),
        slice(
            0 - min(0, pos[1]),
            fore.shape[1] + min(0, (back.shape[1] - pos[1]) - fore.shape[1])
        ),
        slice(0,4)
    )

    comb[back_indexes] = fore[fore_indexes]

    return comb, back_indexes, fore_indexes

def imwindow(img, i, j, size):
    """
    Returns the slices corresponding to the rectangle around the
    given pixel (i,j).

    :param img: Numpy array. The image.
    :param i: Integer. The index of the image row.
    :param j: Integer. The index of the image column.
    :param size: Array-like object. The 2D shape of the rectangle.

    :return: Tuple. The slices for the rows and columns. The two
             other elements can be ignored.
    """

    sx = (size[0] - 1)/2
    sy = (size[1] - 1)/2

    slicex = slice(
        int(max(0, i - sx)),
        int(min(img.shape[0], i + sx + 1))
    )

    slicey = slice(
        int(max(0, j - sy)),
        int(min(img.shape[1], j + sy + 1))
    )

    sizex = slicex.stop - slicex.start
    sizey = slicey.stop - slicey.start

    slicekx = slice(
        max(0, sx - sizex),
        min(size[0], sx + sizex + 1)
    )

    sliceky = slice(
        max(0, sy - sizey),
        min(size[1], sy + sizey + 1)
    )

    return slicex, slicey, slicekx, sliceky

def im_stats(img):
    """
    Utility function that returns the following statistics
    of an image: average, standard deviation, maximum and
    minimum value.

    :param img: Numpy array. The image.

    :return: Dictionary. The 4 values.
    """

    stats = dict(
        mean = img.mean(),
        std = img.std(),
        max = img.max(),
        min = img.min()
    )

    return stats

def safe_list_range(l, r):
    """
    Utility function that returns a transformed list where each
    element is trimmed in the given range.

    e.g. input list = [20, 40, -4, 7], range = [0, 20], output = [20, 20, 0, 7]

    :param l: Array-like object. 1D array, the list to be trimmed.
    :param r: Array-like object. 2 elements array with the minimum and maximum
              desired values.

    :return: List. The elements of 'l' trimmed with the given range.
    """

    new_l = [min(max(i, r[0]), r[1]) for i in l]

    return new_l

def save_boxes(slices, img_shape, image_name):
    """
    Saves the bounding box information in a ".boxes" file in the
    FDDB (http://vis-www.cs.umass.edu/fddb/) format.

    :param slices: Collection of slices. The corresponding slices for all
                   bounding boxes.
    :param img_shape: Array-like object. The 2D shape of the image.
    :param image_name: String. A name to be used when saving the file.
    """

    with open(image_name + '.boxes', 'w') as boxes_file:

        writer = csv.DictWriter(boxes_file, ['x', 'y', 'w', 'h'], delimiter=',')
        writer.writeheader()

        for s in slices:

            x = s[1].start
            dx = s[1].stop - s[1].start

            y = s[0].start
            dy = s[0].stop - s[0].start

            box = dict(
                x=str((x + dx/2)/img_shape[1]),
                y=str((y + dy/2)/img_shape[0]),
                w=str(dx/img_shape[1]),
                h=str(dy/img_shape[0])
            )

            writer.writerow(box)

# Unit Tests 
class ClassTests(TestCase):

    def assertEqual(self, first, second, msg=None):
        if first != second:
            e_msg = ''.join(
                ['\nExpected: ', str(second), ' Found: ', str(first)])
            print(e_msg)

        TestCase.assertEqual(self, first, second, msg)

    def test_1_image_combination(self):

        dataset_root =  '/path_to/dataset/'
        foreground_folder = 'foreground/'
        background_folder = 'background/'

        back = imread(dataset_root + background_folder + 'image_2-200-in-bright.jpg')
        back = rescale(back, 0.4)
        kernel = getGaussianKernel(33, 11) * getGaussianKernel(33, 11).T
        back = filter2D(back, -1, kernel)

        fore = imread(dataset_root + foreground_folder + 'image_2-11-in-bright.png')
        fore = rescale(fore, 0.24)
        kernel = getGaussianKernel(33, 5) * getGaussianKernel(33, 5).T
        fore = filter2D(fore, -1, kernel)
        fore[fore < 0] = 0
        fore[fore > 1] = 1

        print('back shape =', back.shape)
        print('fore shape =', fore.shape)

        print('\nback stats =', im_stats(back))
        print('fore stats =', im_stats(fore))

        comb, back_slice, fore_slice = combine(back, fore, (400, 500))
        print(im_stats(comb))

        plt.imshow(comb, interpolation='none')
        plt.show()

        imsave(dataset_root + 'comb3_nothing_in_focus.jpg', comb)

    def test_2_spot_lights(self):

        dataset_root = '/path_to/dataset/'
        background_folder = 'background/'

        back = imread(dataset_root + background_folder + 'image_7-400-out-dark-lights.jpg')
        back = rescale(back, 0.7)

        blur_with_spot_light(back, level=4)

    def test_3_dataset_creation(self):

        background_path = '/path_to/dataset/background/'
        foreground_path = '/path_to/dataset/foreground/'
        dest_path = '/path_to/images/'

        specs = SpecList()

        generate_dataset(background_path, foreground_path, dest_path, specs)

    def test_4_comb_and_specs(self):

        specs = SpecList()

        specs.add(Spec(
            foreground_specs=[
                ForeSpecs(positions=[-2], depths=[0]),
                ForeSpecs(positions=[1], depths=[2]),
                ForeSpecs(positions=[0], depths=[4]),
            ],
            blur_method='macro',
            blur_level=8,
            noise_level=1,
            compression=0,
            scale=0.8
        ))

        back = ImageLayer('../background/image_1-600-out-dark-lights.jpg', '.jpg')
        fore_layers = LayerPicker('../foreground/', '.png')
        logger = CombLogger('db_logger_test.csv')

        generate_composition(back, fore_layers, '../images/', specs, logger, verbose=3)

    def test_5_logger(self):

        spec = Spec(
            foreground_specs=[
                ForeSpecs(positions=[0, 2], depths=[5, 1]),
                ForeSpecs(positions=[-2], depths=[1]),
            ],
            blur_method='macro',
            blur_level=6,
            noise_level=9,
            compression=10
        )

        logger = CombLogger('db_logger_test.csv')

        fore_list = ['A', 'B']
        back = ImageLayer('image_4-600-out-bright-no_lights.jpg', '.jpg')
        print(logger.add(back, fore_list, spec, 0))

        fore_list = ['C']
        back = ImageLayer('image_4-600-out-bright-no_lights.jpg', '.jpg')
        print(logger.add(back, fore_list, spec, 1))

    def test_6_inventory(self):

        foregrounds = LayerPicker('../foreground/', '.png')
        backgrounds = LayerPicker('../background/', '.jpg')

        foregrounds.print_inventory()
        backgrounds.print_inventory()

    def test_7_pos_and_scale(self):

        bimg = zeros((100,300,3))
        fimg = zeros((50,20,3))

        position, scale = compute_pos_and_scale(fimg, bimg, 100, 500, 0, 0)

        print(position)
        print(scale)

    def test_8_noise(self):

        img = imread('../background/image_200-250-in-bright-no_lights.jpg')
        img = rescale(img, 0.4)
        img *= 255.

        print(im_stats(img))

        img_noise = noise(img)

        plt.imshow(img_noise)
        plt.show()

        Image.fromarray(img_noise).save('img_noise_test.jpg', format='JPEG', quality=100)


if __name__ == '__main__':
    # loads and runs the Unit Tests
    suite = TestLoader().loadTestsFromTestCase(ClassTests)
    TextTestRunner(verbosity=2, ).run(suite)