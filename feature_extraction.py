"""
This module contains functions and classes to manage databases and
the feature extraction process.
"""

# here goes the imports
from unittest import TestLoader, TextTestRunner
from unittest.case import TestCase

from numpy import array, concatenate, where, inf, gradient, absolute
from pre_processors import imread, denoise, normalize_size
from skimage.color import rgb2grey
from skimage.feature import hog
from fft import fft_features
from sharpness_behavior import sharpness_behavior
from yuzhen_method import yuzhen_blur_metric
from sieberth_method import sieberth_blur_metric

import dill
import os
import object_features as of
import csv


def sharpness_class(file_name, **kwargs):
    """
    Extracts the class information from the filename
    of the image. Two classes are expected: "blur" and
    "sharp".

    :param file_name: String. Path to the image file.

    :return: List of integer. The image class.
    """

    if file_name is None:
        return [-1]

    # removes the folder part of the path
    file_name = file_name.split('/')[-1]

    # converts the entire file name/path to lower case
    file_name = file_name.lower()

    if 'blur' in file_name:
        return [0]

    elif 'sharp' in file_name:
        return [1]

    # if none of the above strings are present, there is no class information
    # in the file name/path, then ...
    else:
        # raises an exception
        raise Exception('Invalid class label')


class FW(object):
    """
    Function Wrapper (FW) class. Utility class to wrap a function with its
    parameters.
    """

    def __init__(self, f, **kwargs):
        """
        Creates a function wrapper object.

        Parameters:
            :f: a python function object;
            :params: a dictionary with the parameters for 'f';
        """

        self.name = f.__name__
        self.f = f
        self.params = kwargs

    def __call__(self, *args, **kwargs):
        """
        Applies the wrapped function with its inner parameters to the first
        argument.
        """

        return self.f(args[0], **self.params)


class Feature(object):
    """
    Class to model a feature, descriptor or other information to be extracted
    from an image file. It is composed by a extraction function and its
    parameter and a list of pre-processing functions to be applied before the
    feature extraction function.
    """

    def __init__(self, f, pre_processing=list(), **kwargs):
        """
        Constructs a new Feature object.

        :param f: Function-like object. The feature extraction function. It
                  must return a list and it will receive the output of the
                  last pre-processing function as its input.
        :param pre_processing: A list of pre-processing functions. Usually
               a list of FW objects, since the pre-processing functions
               generally have parameters of their own. They are applied in
               the order they appear in the list and each function receives
               the output of its predecessor as input. The first function
               receives the path of the image file as input.
        :param kwargs: The parameters for *f* and/or its name.
        """

        self.name = f.__name__ if 'name' not in kwargs else kwargs['name']
        self.f = f
        self.params = kwargs
        self.size = len(f(None))
        self.pre_processing = pre_processing

    def __call__(self, *args, **kwargs):
        """
        Extracts the feature vector out of the image after
        applying all pre-processing functions.

        :param args[0]: String. The path of the image file.

        :return: List. The feature/descriptor vector.
        """

        input = args[0]

        for p in self.pre_processing:
            input = p(input)

        return self.f(input, **self.params)


def int_if_possible(text):
    """
    Converts a string into its corresponding integer number,
    if possible, otherwise, returns the input itself.

    :param text: String. A natural number in text format.

    :return: Either the integer corresponding to the input string or the
             input itself.
    """

    if type(text) is not str:
        raise Exception('int_if_possible expects a string as input!\ngot {}'.format(text))

    try:
        return int(text)

    except ValueError as e:

        return text

def hog_features(img, **kwargs):
    """
    Wrapper for the HOG features function so it will return a list
    as our DataBase object requires.

    :param img: Numpy array containing the image.
    :param kwargs: Parameters of the HOG function from Scikit-Learn.

    :return: List of numbers. The HOG features.
    """

    if img is None:
        return [-1] * 225

    f = hog(img, **kwargs)

    return list(f)


def sharpness(filepath, **kwargs):
    """
    A simple sharpness metric based on the estimated gradient of
    the image. This metric is called "GSM" in the dissertation.

    :param filepath: String. Path to the image file.

    :return: List. Containing only the estimated sharpness value.
    """
    if filepath is None:
        return [-1]

    img = imread(filepath)
    img = rgb2grey(img)

    gx, gy = gradient(img)

    sh = (absolute(gx) + absolute(gy)) / 2

    return [sh.mean()]


class DataBase(object):
    """
    Class to represent and manage databases of images.
    """

    def __init__(self):
        """
        Initiates an empty database.
        """

        self.features = dict()
        self.data = dict()
        self.samples_path = []
        self.folder = ''
        self.last_file_extracted = -1

    def add_variable(self, f):
        """
        Add a "variable", which can be values for class information or
        features.

        :param f: Feature object.

        :return: DataBase. The caller itself.
        """

        self.features[f.name] = f

        if f.name in self.data:
            print('WARNING: variable already exists, overwriting!')

        self.data[f.name] = []

        return self

    def get_data(self, variables=list(), samples=None):
        """
        Returns a subset of the data contained in this database. If
        no variables are given, data from all variables are returned.
        Similarly, if no samples' index are given, data from all
        samples are returned.

        :param variables: List of strings. The variable names.
        :param samples: List of integers. The indexes of the samples.

        :return: Numpy array with the returned data.
        """

        if len(variables) == 0:
            variables = self.features.keys()

        if samples is None:
            samples = range(len(self.samples_path))

        data = None

        for v in variables:

            d = array(self.data[v])[samples]

            if data is None:
                data = d.copy()

            else:
                data = concatenate((data, d), axis=1)

        return data

    def add_files_from_folder(self, folder, file_type='jpg'):
        """
        Add all files from a given folder with a specific file extension
        to the database buffer. The features will be extracted from
        these files.

        :param folder: String. Path to the folder.
        :param file_type: String. File extension, e.g. "jpg".
        """

        # gets a complete list of all files inside the 'folder' with the
        # extension given by 'filetype'
        files = [f for f in os.listdir(folder)
                 if os.path.isfile(''.join([folder, f]))
                 and f.lower().endswith(file_type)]

        self.folder = folder
        self.samples_path = []

        # for each file 'f', ...
        for i, f in enumerate(files):
            self.samples_path.append(f)

    def add_metadata(self, metadata_path):
        """
        Get metadata information from a CSV file and add it to
        every sample in the database by its ID. The ID should be
        the sample's filename. Each metadata field is added as a
        variable of the database.

        :param metadata_path: String. Path to the CSV file.
        """

        if len(self.samples_path) == 0:

            raise Exception('There is no file added yet!')

        fieldnames = ['combination', 'background', 'foreground', 'position', 'depths', 'blur_method', 'blur_level',
                      'motion_method', 'motion_level', 'motion_angle', 'noise', 'compression']

        fieldnames_idx_map = dict([(f, i) for i, f in enumerate(fieldnames)])

        for variable in fieldnames:
            if variable not in self.features.keys():

                self.add_variable(Feature(lambda x: [x], name=variable))
                self.data[variable] = [inf] * len(self.samples_path)

        metadata = []

        with open(metadata_path, 'r') as metadata_file:

            reader = csv.DictReader(metadata_file, None, delimiter=',')

            for sample in reader:

                data = []
                for v in fieldnames:
                    data.append(sample[v])

                metadata.append(data)

        metadata = array(metadata)

        for i, sample in enumerate(self.samples_path):

            comb = sample.split('_')[1]
            x = where(metadata[:, 0] == comb)[0]

            if len(x) > 0:

                x = x[0]

                for v in fieldnames:

                    if len(metadata[x, :]) == len(fieldnames):
                        self.data[v][i] = int_if_possible(str(metadata[x, fieldnames_idx_map[v]]))

    def extract(self, variables=list(), from_last=False):
        """
        Runs the extraction function from the feature objects. If no
        variable names are given, it tries to extract for all
        variables added.

        :param variables: List of strings. The names of variables whose
                          values should be extracted.
        :param from_last: Boolean. Whether it should start the process
                          from the last known processed sample or
                          from the beginning.
        """

        if len(variables) == 0:

            variables = self.features.keys()

        initial = self.last_file_extracted + 1 if from_last and self.last_file_extracted >= 0 else 0

        # for each file 'f', ...
        for i in range(initial, len(self.samples_path)):

            f = self.samples_path[i]

            # initiates the image's content variable
            img = '{}{}'.format(self.folder, f)

            # for each feature function, ...
            for v in variables:

                # extract the feature and appends it on 'x'
                self.data[v].append(self.features[v](img))

            self.last_file_extracted = i

            print('extracting features: {} % | {}'.format((i+1) * 100./len(self.samples_path), f))

        self.last_file_extracted = -1

    def safely_extract(self, variables=list(), from_last=False, database_file='database.db'):
        """
        Utility wrapper for the "extract" method. It will detect any
        possible exception and guarantee that the database object
        will always be saved to disc with all the extracted data up
        to the latest successfully extracted sample.

        :param variables: List of strings. The names of variables whose
                          values should be extracted.
        :param from_last: Boolean. Whether it should start the process
                          from the last known processed sample or
                          from the beginning.
        :param database_file: String. Path to the file where the database
                              object will be saved.
        """

        try:
            self.extract(variables, from_last)

        except Exception as e:

            print('ERROR\n--------------------')
            print(e.__cause__, '\n---------------------')
            print(e.args, '\n---------------------')

        finally:

            print('\nSaving Database ...')
            self.save_to_file(database_file)
            print('Done!')

    def save_to_file(self, file_path):
        """
        Saves the current state of this database object to disc.

        :param file_path: String. Path to the file where the
                          object will be saved.
        """

        try:
            # opens the specified file to write as a binary file
            with open(file_path, 'wb') as output:
                # dumps the content of this object into the file
                dill.dump(self, output)

        except Exception as e:
            # if anything went wrong, raises an exception
            raise Exception(''.join(['Error when trying to write the '
                                          'object to the file: ', file_path,
                                          '\n', e.args]))

    @staticmethod
    def load_from_file(file_path):
        """
        Loads a database object from disc.

        :param file_path: String. The path to the file where the
                          database object's data is stored.

        :return: DataBase object.The loaded data.
        """

        try:
            # opens the input file to read as a binary file
            with open(file_path, 'rb') as input:
                # loads its content
                db = dill.load(input)

            # returns the DataBase object loaded
            return db

        except Exception as e:
            # if anything went wrong, raises an exception
            raise Exception(''.join(['Error when trying to read from the '
                                     'file: ', file_path, '\n', e.message]))


# Unit Tests and executions
class ClassTests(TestCase):

    def assertEqual(self, first, second, msg=None):
        if first != second:
            e_msg = ''.join(
                ['\nExpected: ', str(second), ' Found: ', str(first)])
            print(e_msg)

        TestCase.assertEqual(self, first, second, msg)

    def test_1_data_base(self):

        db = DataBase()

        db.add_variable(
            Feature(sharpness_behavior, pre_processing=[
                FW(imread),
                FW(normalize_size),
                FW(rgb2grey)
            ])
        ).add_variable(
            Feature(sharpness_class)
        ).add_variable(
            Feature(fft_features, pre_processing=[
                FW(imread),
                FW(rgb2grey),
                FW(normalize_size),
                FW(denoise)
            ])
        )

        print(db.features.keys())

        db.add_files_from_folder('images_test/', 'jpg')

        db.samples_path[3] = 'non_existing_file.jpg'

        db.safely_extract(variables=['sharpness_class', 'fft_features'], database_file='database_test.db')

        db = DataBase.load_from_file('database_test.db')

        print('last file extracted =', db.last_file_extracted)

        db.add_files_from_folder('images_test/', 'jpg')

        db.safely_extract(variables=['sharpness_class', 'fft_features'], from_last=True, database_file='database_test.db')

        data = db.get_data(variables=['fft_features', 'sharpness_class'])

        print(data.shape)
        print(data)

    def test_run(self):

        db = DataBase()

        db.add_variable(
            Feature(sharpness_behavior, pre_processing=[
                FW(imread),
                FW(rgb2grey),
                FW(normalize_size),
                FW(denoise)
            ])
        ).add_variable(
            Feature(fft_features, pre_processing=[
                FW(imread),
                FW(rgb2grey),
                FW(normalize_size),
                FW(denoise)
            ])
        ).add_variable(
            Feature(of.extract, name='object_features')
        ).add_variable(
            Feature(sharpness_class)
        ).add_variable(
            Feature(of.extract_hog, use_boxes=False, name='hog'),
        ).add_variable(
            Feature(of.extract_hog, use_boxes=True, name='object_hog'),
        )

        db.add_files_from_folder('images/', 'jpg')
        db.add_metadata('dataset_inventory_0.csv')
        db.add_metadata('dataset_inventory_1.csv')

        db.safely_extract(
            variables=[
                'sharpness_class',
                'fft_features',
                'sharpness_behavior',
                'object_features',
                'hog',
                'object_hog'
            ],
            from_last=True,
            database_file='database.db'
        )
        # db.safely_extract(variables=['object_features'], from_last=True, database_file='database.db')

    def test_extract_test_set(self):

        db = DataBase()

        db.add_variable(
            Feature(sharpness_behavior, pre_processing=[
                FW(imread),
                FW(rgb2grey),
                FW(normalize_size),
                FW(denoise)
            ])
        ).add_variable(
            Feature(fft_features, pre_processing=[
                FW(imread),
                FW(rgb2grey),
                FW(normalize_size),
                FW(denoise)
            ])
        ).add_variable(
            Feature(of.extract, name='object_features')
        ).add_variable(
            Feature(sharpness_class)
        ).add_variable(
            Feature(of.extract_hog, use_boxes=False, name='hog'),
        ).add_variable(
            Feature(of.extract_hog, use_boxes=True, name='object_hog'),
        ).add_variable(
            Feature(yuzhen_blur_metric, name='yuzhen')
        )

        db.add_files_from_folder('test_set/images/', 'jpg')
        db.add_metadata('testset_inventory_0.csv')

        db.safely_extract(
            variables=[
                'sharpness_class',
                'fft_features',
                'sharpness_behavior',
                'object_features',
                'hog',
                'object_hog'
            ],
            from_last=True,
            database_file='database_testset.db'
        )

    def test_metadata(self):

        db = DataBase()

        db.add_files_from_folder('../images_test_2/', 'jpg')

        db.add_metadata('../dataset_inventory_1.csv')

        noise = db.get_data(variables=['noise'], samples=[160,161,162,163,164])
        print(db.samples_path[160:165])
        print(noise)

        combs = db.get_data(variables=['combination'], samples=[160, 161, 162, 163, 164])
        print(db.samples_path[160:165])
        print(combs)

    def test_extract_single_feature(self):

        db = DataBase.load_from_file('splited_database_testset_3.db')
        db.add_variable(
            Feature(sieberth_blur_metric, name='sieberth')
        )

        db.safely_extract(variables=['sieberth'], database_file='splited_database_testset_3.db', from_last=False)


if __name__ == '__main__':
    # loads and runs the Unit Tests
    suite = TestLoader().loadTestsFromTestCase(ClassTests)
    TextTestRunner(verbosity=2, ).run(suite)