"""
This module contains functions to split the database and an utility class
to make filtering samples from the database easier.
"""

# here goes the imports
from unittest import TestLoader, TextTestRunner
from unittest.case import TestCase

from feature_extraction import DataBase, Feature
from sklearn.cross_validation import train_test_split
from numpy import array, floor


def split_data(database_file, dest_file='split_database.db'):
    """
    Splits the database into training and testing samples. 2/3 of the
    samples go for training and 1/3 for testing.
    The samples are chosen according to their combination ID, which
    follows the order of creation of the images. Since 3 images were
    created for each different specification, only varying the foreground,
    we assure all variations of type of blur, types of scenes, noise,
    compression and scale are equally represented in both sets.

    :param database_file: String. The path to the database file.
    :param dest_file: String. The path to the new database file with the
                      samples split.
    """
    db = DataBase.load_from_file(database_file)

    categories = get_stratify_info_per_comb_id(db)

    dumb_function = lambda x: [x]
    splits_variable = Feature(dumb_function, name='splits')
    db.add_variable(splits_variable)

    samples = array([i for i in range(len(db.samples_path))])

    assert len(categories) == len(samples)

    train, test = train_test_split(samples, stratify=categories, test_size=2/3.)

    splits = [''] * len(samples)

    for t in train:
        splits[t] = 'train'

    for t in test:
        splits[t] = 'test'

    db.data['splits'] = splits

    db.save_to_file(dest_file)

def get_stratify_info_per_comb_id(db):
    """
    Make groups out of the samples of the database with every
    3 samples in the order they were inserted in the database,
    which also matches its combination ID.

    :param db: The database object.

    :return: Numpy array of integers. The group label of the samples.
    """

    ids = array(get_comb_ids(db))

    idxs = floor(ids/3).astype('int32')

    return idxs

def get_comb_ids(db):
    """
    Extracts the combination IDs of every sample in the database.

    :param db: The database object.

    :return: List of integers. The combination ID of the samples
             in *db*.
    """

    ids = []

    for p in db.samples_path:

        id = int(p.split('/')[-1].split('_')[1])

        ids.append(id)

    return ids


class FilterCondition(object):
    """
    Utility class to help filtering samples from database objects.
    """

    # The supported condition operations:
    ops = {
        '==': lambda x, y: x == y,
        '>': lambda x, y: x > y,
        '<': lambda x, y: x < y,
        '>=': lambda x, y: x >= y,
        '<=': lambda x, y: x <= y,
        '!=': lambda x, y: x != y,
        'in': lambda x, y: array([y in i for i in x]),
        'not in': lambda x, y: array([y not in i for i in x])
    }

    def __init__(self, field, op, value=None):
        """
        Creates a new FilterCondition object.

        :param field: String. Name of the field from the database.
        :param op: String. The corresponding operation string.
        :param value: Anything. The value to be used in the comparisons.
        """
        self.field = field
        self.op = op
        self.value = value
        self.apply = lambda x: FilterCondition.ops[self.op](array(x[self.field]), self.value)

    def __and__(self, other):
        """
        Combines two FilterConditions into one, which will return the
        samples whose comparisons return true for both FilterConditions.

        :param other: FilterCondition object.

        :return: The combined FilterCondition object.
        """
        new_condition = FilterCondition(self.field, 'and', other.field)
        new_condition.apply = lambda x: self.apply(x).__and__(other.apply(x))

        return new_condition

    def __or__(self, other):
        """
        Combines two FilterConditions into one, which will return the
        samples whose comparisons return true for at least one of the
        FilterConditions.

        :param other: FilterCondition object.

        :return: The combined FilterCondition object.
        """
        new_condition = FilterCondition(self.field, 'or', other.field)
        new_condition.apply = lambda x: self.apply(x).__or__(other.apply(x))

        return new_condition

    def __call__(self, *args, **kwargs):
        """
        Filter the samples returning the result of the comparisons.

        :param args[0]: Dictionary. The data property of a DataBase object.

        :return: Numpy array of booleans. Whether the sample matched the
                 conditions.
        """

        return self.apply(args[0])


# Unit Tests 
class ClassTests(TestCase):

    def assertEqual(self, first, second, msg=None):
        if first != second:
            e_msg = ''.join(
                ['\nExpected: ', str(second), ' Found: ', str(first)])
            print(e_msg)

        TestCase.assertEqual(self, first, second, msg)

    def test_filter(self):

        db = {
            'height': [5, 10, 1, 2, 12, 6],
            'type': ['a', 'c', 'b', 'a', 'a', 'b'],
            'name': ['01', '01', '02', '02', '03', '03']
        }

        smaller = FilterCondition('height', '<', 10)
        type_a = FilterCondition('type', '==', 'a')
        name_01 = FilterCondition('name', '==', '01')

        print('smaller =', smaller.apply(db))
        print('type a =', type_a.apply(db))
        print('name 01 =', name_01.apply(db))

        smaller_and_type_a = smaller.__and__(type_a)
        print('smaller and type a =', smaller_and_type_a.apply(db))

        all_together = smaller.__and__(type_a).__and__(name_01)
        print('all together =', all_together.apply(db))

    def test_data_split(self):

        class DB():

            def __init__(self):

                self.samples_path = None

        db = DB()
        db.samples_path = [
            'aldfjasd/asdfasfa/adf/as/combination_1_blur.jpg',
            'aldfjasd/asdfasfa/adf/as/combination_2_blur.jpg',
            'aldfjasd/asdfasfa/adf/as/combination_6_blur.jpg',
            'aldfjasd/asdfasfa/adf/as/combination_7_blur.jpg',
            'aldfjasd/asdfasfa/adf/as/combination_8_sharp.jpg',
            'aldfjasd/asdfasfa/adf/as/combination_10_sharp.jpg',
        ]

        ids = get_comb_ids(db)

        print('\nids =', ids)

        self.assertEqual([1,2,6,7,8,10], ids, 'IDs were incorrectly computed!')

        strat = [0, 0, 2, 2, 2, 3]

        idxs = get_stratify_info_per_comb_id(db)

        print('\nidxs =', idxs)

        self.assertEqual(strat, idxs.tolist(), 'Stratify info are incorrect!')


if __name__ == '__main__':
    # loads and runs the Unit Tests
    suite = TestLoader().loadTestsFromTestCase(ClassTests)
    TextTestRunner(verbosity=2, ).run(suite)