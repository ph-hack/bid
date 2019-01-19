"""
This module contains the functions that implement the blurriness
metric (BM) of Hong et al.
"""
from numpy import array, zeros, zeros_like, where, unique, dot, gradient, absolute, isnan
from pre_processors import imread
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import resize
from scipy.spatial.distance import euclidean
from scipy.optimize import curve_fit
from scipy import exp

import cv2 as cv
import math

def yuzhen_blur_metric(img_path, **kwargs):
    """
    The blur metric (BM) of Hong et al.

    Y. Hong, G. Ren, E. Liu, and J. Sun. A Blur Estimation
    and Detection Method for Out-of-Focus Images. Springer
    Science and Business Media New York, i75(18):10807–10822,
    2015.

    :param img_path: String. The path to the image file.
    :param std_threshold: Float. The threshold for the
                          homogeneity of the lines. Modeled in
                          our dissertation as
                          u(l) = maximum(std(S0), std(S1)).
    :param method: Integer. Either 1 (our adapted version) or
                   2 (our modified version).

    :return: List. The BM value.
    """

    std_th = 12 if 'std_threshold' not in kwargs else kwargs['std_threshold']
    method = 1 if 'method' not in kwargs else kwargs['method']

    if img_path is None:

        return [-1]

    # Reads the image
    img = imread(img_path)
    img = resize(img, (500, 500))

    # Apply canny
    gray = rgb2gray(img)
    edges = canny(gray, 0.1).astype('uint8')

    # Apply Hough transform for lines
    gap = 0
    max_gap = 5
    th = 50
    min_th = 10
    lines_cv = cv.HoughLinesP(edges, 1, math.pi/180, th, None, minLineLength=12, maxLineGap=gap)

    box_width = 11
    max_box_height = 10

    while lines_cv is None:

        if th == min_th and gap == max_gap:
            break

        if th == min_th and gap < max_gap:
            gap += 1

        if th > min_th:
            th -= 10

        print('tyring again th', th, 'gap', gap)
        lines_cv = cv.HoughLinesP(edges, 1, math.pi / 180, th, None, minLineLength=12, maxLineGap=gap)

    if lines_cv is None:
        return [255]

    lines = []
    # Convert the output of the hough transform to the original used format
    for l in lines_cv:
        lines.append((
            [l[0][0], l[0][1]],
            [l[0][2], l[0][3]]
        ))

    # removes the lines according to how similar are both sides of the lines
    line_mask = zeros_like(edges, dtype='float32')
    new_lines = []

    i = 0

    im = gray * 255
    shape = edges.shape
    max_stds = []

    for line in lines:

        p0, p1 = line
        box = Box(p0, p1, shape=(max_box_height, box_width))
        w1, w2 = box.get_grid(shape)
        try:
            a = max(im[w1].std(), im[w2].std())
            max_stds.append(a)

        except IndexError:

            print(box.ml.a, box.ml.b, box.ml.image_axis)

        if a < std_th:
            i += 1

            if box.ml.image_axis:
                xs, ys = box.ml.sample_discrete(shape[1] - 1, 0, min(p0[0], p1[0]),
                                                max(p0[0], p1[0]) - min(p0[0], p1[0]), 1)
            else:
                xs, ys = box.ml.sample_discrete(shape[0] - 1, 0, min(p0[1], p1[1]),
                                                max(p0[1], p1[1]) - min(p0[1], p1[1]), 1)

            line_mask[ys.astype('int'), xs.astype('int')] = i

            new_lines.append(line)

    # second criterion for keeping the lines: no other edges in the gradient direction within 16 pixels of distance
    to_be_removed = []

    for i, l1 in enumerate(new_lines):

        # make box of L1
        l1_p1, l1_p2 = l1
        l1_box_th = Box(l1_p1, l1_p2, (euclidean(l1_p1, l1_p2), 16 * 2))

        # get the pixels of box of L1
        w1, w2 = l1_box_th.get_grid(edges.shape)
        w = (w1[0] + w2[0], w1[1] + w2[1])

        l2_lines = unique(line_mask[w]).astype('int')
        l2_lines = l2_lines[l2_lines != 0] - 1

        for j in l2_lines:

            l2 = new_lines[j]

            if l1 != l2:

                l1_std = max_stds[i]
                l2_std = max_stds[j]

                # remove the one if the biggest difference or L2
                if l1_std > l2_std:
                    to_be_removed.append(i)

    final_lines = []

    for i, line in enumerate(new_lines):

        if i not in to_be_removed:
            final_lines.append(line)

    # compute the sigmas of all remaining lines
    sigmas = []

    ultimate_lines = []

    if len(final_lines) == 0:
        if len(new_lines) == 0:
            ultimate_lines = lines
        else:
            ultimate_lines = new_lines
    else:
        ultimate_lines = final_lines

    if len(ultimate_lines) == 0:
        return [255]

    for l in ultimate_lines:
        p0 = l[0]
        p1 = l[1]
        w = int(min(euclidean(p0, p1), max_box_height))
        shape = (w, box_width)

        box = Box(p0, p1, shape)

        box_img = get_vertical_box(gray, box)

        if box_img is not None:
            box_img *= 255
            try:
                if method == 1:
                    sigma, gamma, c = get_sigma(box_img)
                else:
                    sigma, gamma, c = get_sigma2(box_img)

                sigmas.append(sigma)
            except Exception as e:
                print(e.__cause__)
                print('some error on sigma function')
                pass
        else:
            pass

    bm = blur_metric(sigmas)

    if isnan(bm):
        print('========NAN Found========')
        print('sigmas =', sigmas)
        print('=========================')

    return [bm]

def get_transf_params(box):
    """
    Computes the transformation parameters to rotate the
    lines' boxes so they get in the vertical position.

    :param box: Box object. The original detected box.

    :return: Tuple. The translation and the rotation
             parameters.
    """

    to_origin_point = box.p0

    if box.p1[1] < box.p0[1]:

        to_origin_point = box.p1

    elif box.p1[1] == box.p0[1] and box.p1[0] < box.p0[0]:

        to_origin_point = box.p1

    translation_params = (-to_origin_point[0], -to_origin_point[1])

    # check whether the box is already in the correct alignment
    if box.ml.image_axis == 0:
        # no rotation needed, then
        return translation_params, 0.

    # convert the angular coefficient to radian angle
    angle = math.atan(box.ml.a)

    # compute the angles' delta

    if box.ml.a > 0:

        return translation_params, math.pi / 2. - angle

    elif box.ml.a < 0:

        return translation_params, 1.5 * math.pi - angle

    else:
        return translation_params, math.pi / 2.

def rotation_adjustment(point, translation, rotation):
    """
    Applies the translation and rotation operations to
    the given point.

    :param point: Array-like object. The 2D original point.
    :param translation: Array-like object. The delta in
                        both dimensions.
    :param rotation: Float. Rotation angle in radians.

    :return: Numpy array. The new 2D point, after the
             transformations.
    """

    # first the rotation
    rot_mat = array([
        [math.cos(-rotation), -math.sin(-rotation)],
        [math.sin(-rotation), math.cos(-rotation)]
    ])

    point = array([[point[0], point[1]]])

    new_point = dot(point, rot_mat)

    new_point += array(translation)

    return new_point

def get_pixel_value(img, point):
    """
    Gets the value of the pixel for the adjusted point.
    Implementation of Equation 2.2 in our dissertation.

    :param img: Numpy array. The image.
    :param point: Array-like object. The adjusted 2D point.

    :return: Float. The corresponding pixel's value.
    """

    x = point[0]
    y = point[1]
    xint = math.floor(x)
    yint = math.floor(y)

    if xint == img.shape[1] - 1 or yint == img.shape[0] - 1:
        raise Exception('box outside the image boundaries!')

    a = (xint + 1 - x) * (yint + 1 - y)
    b = (x - xint) * (yint + 1 - y)
    c = (xint + 1 - x) * (y - yint)
    d = (x - xint) * (y - yint)

    v = a * img[yint, xint] + b * img[yint, xint + 1] + c * img[yint + 1, xint] + d * img[
        yint + 1, xint + 1]

    return v

def get_vertical_box(img, box):
    """
    Returns the region of the image given by
    the *box*, but adjusted to the vertical
    position.

    :param img: Numpy array. The image.
    :param box: Box object.

    :return: Numpy array. The region of the
             image corresponding to the box.
    """

    t, r = get_transf_params(box)
    t = (-t[0], -t[1])
    r = -r

    box_img = zeros(box.shape)

    for i in range(box.shape[0]):
        for j in range(box.shape[1]):
            point = rotation_adjustment([j - box.shape[1] / 2, i], t, r).reshape(2)

            try:
                box_img[i, j] = get_pixel_value(img, point)
            except Exception:
                return None
    return box_img


def signal(x):
    """
    Returns -1 if *x* is negative and +1
    if *x* is positive and different than
    zero. Returns *x* itself, otherwise.

    :param x: Number.
    :return: Number.
    """
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1


def get_slope(x):
    """
    Corresponds to our modification in Hong et al.'s method.
    See Section 3.2.3 of our dissertation for details.

    :param x: Array-like object. The input curve/profile.
    :return: Array-like object. Only the slope part extracted.
    """
    mid = int(len(x) / 2)

    start = -1
    end = len(x) + 1

    s1 = 0
    s2 = 0

    for i, j in zip(range(mid - 1, 0, -1), range(mid + 1, len(x))):

        g1 = x[i + 1] - x[i]
        g2 = x[j] - x[j - 1]

        if s1 == 0:
            s1 = signal(g1)
        elif start >= 0:
            pass
        elif s1 != signal(g1):
            start = i + 1

        if s2 == 0:
            s2 = signal(g2)
        elif end <= len(x):
            pass
        elif s2 != signal(g2):
            end = j

        if start >= 0 and end <= len(x):

            break

    start = max(0, start)
    end = min(len(x), end)

    slope = zeros(len(x[start:end]) + 2)
    slope[0] = x[start]
    slope[1:-1] = x[start:end]
    slope[-1] = x[end - 1]

    return slope

# Original implementation
#
# def get_sigma_yuzhen(box_img):
#
#     W = box_img.shape[1]
#     m = box_img.mean(axis=0)
#     g = absolute(gradient(m))
#     A = concatenate([
#         array([1] * W).reshape((W, 1)),
#         m.reshape((W, 1)),
#         power(m, 2).reshape((W, 1))
#     ], axis=1)
#
#     b = dot(dot(inv(dot(A.transpose(), A)), A.transpose()), g)
#
#     if b[2] < 0:
#         sigma = 1 / math.sqrt(-2 * b[2])
#     else:
#         sigma = 1 / math.sqrt(2 * b[2])
#     gamma = b[1] * math.pow(sigma, 2.)
#     c = math.exp(b[0]) + math.pow(gamma, 2.) / (2 * math.pow(sigma, 2.))
#
#     return sigma, gamma, c

def get_sigma(box_img):
    """
    Computes the sigma parameter of this box of the image.
    This follows our adapted version.

    :param box_img: Numpy array. A region of the image.

    :return: Tuple. The Gaussian function parameters:
             sigma, gamma and C.
    """

    m = box_img.mean(axis=0)
    g = gradient(m)
    x = array(list(range(len(g))))

    popt, _ = curve_fit(gaussian, x, g, p0=[1, 1, 1])

    sigma, gamma, c = popt

    if isnan(sigma):
        print('===================')
        print('box_img shape', box_img.shape)
        print('g =', g)
        print('x =', x)
        print('===================')

    return sigma, gamma, c

def get_sigma2(box_img):
    """
    Computes the sigma parameter of this box of the image.
    This follows our modified version.

    :param box_img: Numpy array. A region of the image.

    :return: Tuple. The Gaussian function parameters:
             sigma, gamma and C.
    """

    m = box_img.mean(axis=0)

    # the modification part
    m = get_slope(m)

    g = gradient(m)
    x = array(list(range(len(g))))

    popt, _ = curve_fit(gaussian, x, g, p0=[1, 1, 1])

    sigma, gamma, c = popt

    if isnan(sigma):
        print('===================')
        print('box_img shape', box_img.shape)
        print('g =', g)
        print('x =', x)
        print('===================')

    return sigma, gamma, c

def gaussian(x, sigma, gamma, c):
    """
    An implementation of the Gaussian function.

    :param x: Numpy array. The input/variable/domain
              for the Gaussian function.
    :param sigma: Float.
    :param gamma: Float.
    :param c: Float.

    :return: Numpy array. G(x).
    """

    return c * exp(- ((x - gamma) ** 2.) / (2. * sigma ** 2.))

def blur_metric(sigmas):
    """
    The final part of Hong et al.'s method.
    Computes the final metric from the sigmas
    of the edges/lines.

    :param sigmas: Array-like object.

    :return: Float. The BM value.
    """

    if len(sigmas) == 0:
        return 255

    sigmas = array(sigmas)
    sigmas = absolute(sigmas)
    m = sigmas.mean()
    s = sigmas.std()

    sigmas = sigmas[where(absolute(sigmas - m) <= s)]

    return sigmas.mean()


# some useful classes ##################################################

class Line(object):
    """
    Class to represent and manipulate lines in a 2D domain.
    The line object is modeled as a polynomial function of
    degree 1: f(x) = ax + b.

    Properties::
        :a: Float.
        :b: Float.
        :image_axis: The index of the dimension that repre-
                     sents the domain of the function.
                     Either 0 or 1.
    """
    def __init__(self, p0=None, p1=None):
        """
        Creates a new line object from two 2D points.

        :param p0: Array-like object. First point.
        :param p1: Array-like object. Second point.
        """

        if p0 is not None and p1 is not None:

            dx = p1[0] - p0[0]

            if dx != 0:
                self.a = (p1[1] - p0[1]) / (dx)
                self.b = p0[1] - self.a * p0[0]
                self.image_axis = 1
            else:
                self.a = (dx) / (p1[1] - p0[1])
                self.b = p0[0] - self.a * p0[1]
                self.image_axis = 0
        else:
            self.a = 0
            self.b = 0
            self.image_axis = 1

    def __call__(self, *args, **kwargs):
        """
        Returns the value for f(x).

        :param args[0]: Numpy array or float. The input ("x").

        :return: Numpy array or float.
        """

        return self.a * args[0] + self.b

    def __contains__(self, point):
        """
        Checks whether the given point lies in this line.

        :param point: Array-like object. 2D point.

        :return: Boolean. True if the point lies in the line.
                 False, otherwise.
        """

        x, y = point

        if self.image_axis == 1:

            return y == self(x)

        else:
            return x == self(y)

    def __eq__(self, other):
        """
        Compares this line object with another and returns
        True if they are equal. i.e. they have the same
        attributes' values.

        :param other: Line object.

        :return: Boolean.
        """

        return self.a == other.a and self.b == other.b and self.image_axis == other.image_axis

    def __ne__(self, other):
        """
        Compares this line object with another and returns
        True if they are different. i.e. they have any of
        the attributes with a different value from one
        another.

        :param other: Line object.

        :return: Boolean.
        """

        return self.a != other.a or self.b != other.b or self.image_axis != other.image_axis

    def copy(self):
        """
        Creates another line object that is equal to this one.

        :return: Line object.s
        """

        c = Line()
        c.a = self.a
        c.b = self.b
        c.image_axis = self.image_axis

        return c

    def parallel(self, offset):
        """
        Creates another line object that is parallel to this
        one and separated by a given offset.

        :param offset: Float. The distance between this line
                       and the new one.

        :return: Line object.
        """

        p = self.copy()

        alpha = math.atan(self.a)
        x = - offset * math.sin(alpha)
        y = offset * math.cos(alpha) + self.b

        p.b = y - p.a * x

        return p

    def perpendicular(self, intersection):
        """
        Creates another line object that is perpendicular
        to this one and they intersect at a given point.

        :param intersection: Array-like object. The
                             intersection point.

        :return: Line object.
        """

        p = self.copy()

        if self.a != 0 and self.image_axis:

            alpha = math.atan(self.a)
            x = intersection[0]
            y = intersection[1]

            p.a = math.tan(alpha + math.pi / 2)
            p.b = y - p.a * x

        elif self.a == 0 and self.image_axis:

            p.b = intersection[0]
            p.image_axis = 0

        elif self.image_axis == 0:

            p.b = intersection[1]
            p.image_axis = 1

        return p

    def sample(self, start=0, n=0, step=1):
        """
        Gets some points that lies in this line. They start with
        the domain *start* and goes up to *n* at the given *step*.

        :param start: Float. Initial domain value.
        :param n: Float. Final domain value (not inclusive).
        :param step: Float. Size of the step.

        :return: Tuple. Two numpy arrays with the domain and the image
                 dimensions, respectively.
        """

        if self.image_axis:

            x = array([i * step + start for i in range(n)])
            y = self(x)

        else:
            y = array([i * step + start for i in range(n)])
            x = self(y)

        return x, y

    def sample_discrete(self, max_value, min_value=0, start=0, n=0, step=1):
        """
        Same as the *sample* method, but constrains the domain values inside
        the range [*min_value*, *max_value*].

        :param max_value: Float. Maximum value for the domain.
        :param min_value: Float. Minimum value for the domain.
        :param start: Float. Initial domain value.
        :param n: Float. Final domain value (not inclusive).
        :param step: Float. Size of the step.

        :return: Tuple. Two numpy arrays with the domain and the image
                 dimensions, respectively.
        """

        safe = lambda x: int(max(min(x, max_value), min_value))

        if self.image_axis:

            x = array([safe(i * step + start) for i in range(n)])
            y = self(x)

        else:
            y = array([safe(i * step + start) for i in range(n)])
            x = self(y)

        return x, y

    def inverse(self, value):
        """
        Compute the inverse of f(x). i.e. x = g(y) = (y - b)/a.

        :param value: Numpy array or float. The image value(s).

        :return: Numpy array or float. The corresponding domain value(s).
        """

        return (value - self.b) / self.a

    def intersection(self, other):
        """
        Computes the intersection point between this line and another one.

        :param other: Line object.

        :return: Tuple. The 2D intersection point.
        """

        if self.a == other.a and self.image_axis == other.image_axis:
            return None

        if self.image_axis:

            if other.image_axis:

                x = (other.b - self.b) / (self.a - other.a)
                y = other(x)

                return (x, y)

            else:
                x = other.b
                y = self(x)

                return (x, y)
        else:
            if other.image_axis:
                x = self.b
                y = other(x)

                return (x, y)


class Box(object):
    """
    Class to represent and manipulate boxes in a 2D space.
    """
    def __init__(self, p0, p1, shape=(20, 17)):
        """
        Creates a new box object from two points, that gives the
        orientation of the box, and its shape. The first dimension
        in the shape corresponds to the perpendicular direction of
        the one given by the points. Therefore, the second dimension
        corresponds to the size of the box in the direction given
        by the points.

        :param p0: Array-like object. 2D first point.
        :param p1: Array-like object. 2D second point.
        :param shape: Array-like object. 2D shape.
        """

        self.p0 = array(p0)
        self.p1 = array(p1)
        self.shape = shape

        # mean point
        self.mp = (self.p0 + self.p1) / 2
        # main line
        self.ml = Line(p0, p1)

        perp = self.ml.perpendicular(self.mp)

        self.lines = [
            self.ml.parallel(shape[1] / 2),
            self.ml.parallel(-shape[1] / 2),
            perp.parallel(shape[0] / 2),
            perp.parallel(-shape[0] / 2)
        ]

    def __contains__(self, point):
        """
        Checks whether a point lies inside this box.

        :param point: Array-like object. 2D point.

        :return: Boolean. True if the given point lies inside this
                 box. False, otherwise.
        """

        tests = []

        # first edge
        if self.lines[0].image_axis:
            tests.append(point[1] < self.lines[0](point[0]))
        else:
            tests.append(point[0] < self.lines[0](point[1]))

        # second edge
        if self.lines[1].image_axis:
            tests.append(point[1] > self.lines[1](point[0]))
        else:
            tests.append(point[0] > self.lines[1](point[1]))

        # third edge
        if self.lines[2].image_axis:
            tests.append(point[1] < self.lines[2](point[0]))
        else:
            tests.append(point[0] < self.lines[2](point[1]))

        # fourth edge
        if self.lines[3].image_axis:
            tests.append(point[1] > self.lines[3](point[0]))
        else:
            tests.append(point[0] > self.lines[3](point[1]))

        tests = array(tests)
        return tests.all()

    def get_grid(self, shape):
        """
        Returns the coordinates of the pixels of a grid
        (image) that lies inside this box. The given shape
        defines the grid.

        :param shape: Array-like object. 2D shape.

        :return: Tuple. Containing two other tuples in the format
                 (rows1, cols1), (rows2, cols2). Where the first
                 one contains the vertical and horizontal
                 coordinates of the first half of the box, which
                 is divided by the line formed by the two initial
                 points; the second tuple returns the same, but
                 for the other half.
        """

        rows1, cols1 = [], []
        rows2, cols2 = [], []

        for i in range(shape[0]):

            if self.ml.a != 0:

                l0 = self.lines[0](i) if self.lines[0].image_axis == 0 else self.lines[0].inverse(i)
                l1 = self.lines[1](i) if self.lines[1].image_axis == 0 else self.lines[1].inverse(i)
                l2 = self.lines[2](i) if self.lines[2].image_axis == 0 else self.lines[2].inverse(i)
                l3 = self.lines[3](i) if self.lines[3].image_axis == 0 else self.lines[3].inverse(i)
                lm = self.ml(i) if self.ml.image_axis == 0 else self.ml.inverse(i)

                if self.ml.a > 0:

                    ms = (l0, l3)
                    Ms = (l1, l2)

                elif self.ml.a < 0:

                    ms = (l1, l2)
                    Ms = (l0, l3)

                # minimum col
                m = int(math.floor(max(0, *ms)))

                # maximum col
                M = int(math.floor(min(lm, shape[1] - 1, *Ms)))

                if M >= m:
                    cs = list(range(m, M + 1))

                    cols1.extend(cs)
                    rows1.extend([i for c in cs])

                # minimum col
                m = int(math.floor(max(lm, 0, *ms)))

                # maximum col
                M = int(math.floor(min(shape[1] - 1, *Ms)))

                if M >= m:
                    cs = list(range(m, M + 1))

                    cols2.extend(cs)
                    rows2.extend([i for c in cs])

            elif self.ml.image_axis:

                l0 = self.lines[0].b
                l1 = self.lines[1].b
                ml = self.ml.b

                l2 = int(math.floor(self.lines[2](i)))
                l3 = int(math.floor(self.lines[3](i)))

                if i <= ml and i >= l1:
                    cs = list(range(max(0, l3), min(l2 + 1, shape[1])))

                    cols1.extend(cs)
                    rows1.extend([i for c in cs])

                if i <= l0 and i >= ml:
                    cs = list(range(max(0, l3), min(l2 + 1, shape[1])))

                    cols2.extend(cs)
                    rows2.extend([i for c in cs])

            elif self.ml.image_axis == 0:

                l0 = int(math.floor(self.lines[0](i)))
                l1 = int(math.floor(self.lines[1](i)))
                ml = int(math.floor(self.ml(i)))

                l2 = self.lines[2].b
                l3 = self.lines[3].b

                if i <= l2 and i >= l3:
                    cs = list(range(max(0, l1), min(ml + 1, shape[1])))

                    cols1.extend(cs)
                    rows1.extend([i for c in cs])

                if i <= l2 and i >= l3:
                    cs = list(range(max(0, ml), min(l0 + 1, shape[1])))

                    cols2.extend(cs)
                    rows2.extend([i for c in cs])

        return (rows1, cols1), (rows2, cols2)

    def get_ranges(self):
        """
        Computes the range of values occupied by this box's points
        in both dimensions.

        :return: Tuple. Containing two tuples: the range in the first
                 dimension and the range in the second dimension,
                 respectively.
        """

        if self.ml.a > 0:
            #             ms = (l0, l3)
            #             Ms = (l1, l2)
            pm = self.lines[0].intersection(self.lines[3])
            pM = self.lines[1].intersection(self.lines[2])

            return (pm[0], pM[0]), (pm[1], pM[1])

        elif self.ml.a < 0:
            #             ms = (l1, l2)
            #             Ms = (l0, l3)
            pm = self.lines[1].intersection(self.lines[2])
            pM = self.lines[0].intersection(self.lines[3])

            return (pm[0], pM[0]), (pm[1], pM[1])

        elif self.ml.image_axis:

            return (self.lines[1].b, self.lines[0].b), (self.lines[3].b, self.lines[2].b)

        else:
            return (self.lines[3].b, self.lines[2].b), (self.lines[1].b, self.lines[0].b)
