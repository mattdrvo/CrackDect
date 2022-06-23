"""
Routines for preprocessing image stacks.

All functions in this module are designed to take an image stack and additional arguments as input.

The main functionality consists of different methods for shift correction and change detection for consecutive images.
"""
import numpy as np
import warnings
from scipy.fft import fftn
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, ProjectiveTransform, warp
from .image_functions import detect_changes_division, detect_changes_subtraction


def _stack_operation(stack, function, *args, **kwargs):
    """
    Perform an operation for all images of an image stack.

    This is just a wrapper for a function which can perform a procedure for one image to
    perform it for all images of a stack instead.

    Parameters
    ----------
    stack: ImageStack
        The image stack the function should be performed on
    function: function
        A function which takes ONE image as input and returns ONE image
    args:
        args are forwarded to the function.
    kwargs:
        kwargs are forwarded to the function.
    """
    for ind, img in enumerate(stack):
        stack[ind] = function(img, *args, **kwargs)
    return stack


def _rolling_stack_operation(stack, function, keep_first=False, *args, **kwargs):
    """
    Perform an rolling operation for all images of an image stack.

    :math:`I_{new} = func(I_{n-1}, I_n)`

    This is just a wrapper for a function which can perform a procedure for two subsequent images
    for a whole stack.

    Parameters
    ----------
    stack: ImageStack
        The image stack the function should be performed on
    function: function
        A function which takes TWO subsequent images as input and returns ONE image
    keep_first: bool
        If True, keep the first image of the stack.
        The function will not be performed on the first image alone!
    args:
        args are forwarded to the function.
    kwargs:
        kwargs are forwarded to the function.
    """
    img_minus1 = stack[0]
    for ind, img in enumerate(stack[1:]):
        stack[ind+1] = function(img_minus1, img, *args, **kwargs)
        img_minus1 = img
    if not keep_first:
        del stack[0]
    return stack


def region_of_interest(images, x0=0, x1=None, y0=0, y1=None):
    """
    Crop all images in a stack to the desired shape.

    This function changes the images in the stack.
    If the input images should be preserved copy the input to a separate object before!

    The coordinate system is the following: x0->x1 = width, y0->y1 = height from the top left corner of the image

    Parameters
    ----------
    images: list, ImageStack
    x0: int
    x1: int
    y0: int
    y1: int

    Returns
    -------
    out: list, ImageStack
        ImageStack or list with the cropped images
    """
    for ind, img in enumerate(images):
        images[ind] = img[y0:y1, x0:x1]
    return images


def cut_images_to_same_shape(images):
    """
    Cuts all images in a stack to the same shape.

    The images are cut to the shape of the smallest image in the stack. The top left corner is 0,0 and the

    Parameters
    ----------
    images: list, ImageStack

    Returns
    -------
    out: list, ImageStack
        list or ImageStack with all images in the same shape.
    """
    shapes = np.array([img.shape[:2] for img in images])
    height, width = shapes.min(axis=0)
    if not (np.all(shapes[:, 0] == height) and np.all(shapes[:, 1] == width)):
        images = region_of_interest(images, 0, width, 0, height)
    return images


def image_shift(images):
    """
    Compute the shift of all images in a stack.

    The shift of the n+1st image relative to the n-th is computed. The commutative sum of these shifts
    is the shift relative to the 0th image in the stack.

    All input images must have the same width and height!
    Parameters
    ----------
    images: ImageStack, list

    Returns
    -------
    out: list
        [(0,0), (y1, x1), ...(yn, xn)] The shift in x and y direction relative to the first image in the stack.
    """
    n_minus_1 = fftn(images[0], workers=-1)
    shift = [np.zeros(len(images[0].shape))]

    for img in images[1:]:
        fft_n = fftn(img, workers=-1)
        shift.append(phase_cross_correlation(n_minus_1, fft_n, space='fourier', upsample_factor=5)[0])
        n_minus_1 = fft_n

    return np.cumsum(np.array(shift), axis=0)[:, :2]


def biggest_common_sector(images):
    """
    Biggest common sector of the image stack

    This function computes the relative translation between the images with the first image in the stack as
    the reference image. Then the biggest common sector is cropped from the images. The cropping window
    moves with the relative translation of the images so that the translation is corrected.

    Warping of the images which could be a result of strain is not accounted for. If the warp cant be neglected
    do not use this method!!

    Parameters
    ----------
    images: list or ImageStack
        Images represented as np.ndarray. All images must have the same dimensionality! If the width and height of
        the images is not the same, they are cut to the shape of the smallest image.

    Returns
    -------
    out: list, ImageStack
        list or ImageStack with the corrected images.
    """
    # if all images are the same shape, no need to crop them
    cut_images_to_same_shape(images)
    height, width = images[0].shape[:2]

    # compute shift relative to the 0th image
    total_shift = (np.round(image_shift(images)) * -1).astype(int)

    # minimal and maximal boarders to cut after shift
    h_min, w_min = np.abs(np.min(total_shift, axis=0)).astype(int)
    h_max, w_max = np.abs(np.max(total_shift, axis=0)).astype(int)

    # cutting out the image
    for ind, (n, (t_h, t_w)) in enumerate(zip(images, total_shift)):
        images[ind] = n[t_h + h_min:height + t_h - h_max, t_w + w_min: width + t_w - w_max]

    return images


def shift_correction(images):
    """
    Shift correction of all images in a stack. This function is more precise than :func:`biggest_common_sector`
    but more time consuming. The memory footprint is the same.

    This function computes the relative translation between the images with the first image in the stack as
    the reference image. The images are translated into the coordinate system of the 0th image form the stack.

    Warping of the images which could be a result of strain is not accounted for. If the warp cant be neglected
    do not use this function!

    Parameters
    ----------
    images: list, ImageStack
        Images represented as np.ndarray. All images must have the same dimensionality! If the width and height of
        the images is not the same, they are cut to the shape of the smallest image.

    Returns
    -------
    out: list, ImageStack
        list or ImageStack with the corrected images.
    """

    cut_images_to_same_shape(images)
    height, width = images[0].shape[:2]

    # compute shift relative to the 0th image
    total_shift = np.round(image_shift(images)) * -1

    h_min, w_min = np.abs(np.min(total_shift, axis=0).astype(int))
    h_max, w_max = np.abs(np.max(total_shift, axis=0).astype(int))

    for ind, (img, t) in enumerate(zip(images, total_shift)):
        if not (t[0] == 0 and t[1] == 0):
            shift = AffineTransform(translation=t[::-1])
            temp = warp(img, shift, mode='constant', cval=0.5,
                        preserve_range=True)[h_min: height - h_max, w_min: width - w_max].astype(img.dtype.type)
        else:
            temp = img[h_min: height - h_max, w_min: width - w_max]
        images[ind] = temp
    return images


def shift_distortion_correction(images,
                                reg_regions=((0, 0, 0.1, 0.1), (0.9, 0, 1, 0.1), (0, 0.9, 0.1, 1), (0.9, 0.9, 1, 1)),
                                absolute=False):
    """
    Shift and distortion (=strain) correction for all images in a stack.

    This function computes the relative translation and the warp between the images with the first image in the stack as
    the reference image. The images are translated into the coordinate system of the first image.
    Black areas in the resulting images are the result of the transformation into the reference coordinate system.

    Four rectangular areas must be chosen from which the global shift and distortion relative to each other is computed.

    Notes
    -----
    For this algorithm to work properly it is advantageous to have markers on reach corner of the images (like crosses).
    Make sure that the rectangular areas cover the markers in each image. If no distinct features that
    can be tracked are given, this function can result in wrong shift and distortion corrections. In this case,
    make sure to check the results before further image processing steps.

    Parameters
    ----------
    images: list, ImageStack
        Images represented as np.ndarray. All images must have the same dimensionality! If the width and height of
        the images is not the same, they are cut to the shape of the smallest image.
    reg_regions: tuple
        A tuple of four tuples containing the upper left and lower right corners of the rectangles for the areas where
        the phase-cross-correlation is computed. E.g. ((x1, y1, x2, y2), (...), (...), (...)) where x1, y1 etc. can be
        relative dimensions of the image or the absolute coordinates in pixel. Default is relative.
        For relative coordinates enter values from 0-1. If values above 1 are given, absolute coordinate values
        are assumed.
    absolute: bool
        True if absolute and False if relative values are given in reg_regions

    Returns
    -------
    out: list, ImageStack
        list or ImageStack with the corrected images.
    """
    # check if reg_regions is valide
    reg = np.array(reg_regions)
    if reg.shape != (4, 4):
        raise ValueError('Regions for distortion registration must be given as ((x1, y1, x2, y2), (..), (..), (..))')
    # split into x and y coordinates
    x = reg[:, [0, 2]]
    y = reg[:, [1, 3]]
    # check shape of the images in the stack. If the shape is different, cut to smallest image
    cut_images_to_same_shape(images)

    # if reg_regions are given as relative lengths, compute the absolute coordinates
    height, width = images[0].shape[:2]
    if not absolute:
        # if any entry in reg_regions > 1 but flag = False => it is assumed that the User forgot to set flag to True
        if np.any(reg > 1):
            msg = 'Values in reg_regions > 1. Therefore it is assumed that the regions are given in ' \
                  'absolute coordinates and not relative to the image dimensions'
            warnings.warn(msg)
        else:
            x = (x * width).astype(int)
            y = (y * height).astype(int)

    # clean up absolute coordinates
    x[x > width] = width
    x[x < 0] = 0
    y[y > height] = height
    y[y < 0] = 0

    # source coordinates (middle points of the regions in the first image)
    src = np.array((np.mean(x, axis=1, dtype=int), np.mean(y, axis=1, dtype=int))).T

    # fft for all regions of the first image
    img = images[0]
    p1_nminus1 = fftn(img[y[0, 0]:y[0, 1], x[0, 0]:x[0, 1]], workers=-1)
    p2_nminus1 = fftn(img[y[1, 0]:y[1, 1], x[1, 0]:x[1, 1]], workers=-1)
    p3_nminus1 = fftn(img[y[2, 0]:y[2, 1], x[2, 0]:x[2, 1]], workers=-1)
    p4_nminus1 = fftn(img[y[3, 0]:y[3, 1], x[3, 0]:x[3, 1]], workers=-1)

    dx = []
    for i in range(1, len(images)):
        img = images[i]
        # fft for all regions of the nth image (n>=1)
        p1_n = fftn(img[y[0, 0]:y[0, 1], x[0, 0]:x[0, 1]], workers=-1)
        p2_n = fftn(img[y[1, 0]:y[1, 1], x[1, 0]:x[1, 1]], workers=-1)
        p3_n = fftn(img[y[2, 0]:y[2, 1], x[2, 0]:x[2, 1]], workers=-1)
        p4_n = fftn(img[y[3, 0]:y[3, 1], x[3, 0]:x[3, 1]], workers=-1)

        # compute the relative shift of the regions from the n-1st to the nth image
        dx1 = phase_cross_correlation(p1_nminus1, p1_n, space='fourier', upsample_factor=5)[0]
        dx2 = phase_cross_correlation(p2_nminus1, p2_n, space='fourier', upsample_factor=5)[0]
        dx3 = phase_cross_correlation(p3_nminus1, p3_n, space='fourier', upsample_factor=5)[0]
        dx4 = phase_cross_correlation(p4_nminus1, p4_n, space='fourier', upsample_factor=5)[0]

        # n-1st regions
        p1_nminus1, p2_nminus1, p3_nminus1, p4_nminus1 = p1_n, p2_n, p3_n, p4_n

        dx.append(np.array((dx1[:2], dx2[:2], dx3[:2], dx4[:2])))

    # total shift of all regions
    dx_total = np.cumsum(np.array(dx), axis=0)[:, :, [1, 0]]
    p = ProjectiveTransform()
    for i, dx in zip(range(1, len(images)), dx_total):
        # backtransformation of the nth image (n>=1) to the coordinate system of the 1st image
        p.estimate(src + dx, src)
        images[i] = warp(images[i], p, mode='constant', cval=0, preserve_range=True)

    return images


def change_detection_division(images, output_range=None):
    """
    Change detection for all images in an image stack.

    Change detection with image rationing is applied to an image stack.
    The new images are the result of the change between the n-th and the n-1st image.

    The first image will be deleted from the stack.

    Parameters
    ----------
    images: ImageStack, list
    output_range: tuple, optional
        The resulting images will be rescaled to the given range. E.g. (0,1).

    Returns
    -------
    out: ImageStack, list
    """
    return _rolling_stack_operation(images, detect_changes_division, output_range=output_range)


def change_detection_subtraction(images, output_range=None):
    """
    Change detection for all images in an image stack.

    Change detection with image differencing is applied to an image stack.
    The new images are the result of the change between the n-th and the n-1st image.

    The first image will be deleted from the stack.

    Parameters
    ----------
    images: ImageStack, list
    output_range: tuple, optional
        The resulting images will be rescaled to the given range. E.g. (0,1).

    Returns
    -------
    out: ImageStack, list
    """
    return _rolling_stack_operation(images, detect_changes_subtraction, output_range=output_range)
