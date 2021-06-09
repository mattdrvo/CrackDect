"""
Routines for preprocessing image stacks.

All functions in this module are designed to take an image stack and additional arguments as input.
"""
import numpy as np
from scipy.fft import fft2
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, warp
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
    n_minus_1 = fft2(images[0])
    shift = [(0, 0)]

    for img in images[1:]:
        fft_n = fft2(img)
        shift.append(phase_cross_correlation(n_minus_1, fft_n, space='fourier', upsample_factor=5)[0])
        n_minus_1 = fft_n

    return np.cumsum(shift, axis=0)


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

    Returns
    -------
    out: list, ImageStack
        list or ImageStack with the corrected images.
    """
    # if all images are the same shape, no need to crop them
    shapes = np.array([img.shape for img in images])
    height, width = shapes.min(axis=0)
    if not (np.all(shapes[:, 0] == height) and np.all(shapes[:, 1] == width)):
        images = region_of_interest(images, 0, width, 0, height)

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

    Returns
    -------
    out: list, ImageStack
        list or ImageStack with the corrected images.
    """

    shapes = np.array([img.shape for img in images])
    height, width = shapes.min(axis=0)
    if not (np.all(shapes[:, 0] == height) and np.all(shapes[:, 1] == width)):
        images = region_of_interest(images, 0, width, 0, height)

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


def overload_images(images):
    """
    Combines the nth image with the n-1st with logical_or.

    :math:`I_n^{new} = I_n | I_{n-1}`

    Parameters
    ----------
    images: ImageStack
        Image stack with image-dtype bool

    Returns
    -------
    out: ImageStack
    """
    if images._dtype != bool:
        raise TypeError('The stack must contain only bool images!')

    def fun(img1, img2):
        return np.logical_or(img1, img2)
    return _rolling_stack_operation(images, fun, True)
