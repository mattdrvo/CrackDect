"""
Preprocessing functions for single images

In this module image processing functions for single images are grouped as well as helper functions.
"""
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.util.dtype import dtype_range


def scale_to(x, x_min=0, x_max=1):
    """
    Linear scaling to new range of values.

    This function scales the input values from their current range to the given range.

    Parameters
    ----------
    x: array-like
    x_min: float
        Lower boarder of the new range
    x_max: float
        Upper boarder of the new range

    Returns
    -------
    out:
        Scaled values

    Examples
    --------
    >>> y = np.array((-1,2,5,7))
    >>> scale_to(y, x_min=0, x_max=1)
    array([0.   , 0.375, 0.75 , 1.   ])
    """
    if x.min() == x_min and x.max() == x_max:
        return x
    else:
        temp = (x - np.min(x)) / (np.max(x) - np.min(x))
        return temp * (x_max - x_min) + x_min


def _fast_astype(img, dtype):
    if np.issubdtype(img.dtype.type, dtype):
        return img
    else:
        return img.astype(dtype)


def detect_changes_division(img1, img2, output_range=None):
    """
    Detect the changes between two images by division.
    Areas with a lot of change will appear darker and areas with little change bright.

    The change from img1 to img2 is computed with.

    :math:`I_d = \\frac{ img2 + 1}{img1 + 1}`

    The range of the output image is scaled to the range of the datatype of the input image.

    Parameters
    ----------
    img1: np.ndarray
    img2: np.ndarray
    output_range: tuple, optional
        Range of the output image. This range must be within the possible range of the dtype of the input images.
        E.g. (0,1) for float images.

    Returns
    -------
    out: np.ndarray
        Input dtpye is only converved when an output_range is given. Else the result will have
        dtype float with a maximal possible range of 0.5-2
    """
    inp_dtype = img1.dtype.type
    img1 = _fast_astype(img1, np.floating)
    img2 = _fast_astype(img2, np.floating)
    temp = (scale_to(img2) + 1) / (scale_to(img1) + 1)
    if output_range is not None:
        return _fast_astype(rescale_intensity(temp, out_range=output_range), inp_dtype)
    else:
        return temp


def detect_changes_subtraction(img1, img2, output_range=None):
    """
    Simple change detection by image subtracting.
    Areas with a lot of change will appear darker and areas with little change bright.

    :math:`I_d = I_2 - I_1`

    Parameters
    ----------
    img1: np.ndarray
    img2: np.ndarray
    output_range: tuple, optional
        Range of the output image. This range must be within the possible range of the dtype of the input images
        E.g. (0,1) for float images.

    Returns
    -------
    out: np.ndarray
        Image as input dtype
    """
    inp_dtype = img1.dtype.type
    img1 = _fast_astype(img1, np.floating)
    img2 = _fast_astype(img2, np.floating)
    if output_range is not None:
        return _fast_astype(rescale_intensity(img2 - img1, out_range=output_range[::-1]), inp_dtype)
    else:
        return _fast_astype(img2-img1, inp_dtype)
