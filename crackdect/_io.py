"""
IO module

Convenience functions for handling image paths, sorting paths and loading images.

The default dtype for the images is np.float32. This saves memory compared to np.float64 without
significant losses in accuracy since these images are normally represented in a range from 0 -> 1 or -1 -> 1.
"""

import os
import numpy as np
import re
from .imagestack import ImageStack
from .crack_detection import CrackDetectionTWLI
from skimage.io import imsave
import matplotlib.pyplot as plt


def image_paths(img_dir, image_types=('jpg', 'png', 'bmp')):
    """
    Selects all images given in the image_types list from a directory

    Parameters
    ----------
    img_dir: str
        path to the directory which includes the images
    image_types: list
        a list of strings of the image types to select

    Returns
    -------
    image_paths: list
    """
    files = os.listdir(img_dir)
    paths = np.array([
        os.path.abspath(os.path.join(img_dir, name))
        for name in files
        if name.lower().split('.')[-1] in image_types])
    return paths


def general_path_sorter(path_list, pattern):
    """
    General sorting of paths in ascending order with regex pattern.

    The regex pattern must contain a "group1" which matches any number. The paths are then sorted
    with this number in ascending order.

    Parameters
    ----------
    path_list: list
        List of filenames or paths
    pattern: str
        regex pattern that matches any number. The group must be marked as "group1".
        E.g. "(?P<group1>[0-9]+)cycles" would match the number 1234 in "1234cycles".

    Returns
    -------
    paths: array
        Sorted paths in ascending order
    numbers: array
        The numbers from the match corresponding to the paths.
    """
    if 'group1' not in pattern:
        raise ValueError('The regex pattern must have a group called "group1".')
    
    x = np.ones(len(path_list), dtype=bool)
    var = []
    for ind, name in enumerate(path_list):
        try:
            var.append(int(re.search(pattern, os.path.split(name)[1]).group('group1')))
        except AttributeError:
            x[ind] = False

    index_array = np.argsort(var)
    return np.array(path_list)[x][index_array], np.array(var)[index_array]


def sort_paths(path_list, sorting_key='cycles'):
    """
    Sorts the given paths according to a sorting keyword.

    The paths must have the following structure:
    /xx/xx.../[SORTING_KEY][NUMBER].*
    E.g. test_3cycles_force1.png.
    This will extract the number 3 from the filename and sort other similar filenames in ascending order.

    Parameters
    ----------
    path_list: list
        list of strings of the image names
    sorting_key: str
        A sorting keyword. The number for sorting must be before the keyword. E.g. "cycles" will
        sort all paths with [NUMBER]cycles.* e.g. 123cycles, 234cycles,.. etc

    Returns
    -------
    paths: array
        Sorted paths in ascending order
    numbers: array
        The numbers from the match corresponding to the paths.

    Examples
    --------
    >>> paths = ['A_1cycles.jpg', 'A_50cycles.jpg', 'A_2cycles.jpg', 'A_test.jpg']
    >>> sort_paths(paths, 'cycles')
    (array(['A_1cycles.jpg', 'A_2cycles.jpg', 'A_50cycles.jpg']), array([ 1,  2, 50]))
    """
    pattern = '((?<![0-9])|^)(?P<group1>[0-9]+){}'.format(sorting_key)
    return general_path_sorter(path_list, pattern)


def load_images(paths, dtype=None, **kwargs):
    """
    Loads all images from the paths into memory and stores them in an ImageStack.

    Parameters
    ----------
    paths: list
        Paths of the images
    dtype: dtype, optional
        The dtype all images will be converted to.
    kwargs:
        All kwargs are forwarded to :func:`crackdect.imagestack.ImageStack.from_paths`

    Returns
    -------
    out: ImageStack
    """
    return ImageStack.from_paths(paths, dtype, **kwargs)


def save_images(images, image_format='jpg', names="", directory=None):
    """
    Save all images.

    This saves all images in a stack to a given directory in the given file format.

    Parameters
    ----------
    images: iterable
        An iterable like ImageStack with images represented as np.ndarray
    image_format: str, optional
        The format in which to save the images. Default is jpg.
    names: list, optional
        A list of names corresponding to the images. If none is entered the images will be saved as 1.jpg, 2.jpg, ...
        If these names have extensions for the image format, this extension is ignored. All images are saved with
        image_format
    directory: str, optional
        Path to the directory where the images should be saved. If the path does not exist it is created.
        Default is the current working directory
    """
    if names == "":
        names = np.arange(len(images))
    if directory is None:
        directory = os.getcwd()
    elif not os.path.exists(directory):
        os.mkdir(directory)
    for name, image in zip(names, images):
        imsave(os.path.join(directory, os.path.splitext(str(name))[0] + f'.{image_format}'), image, check_contrast=False)


def plot_cracks(image, cracks, **kwargs):
    """
    Plots cracks in the foreground with an image in the background.

    Parameters
    ----------
    image: np.ndarray
        Background image
    cracks: np.ndarray
        Array with the coordinates of the crack with the following structure:
        ([[x0, y0],[x1,y1], [...]]) where x0 and y0 are the starting coordinates and x1, y1
        the end of one crack. Each crack is represented by a 2x2 array stacked into a bigger array (x,2,2).
    kwargs:
        Forwarded to `plt.figure() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html>`_

    Returns
    -------
    fig: Figure
    ax: Axes
    """
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    for (y0, x0), (y1, x1) in cracks:
        ax.plot((x0, x1), (y0, y1), color='red', linewidth=1)
    ax.set_ylim(image.shape[0], 0)
    ax.set_xlim(0, image.shape[1])
    return fig, ax


def plot_gabor(kernel, cmap='plasma', **kwargs):
    """
    Plot the real part of the Gabor kernel

    Parameters
    ----------
    kernel: CrackDetectionTWLI, np.ndarray
    cmap: str
        Identifier for a cmap from matplotlib
    kwargs:
        Forwarded to plt.figure()

    Returns
    -------
    fig: Figure
    ax: Axes
    """
    if isinstance(kernel, CrackDetectionTWLI):
        gk = kernel._gk_real
    if isinstance(kernel, np.ndarray):
        gk = np.real(kernel)

    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(111)

    ax.imshow(gk, cmap=cmap)
    return fig, ax
