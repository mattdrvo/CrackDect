"""
Crack detection algorithms

These module contains the different functions for the crack detection. This includes functions for different
sub-algorithms which are used in the final crack detection as well as different methods for the crack detection.
The different crack detection methods are available as functions with an image stack and additional arguments
as input.

"""
import numpy as np
from skimage.morphology._skeletonize_cy import _fast_skeletonize
from skimage.morphology._skeletonize import skeletonize_3d
from skimage.transform import rotate
from skimage.filters import gabor_kernel, threshold_otsu, threshold_yen
from scipy.signal import convolve
from numba import jit, int32
from .imagestack import _add_to_docstring
from skimage.filters._gabor import _sigma_prefactor


_THRESHOLDS = {'yen': threshold_yen,
               'otsu': threshold_otsu}


def rotation_matrix_z(phi):
    """
    Rotation matrix around Z

    Computes the rotation matrix for the angle phi(radiant) around the z-axis

    Parameters
    ----------
    phi: float
        rotation angle (radiant)

    Returns
    -------
    R: array
        3x3 rotation matrix
    """
    s, c = np.sin(phi), np.cos(phi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def sigma_gabor(lam, bandwidth=1):
    """
    Compute the standard deviation for the gabor filter in dependence of the wavelength and the
    bandwidth.

    A bandwidth of 1 has shown to lead to good results. The wavelength should be the
    average width of a crack in pixel. Measure the width of one major crack in the image
    if the cracks are approximately the same width, this is a good approximation. If the cracks
    differ vastly in width lean more to the thinner cracks to get a reasonable approximation.

    Parameters
    ----------
    lam: float
        Wavelength of the gabor filter. This should be approximately the with in pixel of the structures to detect.
    bandwidth: float, optional
        The bandwidth of the gabor filter.

    Returns
    -------
    simga: float
        Standard deviation of the gabor kernel.
    """
    return _sigma_prefactor(bandwidth) * lam


@jit(nopython=True, cache=True)
def find_crack_end(sk_image, start_row, start_col):
    """
    Finde the end of one crack.

    This algorithm finds the end of one crack. The crack must be aligned approximately vertical. The input image
    is scanned and if a crack is found, the algorithm follows it down until the end and overwrites all pixels which
    belong to the crack. This is done that the same crack is not found again.

    Parameters
    ----------
    sk_image: np.ndarray
        Bool-Image where False is background and True is the 1 pixel wide representation of the crack.
    start_row: int
        Row from where the crack searching begins.
    start_col: int
        Column from where the crack searching begins.

    Returns
    -------
    crack_end_x: int
        X-coordinate of the crack end
    crack_end_y: int
        Y-coordinate of the crack end.
    """
    row_num, col_num = int32(sk_image.shape)

    active_row = start_row
    active_col = start_col

    def check_columns(row, col_numbers):
        for i in col_numbers:
            if sk_image[row][i]:
                return True, i
        return False, i

    rn = row_num - 1
    cn = col_num - 1

    while active_row < rn:
        sk_image[active_row][active_col] = False
        if active_col == 0:
            check_cols = [0, 1]
        elif active_col == cn:
            check_cols = [active_col, active_col-1]
        else:
            check_cols = [active_col, active_col-1, active_col+1]

        b, new_col = check_columns(active_row+1, check_cols)
        if b:
            active_col = new_col
        else:
            return active_row, active_col
        active_row += 1
    return active_row, active_col


@jit(nopython=True, cache=True)
def find_cracks(skel_im, min_size):
    """
    Find the cracks in a skeletonized image.

    This function finds the start and end of the cracks in a skeletonized image. All cracks must be aligned
    approximately vertical.

    Parameters
    ----------
    skel_im: np.ndarray
        Bool-Image where False is background and True is the 1 pixel wide representation of the crack.
    min_size: int
        Minimal minimal crack length in pixels that is detected.

    Returns
    -------
    cracks: np.ndarray
        Array with the coordinates of the crack with the following structure:
        ([[x0, y0],[x1,y1], [...]]) where x0 and y0 are the starting coordinates and x1, y1
        the end of one crack. Each crack is represented by a 2x2 array stacked into a bigger array (x,2,2).
    """
    image = skel_im.copy()
    row_num, col_num = image.shape

    rows = np.arange(0, row_num)
    cols = np.arange(0, col_num)

    cracks = []

    for row in rows:
        for col in cols:
            if image[row][col]:
                # indicating a crack start
                crack_start = np.array((row, col), dtype=np.int32)

                # search row wise for the crack end
                # crack_end = np.array(_find_crack_end_fast(image, rows, crack_start), dtype=np.int32)
                crack_end = np.array(find_crack_end(image, row, col), dtype=np.int32)

                # apply a min_size criterion
                x, y = np.subtract(crack_start, crack_end)
                if np.hypot(x, y) >= min_size:
                    # add to cracks
                    cracks.append((crack_start, crack_end))

    return cracks


def cracks_skeletonize(pattern, theta, min_size=5):
    """
    Get the cracks and the skeletonized image from a pattern.

    Parameters
    ----------
    pattern: array-like
        True/False array representing the white/black image
    theta: float
        The orientation angle of the cracks in degrees!!
    min_size: int
        The minimal length of pixels for which will be considered a crack

    Returns
    -------
    cracks: np.ndarray
        Array with the coordinates of the crack with the following structure:
        ([[x0, y0],[x1,y1], [...]]) where x0 and y0 are the starting coordinates and x1, y1
        the end of one crack. Each crack is represented by a 2x2 array stacked into a bigger array (x,2,2).
    skeletonized: np.ndarray
        skeletonized image
    """
    # skeletonize for crack finding

    # sk = _fast_skeletonize(rotate(pattern, theta, resize=True))  #quicker but results are worse

    sk = skeletonize_3d(rotate(pattern, theta, resize=True)).astype(bool)

    # backrotate skeletonized image (sk must be of dtype bool)
    t = rotate(sk, -theta, resize=True)
    y0, x0 = pattern.shape
    y1, x1 = t.shape
    t = t[int((y1 - y0) / 2): int((y1 + y0) / 2), int((x1 - x0) / 2): int((x1 + x0) / 2)]

    # backrotate crack coords
    y1, x1 = sk.shape
    crack_coords = np.array(find_cracks(sk, min_size)).reshape(-1, 2) - np.array(
        (y1 / 2, x1 / 2))
    R = rotation_matrix_z(np.radians(-theta))[0:2, 0:2]
    return (R.dot(crack_coords.T).T + np.array((y0 / 2, x0 / 2))).reshape(-1, 2, 2), t


def crack_density(cracks, area):
    """
    Compute the crack density from an array of crack coordinates.

    The crack density is the combined length of all cracks in a given area.
    Therefore, its unit is m^-1.

    Parameters
    ----------
    cracks: array-like
        Array with the coordinates of the crack with the  following structure:
        ([[x0, y0],[x1,y1]], [[...]]) where x0 and y0 are the starting coordinates and x1, y1
        the end of one crack. Each crack is represented by a 2x2 array stacked into a bigger array (x,2,2).
    area: float
        The area to which the density is refered to.

    Returns
    -------
    crack density: float
    """
    v = cracks[:, 1, :] - cracks[:, 0, :]
    return np.sum(np.hypot(*v.T)) / area


class CrackDetectionTWLI:
    r"""
    The basic method from Glud et al. for crack detection without preprocessing.

    This is the basis for a crack detection with this method. Each object from this class
    can be used to detect cracks from images. The workflow of objects from this class is quite easy.

    #. Object instantiation. Create an object from with the input parameter for the crack detection.
    #. Call the method :meth:`~.detect_cracks` with an image as input.
       This method will call all sub-functions of the crack detection.

       #. apply the gabor filter
       #. apply otsu´s threshold to split the image into foreground and background.
       #. skeletonize the foreground
       #. find the cracks in the skeletonized image.

    Shift detection, normalization, and other preprocessing procedures are not performed! It is assumed that
    all the necessary preprocessing is already done for the input image. For preprocessing please use
    the :mod:`~.stack_operations` or other means.

    Parameters
    ----------
    theta: float
        Angle of the cracks in respect to a horizontal line in degrees
    frequency: float, optional
        Frequency of the gabor filter. Default: 0.1
    bandwidth: float, optional
        The bandwidth of the gabor filter, Default: 1
    sigma_x: float, optional
        Standard deviation of the gabor kernel in x-direction. This applies to the kernel before rotation. The
        kernel is then rotated *theta* degrees.
    sigma_y: float, optional
        Standard deviation of the gabor kernel in y-direction. This applies to the kernel before rotation. The
        kernel is then rotated *theta* degrees.
    n_stds: int, optional
        The size of the gabor kernel in standard deviations. A smaller kernel is faster but also less accurate.
        Default: 3
    min_size: int, optional
        The minimal number of pixels a crack can be. Cracks under this size will not get counted. Default: 1
    threshold: str
        Method of determining the threshold between foreground and background. Choose between 'otsu' or 'yen'.
        Generally, yen is not as sensitive as otsu. For blurry images with lots of noise yen is nearly always
        better than otsu.
    sensitivity: float, optional
        Adds or subtracts x percent of the input image range to the Otsu-threshold. E.g. sensitivity=-10 will lower
        the threshold to determine foreground by 10 percent of the input image range. For crack detection with
        bad image quality or lots of artefacts it can be helpful to lower the sensitivity to avoid too much false
        detections.
    """

    def __init__(self, theta=0, frequency=0.1, bandwidth=1, sigma_x=None, sigma_y=None, n_stds=3,
                 min_size=5, threshold='yen', sensitivity=0):
        self.min_size = min_size
        self.sensitivity = sensitivity
        self._theta = np.radians(theta)
        self.theta_deg = theta
        self.threshold = threshold
        # Gabor kernel
        self.gk = gabor_kernel(frequency, self._theta, bandwidth, sigma_x, sigma_y, n_stds)
        self._gk_real = np.real(self.gk)
        h, w = self.gk.shape
        self.h = int(h / 2)
        self.w = int(w / 2)

    def detect_cracks(self, image, out_intermediate_images=False):
        """
        Compute all steps of the crack detection

        Parameters
        ----------
        image: np.ndarray
        out_intermediate_images: bool, optional
            If True the result of the gabor filter, the foreground pattern as a result of the otsu´s threshold
            and the skeletonized image are also included in the output.
            As this are three full sized images the default is False.

        Returns
        -------
        crack_density: float
        cracks: np.ndarray
            Array with the coordinates of the crack with the following structure:
            ([[x0, y0],[x1,y1], [...]]) where x0 and y0 are the starting coordinates and x1, y1
            the end of one crack. Each crack is represented by a 2x2 array stacked into a bigger array (x,2,2).
        threshold_density: float
            A measure how much of the area of the input image is detected as foreground. If the gabor filter can not
            distinguish between cracks with very little space in between the crack detection will break down and
            lead to false results. If this value is high but the crack density is low, this is an indicator that
            the crack detection does not work with the given input parameters and the input image.
        gabor: np.ndarray, optional
            The result of the Gabor filter.
        pattern: np.ndarray, optional
            A bool image the crack detection detects as cracked area.
        skel_image: np.ndarray, optional
            The skeletonized pattern as bool image.
        """
        # gabor = convolve(image, self._gk_real, mode='same', method='fft')
        gabor = self._gabor_image(image)
        # apply otsu threshold
        pattern = self.foreground_pattern(gabor, self.threshold, self.sensitivity)
        # compute threshold density
        y, x = pattern.shape
        threshold_area = np.sum(pattern)
        threshold_density = threshold_area / (x * y)
        # find cracks
        cracks, skel_img = cracks_skeletonize(pattern, self.theta_deg, self.min_size)
        cd = crack_density(cracks, x * y)
        if out_intermediate_images:
            return cd, cracks, threshold_density, gabor, pattern, skel_img
        else:
            return cd, cracks, threshold_density

    def _gabor_image(self, image):
        """
        Apply the gabor filter to an image.

        Parameters
        ----------
        image: np.ndarray

        Returns
        -------
        out: Result of the gabor filter for the image.
        """
        temp = np.pad(image, ((self.h, self.h), (self.w, self.w)), mode='edge')
        return convolve(temp, self._gk_real, mode='same', method='fft')[self.h:-self.h, self.w:-self.w]

    @staticmethod
    def foreground_pattern(image, method='yen', sensitivity=0):
        """
        Apply the threshold to an image do determine foreground and background of the image.

        The result is a bool array with where True is foreground and False background of the image.
        The image can be split with image[pattern] into foreground and image[~pattern] into background.

        Parameters
        ----------
        image: array-like
        method: str
            Method of determining the threshold between foreground and background. Choose between 'otsu' or 'yen'.
        sensitivity: float, optional
            Adds or subtracts x percent of the input image range to the threshold. E.g. sensitivity=-10 will lower
            the threshold to determine foreground by 10 percent of the input image range.
        Returns
        -------
        pattern: numpy.ndarray
            Bool image with True as foreground.
        """
        threshold = _THRESHOLDS[method](image)

        if sensitivity:
            i_min, i_max = image.min(), image.max()
            threshold += (i_max - i_min) * sensitivity / 100

        # check if yen falls on the wrong side of the histogram (swaps foreground and background)
        if method == 'yen' and threshold > 0:
            histogram, bin_edges = np.histogram(image, bins=256)
            temp = bin_edges[np.argmax(histogram)]
            if not threshold < temp:
                threshold = temp - np.abs(temp - threshold)
        pattern = np.full(image.shape, False)
        pattern[image <= threshold] = True
        return pattern

    def __call__(self, image, **kwargs):
        return self.detect_cracks(image, **kwargs)


def detect_cracks(images, theta=0, crack_width=10, ar=2, bandwidth=1, n_stds=3,
                  min_size=5, threshold='yen', sensitivity=0):
    """
    Crack detection for an image stack. 
    
    All images are treated independent. The crack detection is performed for all images according to the 
    input parameters.

    Parameters
    ----------
    theta: float
        Angle of the cracks in respect to a horizontal line in degrees
    crack_width: int
        The approximate width of an average crack in pixel. This determines the width of the detected features.
    ar: float
        The aspect ratio of the gabor kernel. Since cracks are a lot longer than wide a longer gabor kernel will
        automatically detect cracks easier and artifacts are filtered out better. A too large aspect ratio will
        result in an big kernel which slows down the computation. Default: 2
    bandwidth: float, optional
        The bandwidth of the gabor filter, Default: 1
    n_stds: int, optional
        The size of the gabor kernel in standard deviations. A smaller kernel is faster but also less accurate.
        Default: 3
    min_size: int, optional
        The minimal number of pixels a crack can be. Cracks under this size will not get counted. Default: 1
    threshold: str
        Method of determining the threshold between foreground and background. Choose between 'otsu' or 'yen'.
        Generally, yen is not as sensitive as otsu. For blurry images with lots of noise yen is nearly always
        better than otsu.
    sensitivity: float, optional
        Adds or subtracts x percent of the input image range to the Otsu-threshold. E.g. sensitivity=-10 will lower
        the threshold to determine foreground by 10 percent of the input image range. For crack detection with
        bad image quality or lots of artefacts it can be helpful to lower the sensitivity to avoid too much false
        detections.

    Returns
    -------
    rho_c: float
        Crack density [1/px]
    cracks: np.ndarray
        Array with the coordinates of the crack with the following structure:
        ([[x0, y0],[x1,y1], [...]]) where x0 and y0 are the starting coordinates and x1, y1
        the end of one crack. Each crack is represented by a 2x2 array stacked into a bigger array (x,2,2).
    rho_th: float
        A measure how much of the area of the input image is detected as foreground. If the gabor filter can not
        distinguish between cracks with very little space in between the crack detection will break down and
        lead to false results. If this value is high but the crack density is low, this is an indicator that
        the crack detection does not work with the given input parameters and the input image.
    """

    frequency = 1 / crack_width
    sig = sigma_gabor(crack_width, bandwidth)

    temp = CrackDetectionTWLI(theta, frequency, bandwidth, sig, sig*ar, n_stds, min_size, threshold, sensitivity)
    rho_c, cracks, rho_th = [], [], []
    for ind, img in enumerate(images):
        x, y, z = temp.detect_cracks(img)
        rho_c.append(x)
        cracks.append(y)
        rho_th.append(z)
    return rho_c, cracks, rho_th


def detect_cracks_overloaded(images, theta=0, crack_width=10, ar=2, bandwidth=1, n_stds=3,
                             min_size=5, threshold='yen', sensitivity=0):
    """
    Crack detection with overloaded gabor pattern.

    The gabor pattern is the foreground of the gabor image. The pattern of the nth image gets overloaded
    with the n-1 pattern.

    :math:`P_n = P_n | P_{n-1}`

    Essentially, this means that the area detected as crack one image before is added to the current crack area.
    The cracks are then detected form this overloaded pattern.

    Parameters
    ----------
    theta: float
        Angle of the cracks in respect to a horizontal line in degrees
    crack_width: int
        The approximate width of an average crack in pixel. This determines the width of the detected features.
    ar: float
        The aspect ratio of the gabor kernel. Since cracks are a lot longer than wide a longer gabor kernel will
        automatically detect cracks easier and artifacts are filtered out better. A too large aspect ratio will
        result in an big kernel which slows down the computation. Default: 2
    bandwidth: float, optional
        The bandwidth of the gabor filter, Default: 1
    n_stds: int, optional
        The size of the gabor kernel in standard deviations. A smaller kernel is faster but also less accurate.
        Default: 3
    min_size: int, optional
        The minimal number of pixels a crack can be. Cracks under this size will not get counted. Default: 1
    threshold: str
        Method of determining the threshold between foreground and background. Choose between 'otsu' or 'yen'.
        Generally, yen is not as sensitive as otsu. For blurry images with lots of noise yen is nearly always
        better than otsu.
    sensitivity: float, optional
        Adds or subtracts x percent of the input image range to the Otsu-threshold. E.g. sensitivity=-10 will lower
        the threshold to determine foreground by 10 percent of the input image range. For crack detection with
        bad image quality or lots of artefacts it can be helpful to lower the sensitivity to avoid too much false
        detections.

    Returns
    -------
    rho_c: float
        Crack density [1/px]
    cracks: np.ndarray
        Array with the coordinates of the crack with the following structure:
        ([[x0, y0],[x1,y1], [...]]) where x0 and y0 are the starting coordinates and x1, y1
        the end of one crack. Each crack is represented by a 2x2 array stacked into a bigger array (x,2,2).
    rho_th: float
        A measure how much of the area of the input image is detected as foreground. If the gabor filter can not
        distinguish between cracks with very little space in between the crack detection will break down and
        lead to false results. If this value is high but the crack density is low, this is an indicator that
        the crack detection does not work with the given input parameters and the input image.
    """

    frequency = 1 / crack_width
    sig = sigma_gabor(crack_width, bandwidth)

    temp = CrackDetectionTWLI(theta, frequency, bandwidth, sig, sig*ar, n_stds, min_size, sensitivity)
    rho_c, cracks, rho_th = [], [], []

    # pattern of the n-1st image
    pattern_nminus1 = np.full(images[0].shape, False)
    for ind, img in enumerate(images):
        gabor = temp._gabor_image(img)
        pattern = temp.foreground_pattern(gabor, threshold, sensitivity)
        pattern = pattern | pattern_nminus1
        pattern_nminus1 = pattern
        y, x = pattern.shape
        threshold_area = np.sum(pattern)
        rho_th.append(threshold_area / (x * y))
        # find cracks
        c, skel_img = cracks_skeletonize(pattern, temp.theta_deg, temp.min_size)
        rho_c.append(crack_density(c, x * y))
        cracks.append(c)
    return rho_c, cracks, rho_th
