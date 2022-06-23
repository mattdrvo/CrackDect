"""
Crack detection algorithms

These module contains the different functions for the crack detection. This includes functions for different
sub-algorithms which are used in the final crack detection as well as different methods for the crack detection.
The different crack detection methods are available as functions with an image stack and additional arguments
as input.

"""
import numpy as np
from numpy.lib.utils import deprecate_with_doc
from scipy.signal import convolve
from numba import jit, int32
# from skimage.morphology._skeletonize_cy import _fast_skeletonize
from skimage.morphology._skeletonize import skeletonize_3d
from skimage.morphology import closing
from skimage.transform import rotate
from skimage.filters import gabor_kernel, threshold_otsu, threshold_yen, gaussian, unsharp_mask
from skimage.util import img_as_float
from skimage.filters._gabor import _sigma_prefactor


_THRESHOLDS = {'yen': threshold_yen,
               'otsu': threshold_otsu}


def rotation_matrix_z(phi):
    """
    Rotation matrix around the z-axis.

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


def _sigma_gabor(lam, bandwidth=1):
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
def _find_crack_end(sk_image, start_row, start_col):
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
            check_cols = [active_col, active_col - 1]
        else:
            check_cols = [active_col, active_col - 1, active_col + 1]

        b, new_col = check_columns(active_row + 1, check_cols)
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
                crack_end = np.array(_find_crack_end(image, row, col), dtype=np.int32)

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


def anisotropic_gauss_kernel(sig_x, sig_y, theta=0, truncate=3):
    """
    Gaussian kernel with different standard deviations in x and y direction.

    Parameters
    ----------
    sig_x: int
        Standard deviation in x-direction.
        A value of e.g. 5 means that the Gaussian kernel will reach a standard deviation of 1 after 5 pixel.
    sig_y: int
        Standard deviation in y-direction.
    theta: float
        Angle in degrees
    truncate: float
        Truncate the filter at this many standard deviations.
        Default is 4.0.
    Returns
    -------
    kernel: ndarray
        The Gaussian kernel as a 2D array.
    """
    r_x = int(truncate * sig_x + 0.5)
    r_y = int(truncate * sig_y + 0.5)
    xx = np.arange(-r_x, r_x + 1)
    yy = np.arange(-r_y, r_y + 1)
    sig_x2 = sig_x * sig_x
    sig_y2 = sig_y * sig_y

    phi_x = np.exp(-0.5 / sig_x2 * xx ** 2)
    phi_y = np.exp(-0.5 / sig_y2 * yy ** 2)
    kernel = np.outer(phi_x, phi_y)
    if theta != 0:
        kernel = rotate(kernel, theta, resize=True, order=3)
    return kernel / np.sum(kernel)


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
        Adds or subtracts x percent of the input image range to the threshold. E.g. sensitivity=-10 will lower
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


class CrackDetectionBender:
    r"""
    Base class for the crack detection `method by J.J. Bender
    <https://www.researchgate.net/publication/350967596_Effect_of_variable_amplitude_block_loading_on_intralaminar_crack_initiation_and_propagation_in_multidirectional_GFRP_laminate>`_.

    This crack detection algorithm only works on an image stack with consecutive images of one specimen.
    The first image is used as the background image.
    No cracks are detected in the first image. The images must be aligned for this algorithm to
    work correctly. Cracks can only be detected in grayscale images with the same shape.

    Following filters are applied to the images:
    1: Apply image history. Images must become darker with time. This subsequently reduces noise in the imagestack
    2: Image division with reference image (first image of the stack) to remove constant objects.
    3: The image is divided by a blurred version of itself to remove the background.
    4: A directional Gaussian filter is applied to diminish cracks in other directions.
    5: Images are sharpened with an `unsharp_mask <https://scikit-image.org/docs/stable/auto_examples/filters/plot_unsharp_mask.html>`_.
    6: A threshold is applied to remove falsely identified cracks or artefacts with a weak signal.
    7: Morphological closing of the image with a crack-like footprint.
    8: Binarization of the image
    9: The n-1st binaritzed image is added.
    10: Finde the cracks with the skeletonizing and scanning method.

    Parameters
    ----------
    theta: float
        Angle of the cracks in respect to a horizontal line in degrees
    crack_width: int
        The approximate width of an average crack in pixel. This determines the width of the detected features.
    threshold: float, optional
        Threshold of what is perceived as a crack after all filters. E.g. 0.96 means that all gray values over 0.96
        in the filtered image are cracks. This value should be close to 1. A lower value will detect cracks with a
        weak signal but more artefacts as well. Default: 5
    min_size: int, optional
        The minimal number of pixels a crack can be. Cracks under this size will not get counted. Default: 5

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

    def __init__(self, theta=0, crack_width=10, threshold=0.96, min_size=None):
        self.crack_width = int(crack_width)
        self.theta = theta % 360
        if threshold >= 1:
            raise ValueError('The threshold must be lower than 1!')
        self.threshold = threshold

        self.gk = anisotropic_gauss_kernel(crack_width, crack_width / 2, -theta, truncate=3)
        self.closing_footprint = self.make_footprint(self.crack_width, self.theta)
        if min_size is None:
            self.min_size = crack_width * 4
        else:
            self.min_size = min_size

    def detect_cracks(self, images):
        if len(images) <= 1:
            raise ValueError('This crack detection algorithm needs more than one image to detect cracks. The first'
                             'image in the input stack must be the reference and no cracks will be detected in this'
                             'image.')
        img_0 = img_as_float(images[0], force_copy=True)
        hist_img = img_0.copy()
        pattern_n1 = np.zeros(img_0.shape, dtype=bool)

        cd, cracks, threshold = [0], [np.array([])], [0]
        for ind in range(1, len(images)):
            # step 1: applying image history on the n-th image. (0, 1) = (black, white)
            img_n = img_as_float(images[ind], force_copy=True)
            if img_n.shape != hist_img.shape:
                raise ValueError(f'The shape of image {ind} and {ind - 1} is {img_n.shape} and {hist_img.shape}.'
                                 f'This is not allowed since all images must have the same shape for this algorithm to'
                                 f'work.')
            mask = img_n > hist_img
            img_n[mask] = hist_img[mask]
            hist_img = img_n.copy()

            # step 2: change detection with division. No need for cutoff since values can only range from >0 to 1.
            # With history, nth image is always lower than n-1st. -> white = no change, black = change
            img_n = np.divide(img_n, img_0, out=np.ones_like(img_n), where=img_0 != 0)

            # step 3: division with blurred image
            img_n = img_n / gaussian(img_n, sigma=self.crack_width)

            # step 4: Directional Gaussian filter
            img_n = self.anisotropic_gauss_filter(img_n, self.gk)

            # step 5: sharpening image -> will rescale to 0-1
            img_n = unsharp_mask(img_n, radius=self.crack_width, amount=2, preserve_range=False)

            # step 6: apply threshold to % of the current range of the image
            img_n[img_n > self.threshold] = 1

            # step 7: morphological closing with line element
            img_n = closing(img_n, self.closing_footprint)

            # step 8: binarization with threshold of 99% -> only 0 and 1 in image
            img_n[img_n < 0.99] = 0

            # step 9: computing threshold density and crack density
            pattern = ~img_n.astype(bool)
            pattern = np.logical_or(pattern, pattern_n1)
            pattern_n1 = pattern
            y, x = pattern.shape
            threshold.append(np.sum(pattern) / (x * y))
            c, skel_img = cracks_skeletonize(pattern, self.theta, self.min_size)
            cd.append(crack_density(c, x * y))
            cracks.append(c)
        return cd, cracks, threshold

    # TODO find faster convolution method or separate gauss kernel.
    # Convolution form scipy.signal is not exactly the same as from scipy.ndimage but takes much longer.
    # Convolution from scipy.signal is used. The only difference occurs at the edges of the image but it is ~50x
    # faster form testing with 1900x1800 images and a kernel of 77x77. The mean squared error between tests was ~10e-7
    # It seems that the padding in scipy.signal.convolve is different since only the edges are affected.
    # With the additional padding in this function the difference is even smaller.
    @staticmethod
    def anisotropic_gauss_filter(image, kernel):
        h, w = kernel.shape
        h = int(h / 2)
        w = int(w / 2)
        temp = np.pad(image, ((h, h), (w, w)), mode='reflect')
        return convolve(temp, kernel, mode='same', method='fft')[h:-h, w:-w]

    @staticmethod
    def make_footprint(width, theta):
        p_w = max(int(width / 4), 1)
        closing_footprint = rotate(np.pad(np.ones((width * 4, p_w), dtype=bool), (p_w, p_w)), -theta, resize=True)
        ind = np.argwhere(closing_footprint == 1)
        c_min, c_max = np.min(ind, axis=0), np.max(ind, axis=0) + 1
        return closing_footprint[c_min[0]: c_max[0], c_min[1]: c_max[1]]


def detect_cracks(images, theta=0, crack_width=10, ar=2, bandwidth=1, n_stds=3,
                  min_size=5, threshold='yen', sensitivity=0):
    """
    Crack detection based on a simpler version of the algorithm by J.A. Glud.
    All images are treated separately.

    Parameters
    ----------
    images: ImageStack, list
        Image stack or list of grayscale images on which the crack detection will be performed. This algorithm treats
        each image separately as no image influences the results of the other images.
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
        The minimal number of pixels a crack can be. Cracks under this size will not get counted. Default: 5
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
    sig = _sigma_gabor(crack_width, bandwidth)

    temp = CrackDetectionTWLI(theta, frequency, bandwidth, sig, sig * ar, n_stds, min_size, threshold, sensitivity)
    rho_c, cracks, rho_th = [], [], []
    for ind, img in enumerate(images):
        x, y, z = temp.detect_cracks(img)
        rho_c.append(x)
        cracks.append(y)
        rho_th.append(z)
    return rho_c, cracks, rho_th


def detect_cracks_glud(images, theta=0, crack_width=10, ar=2, bandwidth=1, n_stds=3,
                       min_size=5, threshold='yen', sensitivity=0):
    """
    Crack detection using a slightly modified version of the `algorithm from J.A. Glud.
    <https://www.researchgate.net/publication/292077678_Automated_counting_of_off-axis_tunnelling_cracks_using_digital_image_processing>`_

    This crack detection algorithm only works on an image stack with consecutive images of one specimen.
    In contrast to the original algorithm from J.A. Glud, no change detection is applied in this implementation since
    it can be easily applied as a preprocessing step if needed (see :mod:`~.stack_operations`)

    Parameters
    ----------
    images: ImageStack, list
        Image stack or list with consecutive grayscale images (np.ndarray) of the same shape and aligned.
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
        The minimal number of pixels a crack can be. Cracks under this size will not get counted. Default: 5
    threshold: str
        Method of determining the threshold between foreground and background. Choose between 'otsu' or 'yen'.
        Generally, yen is not as sensitive as otsu. For blurry images with lots of noise yen is nearly always
        better than otsu.
    sensitivity: float, optional
        Adds or subtracts x percent of the input image range to the threshold. E.g. sensitivity=-10 will lower
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
    sig = _sigma_gabor(crack_width, bandwidth)

    temp = CrackDetectionTWLI(theta, frequency, bandwidth, sig, sig * ar, n_stds, min_size, sensitivity)
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


@deprecate_with_doc(msg='This function is deprecated in version 0.2 and will be removed in the next version! Use '
                        '"detect_cracks_glud" instead!')
def detect_cracks_overloaded(images, theta=0, crack_width=10, ar=2, bandwidth=1, n_stds=3,
                             min_size=5, threshold='yen', sensitivity=0):
    return detect_cracks_glud(images, theta, crack_width, ar, bandwidth, n_stds, min_size, threshold, sensitivity)


def detect_cracks_bender(images, theta=0, crack_width=10, threshold=0.96, min_size=None):
    r"""
    Crack detection `algorithm by J.J. Bender.
    <https://www.researchgate.net/publication/350967596_Effect_of_variable_amplitude_block_loading_on_intralaminar_crack_initiation_and_propagation_in_multidirectional_GFRP_laminate>`_

    This crack detection algorithm only works on an image stack with consecutive images of one specimen.
    The first image is used as the background image.
    No cracks are detected in the first image. The images must be aligned for this algorithm to
    work correctly. Cracks can only be detected in grayscale images with the same shape.

    Parameters
    ----------
    images: ImageStack, list
        Image stack or list with consecutive grayscale images (np.ndarray) of the same shape and aligned.
    theta: float
        Angle of the cracks in respect to a horizontal line in degrees
    crack_width: int
        The approximate width of an average crack in pixel. This determines the width of the detected features.
    threshold: float, optional
        Threshold of what is perceived as a crack after all filters. E.g. 0.96 means that all gray values over 0.96
        in the filtered image are cracks. This value should be close to 1. A lower value will detect cracks with a
        weak signal but more artefacts as well. Default: 5
    min_size: int, optional
        The minimal number of pixels a crack can be. Cracks under this size will not get counted. Default: 5

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
    cd = CrackDetectionBender(theta, crack_width, threshold, min_size)
    return cd.detect_cracks(images)
