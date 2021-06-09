Crack Detection
===============

:mod:`~.imagestack` and :mod:`~.stack_operations` are not necessary restrained for the usage in a
crack detection. :mod:`~.imagestack` provides the framework for image processing with
whole stacks. The module :mod:`~.crack_detection` provides the core functionality for the
crack detection algorithm proposed by
`Glud et al. <https://www.sciencedirect.com/science/article/abs/pii/S0266353816300197>`_.

Theory
------

The method from `Glud et al. <https://www.sciencedirect.com/science/article/abs/pii/S0266353816300197>`_
is designed to detect off axis tunneling cracks in composite materials. It is limited to
transparent or semi-transparent composites since transilluminated white light imaging (TWLI) is used to
capture the images. The following image shows the basic principle of TWLI.

.. figure:: images/tiwli.png
    :width: 300

This technique results in a bright image of the specimen with dark crack like this.

.. figure:: images/input_image.png
    :width: 500

The basic steps for this method after preprocessing are:

- **Gabor Filter:** The Gabor filter is applied which detects lines in a set direction. Cracks are only detected in
  the given direction. This allows to separate crack densities from different layers of the laminate.
- **Threshold:** A threshold is applied on the result of the Gabor filter. This separates foreground
  and background in an image. The default is
  `Yen´s threshold <https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.threshold_yen>`_.
  In the case of the crack detection it separates cracked and intact area. The
  Result of the threshold for the image is shown in the next image.

  .. figure:: images/pattern.png
      :width: 300

- **Skeletonizing:** Since off axis tunneling cracks are aligned with the fibers they are straight. The white bands
  from the threshold are thinned to a width of one pixel. The algorithm which determines the start and end of
  each crack relies on only one pixel wide lines. The result of this skeletonizing for a part af the
  threshold image from above is shown in the next image. This
  The lines in this image are not continuous. The skeletonizing is done in a rotated coordinate system. This
  image is rotated back which creates this effect.

  .. figure:: images/skeleton.png
      :width: 300

- **Crack Counting:** The cracks are counted in the skeletonized image. The skeletonized image is rotated into
  a coordinate system where all cracks are vertical (y-direction). Then a loop
  scans each pixel in each line of pixels in the image. If a crack is found, it follows it down the ydirection
  until the end of the crack. The coordinates of the beginning and end are saved. After
  one crack has been detected, it is removed from the image to avoid double detection when the
  loop runs over the next line of pixels. The following image shows this process.

  .. figure:: images/crack_counting.png
      :width: 300

- **Crack Density:** The crack density is computed from the detected cracks with

  :math:`\rho_c = \frac{\sum_{i=1}^{n} L_i}{AREA}`

  with :math:`L_i` as the length of the i-th crack and :math:`AREA` as the area of the image.

- **Threshold Density:** The threshold density is the area which is detected as cracked divided by the total image area.
  It simply is the ratio of white pixels to the total number of pixels in the threshold image. For series of related
  images from the same specimen where the cracks grow and new cracks initiate this measure can be taken as an sanity
  check. If the cracks grow too close to each other the white bands in the threshold image merge. Then the
  crack density fails to detect two individual cracks since the skeletonizing will result in only one line for two
  merged bands. The crack density starts to decrease even tho the threshold density still rises. This is a sign that the
  crack detection reached its limit and the cracks in the images are too close to each other.

The crack density, crack coordinates (start- and endpoints) and the threshold density are the main results of the crack
detection.

Crack Detection Variants
------------------------

.. py:currentmodule:: crackdect.crack_detection

Up to now two main crack detection methods are available in this package.

- **Basic crack detection:**  :func:`~.detect_cracks`

  Crack detection for images without cross -influence between them.

- **Crack detection for related images:**.  :func:`~.detect_cracks_overloaded`

  Here, the initiation and growth of cracks for a series of related images is
  tracked. The area detected as cracked from the n-1st image is added to the cracked area of the n-th image.
  This is done after the image is split into fore- and background with the threshold. This method is for the detection
  of cracks in images from a time series. A change detection must be applied on the image series. The change detection
  deletes all the same features from the n-1st to the n-th image. Then only cracks that formed
  in the n-th image are recognised with the Gabor filter. The cracks from the n-1st image are then added to the
  threshold image. Then the normal procedure with skeletonizing and crack counting is continued. It is also
  important that all the images in the series are aligned in a global coordinate system. If not, the change
  detection in the preprocessing will add a lot of artifacts.

The second method is a variation of the algorithm proposed by
`Glud et al. <https://www.sciencedirect.com/science/article/abs/pii/S0266353816300197>`_. The
first method is essentially what
`Glud et al. <https://www.sciencedirect.com/science/article/abs/pii/S0266353816300197>`_ described as the crack
counting algorithm. If it is exactly the same can not be verified since the implementation details are not included in
the publication and small differences in presets for filters can make a difference. This implementation was
tested with several image series from fatigue tests with varying image quality. The results are sensitive to the input
parameters especially to the parameters which control the gabor filter. Therefore it is a good practice to
try the input parameters on a few images from the preprocessed stack before running the crack detection for the whole
stack. The crack detection is resource intensive and can take quite a long time if a lot of images are processed at
once.

