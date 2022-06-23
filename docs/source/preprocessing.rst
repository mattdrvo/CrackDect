Preprocessing
=============

.. currentmodule:: crackdect

The preprocessing for the images is a modular process. Since each user might capture the images in a slightly different
way it´s impossible to just set up one preprocessing routine and expect it to work for all circumstances. Therefore,
the preprocessing is modular. The preprocessing routines included in this package are defined in
:mod:`~.stack_operations`. But these are just some predefined functions for the most important preprocessing steps.
Here an example of how to use custom image processing functions with the image stack (:mod:`~.imagestack`) is shown.

Apply functions
---------------

An arbitrary function that takes one image and other arguments can be applied to the whole image stack. The function
must return an image and nothing else. Applying such an function to the whole image stack will alter all the images
in the stack since the images from the stack are taken as input and are replaced with the output of the function.
E.g histogram equalisation for the whole image stack can be done in one line of code.

.. code-block::

    import crackdect as cd
    from skimage import exposure

    stack.execute_function(exposure.equalize_adapthist, clip_limit=0.03)

This performs *equalize_adapthist* on all images in the stack with a clip limit of 0.03. *clip_limit* is a
keyword argument of *equalize_adapthist*.

With this functionality custom functions can be defined easily without worrying about the image stack. A cascade
of different preprocessing functions can be performed on one image stack. This enables a really modular approach
and the most flexibility.

.. code-block::

    def contrast_stretching(img):
        p5, p95 = np.percentile(img, (5, 95))
        return exposure.rescale_intensity(img, in_range=(p5, p95))

    stack.execute_function(custom_stretching)

Rolling Operations
------------------
Another way to preprocess the images in stacks is to perform an rolling operation on them. A function for rolling
operations takes two images as input and returns just one image. Change detection with image differencing is an example.

:math:`I_d = I_2 - I_1`

Applying this function to the image stack would look like this:

.. code-block::

    def simple_differencing(img1, img2):
        return img2-img1

    stack.execute_rolling_function(simple_differencing, keep_first=False)

This will evaluate *simple_differencing* for all images starting from the second image in the stack. The n-th image
in the stack is computed with this schema.

:math:`I_{new}^n = f(I^{n-1}, I^n)`

Since this schema can only start at the second image, the argument *keep_first* defines is the first image is deleted
after the rolling operation or not. The first image will not be changed since the function is not applied on
it.

Predefined Preprocessing Functions
----------------------------------

.. currentmodule:: crackdect

The most important preprocessing for the crack detection is the change detection and
shift correction. This package comes with functions for these routines. There are variations for both routines and
other useful functions like cutting to the region of interest in :mod:`~.stack_operations`.

All the functions in :mod:`~.stack_operations` take an image stack and return the stack with the results of the
routine. The images in the stack get changed. If the state of the image stack prior to applying a routine should
be kept, copy the stack before.

For more information see the documentation from :mod:`~.stack_operations`.

