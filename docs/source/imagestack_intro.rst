The Image Stack
===============

Since this package is build for processing multiple images the efficient handling of image collections is important. The
whole functionality of the package as also available for working with single images but the API of most top level
functions is built for image stacks.

The image stack is the core of this package. It works as a container for collections of images. It can hold images of
any size and color/grayscale images can be mixed. The only restriction is that all images are from the same data type
e.g *np.float32*, *np.unit8*, etc. The data type of incoming images is checked automatically and if the types do not
match the incoming image is converted.

.. py:currentmodule:: crackdect.imagestack

All image stack classes have the same io structure. Images can be added, removed and altered. Single images are
accessed with indices and groups of images via slicing. Currently these image stacks are available:

* :class:`ImageStack`: The most basic image stack. It just manages the dtype checks and conversion of incoming
  images. All images are held in memory (RAM) at all times. When working with just a few images this is the best choice
  since it adds nearly zero overhead.


* :class:`ImageStackSQL`: When working with a large number of images available RAM can become a problem.
  This container manages the used RAM of the stack. When exceeding set limits it automatically saves the current state
  of the images to an SQL database. It is built with `sqlalchemy <https://www.sqlalchemy.org/>`_ for maximum flexibility.
  It can als be used to save the current state of an image stack for later usage or transferring the data to other
  locations. Image stacks can be constructed directly from the created databases. The creation of the database is
  automated and does not need user input. One database can hold multiple stacks so more than one image stack can
  interact with one database at once. `Sqlalchemy <https://www.sqlalchemy.org/>`_ handles all transactions
  with the database.

This is a quick introduction on how the image stack works. The full API documentation is :ref:`here <imagestack>`.

Now a few examples are given on how to use an image stack. Image stacks can be constructed from directly or
via convenience methods to automatically load all images from a list of paths.

Basic functionality
-------------------

This is an example of the basic functionality all image stacks must have to work with the preprocessing funcitons and
the crack detection.

Directly construct an image stack. The dtype of the images in the stack should be set. The default is *np.float32* since
all functions and the crack detection are optimised for handling float images.

.. code-block:: python

    import crackdect as cd
    stack = cd.ImageStack(dtype=np.float32)

Adding images to the stack directly. Numpy arrays and `pillow <https://pillow.readthedocs.io/en/stable/>`_
image objects can be added. PIL images will be converted to numpy arrays.

.. code-block:: python

    stack.add_image(img)

To access images from the stack use indices or slices if multiple images should be accessed. The return when slicing
will be a new image stack.

.. code-block:: python

    stack[0]  # => first image = numpy array
    stack[1:4]  # => image stack of the images with index 1-4(not included).
    stack[-1]  # => last image of the stack.

Overriding images in a stack works also like for objects in normal lists.

.. code-block:: python

    stack[1] = np.random.rand(200,200) * np.linspace(0,1,200)

This overrides the 2nd image in the stack. If the dtype does not fit the image is converted.
Multiple images can be overridden at once

.. code-block:: python

    stack[1:5] = ['list of 4 images']

But unlike lists 4 images must be given to replace 4 images in the stack. There is no thing as sub-stacks. Removing
images also works like for lists.

.. code-block:: python

    del stack[4]  # removes the 5th image of the stack
    del stack[-3:]  # removes the last 3 images.
    stack.remove_image(4)  # the same as del stack[4] but no slicing possible
    stack.remove_image()  # removes per default the last image


Advanced Features
-----------------

:class:`ImageStackSQL` has more functionality to it. It can be created like the normal :class:`ImageStack`
but it is it is recommended to set the name of the database and the name of the table the images
will be stored in. With this it is easy to identify saved results. If no
names are set, the object id is taken. The database is created in the current
working directory.

.. code-block:: python

    stack = cd.ImageStackSQL()  # completely default creation
    stack = cd.ImageStackSQL(database='test', stack_name='test_stack1')

Multiple stacks can be connected with one database

.. code-block:: python

    stack2 = cd.ImageStackSQL(database='test', stack_name='test_stack2')
    stack3 = cd.ImageStackSQL(database='test', stack_name='test_stack3')

Saving and loading is done automatically but only when needed. So it is possible that
the stack was altered but the current state is not saved jet. To save the current state call

.. code-block:: python

    stack.save_state()

This will save all changes and free the RAM the images used. When images are accessed after this, they
are loaded form the databased again.

All stacks can be copied.

.. code-block:: python

    new_stack = stack.copy()  # works for all stacks

Stacks with sql connection should be named

.. code-block:: python

    new_sql_stack = sql_stack.copy(stack_name='test_stack4')

Copying a normal stack will not use more ram until the images in the new stack are overridden.
Copying a stack with sql-connection will create a new table in the database and copy all
images to the new table. For big image stacks, this is a costly operation since all images
will be loaded at some point, copied to the other table and saved there. It the image stack exceeds
its set RAM limits multiple rounds of loading parts of the stack and saving them in
the new table may be required.

Convenience Creation
--------------------

To avoid manually loading all images and putting them into an image stack
there are several options to automatically create an image stack. Images are loaded with
`skimage.io.imread <https://scikit-image.org/docs/dev/api/skimage.io.html#skimage.io.imread>`_
so a huge flexibility is provided to control the loading process which can be controlled with kwargs.

.. code-block:: python

    # create from a list of image paths
    stack = cd.ImageStack.from_paths(['list of paths'])
    # create image stack with database connection. Database and stack_name are optional
    stack = cd.ImageStackSQL.from_paths(['list of paths'], 'database', 'stack_name')
    # create from previously saved database.
    stack = cd.ImageStackSQL.load_from_database('database', 'stack_name')

The simplest form of creating a basic :class:`ImageStack` is

.. code-block:: python

    stack = cd.load_images(['list of paths'])

For more information and more control over the behaviour of the full documentation for :ref:`imagestacks <imagestack>`.

