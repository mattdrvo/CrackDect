"""
This module provides the core functionality for handling a stack of images at once.

Image stacks are objects that hold multiple images and act in many cases like python lists. They can
be indexed and images in the stack can be changed. All images in one image stack must have the same dtype. If
an image with another dtype is added or an image in the stack is replaced with an other image with different dtype,
the incoming image is automatically converted to match the dtype of the image stack.

It is strongly recommended that *np.float32* is used when performing a crack detection. The crack detecion is
tested and developed for images of dtypes *float*, *np.float64*, *np.float32* or *np.float16*.

Currently, there are two image stack objects that can be used. All image stack have the same structure.
Accessing images, replacing images in the stack and adding new images works the same for all image stacks.

.. currentmodule:: crackdect.imagestack

* :class:`ImageStack`: A simple wrapper around a list. This container holds all images in the system memory (RAM).

* :class:`ImageStackSQL`: Manages RAM usage of the image stack. Images are held in memory as long as the
  total available memory does not exceed a certain percentage of available memory or the image stack
  exceeds a set number of MB. If any more images are added, all current loaded images get stored in a database and only
  references to the images are kept in memory. The images are only loaded when directly accessed. This allows working and
  changing images of a stack even if the stack is too big to fit into the memory. The loaded images will be kept in
  memory until the stack exceeds the RAM limits again. This reduces the number loading and storing operations and
  therefore saves time since this can be quite time consuming for a lot of images.

The image stack is quite easy to use.
"""
import io
from sqlalchemy import Column, Integer, create_engine, TypeDecorator, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import deferred, sessionmaker
import numpy as np
import psutil
from skimage.util.dtype import _convert
from skimage.io import imread


class NumpyType(TypeDecorator):
    """
    Numpy Type for sql databases when using sqlalchemy.

    This handles the IO with a sql database and sqlalchemy.

    Inside the database, an numpy array is stored as LargeBinary. sqlalchemy handles
    loading and storing of entries for columns marked with this custom type. All arrays are
    converted to numpy arrays when loading and converted to binary when storing in the database automatically.
    """

    impl = LargeBinary

    def __init__(self):
        super(NumpyType, self).__init__()

    def bind_processor(self, dialect):
        impl_processor = self.impl.bind_processor(dialect)
        if impl_processor:

            def process(value):
                if value is not None:
                    out = io.BytesIO()
                    np.save(out, value)
                    out.seek(0)
                    return impl_processor(out.read())

        else:

            def process(value):
                if value is not None:
                    out = io.BytesIO()
                    np.save(out, value)
                    out.seek(0)
                    return out.read()

        return process

    def result_processor(self, dialect, coltype):
        impl_processor = self.impl.result_processor(dialect, coltype)
        if impl_processor:

            def process(value):
                value = impl_processor(value)
                if value is None:
                    return None
                value = io.BytesIO(value)
                value.seek(0)
                return np.load(value)

        else:

            def process(value):
                if value is None:
                    return None
                value = io.BytesIO(value)
                value.seek(0)
                return np.load(value)

        return process


def _add_docstring(func):
    def inner(function):
        if function.__doc__ is None:
            function.__doc__ = func.__doc__
        else:
            function.__doc__ = func.__doc__ + function.__doc__
        return function

    return inner


def _add_to_docstring(docstring):

    def docstring_decorator(func):
        if func.__doc__ is None:
            func.__doc__ = docstring
        else:
            func.__doc__ = func.__doc__ + docstring
        return func

    return docstring_decorator


def _fast_convert(img, dtype):
    """
    Check if the image is already the right dtype.

    This will ignore value limits if the image is already the right dtype

    Parameters
    ----------
    img: array-like
    dtype:
        dtype the image should be converted to

    Returns
    -------
    image: np.ndarray
    """
    if img.dtype.type is dtype:
        return img
    else:
        return _convert(img, dtype)


class ImageStack:
    """
    This object holds multiple images. All images are converted to the same datatype. This ensures that all
    images have the same characteristics for further processing.

    All images are represented as numpy arrays. The same convention for representing images is used as in
    skimage.

    If an image with mismatching dtype is added it is automatically converted to match the dtype.
    Read more about conversion details at skimage.util.dtype.

    This object behaves a lot like a list. Individual images or groups of images can be retrieved with slicing.
    Setitem and delitem behaviour is like with normal python lists but mages can only be added with add_image.

    Parameters
    ----------
    dtype: optional, default=np.float32
        The dtype all images will be converted to. E.g. np.float32, bool, etc.

    Examples
    --------
    >>> # make an ImageStack object where all images are represented as unsigned integer arrays [0-255]
    >>> stack = ImageStack(dtype=np.uint8)
    >>> # Add an image to it.
    >>> img = (np.random.rand(200,200) * np.arange(200))/200  # floating point images must be in range [-1,1]
    >>> stack.add_image(img)
    This ImageStack can be indexed.
    >>> stack[0]  # getting the image with index 0 from the stack
    Changing an image in the stack. The input will also be converted to the dtype of the stack.
    >>> stack[0] = (np.random.rand(200,200) * np.arange(200))/200[::-1]  # setting an image in the stack
    Or deleting an image form the stack
    >>> del stack[0]
    """
    def __init__(self, dtype=np.float32):
        self._dtype = dtype
        self._images = []

    def add_image(self, img):
        """
        Add an image to the stack. The image must be a numpy array

        The input array will be converted to the dtype of the ImageStack

        Parameters
        ----------
        img: np.ndarray
        """
        self._images.append(_fast_convert(img, dtype=self._dtype))

    def remove_image(self, i=-1):
        """
        Remove an image from the stack.

        Parameters
        ----------
        i: int
            Index of the image to be removed
        """
        self._images.pop(i)

    def __len__(self): return self._images.__len__()
    def __repr__(self): return 'ImageStack: {}images, {}'.format(len(self), np.dtype(self._dtype).name)

    def __getitem__(self, i):
        if hasattr(i, '__index__'):
            i = i.__index__()

        if type(i) is int:
            return self._images[i]
        elif type(i) is slice:
            temp_stack = ImageStack(self._dtype)
            temp_stack._images = self._images[i]
            return temp_stack
        else:
            raise TypeError('slicing must be with an int or slice object')

    def __delitem__(self, i): del self._images[i]

    def __setitem__(self, i, item):
        if type(i) is int:
            return self._images.__setitem__(i, _fast_convert(item, dtype=self._dtype))
        elif type(i) is slice:
            if len(item) == len(self._images[i]):
                item = [_fast_convert(j, self._dtype) for j in item]
                self._images[i] = item
            else:
                raise ValueError('{} images provided to override {} images!'.format(len(item), len(self._images[i])))

    # def __add__(self, other):
    #     if isinstance(other, self.__class__) and self._dtype is other._dtype:
    #         self._images = self._images.__add__(other._images)
    #         return self
    #     else:
    #         raise TypeError('Only two image stacks with the same image format can be combined!')

    @classmethod
    def from_paths(cls, paths, dtype=None, **kwargs):
        """
        Make an ImageStack object directly form paths of images. The images will be loaded, converted to the
        dtype of the ImageStack and added.

        Parameters
        ----------
        paths: list
            paths of the images to be added
        dtype: optional
            The dtype all images will be converted to. E.g. np.float32, bool, etc.
            If this is not set, the dtype of the first image loaded will determine the dtype of the stack.
        kwargs:
            kwargs are forwarded to
            `skimage.io.imread <https://scikit-image.org/docs/dev/api/skimage.io.html#skimage.io.imread>`_
            For grayscale images simply add **as_gray = True**. For the kwargs for colored images use
            `parameters for reading <https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.pillow_legacy.html#module-imageio.plugins.pillow_legacy>`_.
            Keep in mind that some images might have alpha channels and some not even if they have the same format.

        Returns
        -------
        out: ImageStack
            An ImageStack with all images from paths as arrays.

        Examples
        --------
        >>> paths = ['list of image paths']
        >>> stack = ImageStack.from_paths(paths, as_gray=True)
        """
        temp = imread(paths[0], **kwargs)
        if dtype is None:
            c = cls(temp.dtype.type)
        else:
            c = cls(dtype)

        c.add_image(temp)
        for p in paths[1:]:
            c.add_image(imread(p, **kwargs))
        return c

    def change_dtype(self, dtype):
        """
        Change the dtype of all images in the stack. All images will be converted to the new dtype.

        Parameters
        ----------
        dtype
        """
        if self._dtype == dtype:
            return

        for i in range(len(self._images)):
            self._images[i] = _convert(self._images[i], dtype)

        self._dtype = dtype

    def copy(self):
        """
        Copy the current image stack.

        The copy is shallow until images are changed in the new stack.

        Returns
        -------
        out: ImageStack
        """
        temp = ImageStack(self._dtype)
        for i in self._images:
            temp.add_image(i)
        return temp

    def execute_function(self, func, *args, **kwargs):
        """
        Perform an operation on all the images in the stack.

        The operation can be any function which takes one images and other arguments as input and returns
        only one image.

        This operation changes the images in the stack. If the current state should be kept copy the stack first.

        Parameters
        ----------
        func: function
            A function which takes ONE image as first input and returns ONE image.
        args:
            args are forwarded to the func.
        kwargs:
            kwargs are forwarded to the func.

        Examples
        --------
        >>> def fun(img, to_add):
        >>>     return img + to_add

        >>> stack.execute_function(fun, to_add=4)
        This will apply the function *fun* to all images in the stack.
        """
        for ind, img in enumerate(self._images):
            self._images[ind] = _fast_convert(func(img, *args, **kwargs), self._dtype)

    def execute_rolling_function(self, func, keep_first=False, *args, **kwargs):
        """
        Perform an rolling operation on all the images in the stack.

        The operation can be any function which takes two images and other arguments as input and returns
        only one image.

        :math:`I_{new} = func(I_{n-1}, I_n)`

        This operation changes the images in the stack. If the current state should be kept copy the stack first.

        Since the 0-th image in the stack will remain unchanged because the rolling operation starts at the 1-st image,
        the 0-th image is removed if *keep_first* is set to *False* (default).

        Parameters
        ----------
        func: function
            A function which takes TWO images and other arguments as input and returns ONE image. The function must
            have the following input structure: `fun(img1, img2, args, kwargs)`. *img1* will be the n-1st image in
            the calls.
        keep_first: bool
            If True, keeps the first image in the stack. Delete it otherwise.
        args:
            args are forwarded to the func.
        kwargs:
            kwargs are forwarded to the func.

        Examples
        --------
        >>> def fun(img1, img2):
        >>>     mask = img1 > img1.max()/2
        >>>     return img2[mask]

        >>> stack.execute_rolling_function(fun, keep_first=False)
        This will apply the function *fun* to all images in the stack.

        *img1* is always the n-1st image in the rolling operation.
        """
        img_minus1 = self._images[0]
        for ind, img in enumerate(self._images[1:]):
            self._images[ind + 1] = _fast_convert(func(img_minus1, img, *args, **kwargs), self._dtype)
            img_minus1 = img
        if not keep_first:
            del self._images[0]


class ImageStackSQL:
    """
    This class works the same as ImageStack.

    ImageStackSQL objects will track the amount of memory the images occupy. When the memory limit if
    surpassed, all data will be stored in an sqlite database and the RAM will be cleared. Only a lazy loaded object
    is left in the image stack. Only when directly accessing the images in the stack they will be loaded into RAM
    again. sqlalchemy is used to connect to the database in which all data is stored.

    This makes this container suitable for long term storage and transfer of a lot of images.
    The images can be loaded into an ImageStackSQL object in a new python session.

    Parameters
    ----------
    database: str, optional
        Path of the database. If it does not exist, it will be created. If none is entered, the name is id(object)
    stack_name: str, optional
        The name of the table the images will be saved. If none is entered it will be id(object)
    dtype: optional, default=np.float32
        The dtype all images will be converted to. E.g. np.float32, bool, etc.
    max_size_mb: float, optional
        The maximal size in mb the image stack is allowed to be. If a new image is added after surpassing this
        size all images will be saved in the database and the occupied RAM is cleared. All images are still accessible
        but will be loaded only when directly accessed.
    cache_limit: float, optional, default=90
        The limit of the RAM usage in percent of the available system RAM. When the RAM usage of the system
        surpasses this limit, all images will be saved in the database and RAM is freed again even it
        max_size_mb is not reached. This makes sure that the system never runs out of RAM.
        Values over 100 will effectively deactivate this behaviour. If the total size of the image stack is
        too small to free enough RAM to reach the cache limit newly added images will be saved immediately in
        the database. This also lead to constant reads from the database as no images will be kept im RAM. Therefore
        it is recommended to set this well over the current RAM usage of the system when instantiating an object.
    """
    def __init__(self, database='', stack_name='', dtype=np.float32, max_size_mb=None, cache_limit=80):
        self._dtype = dtype

        # stack name is the name of the table in the sql database. The table must have a name.
        self.stack_name = stack_name if stack_name != '' else 'table'+str(id(self))
        # database name must end with .db
        self.database = database if database != '' and database.endswith('.db') else 'db'+str(id(self)) + '.db'

        # sqlalchemy connection
        self.engine = create_engine('sqlite:///{}'.format(self.database), echo=False)
        self.session = sessionmaker(bind=self.engine)()
        self.base = declarative_base()
        self.table = type(stack_name, (self.base,), {'__tablename__': self.stack_name,
                                                     'id': Column('id', Integer, primary_key=True),
                                                     'image': deferred(Column('image', NumpyType))})
        self.base.metadata.create_all(self.engine)

        # list for easy access to the images.
        self._images = []

        # ram limits
        self._max_nbytes = max_size_mb * 1e6 if max_size_mb is not None else np.inf
        self._cache_limit = cache_limit

        # nbytes and counter for caching logic
        self._nbytes = 0
        self.__counter = 0

    @classmethod
    def load_from_database(cls, database='', stack_name=''):
        """
        Load an image stack from a database.

        A table of a database which was made with an ImageStackSQL object can be loaded and an ImageStackSQL
        object with all the images is made. The dtype of the images in the new object is the same as the images in the
        table. All images, which will be added to the object will be converted to match the dtype.

        Parameters
        ----------
        database: str
            Path of the database.
        stack_name: str
            Name of the table


        Returns
        -------
        out: ImageStackSQL
            The image stack object with connection to the database.
        """
        c = cls(database, stack_name, dtype=bool)
        dtype = c.session.query(c.table).first().image.dtype
        c._dtype = dtype
        c._images = c.session.query(c.table).all()
        return c

    @classmethod
    def from_paths(cls, paths, database='', stack_name='', dtype=None, max_size_mb=None, cache_limit=80, **kwargs):
        """
        Make an ImageStackSQL object directly form paths of images. The images will be loaded, converted to the
        dtype of the ImageStack and added.

        Parameters
        ----------
        paths: list
            paths of the images to be added
        database: str, optional
            Path of the database. If it does not exist, it will be created. If none is entered, the name is id(object)
        stack_name: str, optional
            The name of the table the images will be saved. If none is entered it will be id(object)
        dtype: optional
            The dtype all images will be converted to. E.g. np.float32, bool, etc.
            If this is not set, the dtype of the first image loaded will determine the dtype of the stack.
        max_size_mb: float, optional
            :class:`ImageStackSQL` for more details.
        cache_limit: float, optional, default=90
            :class`ImageStackSQL` for more details.
        kwargs:
            kwargs are forwarded to
            `skimage.io.imread <https://scikit-image.org/docs/dev/api/skimage.io.html#skimage.io.imread>`_
            For grayscale images simply add **as_gray = True**. For the kwargs for colored images use
            `parameters for reading <https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.pillow_legacy.html#module-imageio.plugins.pillow_legacy>`_.
            Keep in mind that some images might have alpha channels and some not even if they have the same format.

        Returns
        -------
        out: ImageStackSQL
            A new ImageStackSQL object with connection to the database.
        """
        stack_name = stack_name if stack_name != '' else 'table'+str(id(cls))
        database = database if database != '' and database.endswith('.db') else 'db'+str(id(cls)) + '.db'

        temp = imread(paths[0], **kwargs)
        if dtype is None:
            c = cls(database, stack_name, temp.dtype.type, max_size_mb, cache_limit)
        else:
            c = cls(database, stack_name, dtype, max_size_mb, cache_limit)

        c.add_image(temp)
        for p in paths[1:]:
            c.add_image(imread(p, **kwargs))
        return c

    @property
    def nbytes(self):
        """
        Sum of bytes for all currently fully loaded images.

        This tracks the used RAM from the images. The overhead of the used RAM from sqlalchemy is not included and
        will not be tracked.
        """
        return self._nbytes

    @nbytes.setter
    def nbytes(self, x):
        self._nbytes = x
        if x > self._max_nbytes:
            self.save_state()
        elif self.__counter > 50:
            if psutil.virtual_memory().percent > self._cache_limit:
                self.save_state()
            self.__counter = 0
        self.__counter += 1

    @staticmethod
    def __is_loaded(sql_obj):
        return False if 'image' not in sql_obj.__dict__ else True

    def __get_image(self, sql_obj):
        if 'image' not in sql_obj.__dict__:
            out = sql_obj.__getattribute__('image')
            self.nbytes += out.nbytes
            return out
        else:
            return sql_obj.__getattribute__('image')

    def __set_image(self, img, sql_obj):
        temp = _fast_convert(img, self._dtype)
        if not self.__is_loaded(sql_obj):
            self.nbytes += temp.nbytes
        else:
            self.nbytes += temp.nbytes - sql_obj.image.nbytes
        sql_obj.image = temp

    def __clean_remove(self, sql_obj):
        if sql_obj._sa_instance_state.pending:
            self.session.expunge(sql_obj)
        else:
            self.session.delete(sql_obj)

    def __getitem__(self, i):
        if hasattr(i, '__index__'):
            i = i.__index__()

        if type(i) is int:
            return self.__get_image(self._images[i])
        elif type(i) is slice:
            temp_objects = self._images[i]
            temp_stack = ImageStack(self._dtype)
            temp_stack._images = [self.__get_image(j) for j in temp_objects]
            return temp_stack
        else:
            raise TypeError('slicing must be with an int or slice object')

    def __setitem__(self, i, value):
        if hasattr(i, '__index__'):
            i = i.__index__()

        if type(i) is int:
            self.__set_image(value, self._images[i])
        elif type(i) is slice:
            temp = self._images[i]
            if len(value) == len(temp):
                for image, sql_obj in zip(value, temp):
                    self.__set_image(image, sql_obj)
            else:
                raise ValueError('{} images provided to override {} images!'.format(len(value), len(temp)))

    def __delitem__(self, i):
        if hasattr(i, '__index__'):
            i = i.__index__()

        if type(i) is int:
            self.remove_image(i)
        elif type(i) is slice:
            temp = self._images[i]
            for j in temp:
                if self.__is_loaded(j):
                    self._nbytes -= j.image.nbytes
                self.__clean_remove(j)
            del self._images[i]

    def __repr__(self):
        s = 'ImageStack: {}images, {}, {:.2f}/{:.2f} MB RAM used, caching at {} percent of system memory'
        return s.format(len(self), np.dtype(self._dtype).name, self._nbytes / 1e6, self._max_nbytes / 1e6, self._cache_limit)

    def __len__(self): return self._images.__len__()

    def relaod(self):
        """
        Reload the images form the table. All not saved changes will be lost.
        """
        self.session.expire_all()
        self._images = self.session.query(self.table).all()
        self._nbytes = 0

    def save_state(self):
        """
        Saves the current state of the image stack to the database.

        This commits all changes and adds all new images to the table. All currently loaded images are expired. This
        means, that all RAM used by the images is freed.

        Call this method before closing the python session if the changes made to the image stack should be
        saved permanently.
        """
        self.session.commit()
        self.session.expire_all()
        self._nbytes = 0

    @_add_docstring(ImageStack.add_image)
    def add_image(self, img):
        temp = self.table(image=_fast_convert(img, dtype=self._dtype))
        self._images.append(temp)
        self.session.add(temp)
        self.nbytes += temp.image.nbytes

    @_add_docstring(ImageStack.remove_image)
    def remove_image(self, i=-1):
        temp = self._images[i]
        if self.__is_loaded(temp):
            self._nbytes -= temp.image.nbytes
        self.__clean_remove(temp)
        self._images.pop(i)

    @_add_docstring(ImageStack.change_dtype)
    def change_dtype(self, dtype):
        if dtype == self._dtype:
            return

        for i in self._images:
            bytes_old = i.image.nbytes
            i.image = _convert(i.image, dtype)
            self.nbytes += i.image.nbytes - bytes_old

        self._dtype = dtype

    def copy(self, stack_name=''):
        """
        Copy the current image stack.

        A new table in the database is created where all images are stored.

        Parameters
        ----------
        stack_name: str
            The name of the stack. This is also the name of the new table in the database

        Returns
        -------
        out: ImageStack
        """

        temp = ImageStackSQL(self.database, stack_name, self._dtype, self._max_nbytes/1e6, self._cache_limit)
        for i in self._images:
            temp.add_image(i.image)
        return temp

    @_add_docstring(ImageStack.execute_function)
    def execute_function(self, func, *args, **kwargs):
        for i in self._images:
            bytes_old = i.image.nbytes
            i.image = _fast_convert(func(i.image, *args, **kwargs), self.dtype)
            self.nbytes += i.image.nbytes - bytes_old

    @_add_docstring(ImageStack.execute_rolling_function)
    def execute_rolling_function(self, func, keep_first=False, *args, **kwargs):
        img_minus1 = self._images[0].image
        for i in self._images[1:]:
            temp = i.image.copy()
            bytes_old = i.image.nbytes
            i.image = _fast_convert(func(img_minus1, i.image, *args, **kwargs), self._dtype)
            img_minus1 = temp
            self.nbytes += i.image.nbytes - bytes_old
        if not keep_first:
            self.remove_image(0)

