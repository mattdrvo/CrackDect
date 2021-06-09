"""
Viewer for an image stack.

This viewer will run in an separate thread. Therefore it does not function in all versions of jupyter.

Call the viewer with

>>> from crackdect.visualization import show_images
>>> show_images(stack)

and navigate the viewer with the arrow keys <- and -> or the mouse wheel.

Additional kwargs for the plot can be set like this

>>> show_images(stack, plt_args=dict(cmap='gray'))
"""
import sys
from PyQt5.QtGui import QWheelEvent, QKeyEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from skimage.viewer.qt import QtWidgets, QtCore

# import matplotlib
# matplotlib.use('Qt5Agg')


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class ImageStackViewer(QtWidgets.QWidget):
    def __init__(self, image_stack, plt_args=None, *args, **kwargs):
        super(ImageStackViewer, self).__init__(*args, **kwargs)
        # main layout
        self.layout = QtWidgets.QVBoxLayout()
        # image stack
        self._stack = image_stack
        self._plt_args = plt_args if plt_args is not None else {}

        # matplotlib canvas and toolbar
        self.canvas = MplCanvas(self)
        toolbar = NavigationToolbar(self.canvas, self)

        self.layout.addWidget(toolbar)
        self.layout.addWidget(self.canvas)

        # toggle axis on and off

        self.setLayout(self.layout)

        self._plot_ref = None
        self.__update_plot(0)

        self.ind = 0

        # self.show()

    def __update_plot(self, ind):
        if self._plot_ref is None:
            self._plot_ref = self.canvas.axes.imshow(self._stack[ind], **self._plt_args)
        else:
            self._plot_ref.set_data(self._stack[ind])
        self.canvas.draw()

    def __add_to_index(self, number):
        if self.ind + number > len(self._stack) - 1:
            self.ind = 0
        elif self.ind + number < 0:
            self.ind = len(self._stack) - 1
        else:
            self.ind = self.ind + number
            self.__update_plot(self.ind)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        x = (delta and delta // abs(delta))
        self.__add_to_index(x)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Left:
            self.__add_to_index(-1)
        elif event.key() == QtCore.Qt.Key_Right:
            self.__add_to_index(1)

    def set_axis_visible(self):
        self.canvas.axes.get_xaxis().set_visible(False)
        self.canvas.axes.get_yaxis().set_visible(False)


def show_images(image_stack, plt_args=None, **kwargs):
    """
    Create a viewer for multiple images.

    Parameters
    ----------
    image_stack: ImageStack or list
    plt_args: dict
        Forwarded to
        `imshow <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html#matplotlib.pyplot.imshow>`_.
        E.g. dict(cmap='gray') for plotting in grayscale
    kwargs:
        forwarded to QtWidgets.QWidget
    """
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    main = ImageStackViewer(image_stack, plt_args=plt_args, **kwargs)
    main.show()
    return main
