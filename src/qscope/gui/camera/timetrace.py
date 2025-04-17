import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmap import Colormap
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

mpl.use("Qt5Agg")

from loguru import logger

from qscope.gui.figures.matplotlib_canvas import MplCanvas
from qscope.gui.util.cbar import add_colorbar
from qscope.gui.util.error_handling import show_critial_error, show_info, show_warning


class VideoTimeTrace(QWidget):
    def __init__(self, mw: QMainWindow = None):
        super().__init__()

        # add a matplotlib linetrace to the video tab
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self._ax_ = self.canvas.axes
        self._fig_ = self.canvas.figure

        self._x_ = np.arange(100)
        self._y_ = np.random.rand(100)

        self.init_plot()

    def init_plot(self):
        self._ax_.clear()

        (self._line_,) = self._ax_.plot(
            self._x_,
            self._y_,
            "-o",
            c="#59b6e6",
            mfc=(0, 0, 0, 0),
            mec="#59b6e6",
            mew=2,
            ms=4,
        )

        self._ax_.set_xlabel("Time (s)")
        self._ax_.set_ylabel("Intensity (cps)")

        self.canvas.draw()

    def update_data(self, xdata, ydata):
        self._x_ = xdata
        self._y_ = ydata
        # update the plot
        self.init_plot()

    def fast_update(self, xdata, ydata):
        # set the data
        self._line_.set_ydata(ydata)
        self._line_.set_xdata(xdata)
        # Set the limits
        if xdata.min() < xdata.max():
            self._ax_.set_xlim(xdata.min(), xdata.max())
        if ydata.min() < ydata.max():
            self._ax_.set_ylim(ydata.min(), ydata.max())
        # Draw
        self.canvas.draw()
        self.canvas.flush_events()
