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


class ImageFigure(QWidget):
    def __init__(self, parent=None, cbar_type="horizontal"):
        super().__init__(parent)
        self.parent = parent

        self.cbar_type = cbar_type
        self.cbar = None

        self.canvas = MplCanvas(self, width=3, height=4, dpi=100)
        self.ax = self.canvas.axes
        self.fig = self.canvas.figure

        self.frame = np.random.rand(100, 100) + 1
        self.frame_to_plt = self.frame.copy()

        self.x_data = np.arange(self.frame.shape[1])
        self.y_data = np.arange(self.frame.shape[0])

        self.norm_type = "signal"

        self.bin_x = 1
        self.bin_y = 1

        self.b_use_limits = False
        self.vmin = 0
        self.vmax = 10000

        self.init_plot(self.frame)

        self.rect = None

    def init_plot(self, frame=None):
        if frame is not None:
            self.frame = frame

        # clear the figure
        self.ax.clear()
        if self.cbar is not None:
            try:
                self.cbar.remove()
                del self.cbar
                self.cbar = None
            except:
                pass

        # make the plot
        self.plot = self.ax.imshow(
            self.frame,
            # aspect='auto',
            # cmap="viridis",
            cmap=Colormap("seaborn:mako").to_mpl(),
            origin="upper",
            extent=[0, self.frame.shape[1], self.frame.shape[0], 0],
            interpolation="nearest",
        )

        self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")
        # set the ticks size
        self.ax.tick_params(axis="x", labelsize=8, direction="in")
        self.ax.tick_params(axis="y", labelsize=8, direction="in")

        # make the figure have a tight layout
        self.fig.get_layout_engine().set(w_pad=0.5, h_pad=0.5)

        if self.cbar is None:
            if self.cbar_type == "horizontal":
                self.cbar = add_colorbar(
                    self.plot,
                    self.fig,
                    self.ax,
                    aspect=25,
                    orientation="horizontal",
                    location="top",
                )
                self.cbar.ax.tick_params(axis="x", labelsize=8, direction="in")
                self.cbar.set_label("Intensity (cps)")
            else:
                self.cbar = add_colorbar(
                    self.plot, self.fig, self.ax, aspect=25, orientation="vertical"
                )

                self.cbar.set_label("Intensity (cps)", rotation=270)
                self.cbar.ax.tick_params(axis="y", labelsize=8, direction="in")

    def update_data(self, frame):
        # apply the requested binning
        # change type to np.array if frame is a list
        if type(frame) == list:
            frame = np.array(frame)

        self.frame_to_plt = frame
        # # update the plot extent
        self.plot.set_extent(
            [0, self.frame_to_plt.shape[1], self.frame_to_plt.shape[0], 0]
        )

        self.update_plot()

    def update_plot(self, frame_to_plt=None, *args, **kwargs):
        if frame_to_plt is None:
            frame_to_plt = self.frame_to_plt
        else:
            self.frame_to_plt = frame_to_plt

        if self.b_use_limits:
            self.plot.set_data(frame_to_plt)
            self.plot.set_clim(self.vmin, self.vmax)
        else:
            self.plot.set_data(frame_to_plt)
            self.plot.autoscale()
        self.canvas.draw()

    def set_binning(self, bin_x, bin_y):
        self.bin_x = bin_x
        self.bin_y = bin_y
        # update the plots tick labels
        self.update_data(self.frame)

    def set_cmap(self, cmap):
        self.plot.set_cmap(cmap)
        self.update_plot()

    def set_color_limits(self, state, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax
        self.b_use_limits = state
