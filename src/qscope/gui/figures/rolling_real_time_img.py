from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qscope.gui.main_window import MainWindow

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

mpl.use("Qt5Agg")
from cmap import Colormap

from qscope.gui.figures.matplotlib_canvas import MplCanvas
from qscope.gui.util.error_handling import show_critial_error, show_info, show_warning


class RollingImageFigure(QWidget):
    def __init__(self, parent: MainWindow, cbar_type="horizontal"):
        super().__init__(parent)
        self.parent = parent

        self.cbar_type = cbar_type
        self.cbar = None

        self.canvas = MplCanvas(self, width=8, height=5, dpi=100)
        self.ax = self.canvas.axes
        self.fig = self.canvas.figure

        def lorentzian_noise(x, x0, gamma, A, noise_level):
            return A * gamma**2 / (
                (x - x0) ** 2 + gamma**2
            ) + noise_level * np.random.rand(len(x))

        x = np.linspace(2770, 2970, 100)
        self.y = list(np.linspace(0, 5, 5))
        for i in range(len(self.y)):
            if i == 0:
                data = lorentzian_noise(x, 2870, 10, 100, 10)
            else:
                data = np.vstack((data, lorentzian_noise(x, 2870, 10, 100, 10)))

        self.data = data.copy()
        self.last_sweep = self.data[-1]  # for rolling plot in QMeas LinePlot
        self.init_data = True

        self.norm_type = "signal"

        self.init_plot()

        self.rect = None

    def init_plot(self):
        # clear the figure
        self.ax.clear()

        # make the plot
        self.plot = self.ax.pcolormesh(
            np.arange(self.data.shape[1]),
            self.y,
            self.data,
            cmap=Colormap("seaborn:mako").to_mpl(),
            shading="nearest",
        )

        # set the ticks size
        self.ax.tick_params(axis="y", labelsize=8, direction="in")
        # turn off the x ticks
        self.ax.set_xticks([])

        # make the plot background transparent
        self.ax.patch.set_alpha(0)
        # Set the axis size
        self._set_size(4, 1, self.ax)
        # self.fig.get_layout_engine().set(w_pad=0.1, h_pad=0.2)

    def _set_size(self, w, h, ax=None):
        """w, h: width, height in inches"""
        if not ax:
            ax = plt.gca()
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w) / (r - l)
        figh = float(h) / (t - b)
        ax.figure.set_size_inches(figw, figh)

    def init_data(self, x):
        self.data = None

    def update_data(self, data, y, x):
        self.data = data
        self.y = y
        self.last_sweep = self.data[-1]
        self.update_plot()

    def update_plot(self):
        try:
            self.ax.clear()

            self.plot = self.ax.pcolormesh(
                np.arange(self.data.shape[1]),
                self.y,
                self.data,
                cmap=Colormap("seaborn:mako").to_mpl(),
                shading="nearest",
            )
            # set the ticks size
            self.ax.tick_params(axis="y", labelsize=8, direction="in")
            # turn off the x ticks
            self.ax.set_xticks([])
            self.canvas.draw()
        except Exception as e:
            logger.exception("Error updating rolling meas plot, continuing")

    def set_cmap(self, cmap):
        self.plot.set_cmap(cmap)
        self.update_plot()
