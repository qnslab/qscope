import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from qscope.gui.util import get_contrasting_text_color

mpl.use("Qt5Agg")
import asyncio

from matplotlib.widgets import Cursor

import qscope.fitting
from qscope.gui.figures.matplotlib_canvas import MplCanvas


class LineFigure(QWidget):
    # cursor_x: float

    def __init__(self, mw: QMainWindow):
        super().__init__()
        self.mw = mw

        # Initialise parameters
        self.norm = "None"
        self.bfit = False
        self.fit_param_results = "Fit Parameters"
        self.fit_type = "Lorentzian"

        self.cursor_color = (1, 0.33, 1)

        self.y_label_key = {
            "Normalise (%)": "Intensity (%)",
            "sig / ref": "Intensity (norm.)",
            "ref / sig": "Intensity (norm.)",
            "sig - ref": "Intensity (cps)",
            "ref - sig": "Intensity (cps)",
            "norm ref / sig": "Intensity (%)",
            "norm sig / ref": "Intensity (%)",
            "intensity": "Intensity (norm.)",
            "None": "Intensity (cps)",
        }

        self.x_multiplier = 1
        self.b_realtime_plot = False
        self.b_rolling_plot = False
        self.b_comp = False
        self.label_comp = None

        self.canvas = MplCanvas(self, width=8, height=5, dpi=100)
        self.x_data = np.linspace(2770, 2970, 100)
        self.y_data = (
            -1 / (1 + (self.x_data - 2870) ** 2 / 10**2)
            + 1000
            + 0.1 * np.random.randn(100)
        )
        # self.cursor_x = np.min(self.x_data)

        self.y_data_ref = 1000 + 0.1 * np.random.randn(100)
        self.set_data(self.x_data, self.y_data, self.y_data_ref)

        self.set_x_label("Frequency (MHz)")

        # Make the canvas background transparent
        self.canvas.setStyleSheet("background-color: None")

        self.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        # Create and add toolbar
        # self.toolbar = NavigationToolbar(self.canvas, self)
        

        def set_size(w, h, ax=None):
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

        bg_color = plt.gca().get_facecolor()

        self.xytext = self.canvas.axes.text(
            0.01,
            0.01,
            "0, 0",
            visible=True,
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=self.canvas.axes.transAxes,
            color=get_contrasting_text_color(bg_color),
        )
        self.xytext.set_visible(False)

        self.hline = self.canvas.axes.axhline(
            y=0,
            color=self.cursor_color,
            alpha=0.6,
            linestyle="-",
            zorder=-10,
            label="_nolegend_",
        )
        self.hline.set_visible(False)
        self.vline = self.canvas.axes.axvline(
            x=0,
            color=self.cursor_color,
            alpha=0.6,
            linestyle="-",
            zorder=-10,
            label="_nolegend_",
        )
        self.vline.set_visible(False)
        (self.cursor_spot,) = self.canvas.axes.plot(
            0,
            0,
            "o",
            mec=(*self.cursor_color, 0.6),
            mfc=(0, 0, 0, 0),
            zorder=-10,
            label="_nolegend_",
        )
        self.cursor_spot.set_visible(False)

        xytext_position = self.xytext.get_position()
        cursor_text_position = (
            xytext_position[0],
            xytext_position[1] + 0.04,
        )  # Adjust the offset as needed

        self.cursor_text = self.canvas.axes.text(
            0,
            0,
            "0; 0",
            color=self.cursor_color,
            transform=self.canvas.axes.transAxes,
            horizontalalignment="left",
            verticalalignment="bottom",
        )
        self.cursor_text.set_position(cursor_text_position)
        self.cursor_text.set_visible(False)

        self.canvas.draw()

        # Set the axis size
        set_size(7, 4, self.canvas.axes)

        self.canvas.setStatusTip(
            "Left click to place cursor, right click to remove cursor"
        )

    def on_move(self, event):
        # printing x. y values
        try:
            if event.inaxes == self.canvas.axes:
                self.xytext.set_text(f"({event.xdata:.4g}; {event.ydata:.4g})")
                self.xytext.set_visible(True)
                self.xytext.set_position((0.01, 0.01))
                self.canvas.draw_idle()
            else:
                self.xytext.set_visible(False)
                self.canvas.draw_idle()
        except Exception as e:
            logger.error(f"Error in on_move: {e}")

    def on_click(self, event):
        try:
            if event.inaxes == self.canvas.axes:
                if event.button == 3:  # Right mouse button
                    self.cursor_spot.set_visible(False)
                    self.cursor_text.set_visible(False)
                    self.hline.set_visible(False)
                    self.vline.set_visible(False)
                    self.canvas.draw_idle()
                else:
                    self.hline.set_ydata([event.ydata])
                    self.vline.set_xdata([event.xdata])
                    self.cursor_spot.set_data([event.xdata], [event.ydata])
                    self.cursor_spot.set_visible(True)
                    self.cursor_text.set_text(f"({event.xdata:.4g}; {event.ydata:.4g})")
                    self.cursor_text.set_color(self.cursor_color)
                    self.cursor_text.set_visible(True)
                    self.hline.set_visible(True)
                    self.vline.set_visible(True)
                    self.canvas.draw_idle()
        except Exception as e:
            logger.error(f"Error in on_click: {e}")

    def set_normalisation(self, norm):
        self.norm = norm
        # self.update_plot()

    def set_fitting(self, bfit):
        self.bfit = bfit
        self.update_plot(force_fit=True)

    def is_fitting(self):
        return self.bfit

    def set_x_label(self, label):
        self.canvas.axes.set_xlabel(label)
        self.canvas.draw()

    def set_y_label(self, label):
        self.canvas.axes.set_ylabel(label)
        self.canvas.draw()

    def set_x_multiplier(self, multiplier):
        self.x_multiplier = multiplier
        self.update_plot()

    def update_realtime_data(self, x_data, y_data, y_data_ref):
        self.realtime_x_data = x_data * self.x_multiplier
        self.realtime_y_data = y_data
        self.realtime_y_data_ref = y_data_ref

    def plot_realtime_data(self, state):
        self.b_realtime_plot = state
        if state:
            self.b_rolling_plot = False

    def plot_rolling_avg(self, state):
        self.b_rolling_plot = state
        if state:
            self.b_realtime_plot = False

    def set_data(self, x_data, y_data, y_data_ref, force_fit=False):
        self.x_data = x_data * self.x_multiplier
        self.y_data = y_data
        self.y_data_ref = y_data_ref

        self.xlims = [np.min(self.x_data), np.max(self.x_data)]

        # For plotting with partial data check if the data contains any zeros and remove them
        if np.any(self.y_data == 0):
            self.x_data = self.x_data[self.y_data != 0]
            self.y_data = self.y_data[self.y_data != 0]
            self.y_data_ref = self.y_data_ref[self.y_data_ref != 0]
        try:
            if self.mw.line_figure_opts.start_index_input.value() != 0:
                start_index = self.mw.line_figure_opts.start_index_input.value()
                self.x_data = self.x_data[start_index:]
                self.y_data = self.y_data[start_index:]
                self.y_data_ref = self.y_data_ref[start_index:]

            if self.mw.line_figure_opts.end_index_input.value() != 0:
                end_index = self.mw.line_figure_opts.end_index_input.value()
                self.x_data = self.x_data[: -end_index - 1]
                self.y_data = self.y_data[: -end_index - 1]
                self.y_data_ref = self.y_data_ref[: -end_index - 1]
        except Exception as e:
            pass

        self.update_plot(force_fit=True)

    def get_plot_data(self):
        return self.x_data, self.y_plot

    def set_comparison_data(self, x_data, y_data, y_data_ref, label=None):
        self.x_data_comp = x_data * self.x_multiplier
        self.y_data_comp = y_data
        self.y_data_ref_comp = y_data_ref
        self.label_comp = label
        self.b_comp = True

    def remove_comparison_data(self):
        self.b_comp = False

    def normalise_data(self, y_data, y_data_ref):
        # check if the plot is normalised
        if self.norm == "None":
            y_plot = y_data
            label = "Signal"

        elif self.norm.lower() == "sig - ref":
            y_plot = y_data - y_data_ref
            label = "sig - ref"

        elif self.norm == "ref - sig":
            y_plot = y_data_ref - y_data
            label = "ref - sig"

        elif self.norm == "sig / ref":
            y_plot = y_data / y_data_ref
            label = "sig / ref"

        elif self.norm == "ref / sig":
            y_plot = y_data_ref / y_data
            label = "ref / sig"

        elif self.norm == "norm sig / ref":
            y_plot = 100 * ((y_data / y_data_ref) - 1)
            label = "norm sig / ref"

        elif self.norm == "norm ref / sig":
            y_plot = 100 * ((y_data_ref / y_data) - 1)
            label = "norm ref / sig"
        elif self.norm == "intensity":
            y_plot = (y_data - y_data_ref) / (y_data + y_data_ref)
            label = "Intensity"

        return y_plot, label

    def update_plot(self, force_fit=False):
        try:
            # self.canvas.axes.cla()
            # Remove previous line plots but keep other artists like xytext
            for line in self.canvas.axes.get_lines():
                if not isinstance(line, mpl.lines.Line2D) or line.get_label() not in [
                    "_nolegend_"
                ]:
                    line.remove()

            # update the plot with the current state of the data

            if not self.y_data.size:  # don't need to plot eh
                return

            sig_plt_options = {
                "c": "#59b6e6",
                "mfc": (0, 0, 0, 0),
                "mec": "#59b6e6",
                "mew": 2,
                "ms": 4,
            }
            ref_plt_options = {
                "c": "#E68959",
                "mfc": (0, 0, 0, 0),
                "mec": "#E68959",
                "mew": 2,
                "ms": 4,
            }
            fit_plt_options = {
                "c": "red",
                "mfc": (0, 0, 0, 0),
                "mec": "#E68959",
                "mew": 2,
                "ms": 4,
            }

            comp_plt_options = {
                "c": "xkcd:light gray",
                "mfc": (0, 0, 0, 0),
                "mec": "xkcd:light gray",
                "mew": 2,
                "ms": 4,
            }

            if self.b_realtime_plot:
                x_data = self.realtime_x_data
                y_data = self.realtime_y_data
                y_data_ref = self.realtime_y_data_ref
            elif self.b_rolling_plot:
                x_data = self.x_data
                y_data = self.parent.rolling_img_figure.last_sweep
                # y_data_ref = self.y_data_ref
            else:
                x_data = self.x_data
                y_data = self.y_data
                y_data_ref = self.y_data_ref

            # check if the plot is normalised
            self.y_plot, label = self.normalise_data(y_data, y_data_ref)
            
            if self.norm == "None":
                y_max = np.max([np.max(y_data), np.max(y_data_ref)])
                y_min = np.min([np.min(y_data), np.min(y_data_ref)])
            else:
                y_max = np.max(self.y_plot)
                y_min = np.min(self.y_plot)

            self.canvas.axes.plot(
                x_data, self.y_plot, "-o", label=label, **sig_plt_options
            )

            # if there is no normalisation, plot the reference data
            if self.norm == "None":
                self.canvas.axes.plot(
                    x_data,
                    y_data_ref,
                    "-o",
                    label="Reference",
                    **ref_plt_options,
                )

            # check if the comparison data should be plotted
            if self.b_comp:
                # plot the comparison data
                self.y_plot_comp, _ = self.normalise_data(
                    self.y_data_comp, self.y_data_ref_comp
                )
                self.canvas.axes.plot(
                    self.x_data_comp,
                    self.y_plot_comp,
                    "-o",
                    label= self.label_comp,
                    zorder=-10,
                    **comp_plt_options,
                )

            # update the y-axis
            y_range = y_max - y_min
            if not np.isnan(y_range):
                self.canvas.axes.set_ylim(
                    [y_min - 0.05 * y_range, y_max + 0.05 * y_range]
                )
                self.canvas.axes.set_xlim(self.xlims)

            # set the y-axis label
            try:
                self.canvas.axes.set_ylabel(self.y_label_key[self.norm])
            except:
                self.canvas.axes.set_ylabel("Unsure (a.u.)")

            # FIXME need to handle x-axis label, maybe send in SweepUpdate?

            self.canvas.axes.set_yticks(np.linspace(y_min, y_max, 5))

            # check if the fit button is pressed
            if self.bfit and force_fit:
                # Set the data for the fit
                self.mw.line_fit_opts.FitModel.set_data(
                    self.x_data / self.x_multiplier, self.y_data, y_data_ref
                )
                self.mw.line_fit_opts.FitModel.set_normalization(self.norm)
                # get the initial guess for the fit
                self.mw.line_fit_opts.FitModel.guess_parameters(
                    fmod=self.mw.meas_opts.freq_mod_input.value()
                )
                # fit the data
                self.mw.line_fit_opts.FitModel.fit()
                # get the best fit results
                x_fit, y_fit = self.mw.line_fit_opts.FitModel.best_fit()
                self.canvas.axes.plot(x_fit, y_fit, label="Fit", **fit_plt_options)
                # Get the results of the fit to display in the gui
                total_counts = (
                    np.nanmean(self.y_data_ref)
                    * self.mw.meas_opts.ith_loop.value()
                    # / self.mw.cam_opts.exposure_time_input.value()
                )
                fit_text = self.mw.line_fit_opts.FitModel.get_fit_results_txt(
                    total_counts,
                    ith_loop=self.mw.meas_opts.ith_loop.value(),
                )
                self.mw.line_fit_opts.update_fit_results(fit_text)

            elif self.bfit:
                self.mw.line_fit_opts.FitModel.set_data(
                    self.x_data / self.x_multiplier, self.y_data, y_data_ref
                )
                self.mw.line_fit_opts.FitModel.set_normalization(self.norm)
                # plot the previous fit
                x_fit, y_fit = self.mw.line_fit_opts.FitModel.best_fit()
                self.canvas.axes.plot(x_fit, y_fit, label="Fit", **fit_plt_options)

            self.canvas.axes.legend(
                # facecolor="None",
                framealpha=0.6,
                edgecolor="None",
                prop={"size": 10},
                labelcolor="linecolor",
                loc="lower right",
            )

            # update the plot
            self.canvas.draw()

        except Exception as e:
            logger.exception("Error updating qmeas_line plot.")
            pass
