# Python script for creating the main window of the QDM GUI

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

mpl.use("Qt5Agg")

from qscope.gui.figures.qmeas_rolling_avg_opts import RollingAvgOpts
from qscope.gui.figures.rolling_real_time_img import RollingImageFigure
from qscope.gui.qmeas.control_opts import QuantumMeasurementOpts
from qscope.gui.qmeas.qmeas_img import QImageAOIOpts, QImageFigure, QImageFigureOpts
from qscope.gui.qmeas.qmeas_line import LineFigure
from qscope.gui.qmeas.qmeas_line_fit_opts import LineFigureFitOpts
from qscope.gui.qmeas.qmeas_line_opts import LineFigureOpts


def init_measurement_tab(mw: QMainWindow, tab: QWidget):
    # Define the box elements for the measurements tab
    tab.layout = QGridLayout()

    # FIXME this method of widget creation is a little unintuitive
    # & note TAB-order is creation-order, which is weird here.
    mw.meas_opts = QuantumMeasurementOpts(mw)
    mw.meas_line_figure = LineFigure(mw)
    mw.line_figure_opts = LineFigureOpts(mw.meas_line_figure)
    mw.line_fit_opts = LineFigureFitOpts(mw.meas_line_figure)
    mw.rolling_img_figure = RollingImageFigure(mw, cbar_type="vertical")
    mw.rolling_img_opts = RollingAvgOpts(mw, mw.rolling_img_figure)

    mw.frame_figure = QImageFigure(mw, cbar_type="horizontal")
    mw.aoi_figure = QImageFigure(mw, cbar_type="horizontal")
    mw.frame_plot_opts = QImageFigureOpts(mw, mw.frame_figure, mw.aoi_figure)
    mw.img_aoi_opts = QImageAOIOpts(mw, mw.frame_figure, mw.aoi_figure)

    tab.layout.addWidget(mw.meas_line_figure)
    # tab.layout.addWidget(mw.meas_line_figure.toolbar)
    tab.layout.addWidget(mw.rolling_img_figure)
    tab.layout.addWidget(mw.frame_figure)
    tab.layout.addWidget(mw.aoi_figure)

    # Layout positioning
    # y, x, h, w

    # Far left column
    tab.layout.addWidget(mw.meas_opts, 0, 0, 1, 1)

    rolling_gbox = QGroupBox("Rolling avg. meas.")
    rolling_gbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
    rolling_gbox.setStyleSheet("QGroupBox { font-weight: bold; }")
    rolling_gbox_layout = QVBoxLayout(rolling_gbox)
    rolling_gbox_layout.addWidget(mw.rolling_img_figure.canvas)
    tab.layout.addWidget(rolling_gbox, 1, 0, -1, 1)

    ## Middle column
    meas_gbox = QGroupBox("Measurement stream")
    meas_gbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
    meas_gbox.setStyleSheet("QGroupBox { font-weight: bold; }")
    meas_gbox_layout = QVBoxLayout(meas_gbox)
    meas_gbox_layout.addWidget(mw.meas_line_figure.canvas, 0)

    tab.layout.addWidget(meas_gbox, 0, 2, 9, 4)

    tab.layout.addWidget(mw.line_figure_opts, 9, 2, 2, 2)

    # tab.layout.addWidget(mw.frame_plot_opts, 9, 8, 2, 2)
    tab.layout.addWidget(mw.img_aoi_opts, 9, 8, 2, 2)

    tab.layout.addWidget(mw.line_fit_opts, 9, 4, 4, 2)

    # Far right column
    aoi_gbox = QGroupBox("Area of interest")
    aoi_gbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
    aoi_gbox.setStyleSheet("QGroupBox { font-weight: bold; }")

    aoi_gbox_layout = QVBoxLayout(aoi_gbox)
    aoi_gbox_layout.addWidget(mw.frame_figure.canvas)
    aoi_gbox_layout.addWidget(mw.aoi_figure.canvas)

    tab.layout.addWidget(aoi_gbox, 0, 8, 9, 2)

    tab.layout.addWidget(mw.rolling_img_opts, 11, 2, 2, 2)

    # tab.layout.addWidget(mw.img_aoi_opts, 11, 8, 2, 2)
    tab.layout.addWidget(mw.frame_plot_opts, 11, 8, 2, 2)

    tab.setLayout(tab.layout)

    return mw
