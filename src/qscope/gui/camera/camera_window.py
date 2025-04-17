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

import qscope.gui.camera


def init_video_tab(mw: QMainWindow, tab: QWidget):
    """
    Function to create the plotting elements for the video tab
    """

    mw.video_fig = qscope.gui.camera.ImageFigure(mw, cbar_type="vertical")
    mw.cam_opts = qscope.gui.camera.CameraOpts(mw, mw.video_fig)
    mw.video_fig_opts = qscope.gui.camera.ImageFigureOpts(mw.video_fig)
    mw.video_timetrace_fig = qscope.gui.camera.VideoTimeTrace(mw)

    # Define the layout for the video tab, add widgets
    tab.layout = QGridLayout()
    tab.layout.addWidget(mw.cam_opts, 0, 0, 4, 2)
    tab.layout.addWidget(mw.video_fig_opts, 4, 0, 4, 2)

    tt_gbox = QGroupBox("Camera trace stream")
    tt_gbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
    tt_gbox.setStyleSheet("QGroupBox { font-weight: bold; }")
    tt_gbox_layout = QVBoxLayout(tt_gbox)
    tt_gbox_layout.addWidget(mw.video_timetrace_fig.canvas)
    tt_gbox_layout.addWidget(mw.video_timetrace_fig)
    tab.layout.addWidget(tt_gbox, 8, 0, 3, 2)

    fig_gbox = QGroupBox("Camera image stream")
    fig_gbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
    fig_gbox.setStyleSheet("QGroupBox { font-weight: bold; }")
    fig_gbox_layout = QVBoxLayout(fig_gbox)
    fig_gbox_layout.addWidget(mw.video_fig.canvas)
    tab.layout.addWidget(fig_gbox, 0, 2, 11, 6)

    # tab.layout.addWidget(mw.video_fig.canvas, 0, 2, 11, 6)
    tab.layout.addWidget(mw.video_fig, 0, 2, 11, 6)

    tab.setLayout(tab.layout)

    return mw
