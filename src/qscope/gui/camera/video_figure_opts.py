import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmap import Colormap
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from qscope.gui.widgets import QuComboBox

mpl.use("Qt5Agg")

from loguru import logger

from qscope.gui.figures.matplotlib_canvas import MplCanvas
from qscope.gui.util.cbar import add_colorbar
from qscope.gui.util.error_handling import show_critial_error, show_info, show_warning


class ImageFigureOpts(QGroupBox):
    def __init__(self, target_figure, parent=None, title="Image plot options"):
        super().__init__(title, parent)
        self.figure = target_figure
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QGroupBox { font-weight: bold; }")

        # add a dropdown menu for the colourmap that change it's items based on the cmap style
        self.cmap_label = QLabel("colour map")
        self.cmap_dropdown = QuComboBox()
        # self.cmap_dropdown.setEditable(True)
        # self.cmap_dropdown.lineEdit().setReadOnly(True)
        # self.cmap_dropdown.lineEdit().setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cmap_dropdown.addItem("seaborn:mako")
        self.cmap_dropdown.addItem("viridis")
        self.cmap_dropdown.addItem("plasma")
        self.cmap_dropdown.addItem("inferno")
        self.cmap_dropdown.addItem("magma")
        self.cmap_dropdown.addItem("cividis")
        self.cmap_dropdown.insertSeparator(6)
        self.cmap_dropdown.addItems(
            ["Greys_r", "Bone", "Blues", "Greens", "Oranges", "Reds"]
        )
        self.cmap_dropdown.insertSeparator(13)
        self.cmap_dropdown.addItems(
            [
                "PuOr",
                "RdGy",
                "RdBu",
                "Spectral",
                "coolwarm",
                "bwr",
                "seismic",
            ]
        )

        # add a row for the colour map options
        self.cmap_row = QHBoxLayout()
        self.cmap_row.addWidget(self.cmap_label)
        self.cmap_row.addWidget(self.cmap_dropdown)

        # add input boxes for binning in x and y
        self.binning_x_label = QLabel("Binning x, y:")
        self.binning_x_input = QDoubleSpinBox()
        self.binning_x_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.binning_x_input.setRange(1, 8)
        self.binning_x_input.setValue(1)
        self.binning_x_input.setSingleStep(1)
        self.binning_x_input.setDecimals(0)

        self.binning_y_input = QDoubleSpinBox()
        self.binning_y_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.binning_y_input.setRange(1, 8)
        self.binning_y_input.setValue(1)
        self.binning_y_input.setSingleStep(1)
        self.binning_y_input.setDecimals(0)

        self.binning_row = QHBoxLayout()
        self.binning_row.addWidget(self.binning_x_label, 50)
        self.binning_row.addWidget(self.binning_x_input, 24)
        self.binning_row.addWidget(self.binning_y_input, 25)

        # add the frame plot options to the frame plot layout
        self.layout = QVBoxLayout()
        # self.layout.addWidget(self.img_plot_label)
        # self.layout.addLayout(self.cmap_type_row)
        self.layout.addLayout(self.cmap_row)
        self.layout.addLayout(self.binning_row)
        self.layout.addStretch()
        self.setLayout(self.layout)

        # connect the dropdowns to the callback functions
        self.cmap_dropdown.currentIndexChanged.connect(self.select_cmap_type)

        self.binning_x_input.valueChanged.connect(self.binning_changed)
        self.binning_y_input.valueChanged.connect(self.binning_changed)

    def binning_changed(self):
        bin_x = int(self.binning_x_input.text())
        bin_y = int(self.binning_y_input.text())
        # check that the binning is valid
        if bin_x < 1 or bin_y < 1:
            show_warning("Binning must be greater than 0")
            return

        # check that if data can be binned by the requested amount,
        # i.e. the data shape is divisible by the binning
        if (
            self.figure.frame.shape[0] % bin_y != 0
            or self.figure.frame.shape[1] % bin_x != 0
        ):
            show_warning("Data shape not divisible by binning")
            return

        # update the plots binning
        self.figure.set_binning(bin_x, bin_y)

    def select_cmap_type(self):
        """Callback for the cmap_dropdown currentIndexChanged event"""

        # update the plots
        if self.cmap_dropdown.currentText() != "" or None:
            # if hasattr(self, 'figure'):
            colormap = Colormap(self.cmap_dropdown.currentText())
            self.figure.set_cmap(colormap.to_mpl())

    def select_meas_cmap_type(self):
        """Callback for the meas_cmap_dropdown currentIndexChanged event"""
        # get the selected cmap
        cmap_type = self.cmap_style_dropdown.currentText().lower()
        # remove all of the items in the cmap dropdown
        self.cmap_dropdown.clear()

        # set the cmap
        if cmap_type == "perceptual":
            cmaps = ["seaborn:mako", "viridis", "plasma", "inferno", "magma"]
            self.cmap_dropdown.addItems(cmaps)
        elif cmap_type == "sequential":
            cmaps = [
                "Greys_r",
                "Purples",
                "Blues",
                "Greens",
                "Oranges",
                "Reds",
                "YlOrBr",
                "YlOrRd",
                "OrRd",
                "PuRd",
                "RdPu",
                "BuPu",
                "GnBu",
                "PuBu",
                "YlGnBu",
                "PuBuGn",
                "BuGn",
                "YlGn",
            ]
            self.cmap_dropdown.addItems(cmaps)
        elif cmap_type == "diverging":
            cmaps = [
                "PiYG",
                "PRGn",
                "BrBG",
                "PuOr",
                "RdGy",
                "RdBu",
                "RdYlBu",
                "RdYlGn",
                "Spectral",
                "coolwarm",
                "bwr",
                "seismic",
            ]
            self.cmap_dropdown.addItems(cmaps)
        elif cmap_type == "cyclic":
            cmaps = ["twilight", "twilight_shifted", "hsv"]
            self.cmap_dropdown.addItems(cmaps)
        elif cmap_type == "qualitative":
            cmaps = [
                "Pastel1",
                "Pastel2",
                "Paired",
                "Accent",
                "Dark2",
                "Set1",
                "Set2",
                "Set3",
                "tab10",
                "tab20",
                "tab20b",
                "tab20c",
            ]
            self.cmap_dropdown.addItems(cmaps)
        elif cmap_type == "misc":
            cmaps = [
                "flag",
                "prism",
                "ocean",
                "gist_earth",
                "terrain",
                "gist_stern",
                "gnuplot",
                "gnuplot2",
                "CMRmap",
                "cubehelix",
                "brg",
                "hsv",
                "gist_rainbow",
                "rainbow",
                "jet",
                "nipy_spectral",
                "gist_ncar",
            ]
            self.cmap_dropdown.addItems(cmaps)
        else:
            cmaps = ["seaborn:mako", "viridis", "plasma", "inferno", "magma", "cividis"]
            self.cmap_dropdown.addItems(cmaps)

        # set the current index to 0 which will trigger the select_cmap_type function
        self.cmap_dropdown.setCurrentIndex(0)
