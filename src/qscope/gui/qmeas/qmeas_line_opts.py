import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

mpl.use("Qt5Agg")
import asyncio

import qscope.fitting
import qscope.gui.util as util
from qscope.gui.figures.matplotlib_canvas import MplCanvas


class LineFigureOpts(QGroupBox):
    REAL_TIME_LABEL = "Real-Time"
    ROLLING_AVG_LABEL = "Rolling Avg"
    COMP_LOAD_LABEL = "Load Comparison"
    COMP_REMOVE_LABEL = "Remove Comparison"
    START_INDEX_LABEL = "Start Index"
    END_INDEX_LABEL = "End Index"
    INDEX_LABEL = "remove pts [start, end]"
    NORM_TYPE_LABEL = "Norm. Type"

    def __init__(self, figure, title="Plot options"):
        super().__init__(title)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.figure = figure

        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        frame = QFrame()
        vbox = QVBoxLayout()

        """Initialize the UI components."""
        self.real_time_button = QPushButton(self.REAL_TIME_LABEL)
        self.real_time_button.setCheckable(True)
        self.real_time_button.setStatusTip(
            "Use the current sweep data to update the plot in real-time"
        )

        self.rolling_avg_button = QPushButton(self.ROLLING_AVG_LABEL)
        self.rolling_avg_button.setCheckable(True)
        self.rolling_avg_button.setStatusTip("Use RollingAvg data to update the plot")

        self.load_comp_button = QPushButton(self.COMP_LOAD_LABEL)   
        self.load_comp_button.setCheckable(True)
        self.load_comp_button.setStatusTip(
            "Load a comparison measurement to the plot"
        )

        self.remove_comp_button = QPushButton(self.COMP_REMOVE_LABEL)
        self.remove_comp_button.setCheckable(True)
        self.remove_comp_button.setStatusTip(
            "Remove the comparison measurement from the plot"
        )

        # options to remove the start or end indices from the plotting
        self.index_label, self.start_index_input = util.get_spin_box_wdiget(
            self.INDEX_LABEL, default_value=1, max_value=1000
        )

        self.end_index_label, self.end_index_input = util.get_spin_box_wdiget(
            self.END_INDEX_LABEL, default_value=0, max_value=1000
        )

        # add a dropdown menu for the plot normalisation options
        self.plot_norm_label, self.plot_norm_dropdown = util.get_dropdown_widget(
            self.NORM_TYPE_LABEL,
            [
                "None",
                "sig - ref",
                # "ref - sig",
                "sig / ref",
                # "ref / sig",
                "sig / ref (%)",
                # "norm ref / sig",
                "intensity",
            ],
        )

        plot_type_frame = QFrame()
        plotting_type_layout = QHBoxLayout(plot_type_frame)
        plotting_type_layout.setSpacing(10)
        plotting_type_layout.setContentsMargins(0, 0, 0, 0)

        # Add the widgets to the layout
        plotting_type_layout.addWidget(self.real_time_button)
        plotting_type_layout.addWidget(self.rolling_avg_button)

        plot_comp_frame = QFrame()
        plotting_comp_layout = QHBoxLayout(plot_comp_frame)
        plotting_comp_layout.setSpacing(10)
        plotting_comp_layout.setContentsMargins(0, 0, 0, 0)

        plotting_comp_layout.addWidget(self.load_comp_button)
        plotting_comp_layout.addWidget(self.remove_comp_button)

        norm_frame = QFrame()
        norm_layout = QHBoxLayout(norm_frame)
        norm_layout.setSpacing(10)
        norm_layout.setContentsMargins(0, 0, 0, 0)

        norm_layout.addWidget(self.plot_norm_label)
        norm_layout.addWidget(self.plot_norm_dropdown)

        # define the plot options frame
        idx_frame = QFrame()
        idx_layout = QHBoxLayout(idx_frame)
        idx_layout.setSpacing(10)
        idx_layout.setContentsMargins(0, 0, 0, 0)

        idx_layout.addWidget(self.index_label)
        idx_layout.addWidget(self.start_index_input)
        idx_layout.addWidget(self.end_index_input)

        vbox.addWidget(norm_frame)
        vbox.addWidget(idx_frame)
        vbox.addWidget(plot_type_frame)
        vbox.addWidget(plot_comp_frame)

        self.setLayout(vbox)
        return

    def connect_signals(self):
        """Connect signals to their respective slots."""
        self.real_time_button.clicked.connect(self._real_time_button_pressed)
        self.plot_norm_dropdown.currentIndexChanged.connect(self.norm_button_pressed)
        self.load_comp_button.clicked.connect(self._load_comp_button_pressed)
        self.remove_comp_button.clicked.connect(self._remove_comp_button_pressed)

    def norm_button_pressed(self):
        """Handle normalisation button press."""
        norm_type = self.plot_norm_dropdown.currentText()
        if norm_type == "sig / ref (%)":
            norm_type = "norm sig / ref"

        self.figure.set_normalisation(norm_type)
        # Check if there is a fit and update the plot
        if self.figure.is_fitting():
            self.figure.set_fitting(True)
        else:
            self.figure.update_plot()

    def _rolling_avg_button_pressed(self, state=None):
        """Handle rolling avg button press."""
        if self.rolling_avg_button.isChecked() and state is None:
            self.rolling_avg_button.setChecked(True)
            self.rolling_avg_button.setStyleSheet("background-color: blue")
            self.figure.plot_rolling_avg(True)
            self._real_time_button_pressed(False)
        else:
            self.rolling_avg_button.setStyleSheet("background-color: None")
            self.rolling_avg_button.setChecked(False)
            self.figure.plot_rolling_avg(False)

    def _real_time_button_pressed(self, state=None):
        """Handle real-time button press."""
        if self.real_time_button.isChecked() and state is None:
            self.real_time_button.setChecked(True)
            self.real_time_button.setStyleSheet("background-color: blue")
            self.figure.plot_realtime_data(True)
            self._rolling_avg_button_pressed(False)
        else:
            self.real_time_button.setStyleSheet("background-color: None")
            self.real_time_button.setChecked(False)
            self.figure.plot_realtime_data(False)

    def _load_comp_button_pressed(self):
        """Handle load comparison button press."""
        if self.load_comp_button.isChecked():
            self.load_comp_button.setChecked(True)
            self.load_comp_button.setStyleSheet("background-color: blue")
            # load the comparison data using a file dialog
            filename = QFileDialog.getOpenFileName(
                self, "Load Comparison Data", "", "numpy data (*.npy)"
            )
            if filename[0] == "":
                self.load_comp_button.setChecked(False)
                return
            # load the comparison data
            try:
                comp_data = np.load(filename[0])
            except Exception as e:
                logger.error(f"Error loading comparison data: {e}")
                return
            
            # split the data into the x and y values and y ref values
            x_data = comp_data[0,::]
            y_data = comp_data[1,::]
            y_ref_data = comp_data[2,::]

            # get the file name after the last /
            filename = filename[0].split("/")[-1]
            # remove the file extension from the file name
            filename = filename.split(".")[0]

            # send the comparison data to the figure
            self.figure.set_comparison_data(x_data, y_data, y_ref_data, label=filename)

            self.remove_comp_button.setStyleSheet("background-color: None")
            self.remove_comp_button.setChecked(False)
            # update the plot
            self.figure.update_plot()

        else:
            self.load_comp_button.setStyleSheet("background-color: None")
            self.load_comp_button.setChecked(False)


    def _remove_comp_button_pressed(self):
        """Handle remove comparison button press."""
        if self.remove_comp_button.isChecked():
            self.remove_comp_button.setChecked(True)
            self.remove_comp_button.setStyleSheet("background-color: blue")
            self.figure.remove_comparison_data()
            # reset the load comparison button
            self.load_comp_button.setStyleSheet("background-color: None")
            self.load_comp_button.setChecked(False)
        else:
            self.remove_comp_button.setStyleSheet("background-color: None")
            self.remove_comp_button.setChecked(False)
        
        self.remove_comp_button.setStyleSheet("background-color: None")
        self.remove_comp_button.setChecked(False)
        # update the plot
        self.figure.update_plot()