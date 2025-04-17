from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
from loguru import logger
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *

mpl.use("Qt5Agg")

import qscope.gui.util as util

if TYPE_CHECKING:
    from qscope.gui.figures.rolling_real_time_img import RollingImageFigure
    from qscope.gui.main_window import MainWindow


class RollingAvgOpts(QGroupBox):
    SWEEP_AVGS_LABEL = "Sweep Avgs"
    TOTAL_SWEEPS_LABEL = "Total Sweeps"
    NORM_TYPE_LABEL = "Norm. Type"

    def __init__(
        self, parent: MainWindow, figure: RollingImageFigure, title="Rolling avg opts"
    ):
        super().__init__(title)
        self.parent = parent
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.figure = figure

        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """Initialize the UI components."""

        self.total_sweeps_label, self.total_sweeps_input = util.get_spin_box_wdiget(
            self.TOTAL_SWEEPS_LABEL,
            default_value=100,
        )
        self.total_sweeps_input.setStatusTip(
            "Number of sweeps in rolling avg fig before decimating/interpolating data."
        )
        self.total_sweeps_label.setStatusTip(
            "Number of sweeps in rolling avg fig before decimating/interpolating data."
        )

        self.sweep_avgs_label, self.sweep_avgs_input = util.get_spin_box_wdiget(
            self.SWEEP_AVGS_LABEL, default_value=1
        )
        self.sweep_avgs_input.setStatusTip(
            "Number of sweeps per spectrum in the rolling average"
        )
        self.sweep_avgs_label.setStatusTip(
            "Number of sweeps per spectrum in the rolling average"
        )

        # add a dropdown menu for the plot normalisation options
        self.norm_label, self.norm_dropdown = util.get_dropdown_widget(
            self.NORM_TYPE_LABEL,
            [
                "None",
                "sig - ref",
                "ref - sig",
                "sig / ref",
                "ref / sig",
                "norm sig / ref",
                "norm ref / sig",
            ],
        )

        # define the plot options frame
        self.opts_frame = QFrame()
        self.opts_layout = QVBoxLayout()

        # Add the widgets to the layout

        util.add_row_to_layout(
            self.opts_layout,
            self.total_sweeps_label,
            self.total_sweeps_input,
            self.sweep_avgs_label,
            self.sweep_avgs_input,
        )

        util.add_row_to_layout(self.opts_layout, self.norm_label, self.norm_dropdown)

        self.opts_frame.setLayout(self.opts_layout)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.opts_frame)

        self.setLayout(self.layout)

    def connect_signals(self):
        """Connect signals to their respective slots."""
        self.sweep_avgs_input.valueChanged.connect(self._sweep_avgs_input_changed)
        self.total_sweeps_input.valueChanged.connect(self._total_sweeps_input_changed)

    def _sweep_avgs_input_changed(self):
        """Slot for when the sweep avgs input is changed."""
        if hasattr(self.parent, "meas_id"):
            meas_id = self.parent.meas_id
            self.parent.connection_manager.measurement_set_rolling_avg_window(
                meas_id, self.sweep_avgs_input.value()
            )

    def _total_sweeps_input_changed(self):
        """Slot for when the total sweeps input is changed."""
        if hasattr(self.parent, "meas_id"):
            meas_id = self.parent.meas_id
            self.parent.connection_manager.measurement_set_rolling_avg_max_sweeps(
                meas_id, self.total_sweeps_input.value()
            )
