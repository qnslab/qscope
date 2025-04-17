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


class LineFigureFitOpts(QGroupBox):
    FIT_BUTTON_LABEL = "Fit"
    NORM_TYPE_LABEL = "Norm. Type"
    FIT_TYPE_LABEL = "Fit Type"

    def __init__(self, figure, title="Fit options"):
        super().__init__(title)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.figure = figure

        self.init_ui()
        self.connect_signals()

        # Need to run this to define the FitModel
        self.fit_type_dropdown.setCurrentIndex(0)
        self.set_fit_type()

    def init_ui(self):
        """Initialize the UI components."""
        self.fit_button = QPushButton(self.FIT_BUTTON_LABEL)
        self.fit_button.setCheckable(True)

        # add a dropdown menu for the fitting options
        self.fit_type_label, self.fit_type_dropdown = util.get_dropdown_widget(
            self.FIT_TYPE_LABEL,
            [
                "Lorentzian",
                "Gaussian",
                "Linear",
                "Differential Lorentzian",
                "Differential Gaussian",
                "Sine",
                "Damped sine",
                "Exp. decay",
                "Guassian decay",
                "Streched exp. decay",
            ],
        )

        # Add a text box to display the fit parameters
        self.fit_parameters = QTextEdit()
        self.fit_parameters.setReadOnly(True)
        self.fit_param_results = "Fit Parameters"
        self.fit_parameters.setText(self.fit_param_results)
        self.fit_parameters.setMaximumHeight(300)

        # Define the fit opts frame
        self.fit_opts_frame = QFrame()
        self.fit_opts_layout = QVBoxLayout()
        util.add_row_to_layout(
            self.fit_opts_layout, self.fit_button, self.fit_type_dropdown
        )

        util.add_row_to_layout(self.fit_opts_layout, self.fit_parameters)

        self.fit_opts_frame.setLayout(self.fit_opts_layout)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.fit_opts_frame)

        self.setLayout(self.layout)

    def connect_signals(self):
        """Connect signals to their respective slots."""
        self.fit_button.clicked.connect(self.fit_button_pressed)
        self.fit_type_dropdown.currentIndexChanged.connect(self.set_fit_type)

    def fit_button_pressed(self):
        if self.fit_button.isChecked():
            # make the fit button have a green background
            self.fit_button.setStyleSheet("background-color: blue")
            self.figure.set_fitting(True)
        else:
            # make the fit button have no background
            self.fit_button.setStyleSheet("background-color: None")
            self.figure.set_fitting(False)
            # clear the fit results
            self.fit_parameters.setText("Fit Parameters")

    def set_fit_type(self, fit_type):
        self.figure.fit_type = fit_type

    def update_fit_results(self, results):
        self.fit_parameters.setText(results)

    def set_fit_type(self):
        """Set the fit type."""
        logger.info(
            "Setting fit type to: {0}".format(self.fit_type_dropdown.currentText())
        )

        if self.fit_type_dropdown.currentText() == "Lorentzian":
            self.FitModel = qscope.fitting.Lorentzian()
        elif self.fit_type_dropdown.currentText() == "Gaussian":
            self.FitModel = qscope.fitting.Gaussian()
        elif self.fit_type_dropdown.currentText() == "Linear":
            self.FitModel = qscope.fitting.Linear()
        elif self.fit_type_dropdown.currentText() == "Differential Lorentzian":
            self.FitModel = qscope.fitting.DifferentialLorentzian()
        elif self.fit_type_dropdown.currentText() == "Differential Gaussian":
            self.FitModel = qscope.fitting.DifferentialGaussian()
        elif self.fit_type_dropdown.currentText() == "Sine":
            self.FitModel = qscope.fitting.Sine()
        elif self.fit_type_dropdown.currentText() == "Damped sine":
            self.FitModel = qscope.fitting.DampedSine()
        elif self.fit_type_dropdown.currentText() == "Exp. decay":
            self.FitModel = qscope.fitting.ExponentialDecay()
        elif self.fit_type_dropdown.currentText() == "Guassian decay":
            self.FitModel = qscope.fitting.GuassianDecay()
        elif self.fit_type_dropdown.currentText() == "Streched exp. decay":
            self.FitModel = qscope.fitting.StretchedExponentialDecay()

        # fit the data
        if self.fit_button.isChecked():
            self.fit_button_pressed()
