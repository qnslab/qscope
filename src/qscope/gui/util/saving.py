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


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        layout = QVBoxLayout()

        self.save_dir_label = QLabel("Current Save Directory:")
        self.save_dir_input = QLineEdit()
        # self.save_dir_input.setReadOnly(True)
        self.save_dir_button = QPushButton("Select Directory")

        self.use_save_dir_button = QPushButton("Use Save Directory only")
        self.use_save_dir_button.setCheckable(True)
        self.use_save_dir_button.setChecked(False)

        self.tag_label = QLabel("Experiment Tag:")
        self.tag_input = QLineEdit()
        self.tag_input.setPlaceholderText("Enter a tag for the measurements")

        layout.addWidget(self.save_dir_label)
        layout.addWidget(self.save_dir_input)
        layout.addWidget(self.save_dir_button)
        layout.addWidget(self.use_save_dir_button)
        layout.addWidget(self.tag_label)
        layout.addWidget(self.tag_input)

        self.save_dir_button.clicked.connect(self.select_save_dir)
        self.use_save_dir_button.clicked.connect(self.use_save_dir_click)

        self.setLayout(layout)

    def getValues(self):
        return (
            self.save_dir_input.text(),
            self.use_save_dir_button.isChecked(),
            self.tag_input.text(),
        )

    # change the background color of the button to blue if checked
    def use_save_dir_click(self):
        if self.use_save_dir_button.isChecked():
            self.use_save_dir_button.setStyleSheet("background-color: blue")
        else:
            self.use_save_dir_button.setStyleSheet("background-color: None")

    def select_save_dir(self):
        # change the background color of the button to blue
        self.save_dir_button.setStyleSheet("background-color: blue")

        self.save_dir_input.setText(
            QFileDialog.getExistingDirectory(self, "Select Directory")
        )
        save_dir = self.save_dir_input.text()

        # change the background color of the button back to None
        self.save_dir_button.setStyleSheet("background-color: None")
        return save_dir
