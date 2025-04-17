# Version: 1.0
# Date: 2021-07-20

# Functions for handling GUI errors

# Import necessary modules
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import QMessageBox


def show_critial_error(error_message):
    """Show an error message box with the given error message"""
    error_box = QMessageBox()
    # set the window size
    error_box.setFixedSize(1000, 500)
    error_box.setIcon(QMessageBox.Icon.Critical)
    error_box.setText("Error")
    error_box.setInformativeText(error_message)
    error_box.setWindowTitle("Error")
    error_box.exec()


def show_warning(error_message):
    """Show an error message box with the given error message"""
    error_box = QMessageBox()
    # set the window size
    error_box.setFixedSize(1000, 500)
    error_box.setIcon(QMessageBox.Icon.Warning)
    error_box.setText("Warning")
    error_box.setInformativeText(error_message)
    error_box.setWindowTitle("Warning")
    error_box.exec()


def show_info(error_message):
    """Show an error message box with the given error message"""
    error_box = QMessageBox()
    # set the window size
    error_box.setFixedSize(1000, 500)
    error_box.setIcon(QMessageBox.Icon.Information)
    error_box.setText("Information")
    error_box.setInformativeText(error_message)
    error_box.setWindowTitle("Information")
    error_box.exec()
