import PyQt6
from loguru import logger
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QInputDialog, QLabel, QMainWindow, QMenu, QMenuBar

import qscope.server


# Class for the server options
class FileMenu(QMenu):
    def __init__(self, parent, menuBar):
        super().__init__()
        self.parent = parent
        self._create_menu(menuBar)

    def _create_menu(self, menuBar):
        # File menu
        fileMenu = menuBar.addMenu("&File")

        # Define Actions
        exit_action = QAction("&Exit", self.parent)
        exit_action.triggered.connect(self._closeEvent)

        # Add actions to the menu
        fileMenu.addAction(exit_action)
        fileMenu.addSeparator()

        return fileMenu

    def _closeEvent(self):
        # close the application
        self.parent.close()
