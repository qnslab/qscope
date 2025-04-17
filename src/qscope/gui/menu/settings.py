import PyQt6
from loguru import logger
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QInputDialog, QLabel, QMainWindow, QMenu, QMenuBar

import qscope.server
from qscope.gui.menu.settings_defs import StatusBarSettings
from qscope.gui.util.saving import SettingsDialog
from qscope.gui.util.settings import GUISettings


# Class for the server options
class SettingsMenu(QMenu):
    def __init__(self, parent, menuBar):
        super().__init__()
        self.parent = parent
        self._create_settings_menu(menuBar)

    def _create_settings_menu(self, menuBar):
        # Settings menu
        SettingsMenu = menuBar.addMenu("&Settings")

        # save_settings_action = QAction("&Settings for saving", self.parent)
        # save_settings_action.triggered.connect(self._open_save_settings)

        settings_system_action = QAction("&Save GUI default values", self.parent)
        settings_system_action.triggered.connect(self._save_gui_settings)

        settings_devices_action = QAction("&Device Settings", self.parent)
        # settings_devices_action.triggered.connect(self.open_device_settings)

        # Add actions to the menu
        # SettingsMenu.addAction(save_settings_action)
        SettingsMenu.addAction(settings_system_action)
        SettingsMenu.addAction(settings_devices_action)

        return SettingsMenu

    def _open_save_settings(self):
        """Open the save settings dialog"""
        dialog = SettingsDialog()
        dialog.exec()
        save_dir, use_save_dir, tag = dialog.getValues()

        return save_dir, use_save_dir, tag

    def _save_gui_settings(self):
        """Save the current system settings"""
        # Get the current system settings

        # Initialize settings manager
        meas_settings = GUISettings(self.parent.meas_opts, "MEASUREMENT OPTS")
        config = meas_settings.make_config()
        config = meas_settings.add_section(config, "MEASUREMENT OPTS")
        # meas_settings.save_current_state()

        cam_settings = GUISettings(self.parent.cam_opts, "CAMERA OPTS")
        config = cam_settings.add_section(config, "CAMERA OPTS")
        cam_settings.save_config(config)

        sbar_settings = GUISettings(self.parent.statusBar, "STATUS BAR OPTS")
        config = sbar_settings.add_section(config, "STATUS BAR OPTS")
        sbar_settings.save_config(config)

        logger.info("Saved GUI default settings")
