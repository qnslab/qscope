import configparser
import os
from pathlib import Path
from typing import Any, Dict

from loguru import logger

# from qscope.gui.control.qmeas_opts import QuantumMeasurementOpts
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox, QWidget

from qscope.gui.widgets import QuComboBox, QuDoubleSpinBox, QuSpinBox
from qscope.gui.widgets.config import WidgetConfig
from qscope.gui.widgets.line_edit import QuLineEdit


class GUISettings:
    """Handles loading/saving of GUI settings using INI files"""

    # Class attributes
    config_dir: Path = Path.home() / ".qscope"

    # gui_handle: QWidget
    # config_section: str = 'MEASUREMENT OPTS'

    def __init__(self, gui_handle, config_section) -> None:
        self.default_ini = self.config_dir / "default.ini"
        self.prev_state_ini = self.config_dir / "prev_state.ini"
        self.config = configparser.ConfigParser()
        self.gui_handle = gui_handle
        self.config_section = config_section

        # Create config directory if it doesn't exist
        self.config_dir.mkdir(exist_ok=True)

        # Load or create default settings
        if not self.default_ini.exists():
            self._create_default_settings()

        # Load previous state or create from default
        if not self.prev_state_ini.exists():
            self._create_prev_state_from_default()

    def _create_default_settings(self):
        """Create default settings INI file"""
        config = self.config
        # Make a new section for each widget in the QuantumMeasurementOpts object
        logger.info("Creating default settings file")

        # if the default_ini file does not exist, create it
        if not self.default_ini.exists():
            config[self.config_section] = {}

            for k, v in self.gui_handle.__dict__.items():
                if isinstance(v, (QuSpinBox, QuDoubleSpinBox, QuComboBox)):
                    config[self.config_section][v.config.name] = str(v.default)
            with open(self.default_ini, "w") as f:
                config.write(f)
        else:
            # open the default_ini file and overwrite the existing config section
            # if it exist or add it if it doesn't
            self.config.read(self.default_ini)
            logger.info(
                f"Default settings file already exists, overwriting {self.config_section} section"
            )

            # if self.config.has_section(self.config_section):
            #     # remove the section
            #     self.config.remove_section(self.config_section)

            config[self.config_section] = {}

            for k, v in self.gui_handle.__dict__.items():
                if isinstance(v, (QuSpinBox, QuDoubleSpinBox, QuComboBox, QuLineEdit)):
                    try:
                        self.config[self.config_section][v.config.name] = str(v.default)
                    except KeyError:
                        self.config[self.config_section] = {
                            v.config.name: str(v.default)
                        }

            with open(self.default_ini, "w") as f:
                self.config.write(f)

    def _create_prev_state_from_default(self):
        """Create previous state INI from default settings"""
        self._create_default_settings()

        if self.default_ini.exists():
            self.config.read(self.default_ini)
            with open(self.prev_state_ini, "w") as f:
                self.config.write(f)

    def make_config(self):
        return configparser.ConfigParser()

    def add_section(self, config, section):
        config[section] = {}

        # Handle widgets with config attribute (measurement widgets)
        for k, v in self.gui_handle.__dict__.items():
            if isinstance(v, (QuSpinBox, QuDoubleSpinBox, QuComboBox, QuLineEdit)):
                if hasattr(v, "config") and v.config is not None:
                    config[self.config_section][v.config.name] = str(v.get_value())
                elif hasattr(v, "objectName") and v.objectName():
                    # Handle widgets without config but with object names
                    config[self.config_section][v.objectName()] = str(v.get_value())
        return config

    def save_config(self, config, filepath=None):
        if filepath == None:
            with open(self.prev_state_ini, "w") as f:
                config.write(f)
        else:
            with open(filepath, "w") as f:
                config.write(f)

    def save_current_state(self) -> None:
        """Save current GUI state to prev_state.ini"""
        config = configparser.ConfigParser()
        config[self.config_section] = {}

        for k, v in self.gui_handle.__dict__.items():
            if isinstance(v, (QuSpinBox, QuDoubleSpinBox, QuComboBox, QuLineEdit)):
                value = v.get_value()
                logger.debug(f"Saving {v.config.name}: {value}")
                config[self.config_section][v.config.name] = str(value)

        logger.debug(f"Writing config to {self.prev_state_ini}")
        with open(self.prev_state_ini, "w") as f:
            config.write(f)

        # Verify written values
        verify_config = configparser.ConfigParser()
        verify_config.read(self.prev_state_ini)
        for k, v in self.gui_handle.__dict__.items():
            if isinstance(v, (QuSpinBox, QuDoubleSpinBox, QuComboBox, QuLineEdit)):
                saved_value = verify_config[self.config_section][v.config.name]
                logger.debug(f"Verified {v.config.name}: {saved_value}")

    def load_prev_state(self) -> None:
        """Load previous state into GUI"""
        if not self.prev_state_ini.exists() or not self._load_config_safe(
            self.prev_state_ini
        ):
            logger.warning("No previous state found or corrupt file, loading defaults")
            self._create_prev_state_from_default()
            if not self._load_config_safe(self.prev_state_ini):
                logger.error("Could not load default settings")
                return
        # Or if the file exists but the section is missing
        if not self.config.has_section(self.config_section):
            self._create_prev_state_from_default()

        try:
            # get a list of all the WidgetConfig objects in the QuantumMeasurementOpts object
            for k, v in self.gui_handle.__dict__.items():
                if isinstance(v, (QuSpinBox, QuDoubleSpinBox, QuComboBox, QuLineEdit)):
                    try:
                        v.set_value(self.config[self.config_section][v.config.name])
                    except Exception as e:
                        logger.error(f"Error loading setting {v.config.name}: {e}")

        except Exception as e:
            logger.error(f"Error loading previous state: {e}")
            logger.info("Loading default settings instead")
            self.reset_to_defaults()

    def _load_config_safe(self, filepath: Path) -> bool:
        """Safely load config file with error handling"""
        try:
            self.config.read(filepath)
        except configparser.Error as e:
            logger.error(f"Error reading config file {filepath}: {e}")
            return False
        return True

    def reset_to_defaults(self) -> None:
        """Reset GUI state to default settings"""
        logger.info("Resetting GUI to default settings")
        self._create_prev_state_from_default()
        # self.load_prev_state()
