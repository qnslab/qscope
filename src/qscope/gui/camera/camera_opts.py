from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from loguru import logger
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

import qscope.server
from qscope.gui.camera.settings_defs import CameraSettings
from qscope.gui.util import add_row_to_layout
from qscope.gui.util.error_handling import *
from qscope.gui.util.settings import GUISettings
from qscope.gui.widgets import (
    QuComboBox,
    QuDoubleSpinBox,
    QuSpinBox,
)
from qscope.gui.widgets.util import WidgetConfig, WidgetType, create_widget_from_config
from qscope.types import MAIN_CAMERA, DeviceRole


@dataclass
class CameraOpts(QGroupBox):
    START_VIDEO_LABEL = "Start video"
    STOP_VIDEO_LABEL = "Stop video"
    TAKE_IMAGE_LABEL = "Take Image"
    SAVE_IMAGE_LABEL = "Save Image"
    LASER_OFF_LABEL = "Laser Off"
    LASER_ON_LABEL = "Laser On"
    MW_OFF_LABEL = "MW Off"
    MW_ON_LABEL = "MW On"
    EXPOSURE_TIME_LABEL = "Exposure Time (s)"
    BINNING_LABEL = "Binning"
    TRIGGER_MODE_LABEL = "Trigger Mode"
    IMAGE_SIZE_LABEL = "Image Size (X, Y)"
    BUTTON_COLOR_BLUE = "background-color: blue"
    BUTTON_COLOR_GREEN = "background-color: green"
    BUTTON_COLOR_NONE = "background-color: None"

    def __init__(self, parent, figure, title="Camera settings", *args, **kwargs):
        super().__init__(title)
        self.parent = parent
        self.figure = figure
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QGroupBox { font-weight: bold; }")

        self.settings = CameraSettings()
        self.init_ui()
        self.connect_signals()

        self.cam_settings = GUISettings(self, "CAMERA OPTS")
        self.cam_settings.load_prev_state()

    def init_ui(self):
        """Initialize the UI components."""
        self.start_video_button = QPushButton(self.START_VIDEO_LABEL)
        self.start_video_button.setCheckable(True)
        self.stop_video_button = QPushButton(self.STOP_VIDEO_LABEL)
        self.stop_video_button.setCheckable(True)
        # self.take_image_button = QPushButton(self.TAKE_IMAGE_LABEL)
        # self.take_image_button.setCheckable(True)
        self.save_image_button = QPushButton(self.SAVE_IMAGE_LABEL)
        self.save_image_button.setCheckable(True)
        self.laser_button = QPushButton(self.LASER_OFF_LABEL)
        self.laser_button.setCheckable(True)
        self.mw_button = QPushButton(self.MW_OFF_LABEL)
        self.mw_button.setCheckable(True)

        buttons_frame = QFrame()
        buttons_vbox = QVBoxLayout()

        self.measurement_buttons_layout = QHBoxLayout()
        self.measurement_buttons_layout.addWidget(self.start_video_button)
        self.measurement_buttons_layout.addWidget(self.stop_video_button)

        self.save_buttons_layout = QHBoxLayout()
        # self.save_buttons_layout.addWidget(self.take_image_button)
        self.save_buttons_layout.addWidget(self.save_image_button)

        self.static_cont_layout = QHBoxLayout()
        self.static_cont_layout.addWidget(self.laser_button)
        self.static_cont_layout.addWidget(self.mw_button)

        buttons_vbox.addLayout(self.measurement_buttons_layout)
        buttons_vbox.addLayout(self.save_buttons_layout)
        buttons_vbox.addLayout(self.static_cont_layout)
        buttons_frame.setLayout(buttons_vbox)

        self.exposure_time_label = QLabel(self.EXPOSURE_TIME_LABEL)
        self.exposure_time_input = create_widget_from_config(
            CameraSettings.exposure_time
        )

        self.binning_label = QLabel(self.BINNING_LABEL)
        self.binning_input = create_widget_from_config(CameraSettings.binning)

        self.image_size_label = QLabel(self.IMAGE_SIZE_LABEL)
        self.image_size_x_input = create_widget_from_config(CameraSettings.image_size_x)
        self.image_size_y_input = create_widget_from_config(CameraSettings.image_size_y)

        camera_settings_frame = QFrame()
        self.camera_settings_layout = QVBoxLayout()
        add_row_to_layout(
            self.camera_settings_layout,
            self.exposure_time_label,
            self.exposure_time_input,
        )

        add_row_to_layout(
            self.camera_settings_layout, self.binning_label, self.binning_input
        )

        add_row_to_layout(
            self.camera_settings_layout,
            *zip(
                [
                    self.image_size_label,
                    self.image_size_x_input,
                    self.image_size_y_input,
                ],
                [51, 24, 25],
            ),
        )

        camera_settings_frame.setLayout(self.camera_settings_layout)

        self.layout = QVBoxLayout()
        self.layout.addWidget(buttons_frame)
        self.layout.addWidget(camera_settings_frame)

        self.setLayout(self.layout)

    def connect_signals(self):
        """Connect signals to their respective slots."""
        self.start_video_button.clicked.connect(self.start_video)
        self.stop_video_button.clicked.connect(self.stop_video)
        # self.take_image_button.clicked.connect(self.take_image)
        self.save_image_button.clicked.connect(self.save_image)
        self.laser_button.clicked.connect(self.laser_button_pressed)
        self.mw_button.clicked.connect(self.mw_button_pressed)

    def laser_button_pressed(self):
        """Handle laser button press."""
        if self.laser_button.isChecked():
            self.laser_button.setText(self.LASER_ON_LABEL)
            self.laser_button.setStyleSheet(self.BUTTON_COLOR_GREEN)
            if self.mw_button.isChecked():
                self.parent.connection_manager.set_rf_state(
                    state=False,
                    freq=self.parent.meas_opts.rf_freq_input.value(),
                    power=self.parent.meas_opts.rf_power_input.value(),
                )  # remove device lock on rf
                self.parent.connection_manager.set_laser_rf_output(
                    state=True,
                    freq=self.parent.meas_opts.rf_freq_input.value(),
                    power=self.parent.meas_opts.rf_power_input.value(),
                )
            else:
                self.parent.connection_manager.set_laser_output(True)
        else:
            self.laser_button.setText(self.LASER_OFF_LABEL)
            self.laser_button.setStyleSheet(self.BUTTON_COLOR_NONE)
            self.parent.connection_manager.set_laser_output(False)

    def mw_button_pressed(self):
        """Handle MW button press."""
        if self.mw_button.isChecked():
            self.mw_button.setText(self.MW_ON_LABEL)
            self.mw_button.setStyleSheet(self.BUTTON_COLOR_GREEN)
            if self.laser_button.isChecked():
                # remove device lock on laser
                self.parent.connection_manager.set_laser_output(False)
                self.parent.connection_manager.set_laser_rf_output(
                    state=True,
                    freq=self.parent.meas_opts.rf_freq_input.value(),
                    power=self.parent.meas_opts.rf_power_input.value(),
                )
            else:
                self.parent.connection_manager.set_rf_state(
                    state=True,
                    freq=self.parent.meas_opts.rf_freq_input.value(),
                    power=self.parent.meas_opts.rf_power_input.value(),
                )
        else:
            self.mw_button.setText(self.MW_OFF_LABEL)
            self.mw_button.setStyleSheet(self.BUTTON_COLOR_NONE)
            self.parent.connection_manager.set_rf_state(False)

    def set_camera_settings(self):
        """Set the camera settings based on the UI inputs."""
        exposure_time = self.exposure_time_input.value()
        binning = int(self.binning_input.currentText().split("x")[0])

        image_size_x = int(self.image_size_x_input.value())
        image_size_y = int(self.image_size_y_input.value())

        try:
            self.parent.connection_manager.camera_set_params(
                exp_t=exposure_time,
                binning=(binning, binning),
                image_size=(image_size_x, image_size_y),
            )
        except Exception as e:
            logger.exception("Error setting camera parameters: {}", e)

    # def take_image(self):
    #     """Take an image using the camera."""
    #     try:
    #         self.set_camera_settings()
    #         self.single_image = self.parent.connection_manager.camera_take_snapshot()
    #         self.figure.update_data(self.single_image)
    #         self.take_image_button.setChecked(False)
    #     except Exception as e:
    #         show_warning("Error taking an image: " + str(e))
    #         logger.exception("Error taking an image.")
    #         self.take_image_button.setChecked(False)

    def save_image(self):
        """Save the latest stream image (& ttrace)."""
        # get the current selected colormap
        color_map = self.parent.video_fig_opts.cmap_dropdown.currentText()
        try:
            path = self.parent.connection_manager.save_latest_stream(
                self.parent.get_project_name(),
                color_map=color_map,
                notes=self.parent.get_notes(),
            )
            self.parent.show_status_msg(f"Image saved to {path}")
        except RuntimeError as e:
            show_warning(e)
        # Reset the button state
        self.save_image_button.setChecked(False)

    def start_video(self, poll_interval_ms: float = 100):
        """Start the video feed."""
        self.start_video_button.setStyleSheet(self.BUTTON_COLOR_BLUE)

        if not self.parent.connection_manager.is_connected():
            self.start_video_button.setChecked(False)
            self.start_video_button.setStyleSheet(self.BUTTON_COLOR_NONE)
            self.show_warning_message("Not connected to the server")
            return
        if MAIN_CAMERA in self.parent.connection_manager.get_device_locks():
            self.start_video_button.setChecked(False)
            self.start_video_button.setStyleSheet(self.BUTTON_COLOR_NONE)
            self.show_warning_message("Camera is locked, cannot start video feed")

        self.parent.timetrace_data = []
        self.parent.timetrace_time = []
        self.set_camera_settings()
        try:
            self.parent.connection_manager.camera_start_video()
        except Exception as e:
            self.start_video_button.setChecked(False)
            self.start_video_button.setStyleSheet(self.BUTTON_COLOR_NONE)
            self.show_warning_message("Error starting video: " + str(e))
            logger.exception("Error starting video.")
            return
        logger.debug("Video started")

    def stop_video(self):
        """Stop the video feed."""
        if not self.parent.connection_manager.is_connected():
            self.stop_video_button.setChecked(False)
            self.show_warning_message("Not connected to the server")
            return
        self.stop_video_button.setStyleSheet(self.BUTTON_COLOR_BLUE)
        self.parent.connection_manager.camera_stop_video()
        logger.debug("Video stopped")
        self.start_video_button.setChecked(False)
        self.stop_video_button.setChecked(False)
        self.start_video_button.setStyleSheet(self.BUTTON_COLOR_NONE)
        self.stop_video_button.setStyleSheet(self.BUTTON_COLOR_NONE)

    def show_warning_message(self, message):
        """Show a warning message box."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(message)
        msg.setWindowTitle("Warning")
        msg.exec()
