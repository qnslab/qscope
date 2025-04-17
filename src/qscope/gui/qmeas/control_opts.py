import configparser
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
from loguru import logger
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import qscope.gui.util as util
import qscope.meas
import qscope.server
from qscope.gui.qmeas.meas_config import get_measurement_config
from qscope.gui.util import show_warning
from qscope.gui.util.layout_building import frame_hbox_widgets, remove_layout_margin
from qscope.gui.util.settings import GUISettings
from qscope.gui.widgets import (
    QuComboBox,
    QuDoubleSpinBox,
    QuSpinBox,
)
from qscope.gui.widgets.util import WidgetConfig, WidgetType, create_widget_from_config
from qscope.types import (
    MockSGAndorESRConfig,
    SGAndorCWESRConfig,
    SGAndorPESRConfig,
    SGAndorRabiConfig,
    SGAndorRamseyConfig,
    SGAndorSpinEchoConfig,
    SGAndorT1Config,
)
from qscope.util import (
    gen_centred_sweep_list,
    gen_exp_centered_list,
    gen_exp_tau_list,
    gen_gauss_sweep_list,
    gen_linear_sweep_list,
)

from ..widgets.config import WidgetConfig
from ..widgets.types import WidgetType
from .settings_defs import MeasurementSettings


@dataclass
class QuantumMeasurementOpts(QGroupBox):
    START_SWEEP_LABEL = "Start Sweep"
    PAUSE_SWEEP_LABEL = "Pause Sweep"
    STOP_NOW_LABEL = "Stop Now!"
    SAVE_SPECTRUM_LABEL = "Save Spectrum"
    SAVE_DATASET_LABEL = "Save Full Dataset"
    STOP_AFTER_SWEEPS_LABEL = "Stop after Y sweeps"
    SAVE_AFTER_SWEEPS_LABEL = "Save after X sweeps"
    MEASUREMENT_ID_LABEL = "Measurement ID"
    # NUMBER_OF_SWEEPS_LABEL = "Number of Sweeps"
    NUMBER_OF_AVERAGES_LABEL = "# Avg. per point"
    SWEEPS_COMPLETED_LABEL = "Sweeps Completed"
    CURRENT_SWEEP_LABEL = "Current Sweep"
    TYPE_LABEL = "Measurement Type"
    LIST_MODE_LABEL = "List Mode"
    FREQUENCY_MIN_MAX_LABEL = "Freq. list (MHz) [Min | Max]"
    FREQUENCY_CENTER_RANGE_LABEL = "Freq. list (MHz) [Center | Range]"
    FREQUENCY_MODULATION_LABEL = "Freq. Mod. (MHz)"
    NUM_POINTS_LABEL = "Points [ #/sweep | Avg./pt ]:"
    Time_MIN_MAX_LABEL = "Time list (ns) [Min | Max]"
    TIME_CENTER_RANGE_LABEL = "Time list (ns) [Center | Range]"
    RF_SETTINGS_LABEL = "RF [Freq. (MHz) | Power (dBm)]"
    RF_DURATION_LABEL = "RF Duration (ns)"
    PULSE_LABEL = "Pulses (ns) [π/2 | π | #]"
    # PI_PULSE_LABEL = "π time (ns)"
    # PI_2_PULSE_LABEL = " π/2 time (ns)"
    # NUMBER_OF_PULSES_LABEL = "Number of pulses"

    # structure:
    # QVBoxLayout (self.vbox) -> groupboxes -> layouts -> widgets

    def __init__(self, _parent, title="Measurement Options"):
        super().__init__(title)
        self.parent = _parent
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QGroupBox { font-weight: bold; }")

        # Initialize settings definitions
        self.settings = MeasurementSettings()
        self.widgets: Dict[str, Any] = {}

        self.init_ui()
        self.connect_signals()
        self.set_measurement_type()

    def validate_settings(self) -> bool:
        """Validate all current settings"""
        validation_results = self.settings.validate_all()
        if not all(validation_results.values()):
            invalid_settings = [k for k, v in validation_results.items() if not v]
            logger.warning(f"Invalid settings: {invalid_settings}")
            return False
        return True

    def get_current_values(self) -> Dict[str, Any]:
        """Get current values of all widgets"""
        values = {}
        for field_name, field in self.settings.__dataclass_fields__.items():
            if isinstance(field.default, WidgetConfig):
                widget = getattr(self, f"{field_name}_input", None)
                if widget:
                    values[field_name] = field.default.get_value(widget)
        return values

    def get_widget_config(self, name: str) -> Optional[WidgetConfig]:
        """Get widget config by name"""
        return getattr(self.settings, name, None)

    def get_all_widget_configs(self) -> Dict[str, WidgetConfig]:
        """Get all widget sysconfig"""
        return {
            k: v
            for k, v in self.settings.__dataclass_fields__.items()
            if isinstance(v.default, WidgetConfig)
        }

    def init_meas_ctrl_frame(self):
        frame = QFrame()
        vbox = QVBoxLayout()

        # Create measurement control widgets from settings
        self.stop_after_sweeps_idx = create_widget_from_config(
            self.settings.stop_after_sweeps
        )
        self.save_after_sweeps_idx = create_widget_from_config(
            self.settings.save_after_sweeps
        )
        self.measurement_type_dropdown = create_widget_from_config(
            self.settings.measurement_type
        )
        self.ref_mode_dropdown = create_widget_from_config(self.settings.ref_mode)

        # Status indicators
        self.meas_id_label = QLabel(self.MEASUREMENT_ID_LABEL)
        self.meas_id_indicator = QLineEdit()
        self.meas_id_indicator.setReadOnly(True)
        self.meas_id_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.state_indicator = QLineEdit()
        self.state_indicator.setReadOnly(True)
        self.state_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Control buttons
        self.start_measurement_button = QPushButton(self.START_SWEEP_LABEL)
        self.start_measurement_button.setCheckable(True)
        self.pause_measurement_button = QPushButton(self.PAUSE_SWEEP_LABEL)
        self.pause_measurement_button.setCheckable(True)
        self.stop_measurement_button = QPushButton(self.STOP_NOW_LABEL)
        self.stop_measurement_button.setCheckable(True)

        # Save buttons
        self.save_measurement_button = QPushButton(self.SAVE_SPECTRUM_LABEL)
        self.save_measurement_button.setCheckable(True)
        self.save_image_button = QPushButton(self.SAVE_DATASET_LABEL)
        self.save_image_button.setCheckable(True)

        # Layout
        meas_info_layout = QHBoxLayout()
        meas_info_layout.addWidget(self.meas_id_label)
        meas_info_layout.addWidget(self.meas_id_indicator)
        meas_info_layout.addWidget(self.state_indicator)

        measurement_buttons_layout = QHBoxLayout()
        measurement_buttons_layout.addWidget(self.start_measurement_button)
        measurement_buttons_layout.addWidget(self.pause_measurement_button)
        measurement_buttons_layout.addWidget(self.stop_measurement_button)

        save_buttons_layout = QHBoxLayout()
        save_buttons_layout.addWidget(self.save_measurement_button)
        save_buttons_layout.addWidget(self.save_image_button)

        vbox.addLayout(meas_info_layout)
        vbox.addLayout(measurement_buttons_layout)
        vbox.addLayout(save_buttons_layout)
        frame.setLayout(vbox)
        return frame

    def init_sweep_frame(self):
        frame = QFrame()
        vbox = QVBoxLayout()

        # self.num_sweeps_label, self.num_sweeps_input = util.get_spin_box_wdiget(
        #     self.NUMBER_OF_SWEEPS_LABEL, default_value=50, max_value=1e4
        # )

        self.ith_loop_label = QLabel(self.SWEEPS_COMPLETED_LABEL)

        self.ith_loop = QDoubleSpinBox()
        self.ith_loop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ith_loop.setValue(0)
        self.ith_loop.setDecimals(0)
        self.ith_loop.setReadOnly(True)
        self.ith_loop.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.ith_loop.setStyleSheet("background-color: black; color: white;")

        self.current_sweep_label = QLabel(self.CURRENT_SWEEP_LABEL)
        self.sweep_progress = QProgressBar()
        self.sweep_progress.setRange(0, 100)
        self.sweep_progress.setValue(0)

        current_sweep_layout = QHBoxLayout()
        current_sweep_layout.addWidget(self.sweep_progress, 80)
        current_sweep_layout.addWidget(self.ith_loop, 20)

        self.stop_after_sweeps = QPushButton(self.STOP_AFTER_SWEEPS_LABEL)
        self.stop_after_sweeps.setCheckable(True)

        self.stop_after_sweeps_idx = create_widget_from_config(
            self.settings.stop_after_sweeps
        )

        self.save_after_sweeps = QPushButton(self.SAVE_AFTER_SWEEPS_LABEL)
        self.save_after_sweeps.setCheckable(True)

        self.save_after_sweeps_idx = create_widget_from_config(
            self.settings.save_after_sweeps
        )

        sweep_stop_opts_layout = QHBoxLayout()
        sweep_stop_opts_layout.addWidget(self.stop_after_sweeps)
        sweep_stop_opts_layout.addWidget(self.stop_after_sweeps_idx)

        sweep_save_opts_layout = QHBoxLayout()
        sweep_save_opts_layout.addWidget(self.save_after_sweeps)
        sweep_save_opts_layout.addWidget(self.save_after_sweeps_idx)

        vbox.addLayout(sweep_save_opts_layout)
        vbox.addLayout(sweep_stop_opts_layout)
        vbox.addLayout(current_sweep_layout)
        frame.setLayout(vbox)
        return frame

    def init_meas_type_frame(self):
        frame = QFrame()
        vbox = QVBoxLayout()

        self.measurement_type_label = QLabel(self.TYPE_LABEL)

        self.measurement_type_dropdown = create_widget_from_config(
            self.settings.measurement_type
        )

        self.ref_mode_dropdown = create_widget_from_config(self.settings.ref_mode)

        measurement_type_layout = QHBoxLayout()
        measurement_type_layout.addWidget(self.measurement_type_label)
        measurement_type_layout.addWidget(self.measurement_type_dropdown)
        measurement_type_layout.addWidget(self.ref_mode_dropdown)

        vbox.addLayout(measurement_type_layout)
        frame.setLayout(vbox)
        return frame

    def init_sequence_frame(self):
        frame = QFrame()
        vbox = QVBoxLayout()

        ### --------------------------- list mode settings

        self.list_mode_frame = QFrame()
        list_mode_layout = QVBoxLayout(self.list_mode_frame)
        list_mode_layout.setSpacing(10)
        list_mode_layout.setContentsMargins(0, 0, 0, 0)

        list_mode_label = QLabel(self.LIST_MODE_LABEL)
        self.list_mode_dropdown = create_widget_from_config(self.settings.list_mode)

        util.add_row_to_layout(
            list_mode_layout, list_mode_label, self.list_mode_dropdown
        )

        ### --------------------------- FREQ SETTINGS

        #  --- Min and Max
        self.freq_min_max_frame = QFrame()
        freq_min_max_vbox = QHBoxLayout(self.freq_min_max_frame)
        freq_min_max_vbox.setSpacing(10)
        freq_min_max_vbox.setContentsMargins(0, 0, 0, 0)

        self.freq_min_max_label = QLabel(self.FREQUENCY_MIN_MAX_LABEL)
        self.start_freq_input = create_widget_from_config(self.settings.start_freq)
        self.stop_freq_input = create_widget_from_config(self.settings.stop_freq)

        util.add_row_to_layout(
            freq_min_max_vbox,
            self.freq_min_max_label,
            self.start_freq_input,
            self.stop_freq_input,
        )

        #  --- Center and range

        self.freq_cen_rang_frame = QFrame()
        freq_cen_range_vbox = QHBoxLayout(self.freq_cen_rang_frame)
        freq_cen_range_vbox.setSpacing(10)
        freq_cen_range_vbox.setContentsMargins(0, 0, 0, 0)

        self.freq_cen_rang_label = QLabel(self.FREQUENCY_CENTER_RANGE_LABEL)
        self.freq_center_input = create_widget_from_config(self.settings.freq_center)
        self.freq_range_input = create_widget_from_config(self.settings.freq_range)

        util.add_row_to_layout(
            freq_cen_range_vbox,
            self.freq_cen_rang_label,
            self.freq_center_input,
            self.freq_range_input,
        )

        #  --- general
        self.freq_mod_frame = QFrame()
        freq_vbox = QVBoxLayout(self.freq_mod_frame)
        freq_vbox.setSpacing(0)
        freq_vbox.setContentsMargins(0, 0, 0, 0)

        self.freq_mod_label = QLabel(self.FREQUENCY_MODULATION_LABEL)
        self.freq_mod_input = create_widget_from_config(self.settings.freq_mod)

        util.add_row_to_layout(
            freq_vbox,
            self.freq_mod_label,
            self.freq_mod_input,
        )

        ### --------------------------- TIME SETTINGS

        #  --- Min and Max
        self.time_min_max_frame = QFrame()
        time_min_max_vbox = QHBoxLayout(self.time_min_max_frame)
        time_min_max_vbox.setSpacing(10)
        time_min_max_vbox.setContentsMargins(0, 0, 0, 0)

        self.time_min_max_label = QLabel(self.Time_MIN_MAX_LABEL)
        self.min_time_input = create_widget_from_config(self.settings.min_time)

        self.max_time_input = create_widget_from_config(self.settings.max_time)

        util.add_row_to_layout(
            time_min_max_vbox,
            self.time_min_max_label,
            self.min_time_input,
            self.max_time_input,
        )

        # ---- center and range
        self.time_cen_rang_frame = QFrame()
        time_cen_range_vbox = QHBoxLayout(self.time_cen_rang_frame)
        time_cen_range_vbox.setSpacing(10)
        time_cen_range_vbox.setContentsMargins(0, 0, 0, 0)

        self.time_cen_rang_label = QLabel(self.TIME_CENTER_RANGE_LABEL)
        self.time_center_input = create_widget_from_config(self.settings.time_center)

        self.time_range_input = create_widget_from_config(self.settings.time_range)

        util.add_row_to_layout(
            time_cen_range_vbox,
            self.time_cen_rang_label,
            self.time_center_input,
            self.time_range_input,
        )

        # --- general

        self.num_pts_settings_frame = QFrame()
        num_pt_settings_layout = QHBoxLayout(self.num_pts_settings_frame)
        num_pt_settings_layout.setSpacing(10)
        num_pt_settings_layout.setContentsMargins(0, 0, 0, 0)

        self.num_points_label = QLabel(self.NUM_POINTS_LABEL)
        self.num_points_input = create_widget_from_config(self.settings.num_points)

        self.num_averages_label = QLabel(self.NUMBER_OF_AVERAGES_LABEL)
        self.num_averages_input = create_widget_from_config(self.settings.num_averages)

        num_pt_settings_layout.addWidget(self.num_points_label, 30)
        num_pt_settings_layout.addWidget(self.num_points_input, 30)
        # num_pt_settings_layout.addWidget(self.num_averages_label, 20)
        num_pt_settings_layout.addWidget(self.num_averages_input, 30)

        ### --------------------------- RF SETTINGS

        self.rf_settings_frame = QFrame()
        rf_settings_layout = QVBoxLayout(self.rf_settings_frame)
        rf_settings_layout.setSpacing(10)
        rf_settings_layout.setContentsMargins(0, 0, 0, 0)

        self.rf_settings_label = QLabel(self.RF_SETTINGS_LABEL)

        self.rf_power_input = create_widget_from_config(self.settings.rf_power)

        self.rf_freq_input = create_widget_from_config(self.settings.rf_freq)

        self.rf_duration_label = QLabel(self.RF_DURATION_LABEL)
        self.rf_duration_input = create_widget_from_config(self.settings.rf_duration)

        util.add_row_to_layout(
            rf_settings_layout,
            self.rf_settings_label,
            self.rf_freq_input,
            self.rf_power_input,
        )

        self.rf_duration_frame = frame_hbox_widgets(
            [self.rf_duration_label, self.rf_duration_input]
        )
        rf_settings_layout.addWidget(self.rf_duration_frame)

        ### --------------------------- Sequence settings

        self.sequence_frame = QFrame()
        sequence_layout = QHBoxLayout(self.sequence_frame)
        sequence_layout.setSpacing(10)
        sequence_layout.setContentsMargins(0, 0, 0, 0)

        self.pulse_label = QLabel(self.PULSE_LABEL)

        self.pi_pulse_input = create_widget_from_config(self.settings.pi_pulse)
        self.pi_2_pulse_input = create_widget_from_config(self.settings.pi_2_pulse)
        self.n_pulses_input = create_widget_from_config(self.settings.n_pulses)

        sequence_layout.addWidget(self.pulse_label, 30)
        sequence_layout.addWidget(self.pi_2_pulse_input, 20)
        sequence_layout.addWidget(self.pi_pulse_input, 20)
        sequence_layout.addWidget(self.n_pulses_input, 20)

        ### Sequence timing settings
        self.seq_timing_frame = QFrame()
        seq_timing_layout = QHBoxLayout(self.seq_timing_frame)
        seq_timing_layout.setSpacing(10)
        seq_timing_layout.setContentsMargins(0, 0, 0, 0)

        self.laser_opts_label = QLabel("Laser (ns) [Dur. | Delay]")
        self.laser_dur_input = create_widget_from_config(self.settings.laser_duration)
        self.laser_delay_input = create_widget_from_config(self.settings.laser_delay)

        seq_timing_layout.addWidget(self.laser_opts_label, 30)
        seq_timing_layout.addWidget(self.laser_dur_input, 30)
        seq_timing_layout.addWidget(self.laser_delay_input, 30)

        self.seq_rf_timing_frame = QFrame()
        seq_rf_timing_layout = QHBoxLayout(self.seq_rf_timing_frame)
        seq_rf_timing_layout.setSpacing(10)
        seq_rf_timing_layout.setContentsMargins(0, 0, 0, 0)

        self.rf_delay_label = QLabel("RF Delay (ns)")
        self.rf_delay_input = create_widget_from_config(self.settings.rf_delay)

        seq_rf_timing_layout.addWidget(self.rf_delay_label, 30)
        seq_rf_timing_layout.addWidget(self.rf_delay_input, 30)

        ### --------------------------- Build Layout & hide frames

        self.rf_duration_frame.hide()
        self.sequence_frame.hide()

        vbox.addWidget(self.list_mode_frame)

        # Frequency settings
        vbox.addWidget(self.freq_min_max_frame)
        vbox.addWidget(self.freq_cen_rang_frame)

        # Time settings
        vbox.addWidget(self.time_min_max_frame)
        vbox.addWidget(self.time_cen_rang_frame)

        vbox.addWidget(self.num_pts_settings_frame)

        vbox.addWidget(self.freq_mod_frame)

        vbox.addWidget(self.rf_settings_frame)

        vbox.addWidget(self.sequence_frame)

        vbox.addWidget(self.seq_timing_frame)
        vbox.addWidget(self.seq_rf_timing_frame)
        frame.setLayout(vbox)
        return frame

    def init_ui(self):
        """Initialize the UI components."""

        self.vbox = QVBoxLayout()

        # meas info, ctrl buttons, save buttons
        meas_ctrl_frame = self.init_meas_ctrl_frame()

        # meas type/ref mode -> in it's own frame
        meas_type_frame = self.init_meas_type_frame()

        # sweep settings
        sweep_frame = self.init_sweep_frame()

        # all sequence settings
        sequence_frame = self.init_sequence_frame()

        self.vbox.addWidget(meas_ctrl_frame)
        self.vbox.addWidget(meas_type_frame)
        self.vbox.addWidget(sweep_frame)
        self.vbox.addWidget(sequence_frame)
        self.setLayout(self.vbox)

        self.meas_settings = GUISettings(self, "MEASUREMENT OPTS")
        self.meas_settings.load_prev_state()

    def create_row_layout(self, label, input_widget):
        """Helper method to create a row layout with a label and input widget."""
        layout = QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(input_widget)
        return layout

    def connect_signals(self):
        """Connect signals to their respective slots."""
        self.start_measurement_button.clicked.connect(self.start_measurement)
        self.pause_measurement_button.clicked.connect(self.pause_measurement)
        self.stop_measurement_button.clicked.connect(self.stop_measurement)
        self.save_measurement_button.clicked.connect(self.save_measurement)
        self.save_image_button.clicked.connect(self.save_image)
        self.stop_after_sweeps.clicked.connect(self.stop_after_n_sweeps)
        self.save_after_sweeps.clicked.connect(self.save_after_n_sweeps)
        self.measurement_type_dropdown.currentIndexChanged.connect(
            self.set_measurement_type
        )
        self.list_mode_dropdown.currentIndexChanged.connect(self.set_list_mode)

    def start_measurement(self):
        self.start_measurement_button.setStyleSheet("background-color: blue")

        # check if the measurement is currently paused
        if hasattr(self.parent, "meas_id"):
            try:
                resp = self.parent.connection_manager.measurement_get_state(
                    self.parent.meas_id
                )
            except:
                pass
            else:
                if resp.lower() == "paused":
                    # Resume the current measurement
                    try:
                        self.parent.connection_manager.start_measurement_wait(
                            self.parent.meas_id
                        )
                        self.start_measurement_button.setText("Start Sweep")
                        self.pause_measurement_button.setChecked(False)
                        self.pause_measurement_button.setStyleSheet(
                            "background-color: None"
                        )
                        return
                    except Exception as e:
                        self.reset_buttons()
                        logger.error(f"Error resuming measurement: {e}")
                        return
                else:
                    # close the previous measurement
                    if hasattr(self.parent, "meas_id"):
                        try:
                            self.parent.connection_manager.close_measurement_wait(
                                self.parent.meas_id
                            )
                        except Exception as e:
                            del self.parent.meas_id
                            self.reset_buttons()

        # Get the sweep list
        x_list = self.get_sweep_x_list()

        # TODO Update the measurement confi to be more automatic.
        # Find the right meas class and then determine the correct config for that class.

        meas_type = self.measurement_type_dropdown.currentText()

        # load the config from the GUI for the selected measurement type
        if meas_type == "Mock Measurement":
            config = get_measurement_config(
                gui_handle=self.parent,
                meas_config_class=MockSGAndorESRConfig,
                meas_class=qscope.meas.MockSGAndorESR,
                sweep_x=x_list,
            )
        elif meas_type == "CW ODMR":
            config = get_measurement_config(
                gui_handle=self.parent,
                meas_config_class=SGAndorCWESRConfig,
                meas_class=qscope.meas.SGAndorCWESR,
                sweep_x=x_list,
            )
        elif meas_type == "Pulsed ODMR":
            config = get_measurement_config(
                gui_handle=self.parent,
                meas_config_class=SGAndorPESRConfig,
                meas_class=qscope.meas.SGAndorPESR,
                sweep_x=x_list,
            )
        elif meas_type == "Rabi":
            config = get_measurement_config(
                gui_handle=self.parent,
                meas_config_class=SGAndorRabiConfig,
                meas_class=qscope.meas.SGAndorRabi,
                sweep_x=x_list,
            )
        elif meas_type == "Ramsey":
            config = get_measurement_config(
                gui_handle=self.parent,
                meas_config_class=SGAndorRamseyConfig,
                meas_class=qscope.meas.SGAndorRamsey,
                sweep_x=x_list,
            )
        elif meas_type == "T1":
            config = get_measurement_config(
                gui_handle=self.parent,
                meas_config_class=SGAndorT1Config,
                meas_class=qscope.meas.SGAndorT1,
                sweep_x=x_list,
            )
        elif meas_type == "Spin Echo":
            config = get_measurement_config(
                gui_handle=self.parent,
                meas_config_class=SGAndorSpinEchoConfig,
                meas_class=qscope.meas.SGAndorSpinEcho,
                sweep_x=x_list,
            )
        elif meas_type == "CPMG":
            qscope.gui.util.show_warning("Measurement type not implemented")
            return
        elif meas_type == "XYN":
            qscope.gui.util.show_warning("Measurement type not implemented")
            return
        else:
            qscope.gui.util.show_warning("Measurement type not implemented")
            return

        if config is None:
            return

        # send the camera settings to the server
        try:
            self.parent.cam_opts.set_camera_settings()
        except Exception as e:
            self.reset_buttons()
            show_warning(f"Error setting camera settings: {e}")
            return

        # check if the laser constant on is checked
        if self.parent.cam_opts.laser_button.isChecked():
            # turn off the laser
            self.parent.connection_manager.set_laser_output(False)
            # Don't adjust the button state here as we will use it to turn the laser back on

        # Prepare the measurement
        self.meas_id = self.parent.connection_manager.add_measurement(config)
        self.meas_info = self.parent.connection_manager.get_all_meas_info()

        # Start the measurement
        try:
            self.parent.connection_manager.start_measurement_wait(self.meas_id)
            self.parent.connection_manager.start_measurement_nowait(self.meas_id)
        except Exception as e:
            self.reset_buttons()
            show_warning(f"Error starting measurement: {e}")
            return

        # Update the line plot x-axis label
        if meas_type in ["Mock Measurement", "CW ODMR", "Pulsed ODMR"]:
            self.parent.meas_line_figure.set_x_label("Frequency (MHz)")
            self.parent.meas_line_figure.set_x_multiplier(1)
        else:
            # Check the maximum time can scale accordingly
            max_time = np.max(x_list)
            if max_time > 1e-3:
                self.parent.meas_line_figure.set_x_label("Time (ms)")
                self.parent.meas_line_figure.set_x_multiplier(1e3)
            elif max_time > 1e-6:
                self.parent.meas_line_figure.set_x_label("Time (μs)")
                self.parent.meas_line_figure.set_x_multiplier(1e6)
            elif max_time > 1e-9:
                self.parent.meas_line_figure.set_x_label("Time (ns)")
                self.parent.meas_line_figure.set_x_multiplier(1e9)
            else:
                self.parent.meas_line_figure.set_x_label("Time (s)")
                self.parent.meas_line_figure.set_x_multiplier(1)

    def pause_measurement(self):
        # set the button to unchecked
        self.pause_measurement_button.setChecked(True)
        # reset the button colour
        self.pause_measurement_button.setStyleSheet("background-color: red")

        # change the start button to display resume if the measurement is paused
        self.start_measurement_button.setText("Resume Sweep")

        # Pause the measurement
        try:
            self.parent.connection_manager.pause_endsweep_measurement(self.meas_id)
        except Exception as e:
            self.reset_buttons()
            show_warning(f"Error pausing measurement: {e}")
            return

    def stop_measurement(self):
        self.stop_measurement_button.setStyleSheet("background-color: red")

        # Stop the measurement
        try:
            self.parent.connection_manager.stop_measurement(self.parent.meas_id)
        except Exception as e:
            self.reset_buttons()
            show_warning(f"Error stopping measurement: {e}")
            return

        # Save state when measurement stops
        self.reset_buttons()

        # check if the laser constant on is checked
        if self.parent.cam_opts.laser_button.isChecked():
            # turn the laser back on
            self.parent.connection_manager.set_laser_output(True)

    def save_measurement(self):
        self.save_measurement_button.setStyleSheet("background-color: blue")

        # get the current normalised data
        # check if there is a comparison line because it can have a different x-axis
        if self.parent.meas_line_figure.b_comp:
            xdata = self.parent.meas_line_figure.x_data
            ydata = self.parent.meas_line_figure.y_plot
            xdata_comp = self.parent.meas_line_figure.x_data_comp
            ydata_comp = self.parent.meas_line_figure.y_plot_comp
        else:
            # xdata, ydata = self.parent.meas_line_figure.get_plot_data()
            xdata = self.parent.meas_line_figure.x_data
            ydata = self.parent.meas_line_figure.y_plot
            xdata_comp, ydata_comp = None, None

        # Get the x-multiplier to convert the x-axis back into seconds
        x_multiplier = self.parent.meas_line_figure.x_multiplier

        # get the raw y_data signal rather than the normalised data
        y_data_sig =  self.parent.meas_line_figure.y_data
        y_data_ref =  self.parent.meas_line_figure.y_data_ref

        # get the information from the fit model
        if self.parent.line_fit_opts.fit_button.isChecked():
            fit_x, fit_y = self.parent.line_fit_opts.FitModel.best_fit()
            total_counts = (
                np.nanmean(y_data_ref)
                * self.parent.meas_opts.ith_loop.value()
                # / self.parent.cam_opts.exposure_time_input.value()
            )
            fit_results = self.parent.line_fit_opts.FitModel.get_fit_results_txt(
                total_counts, ith_loop=self.parent.meas_opts.ith_loop.value()
            )
        else:
            fit_x, fit_y = None, None
            fit_results = ""

        path = self.parent.connection_manager.measurement_save_sweep_w_fit(
            self.parent.meas_id,
            self.parent.get_project_name(),
            xdata / x_multiplier if xdata is not None else xdata,
            ydata,
            fit_x / x_multiplier if fit_x is not None else fit_x,
            fit_y,
            fit_results,
            comparison_x=xdata_comp,
            comparison_y=ydata_comp,
            comparison_label=self.parent.meas_line_figure.label_comp,
            color_map=self.parent.frame_plot_opts.cmap_dropdown.currentText(),
            notes=self.parent.get_notes(), 
        )
        self.parent.show_status_msg(f"Saved sweep to {path}")
        self.save_measurement_button.setStyleSheet("background-color: None")
        self.save_measurement_button.setChecked(False)

    def save_image(self):
        self.save_image_button.setStyleSheet("background-color: blue")
        path = self.parent.connection_manager.measurement_save_full_data(
            self.parent.meas_id,
            self.parent.get_project_name(),
            notes=self.parent.get_notes(),
        )
        self.parent.show_status_msg(f"Saved full dataset to {path}")
        self.save_image_button.setStyleSheet("background-color: None")

    def stop_after_n_sweeps(self):
        """Stop after a specified number of sweeps."""
        if self.stop_after_sweeps.isChecked():
            self.stop_after_sweeps.setChecked(True)
            self.stop_after_sweeps.setStyleSheet("background-color: blue")
        else:
            self.stop_after_sweeps.setChecked(False)
            self.stop_after_sweeps.setStyleSheet("background-color: None")

    def save_after_n_sweeps(self):
        """Save after a specified number of sweeps."""
        if self.save_after_sweeps.isChecked():
            self.save_after_sweeps.setChecked(True)
            self.save_after_sweeps.setStyleSheet("background-color: blue")
        else:
            self.save_after_sweeps.setChecked(False)
            self.save_after_sweeps.setStyleSheet("background-color: None")

    def _freq_or_time_list_mode_type(self):
        """Return the list mode for the current measurement type"""
        meas_type = self.measurement_type_dropdown.currentText()

        match meas_type:
            case "Mock Measurement":
                return "freq"
            case "CW ODMR" | "Pulsed ODMR":
                return "freq"
            case "Rabi" | "Ramsey" | "Spin Echo" | "CPMG" | "XYN" | "T1":
                return "time"
            case _:
                return "time"

    def set_list_mode(self):
        list_mode = self.list_mode_dropdown.currentText()

        if self._freq_or_time_list_mode_type() == "freq":
            self.freq_mod_frame.show()
            min_max_frame = self.freq_min_max_frame
            cen_rang_frame = self.freq_cen_rang_frame
            hide_frames = [self.time_min_max_frame, self.time_cen_rang_frame]

        else:
            self.freq_mod_frame.hide()
            min_max_frame = self.time_min_max_frame
            cen_rang_frame = self.time_cen_rang_frame
            hide_frames = [self.freq_min_max_frame, self.freq_cen_rang_frame]

        for frame in hide_frames:
            frame.hide()

        match list_mode:
            case "Linear (min, max)":
                min_max_frame.show()
                cen_rang_frame.hide()
            case "Linear (center, range)" | "Guassian (center, FWHM)":
                min_max_frame.hide()
                cen_rang_frame.show()
            case _:
                min_max_frame.hide()
                cen_rang_frame.hide()

    def get_sweep_x_list(self):
        """Return the sweep x values based on the current list mode settings"""
        list_type = self._freq_or_time_list_mode_type()

        if list_type == "freq":
            min_val = self.start_freq_input.value()
            max_val = self.stop_freq_input.value()
            num_points = int(self.num_points_input.value())
            cen_val = self.freq_center_input.value()
            range_val = self.freq_range_input.value()
        else:
            min_val = self.min_time_input.value() * 1e-9
            max_val = self.max_time_input.value() * 1e-9
            num_points = int(self.num_points_input.value())
            cen_val = self.time_center_input.value()
            range_val = self.time_range_input.value()

        list_mode = self.list_mode_dropdown.currentText()
        # NOTE DB you could also utilise `gen_multigauss_sweep_list` & `gen_multicentre_sweep_list`
        match list_mode:
            case "Linear (min, max)":
                return gen_linear_sweep_list(min_val, max_val, num_points)

            case "Linear (center, range)":
                return gen_centred_sweep_list(cen_val, range_val, num_points)

            case "Guassian resonance (center, FWHM)":
                return gen_gauss_sweep_list(cen_val, range_val, num_points)

            case "Exponential (min, max)":
                return gen_exp_tau_list(min_val, max_val, num_points)

            case "Exponential (center, range)":
                # Define an exponential spacing for the list
                return gen_exp_centered_list(cen_val, range_val, num_points)

            case _:
                raise ValueError("Invalid list mode")

    def set_measurement_type(self):
        measurement_type = self.measurement_type_dropdown.currentText()

        self.set_list_mode()

        if measurement_type == "Mock Measurement":
            self.freq_mod_frame.show()

            # Disable unused settings
            self.rf_settings_frame.hide()
            self.rf_duration_frame.hide()

        elif measurement_type == "CW ODMR":
            self.freq_mod_frame.show()
            self.rf_settings_frame.show()

            self.rf_duration_frame.hide()
            self.sequence_frame.hide()
            # self.pi_pulse_frame.hide()
            # self.n_pulses_frame.hide()

            # Display only the relavant reference modes
            self.ref_mode_dropdown.clear()
            self.ref_mode_dropdown.addItem("No RF")
            self.ref_mode_dropdown.addItem("Frequency Modulated")
            self.ref_mode_dropdown.addItem("")

        elif measurement_type == "Pulsed ODMR":
            self.sequence_frame.hide()
            # self.pi_pulse_frame.hide()
            # self.n_pulses_frame.hide()

            # Make sure the used settings are enabled and shown
            self.rf_settings_frame.show()
            self.rf_duration_frame.show()

            # Display only the relavant reference modes
            self.ref_mode_dropdown.clear()
            self.ref_mode_dropdown.addItem("No RF")
            self.ref_mode_dropdown.addItem("Frequency Modulated")
            self.ref_mode_dropdown.addItem("")

        elif measurement_type == "Rabi":
            # Disable unused settings
            self.sequence_frame.hide()
            # self.pi_pulse_frame.hide()
            # self.n_pulses_frame.hide()
            self.freq_mod_frame.hide()
            self.rf_duration_frame.hide()

            # Make sure the used settings are enabled and show
            # self.rf_freq_mod_frame.show()
            self.rf_settings_frame.show()

            # Set the list mode to min max
            self.list_mode_dropdown.setCurrentIndex(0)

            # Display only the relavant reference modes
            self.ref_mode_dropdown.clear()
            self.ref_mode_dropdown.addItem("No RF")
            self.ref_mode_dropdown.addItem("π pulse at the end")
            self.ref_mode_dropdown.addItem("π pulse at the start")
            self.ref_mode_dropdown.addItem("")

        elif measurement_type == "Ramsey":
            # Make sure the used settings are enabled and shown
            self.sequence_frame.show()
            self.rf_settings_frame.show()
            # self.rf_freq_mod_frame.show()

            # Disable unused settings
            self.freq_mod_frame.hide()
            self.rf_duration_frame.hide()

            # Display only the relavant reference modes
            self.ref_mode_dropdown.clear()
            self.ref_mode_dropdown.addItem("-π/2 at end")
            self.ref_mode_dropdown.addItem("3π/2 at end")
            self.ref_mode_dropdown.addItem("")

        elif (
            measurement_type == "Spin Echo"
            or measurement_type == "CPMG"
            or measurement_type == "XYN"
        ):
            # Disable unused settings
            self.freq_mod_frame.hide()

            # Make sure the used settings are enabled and shown
            self.rf_settings_frame.show()
            self.sequence_frame.show()
            # self.pi_pulse_frame.show()
            # self.n_pulses_frame.show()
            # self.rf_freq_mod_frame.show()

            # Display only the relavant reference modes
            self.ref_mode_dropdown.clear()

            self.ref_mode_dropdown.addItem("-π/2 at end")
            self.ref_mode_dropdown.addItem("3π/2 at end")
            # self.ref_mode_dropdown.addItem("")

        elif measurement_type == "T1":
            # Disable unused settings
            self.freq_mod_frame.hide()

            # Make sure the used settings are enabled and shown
            self.rf_settings_frame.show()
            self.sequence_frame.show()

            # Display only the relavant reference modes
            self.ref_mode_dropdown.clear()

            self.ref_mode_dropdown.addItem("π at start")
            self.ref_mode_dropdown.addItem("π at end")
            # self.ref_mode_dropdown.addItem("")
        else:
            # Disable unused settings
            self.rf_settings_frame.show()

    def reset_buttons(self):
        self.start_measurement_button.setChecked(False)
        self.pause_measurement_button.setChecked(False)
        self.stop_measurement_button.setChecked(False)
        self.start_measurement_button.setStyleSheet("background-color: None")
        self.pause_measurement_button.setStyleSheet("background-color: None")
        self.stop_measurement_button.setStyleSheet("background-color: None")
        self.start_measurement_button.setText("Start Sweep")
        self.ith_loop.setValue(0)
        self.meas_id_indicator.clear()
        self.state_indicator.clear()
