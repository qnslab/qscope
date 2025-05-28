import asyncio
import time
from typing import Type

import numpy as np
from loguru import logger
from PyQt6.QtWidgets import QMainWindow

import qscope.meas
import qscope.server
import qscope.util
from qscope.gui.util.error_handling import show_critial_error, show_info, show_warning


def get_measurement_config(
    gui_handle,
    meas_config_class: Type[qscope.meas.MeasurementConfig],
    meas_class: Type[qscope.meas.Measurement],
    sweep_x: np.ndarray,
) -> qscope.meas.MeasurementConfig:
    # meas classes are wrapped by a decorator, so this is the only way to get attrs.
    # available_ref_modes = meas_class.__bases__[0].__dict__.get(
    #     "available_ref_modes"
    # )
    available_ref_modes = meas_class.available_ref_modes

    ref_mode = gui_handle.meas_opts.ref_mode_dropdown.currentText()
    conversion_dict = {
        "": "",
        "No RF": "no_rf",
        "Frequency Modulated": "fmod",
        "No Reference": "no_ref",
        "π at end": "π at end",
        "π at start": "π at start",
        "-π/2 at end": "-π/2 at end",
        "3π/2 at end": "3π/2 at end",
    }
    ref_mode = conversion_dict[ref_mode]

    if ref_mode not in available_ref_modes:
        show_critial_error(
            f"Invalid reference mode: {ref_mode}, must be one of: {available_ref_modes}"
        )
        gui_handle.meas_opts.start_measurement_button.setChecked(False)
        gui_handle.meas_opts.stop_measurement_button.setStyleSheet(
            "background-color: None"
        )
        return None

    # TODO remove all passing of the time a frequency values. The sweep_x value should be passed instead.

    # Get the dictionary of the GUI settings
    config_dict = {
        # General settings
        "ref_mode": ref_mode,
        "avg_per_point": int(gui_handle.meas_opts.num_averages_input.value()),
        "fmod_freq": gui_handle.meas_opts.freq_mod_input.value(),
        # Camera settings
        "exposure_time": gui_handle.cam_opts.exposure_time_input.value(),
        "frame_shape": (
            gui_handle.cam_opts.image_size_x_input.value(),
            gui_handle.cam_opts.image_size_y_input.value(),
        ),
        "hardware_binning": (1, 1),  # FIXME
        # "camera_trig_time": 10e-3, # Not needed should be set by the system
        # Sweep
        "sweep_x": sweep_x,
        # Laser settings
        "laser_delay": gui_handle.meas_opts.laser_delay_input.value() * 1e-9,
        "laser_dur": gui_handle.meas_opts.laser_dur_input.value() * 1e-9,
        "laser_to_rf_delay": 0,  # FIXME
        "rf_to_laser_delay": 0,  # FIXME
        # RF settings
        "rf_pow": gui_handle.meas_opts.rf_power_input.value(),
        "rf_freq": gui_handle.meas_opts.rf_freq_input.value(),
        "rf_dur": gui_handle.meas_opts.rf_duration_input.value() * 1e-9,
        "rf_delay": gui_handle.meas_opts.rf_delay_input.value() * 1e-9,
        "pi_dur": gui_handle.meas_opts.pi_pulse_input.value() * 1e-9,
        "pi_2_dur": gui_handle.meas_opts.pi_2_pulse_input.value() * 1e-9,
        # test data params
        "peak_contrasts": (-0.4, -0.4),
        "peak_widths": (18, 22),
        "bg_zeeman": 0.0,
        "ft_zeeman": 50.0,
        "ft_width_dif": 5.0,
        "ft_height_dif": -0.004,
        "ft_centre": (None, None),
        "ft_rad": None,
        "ft_linewidth": 4.0,
        "noise_sigma": 500.0,
    }

    # FIXME delete
    logger.info(
        "RF POWER: "
        + str(gui_handle.meas_opts.rf_power_input.value())
        + " of type: "
        + str(type(gui_handle.meas_opts.rf_power_input.value()))
    )

    logger.info("CONFIG DICT: " + str(config_dict))

    # Make a new dict that only contains the attributes of the config class
    config_dict = {
        key: config_dict[key]
        for key in config_dict
        if key in meas_config_class.__dataclass_fields__.keys()
    }

    return meas_config_class(**config_dict)
