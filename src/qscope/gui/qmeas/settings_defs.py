from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ..widgets.config import WidgetConfig
from ..widgets.types import WidgetType


def confirm_high_power() -> bool:
    """Confirm if user wants to use high RF power"""
    from PyQt6.QtWidgets import QMessageBox

    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Warning)
    msg.setText("High RF power selected")
    msg.setInformativeText("Are you sure you want to use power > 0 dBm?")
    msg.setStandardButtons(
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    return msg.exec() == QMessageBox.StandardButton.Yes


def confirm_long_duration() -> bool:
    """Confirm if user wants to use long laser duration"""
    from PyQt6.QtWidgets import QMessageBox

    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Warning)
    msg.setText("Long laser duration selected")
    msg.setInformativeText("Are you sure you want to use duration > 1ms?")
    msg.setStandardButtons(
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    return msg.exec() == QMessageBox.StandardButton.Yes


class MeasurementSettings:
    """Comprehensive settings definitions for quantum measurements"""

    # Measurement Control Settings
    num_averages: WidgetConfig[int] = WidgetConfig(
        name="num_averages",
        widget_type=WidgetType.INT,
        default=1,
        min_value=1,
        max_value=10000,
    )

    stop_after_sweeps: WidgetConfig[int] = WidgetConfig(
        name="stop_after_sweeps",
        widget_type=WidgetType.INT,
        default=121,
        min_value=1,
        max_value=1000,
    )

    save_after_sweeps: WidgetConfig[int] = WidgetConfig(
        name="save_after_sweeps",
        widget_type=WidgetType.INT,
        default=41,
        min_value=1,
        max_value=1000,
    )

    # Measurement Type Settings
    measurement_type: WidgetConfig[str] = WidgetConfig(
        name="measurement_type",
        widget_type=WidgetType.CHOICE,
        default="CW ODMR",
        choices=[
            "Mock Measurement",
            "CW ODMR",
            "Pulsed ODMR",
            "Rabi",
            "Ramsey",
            "Spin Echo",
            "CPMG",
            "XYN",
            "T1",
        ],
    )

    ref_mode: WidgetConfig[str] = WidgetConfig(
        name="ref_mode",
        widget_type=WidgetType.CHOICE,
        default="No RF",
        choices=[
            "No Reference",
            "No RF",
            "Frequency Modulated",
            "π pulse at the end",
            "π pulse at the start",
        ],
    )

    list_mode: WidgetConfig[str] = WidgetConfig(
        name="list_mode",
        widget_type=WidgetType.CHOICE,
        default="Linear (center, range)",
        choices=[
            "Linear (min, max)",
            "Linear (center, range)",
            "Guassian (center, FWHM)",
        ],
    )

    # Frequency Settings
    start_freq: WidgetConfig[float] = WidgetConfig(
        name="start_freq",
        widget_type=WidgetType.FLOAT,
        default=2770.0,
        min_value=1.0,
        max_value=10000.0,
        transform=lambda x: round(x, 3),
    )

    stop_freq: WidgetConfig[float] = WidgetConfig(
        name="stop_freq",
        widget_type=WidgetType.FLOAT,
        default=2970.0,
        min_value=1.0,
        max_value=10000.0,
        transform=lambda x: round(x, 3),
    )

    num_points: WidgetConfig[int] = WidgetConfig(
        name="num_points",
        widget_type=WidgetType.INT,
        default=100,
        min_value=2,
        max_value=10000,
    )

    freq_center: WidgetConfig[float] = WidgetConfig(
        name="freq_center",
        widget_type=WidgetType.FLOAT,
        default=2870.0,
        min_value=1.0,
        max_value=10000.0,
        transform=lambda x: round(x, 3),
    )

    freq_range: WidgetConfig[float] = WidgetConfig(
        name="freq_range",
        widget_type=WidgetType.FLOAT,
        default=200.0,
        min_value=0.1,
        max_value=10000.0,
        transform=lambda x: round(x, 2),
    )

    freq_mod: WidgetConfig[float] = WidgetConfig(
        name="freq_mod",
        widget_type=WidgetType.FLOAT,
        default=20.0,
        min_value=0.0,
        max_value=10000.0,
        transform=lambda x: round(x, 3),
    )

    # Time Settings
    min_time: WidgetConfig[float] = WidgetConfig(
        name="min_time",
        widget_type=WidgetType.FLOAT,
        default=0.0,
        min_value=0.0,
        max_value=1000000.0,
    )

    max_time: WidgetConfig[float] = WidgetConfig(
        name="max_time",
        widget_type=WidgetType.FLOAT,
        default=1000.0,
        min_value=0.0,
        max_value=10000000.0,
    )

    time_center: WidgetConfig[float] = WidgetConfig(
        name="time_center",
        widget_type=WidgetType.FLOAT,
        default=500.0,
        min_value=0.0,
        max_value=1000000.0,
    )

    time_range: WidgetConfig[float] = WidgetConfig(
        name="time_range",
        widget_type=WidgetType.FLOAT,
        default=1000.0,
        min_value=0.0,
        max_value=1000000.0,
    )

    # RF Settings
    rf_power: WidgetConfig[float] = WidgetConfig(
        name="rf_power",
        widget_type=WidgetType.FLOAT,
        default=-40.0,
        min_value=-60.0,
        max_value=20.0,
        transform=lambda x: round(x, 1),
        validator=lambda x: x <= 0 or confirm_high_power(),
    )

    rf_freq: WidgetConfig[float] = WidgetConfig(
        name="rf_freq",
        widget_type=WidgetType.FLOAT,
        default=2870.0,
        min_value=1000.0,
        max_value=10000.0,
        transform=lambda x: round(x, 3),
    )

    rf_duration: WidgetConfig[int] = WidgetConfig(
        name="rf_duration",
        widget_type=WidgetType.INT,
        default=100,
        min_value=0,
        max_value=10000,
    )

    rf_delay: WidgetConfig[int] = WidgetConfig(
        name="rf_delay",
        widget_type=WidgetType.INT,
        default=50,
        min_value=0,
        max_value=1000000,
    )

    # Pulse Sequence Settings
    pi_pulse: WidgetConfig[int] = WidgetConfig(
        name="pi_pulse",
        widget_type=WidgetType.INT,
        default=200,
        min_value=0,
        max_value=100000,
    )

    pi_2_pulse: WidgetConfig[int] = WidgetConfig(
        name="pi_2_pulse",
        widget_type=WidgetType.INT,
        default=100,
        min_value=0,
        max_value=100000,
    )

    n_pulses: WidgetConfig[int] = WidgetConfig(
        name="n_pulses",
        widget_type=WidgetType.INT,
        default=1,
        min_value=0,
        max_value=100000,
    )

    # Laser Settings
    laser_duration: WidgetConfig[int] = WidgetConfig(
        name="laser_duration",
        widget_type=WidgetType.INT,
        default=10000,
        min_value=0,
        max_value=10000000,
        transform=lambda x: x - (x % 10),  # Round to nearest 10
        validator=lambda x: x <= 5000000 or confirm_long_duration(),
    )

    laser_delay: WidgetConfig[int] = WidgetConfig(
        name="laser_delay",
        widget_type=WidgetType.INT,
        default=340,
        min_value=0,
        max_value=1000000,
    )

    def get_widget_config(self, name: str) -> Optional[WidgetConfig]:
        """Get widget configuration by name"""
        return getattr(self, name, None)

    def validate_all(self) -> Dict[str, bool]:
        """Validate all settings"""
        results = {}
        for field_name, field_value in self.__dataclass_fields__.items():
            if isinstance(field_value.default, WidgetConfig):
                config = field_value.default
                value = getattr(self, field_name, None)
                results[field_name] = config.validate(value)
        return results
