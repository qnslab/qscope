"""Configuration types for measurements and devices."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from mashumaro import DataClassDictMixin
from mashumaro.config import BaseConfig
from mashumaro.types import Discriminator


@dataclass(kw_only=True, repr=False)
class MeasurementConfig(DataClassDictMixin):
    """Base configuration for measurements.

    To be subclassed for each measurement type.
    Provide all the parameters you require here as attributes.
    """

    class Config(BaseConfig):
        discriminator = Discriminator(
            field="meas_type",
            include_subtypes=True,
        )

    meas_type: str
    save_name: str  # REQUIRED per-type
    ref_mode: str
    sweep_x: np.ndarray = field(metadata={"serialize": tuple, "deserialize": np.array})

    def __repr__(self):
        msg = self.__class__.__name__ + "("
        for i, (field, val) in enumerate(self.__dict__.items()):
            if i not in (0, len(self.__dict__)):
                msg += ", "
            if isinstance(val, np.ndarray):
                msg += f"{field}=<Array>"
            else:
                msg += f"{field}={getattr(self, field)}"
        return msg + ")"


@dataclass(kw_only=True, repr=False)
class CameraConfig(MeasurementConfig):
    """Configuration for camera-based measurements."""

    exposure_time: float
    frame_shape: tuple[int, int]
    hardware_binning: tuple[int, int]
    avg_per_point: int


@dataclass(kw_only=True, repr=False)
class MockSGAndorESRConfig(CameraConfig):
    """Configuration for mock ESR measurements."""

    meas_type: str = "MockSGAndorESR"
    save_name: str = "ESR"
    rf_delay: float
    rf_pow: float
    rf_dur: float
    laser_delay: float
    laser_dur: float
    laser_to_rf_delay: float
    # test data params
    peak_contrasts: tuple[float, float]
    peak_widths: tuple[float, float]
    bg_zeeman: float
    ft_zeeman: float
    ft_width_dif: float
    ft_height_dif: float
    ft_centre: tuple[int | None, int | None]
    ft_rad: int | None
    ft_linewidth: int
    noise_sigma: float


@dataclass(kw_only=True, repr=False)
class SGAndorCWESRConfig(CameraConfig):
    """Configuration for CW ESR measurements."""

    meas_type: str = "SGAndorCWESR"
    save_name: str = "ESR"
    avg_per_point: int
    fmod_freq: float
    rf_pow: float
    laser_delay: float
    long_exposure: bool = False


@dataclass(kw_only=True, repr=False)
class SGAndorPESRConfig(CameraConfig):
    """Configuration for pulsed ESR measurements."""

    meas_type: str = "SGAndorPESR"
    save_name: str = "ESR"
    avg_per_point: int
    fmod_freq: float
    rf_delay: float
    rf_pow: float
    rf_dur: float
    laser_delay: float
    laser_dur: float
    laser_to_rf_delay: float


@dataclass(kw_only=True, repr=False)
class SGAndorRabiConfig(CameraConfig):
    """Configuration for Rabi measurements."""

    meas_type: str = "SGAndorRabi"
    save_name: str = "Rabi"
    avg_per_point: int
    laser_dur: float
    rf_dur: float
    laser_delay: float
    rf_delay: float
    rf_pow: float
    rf_freq: float


@dataclass(kw_only=True, repr=False)
class SGAndorT1Config(CameraConfig):
    """Configuration for T1 measurements."""

    meas_type: str = "SGAndorT1"
    save_name: str = "T1"
    laser_dur: float
    pi_dur: float
    laser_delay: float
    rf_delay: float
    rf_pow: float
    rf_freq: float


@dataclass(kw_only=True, repr=False)
class SGAndorRamseyConfig(CameraConfig):
    """Configuration for Ramsey measurements."""

    meas_type: str = "SGAndorRamsey"
    save_name: str = "Ramsey"
    laser_dur: float
    pi_dur: float
    pi_2_dur: float
    laser_delay: float
    rf_delay: float
    rf_pow: float
    rf_freq: float


@dataclass(kw_only=True, repr=False)
class SGAndorSpinEchoConfig(CameraConfig):
    """Configuration for spin echo measurements."""

    meas_type: str = "SGAndorSpinEcho"
    save_name: str = "SpinEcho"
    laser_dur: float
    pi_dur: float
    pi_2_dur: float
    laser_delay: float
    rf_delay: float
    rf_pow: float
    rf_freq: float


# Default test configuration
TESTING_MEAS_CONFIG = MockSGAndorESRConfig(
    ref_mode="no_rf",
    avg_per_point=1,
    exposure_time=0.1,
    frame_shape=(100, 100),
    hardware_binning=(1, 1),
    sweep_x=np.linspace(2860, 2880, 20),
    rf_delay=0.0,
    rf_pow=-20,
    rf_dur=10e-9,
    laser_delay=0.0,
    laser_dur=0.0,
    laser_to_rf_delay=0.0,
    peak_contrasts=(-0.012, -0.018),
    peak_widths=(18, 22),
    bg_zeeman=100,
    ft_zeeman=50,
    ft_width_dif=5,
    ft_height_dif=-0.004,
    ft_centre=(None, None),
    ft_rad=None,
    ft_linewidth=4,
    noise_sigma=7.5e-3,
)
