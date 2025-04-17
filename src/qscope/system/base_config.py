"""Base configuration class for QScope systems."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Type


class ConfigVersion(str, Enum):
    """Configuration version enumeration.

    Versions:
    - LEGACY: Pre-INI format (v1)
    - CURRENT: Initial INI format (v2)
    """

    LEGACY = "v1"
    CURRENT = "v2"


from dataclasses import dataclass
from pathlib import Path

from qscope.device import Device
from qscope.types import DeviceRole, get_valid_device_types

if TYPE_CHECKING:
    from .system import System


@dataclass
class SystemConfig:
    """System configuration loaded from INI files.

    This class represents a system configuration loaded from an INI file.
    It contains all the settings needed to initialize a QScope system.

    Attributes
    ----------
    system_name : str
        Name of the system configuration
    system_type : Type[System]
        Type of system to create
    save_dir : str
        Directory for saving data
    objective_pixel_size : dict[str, float]
        Mapping of objective names to pixel sizes
    devices_config : dict[DeviceRole, tuple[Type[Device], dict[str, Any]]]
        Mapping of device roles to (device_class, parameters) tuples
    """

    system_name: str
    system_type: Type["System"]
    save_dir: str = "C:/ExperimentalData/"
    objective_pixel_size: dict[str, float] = None
    devices_config: dict[DeviceRole, tuple[Type[Device], dict[str, Any]]] = None

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.objective_pixel_size is None:
            self.objective_pixel_size = {}
        if self.devices_config is None:
            self.devices_config = {}
