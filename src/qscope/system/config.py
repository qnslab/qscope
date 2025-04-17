"""System configuration handling for QScope.

This module provides access to system configurations loaded from INI files.
All system configurations are now stored in INI files under sysconfig/systems/.
"""

from pathlib import Path
from typing import Dict, Type

from qscope.system.system import SGCameraSystem, System

from .base_config import SystemConfig
from .sysconfig import load_system_config


def get_system_config(system_name: str) -> SystemConfig:
    """Get configuration for a system by name.

    Parameters
    ----------
    system_name : str
        Name of the system configuration to load

    Returns
    -------
    SystemConfig
        The loaded system configuration

    Raises
    ------
    ValueError
        If system configuration not found
    """
    return load_system_config(system_name)
