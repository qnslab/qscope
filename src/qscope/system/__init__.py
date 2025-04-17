# -*- coding: utf-8 -*-
"""
System configuration and implementation classes for Qscope.

This module provides the System class hierarchy that represents complete
experimental setups, managing multiple hardware devices and their interactions.
It includes:

- Base System class defining the common interface
- Specialized system implementations for different experiment types
- Configuration management for hardware setups
- Device role management and validation

The System class is the central component that coordinates hardware devices
and provides a unified interface for measurements.

Examples
--------
Creating a system instance:
```python
from qscope.system import get_system_config
config = get_system_config("mock")
system = config.create_system()
system.startup()
```

See Also
--------
qscope.device : Hardware device implementations
qscope.meas : Measurement implementations
qscope.types.roles : Device role definitions
"""

from .base_config import SystemConfig
from .config import get_system_config
from .system import SGCameraSystem, SGSystem, System, system_requirements

__all__ = [
    "System",
    "SGSystem",
    "SGCameraSystem",
    "system_requirements",
    "SystemConfig",
]
