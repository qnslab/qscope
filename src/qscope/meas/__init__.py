"""
Measurement implementations for Qscope experiments.

This module provides classes for implementing various quantum sensing
measurements, including:

- Electron Spin Resonance (ESR)
- Rabi oscillation measurements
- T1 relaxation measurements
- Ramsey and Spin Echo sequences
- Frame grabbing and camera control

Each measurement class handles the experiment logic, parameter sweeps,
data collection, and processing for a specific type of measurement.

Examples
--------
Creating a measurement configuration:
```python
from qscope.meas import SGAndorCWESRConfig
config = SGAndorCWESRConfig(
    name="ESR Measurement",
    start_freq=2.7e9,
    stop_freq=3.0e9,
    num_points=101
)
```

See Also
--------
qscope.device : Hardware device implementations
qscope.fitting : Data fitting and analysis
"""

from .framegrabber import FrameGrabber, MockFrameGrabber
from .measurement import (
    ACQ_MODE,
    MEAS_STATE,
    NORM_MODE,
    Measurement,
    MeasurementConfig,
    SGCameraMeasurement,
)
from .mock_sg_andor_esr import MockSGAndorESR, MockSGAndorESRConfig
from .sg_andor_esr import (
    SGAndorCWESR,
    SGAndorCWESRConfig,
    SGAndorPESR,
    SGAndorPESRConfig,
)
from .sg_andor_rabi import SGAndorRabi, SGAndorRabiConfig
from .sg_andor_ramsey import SGAndorRamsey, SGAndorRamseyConfig
from .sg_andor_spin_echo import SGAndorSpinEcho, SGAndorSpinEchoConfig
from .sg_andor_t1 import SGAndorT1, SGAndorT1Config

__all__ = [
    "ACQ_MODE",
    "MEAS_STATE",
    "NORM_MODE",
    "Measurement",
    "MeasurementConfig",
    "SGCameraMeasurement",
    "MockSGAndorESR",
    "MockSGAndorESRConfig",
    "SGAndorCWESR",
    "SGAndorCWESRConfig",
    "SGAndorPESR",
    "SGAndorPESRConfig",
    "SGAndorRabi",
    "SGAndorRabiConfig",
    "SGAndorRamsey",
    "SGAndorRamseyConfig",
    "SGAndorSpinEcho",
    "SGAndorSpinEchoConfig",
    "SGAndorT1",
    "SGAndorT1Config",
]
