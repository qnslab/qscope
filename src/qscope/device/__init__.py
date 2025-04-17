# -*- coding: utf-8 -*-
"""
Hardware device implementations for Qscope.

This module provides classes for interfacing with various hardware devices
used in quantum sensing experiments, including:

- Cameras (Andor, mock implementations)
- RF signal generators (SynthNV, SMB100a)
- Pulse sequence generators (PulseBlaster)
- Digitizers (Picoscope)
- Magnets and other control hardware

Each device class implements a common interface defined by the Device base
class, allowing for consistent interaction regardless of the specific hardware.

Examples
--------
Creating a device instance:
```python
from qscope.device import SynthNV
rf_source = SynthNV(visa_addr="COM3")
rf_source.set_frequency(2.87e9)
```

See Also
--------
qscope.system : System configuration and management
qscope.types.roles : Device role definitions
"""

from .andor import Sona42, Zyla42, Zyla55
from .device import Device
from .magnet import APS100, Magnet
from .mock import MockCamera, MockRFSource, MockSeqGen
from .picoscope import Picoscope5000a
from .seqgen import PulseBlaster
from .smb100a import SMB100a
from .SMU2450 import SMU2450
from .synthNV import SynthNV


# Lazy imports
def get_picoscope():
    """Get Picoscope classes if available."""
    # effectively `from .picoscope import Picoscope5000a`
    from .picoscope import get_picoscope

    return get_picoscope()


__all__ = [
    "Device",
    "MockCamera",
    "MockRFSource",
    "MockSeqGen",
    "SMB100a",
    "SynthNV",
    "PulseBlaster",
    "APS100",
    "Sona42",
    "Zyla42",
    "Zyla55",
    "get_picoscope",
]
