from __future__ import annotations

from .picoscope import PicoSDKImportError, SignalConfig, WaveType
from .picoscope5000a import Picoscope5000a


# Lazy import of device implementations
def get_picoscope():
    """Lazy import of Picoscope classes."""
    try:
        from .picoscope import Picoscope5000a

        return Picoscope5000a
    except ImportError as e:
        raise PicoSDKImportError(f"Failed to import Picoscope classes: {str(e)}")
