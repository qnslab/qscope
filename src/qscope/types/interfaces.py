"""Role interfaces that wrap devices and provide role-specific functionality.

This module defines interface classes that wrap devices and provide role-specific
access to their functionality. Interfaces ensure type safety and provide a clean API
for accessing device capabilities through roles.

The Interface Pattern
-------------------
Interfaces serve several purposes:

1. Type Safety - They ensure devices implement required protocols
2. API Consistency - They provide a standard way to access role-specific functionality
3. Abstraction - They hide device implementation details
4. Documentation - They clearly specify the methods available for each role

Interface Architecture in QScope
------------------------------
Interfaces are a key component of QScope's hardware abstraction layer:

1. Base Interface (RoleInterface)
   - Provides common functionality for all interfaces
   - Wraps a device instance
   - Serves as the foundation for role-specific interfaces

2. Role-Specific Interfaces
   - Inherit from RoleInterface
   - Provide methods specific to their role
   - Add documentation, type hints, and parameter validation
   - Can add additional logic beyond simple device method calls

3. System Integration
   - System.get_device_by_role() returns the appropriate interface
   - Measurements work with interfaces, not raw devices
   - Interface methods call the underlying device methods

Interface Implementation Pattern
------------------------------
Each interface follows a consistent pattern:

1. Constructor accepts a device implementing the corresponding protocol
2. Methods match protocol methods but add documentation and type hints
3. Method implementations call the corresponding device methods
4. Additional logic can be added for parameter validation or processing

This pattern ensures that interfaces provide a consistent, well-documented API
while delegating actual implementation to the device.

Example
-------
Using a device through its role interface:

    # Get device by role - returns appropriate interface
    rf_source = system.get_device_by_role(PRIMARY_RF)

    # Use interface methods - type safe and well documented
    rf_source.set_freq(2.5)  # MHz
    rf_source.set_power(-10)  # dBm

    # Interface ensures device implements required methods
    rf_source.set_state(True)

The interface pattern allows devices to be used safely through their roles while
maintaining type checking and providing clear documentation.

Creating a New Interface
----------------------
To create a new interface:

1. Define a protocol in protocols.py
2. Create an interface class that inherits from RoleInterface
3. Accept a device implementing the protocol in the constructor
4. Implement methods that call the corresponding device methods
5. Add comprehensive documentation and type hints

Example:

```
class MyNewInterface(RoleInterface):
    def __init__(self, device: MyNewProtocol):
        super().__init__(device)

    def do_something(self, param: int) -> str:
        '''Do something with the device.

        Parameters
        ----------
        param : int
            Description of parameter

        Returns
        -------
        str
            Description of return value
        '''
        return self._device.do_something(param)
```

See Also
--------
qscope.types.protocols : Protocol definitions
qscope.types.roles : Role definitions and validation
qscope.device : Device implementations
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from loguru import logger

from qscope.device.device import Device
from qscope.types.protocols import (
    CameraProtocol,
    DigitizerProtocol,
    RFSourceProtocol,
    SeqGenProtocol,
)


class RoleInterface:
    """Base class for role-specific interfaces.

    Role interfaces wrap devices and provide a clean, type-safe API for
    accessing role-specific functionality. Each role has a corresponding
    interface class that defines the methods available for that role.

    Interfaces serve as an abstraction layer between measurements and
    device implementations, allowing measurements to work with roles
    rather than specific devices.

    Attributes
    ----------
    _device : Device
        The wrapped device instance

    Examples
    --------
    Creating a custom interface:

    ```python
    class MyCustomInterface(RoleInterface):
        def __init__(self, device: MyProtocol):
            super().__init__(device)

        def do_something(self, param: int) -> str:
            # Call device method with additional logic
            result = self._device.do_something(param)
            return f"Processed: {result}"
    ```

    Using an interface:

    ```python
    # Get interface from system
    my_device = system.get_device_by_role(MY_ROLE)

    # Use interface methods
    result = my_device.do_something(42)
    ```
    """

    def __init__(self, device: Device):
        self._device = device


class DigitizerInterface(RoleInterface):
    """Interface for digitizer functionality.

    This interface provides methods for configuring and controlling data
    acquisition devices. It wraps a device implementing DigitizerProtocol
    and provides a clean, type-safe API for digitizer operations.

    Typical usage:
    ```python
    # Get digitizer interface from system
    digitizer = system.get_device_by_role(MAIN_DIGITIZER)

    # Configure channels
    digitizer.configure_channels([0, 1], [5.0, 2.0], ["DC", "AC"])

    # Start streaming acquisition
    digitizer.start_streaming(1e-6, 1000)

    # Get acquired data
    times, (ch0_data, ch1_data) = digitizer.get_data()

    # Stop acquisition
    digitizer.stop_streaming()
    ```
    """

    def __init__(self, device: DigitizerProtocol):
        """Initialize the digitizer interface.

        Parameters
        ----------
        device : DigitizerProtocol
            Device implementing the DigitizerProtocol
        """
        super().__init__(device)

    def configure_channels(
        self,
        channels: List[int],
        ranges: List[Union[float, str]],
        coupling: Optional[List[str]] = None,
    ) -> None:
        """Configure input channels.

        Parameters
        ----------
        channels : List[int]
            List of channel indices to configure
        ranges : List[Union[float, str]]
            List of voltage ranges for each channel
        coupling : Optional[List[str]], optional
            List of coupling modes ('AC' or 'DC') for each channel
        """
        return self._device.configure_channels(channels, ranges, coupling)

    def start_streaming(
        self,
        sample_interval_s: float,
        buffer_size: int,
        num_buffers: Optional[int] = None,
        trigger_enabled: bool = False,
        trigger_channel: Optional[int] = None,
        trigger_threshold: Optional[float] = None,
        trigger_direction: str = "RISING",
        trigger_delay: float = 0.0,
    ) -> None:
        """Start streaming data acquisition.

        Parameters
        ----------
        sample_interval_s : float
            Time between samples in seconds
        buffer_size : int
            Number of samples per buffer
        num_buffers : Optional[int], optional
            Number of buffers to use, by default None
        trigger_enabled : bool, optional
            Whether to use hardware triggering, by default False
        trigger_channel : Optional[int], optional
            Channel to use for triggering, by default None
        trigger_threshold : Optional[float], optional
            Voltage threshold for trigger, by default None
        trigger_direction : str, optional
            Trigger direction ('RISING', 'FALLING'), by default "RISING"
        trigger_delay : float, optional
            Delay after trigger in seconds, by default 0.0
        """
        return self._device.start_streaming(
            sample_interval_s,
            buffer_size,
            num_buffers,
            trigger_enabled,
            trigger_channel,
            trigger_threshold,
            trigger_direction,
            trigger_delay,
        )

    def stop_streaming(self) -> None:
        """Stop streaming data acquisition."""
        return self._device.stop_streaming()

    def get_data(
        self, timeout: float = 10.0
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
        """Get acquired data from all enabled channels.

        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait for data in seconds, by default 10.0

        Returns
        -------
        Tuple[np.ndarray, Tuple[np.ndarray, ...]]
            Tuple containing:
            - Time array (np.ndarray)
            - Tuple of channel data arrays (one np.ndarray per channel)
        """
        return self._device.get_data(timeout)


class CameraInterface(RoleInterface):
    """Interface for camera functionality.

    This interface provides methods for controlling scientific cameras
    and retrieving image data. It wraps a device implementing CameraProtocol
    and provides a clean, type-safe API for camera operations.

    Typical usage:
    ```python
    # Get camera interface from system
    camera = system.get_device_by_role(MAIN_CAMERA)

    # Configure camera
    camera.set_exposure_time(0.1)  # 100ms exposure
    camera.set_hardware_binning((2, 2))  # 2x2 binning

    # Take an image
    image = camera.take_snapshot()

    # Get image dimensions
    width, height = camera.get_frame_shape()
    ```
    """

    def __init__(self, device: CameraProtocol):
        """Initialize the camera interface.

        Parameters
        ----------
        device : CameraProtocol
            Device implementing the CameraProtocol
        """
        super().__init__(device)

    def take_snapshot(self) -> np.ndarray:
        """Take a single image.

        Returns
        -------
        np.ndarray
            2D numpy array containing the image data
        """
        return self._device.take_snapshot()

    def set_exposure_time(self, value: float) -> None:
        """Set exposure time in seconds.

        Parameters
        ----------
        value : float
            Exposure time in seconds
        """
        return self._device.set_exposure_time(value)

    def get_frame_shape(self) -> tuple[int, int]:
        """Get current frame dimensions.

        Returns
        -------
        tuple[int, int]
            Frame dimensions as (width, height) in pixels
        """
        return self._device.get_frame_shape()

    def set_hardware_binning(self, binning: tuple[int, int]) -> tuple[int, int]:
        """Set hardware binning factors.

        Parameters
        ----------
        binning : tuple[int, int]
            Binning factors as (horizontal, vertical)

        Returns
        -------
        tuple[int, int]
            Actual binning factors applied
        """
        return self._device.set_hardware_binning(binning)


class RFSourceInterface(RoleInterface):
    """Interface for RF source functionality.

    This interface provides methods for controlling RF signal generators
    used in spectroscopy and control applications. It wraps a device
    implementing RFSourceProtocol and provides a clean, type-safe API
    for RF source operations.

    Typical usage:
    ```python
    # Get RF source interface from system
    rf = system.get_device_by_role(PRIMARY_RF)

    # Configure RF output
    rf.set_freq(2870.0)  # 2.87 GHz
    rf.set_power(-10.0)  # -10 dBm
    rf.set_state(True)   # Turn on output

    # Configure a frequency sweep
    freqs = np.linspace(2800, 2900, 101)  # MHz
    rf.set_freq_list(freqs, step_time=0.05)  # 50ms per point
    ```
    """

    def __init__(self, device: RFSourceProtocol):
        """Initialize the RF source interface.

        Parameters
        ----------
        device : RFSourceProtocol
            Device implementing the RFSourceProtocol
        """
        super().__init__(device)

    def set_freq(self, freq: float) -> None:
        """Set RF frequency in MHz.

        Parameters
        ----------
        freq : float
            Frequency in MHz
        """
        return self._device.set_freq(freq)

    def set_power(self, power: float) -> None:
        """Set RF power in dBm.

        Parameters
        ----------
        power : float
            Power in dBm
        """
        return self._device.set_power(power)

    def set_state(self, state: bool) -> None:
        """Set RF output state.

        Parameters
        ----------
        state : bool
            True to enable output, False to disable
        """
        return self._device.set_state(state)

    def set_freq_list(
        self, rf_freqs: Sequence[float], step_time: float = 0.1
    ) -> Sequence[float]:
        """Configure frequency sweep.

        Parameters
        ----------
        rf_freqs : Sequence[float]
            List of frequencies in MHz
        step_time : float, optional
            Dwell time at each frequency in seconds, by default 0.1

        Returns
        -------
        Sequence[float]
            Actual frequencies set (may differ slightly from requested)
        """
        return self._device.set_freq_list(rf_freqs, step_time)


class SeqGenInterface(RoleInterface):
    """Interface for sequence generator functionality.

    This interface provides methods for controlling sequence generators
    used for timing and pulse control in quantum experiments. It wraps
    a device implementing SeqGenProtocol and provides a clean, type-safe
    API for sequence generator operations.

    Sequence generators typically control:
    - Precise timing of experimental events
    - Digital output patterns
    - Trigger signals for other instruments
    - Synchronization of multiple devices

    Typical usage:
    ```python
    # Get sequence generator interface from system
    seq_gen = system.get_device_by_role(SEQUENCE_GEN)

    # Load a sequence with parameters
    seq_gen.load_seq("rabi", pulse_length=100e-9, wait_time=5e-6)

    # Start the sequence
    seq_gen.start()

    # Later, stop the sequence
    seq_gen.stop()
    ```
    """

    def __init__(self, device: SeqGenProtocol):
        """Initialize the sequence generator interface.

        Parameters
        ----------
        device : SeqGenProtocol
            Device implementing the SeqGenProtocol
        """
        super().__init__(device)

    def load_seq(self, seq_name: str, **seq_kwargs) -> None:
        """Load a sequence.

        Parameters
        ----------
        seq_name : str
            Name of the sequence to load
        **seq_kwargs
            Additional keyword arguments specific to the sequence
        """
        return self._device.load_seq(seq_name, **seq_kwargs)

    def start(self) -> None:
        """Start sequence generation.

        Begins executing the loaded sequence.
        """
        return self._device.start()

    def stop(self) -> None:
        """Stop sequence generation.

        Halts the currently running sequence.
        """
        return self._device.stop()

    def reset(self) -> None:
        """Reset sequence generator.

        Returns the sequence generator to its initial state.
        """
        return self._device.reset()
