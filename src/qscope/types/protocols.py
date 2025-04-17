"""Device role protocols defining required methods for each role.

This module defines the protocol classes that specify the required methods for each device role.
Protocols act as interfaces that devices must implement to fulfill specific roles.

The Protocol Pattern
-------------------
Instead of using inheritance to define device capabilities, we use protocols to specify
required methods. This allows for:

1. Duck typing - devices only need to implement required methods
2. Multiple roles - devices can implement multiple protocols
3. Type safety - mypy can check protocol compliance
4. Runtime validation - @runtime_checkable allows isinstance() checks

Protocol-Based Architecture
--------------------------
QScope uses a protocol-based architecture for hardware abstraction:

1. Protocols (this module)
   - Define method signatures required for each role
   - Use Python's typing.Protocol for static type checking
   - Use @runtime_checkable for dynamic validation
   - Each protocol represents a specific device capability

2. Devices (qscope.device)
   - Implement protocol methods with hardware-specific code
   - Can implement multiple protocols
   - Don't need to inherit from protocol classes

3. Roles (qscope.types.roles)
   - Connect protocols to interfaces
   - Validate that devices implement required methods
   - Provide singleton instances for system configuration

4. Interfaces (qscope.types.interfaces)
   - Wrap devices implementing protocols
   - Provide clean, documented API for accessing functionality
   - Add type safety and additional logic when needed

Benefits of Protocol-Based Design
--------------------------------
- Flexibility: Devices can implement multiple roles
- Testability: Easy to create mock devices for testing
- Type Safety: Static checking with mypy
- Runtime Validation: Verify protocol compliance at runtime
- Clean API: Interfaces provide consistent access patterns
- Separation of Concerns: Devices focus on hardware, interfaces on API

Example
-------
To create a device that can act as an RF source:

    class MyRFDevice(Device):
        def set_freq(self, freq: float) -> None:
            # Implementation
            pass

        def set_power(self, power: float) -> None:
            # Implementation
            pass

        # ... implement other RFSourceProtocol methods

    # Later, in system configuration:
    system.add_device_with_role(MyRFDevice(), PRIMARY_RF)

The system will validate that MyRFDevice implements all required RFSourceProtocol methods.

See Also
--------
qscope.types.roles : Role definitions and validation
qscope.types.interfaces : Interface implementations
qscope.device : Device implementations
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np


@runtime_checkable
class DigitizerProtocol(Protocol):
    """Methods required for digitizer functionality.

    This protocol defines the interface for data acquisition devices that can
    capture analog signals at high sample rates. Digitizers typically support
    multiple input channels, various triggering modes, and different acquisition
    methods (streaming vs. block capture).

    Implementations must provide methods for:
    - Configuring input channels and ranges
    - Setting up acquisition parameters
    - Starting and stopping data capture
    - Retrieving acquired data
    - Managing hardware triggers
    """

    configure_channels: Callable[
        [List[int], List[Union[float, str]], Optional[List[str]]], None
    ]
    """Configure input channels with ranges and coupling.
    
    Parameters:
    - channels: List of channel indices to configure
    - ranges: List of voltage ranges for each channel
    - coupling: Optional list of coupling modes ('AC' or 'DC')
    """

    start_streaming: Callable[
        [float, int, Optional[int], bool, Optional[int], Optional[float], str, float],
        None,
    ]
    """Start continuous data streaming.
    
    Parameters:
    - sample_interval_s: Time between samples in seconds
    - buffer_size: Number of samples per buffer
    - num_buffers: Optional number of buffers to use
    - trigger_enabled: Whether to use hardware triggering
    - trigger_channel: Channel to use for triggering
    - trigger_threshold: Voltage threshold for trigger
    - trigger_direction: Direction ('RISING', 'FALLING', etc.)
    - trigger_delay: Delay after trigger in seconds
    """

    stop_streaming: Callable[[], None]
    """Stop the current streaming acquisition."""

    get_data: Callable[[float], Tuple[np.ndarray, Tuple[np.ndarray, ...]]]
    """Get acquired data from all enabled channels.
    
    Parameters:
    - timeout: Maximum time to wait for data in seconds
    
    Returns:
    - Tuple containing time array and tuple of channel data arrays
    """

    get_timebase: Callable[[], float]
    """Get the current timebase (sample interval) in seconds."""

    set_downsampling: Callable[[int], None]
    """Set downsampling factor for data acquisition."""

    assert_ok: Callable[[], None]
    """Check device status and raise exception if not OK."""

    set_sample_interval: Callable[[float], None]
    """Set the time between samples in seconds."""

    set_resolution: Callable[[int], None]
    """Set the ADC resolution in bits."""

    set_trigger: Callable[[str], None]
    """Set the trigger mode ('AUTO', 'NONE', 'RISING', etc.)."""

    start_block_capture: Callable[[int, float, int, str], None]
    """Start a single block capture.
    
    Parameters:
    - pre_trigger_samples: Number of samples before trigger
    - timebase: Sample interval in seconds
    - post_trigger_samples: Number of samples after trigger
    - trigger_source: Source of trigger signal
    """

    get_block_data: Callable[[], Tuple[np.ndarray, Tuple[np.ndarray, ...]]]
    """Get data from completed block capture.
    
    Returns:
    - Tuple containing time array and tuple of channel data arrays
    """

    check_overflow: Callable[[], bool]
    """Check if data overflow occurred during acquisition."""


@runtime_checkable
class CameraProtocol(Protocol):
    """Methods required for camera functionality.

    This protocol defines the interface for scientific cameras used in imaging
    and spectroscopy applications. Camera implementations must provide methods
    for controlling exposure, acquisition modes, ROI settings, and retrieving
    image data.

    Implementations must support:
    - Single frame acquisition
    - Continuous video mode
    - Hardware triggering
    - ROI and binning configuration
    - Various shutter and readout modes
    """

    take_snapshot: Callable[[], np.ndarray]
    """Capture a single image and return it as a numpy array.
    
    Returns:
    - 2D numpy array containing the image data
    """

    set_exposure_time: Callable[[float], None]
    """Set the exposure time in seconds."""

    get_frame_shape: Callable[[], tuple[int, int]]
    """Get the current frame dimensions (width, height) in pixels."""

    set_hardware_binning: Callable[[tuple[int, int]], tuple[int, int]]
    """Set hardware binning factors and return the actual values set.
    
    Parameters:
    - binning: Tuple of (horizontal, vertical) binning factors
    
    Returns:
    - Tuple of actual (horizontal, vertical) binning factors applied
    """

    get_readout_time: Callable[[], float]
    """Get the camera readout time in seconds."""

    get_all_seq_frames: Callable[[], List[np.ndarray]]
    """Get all frames from a sequence acquisition as a list of arrays."""

    setup_acquisition: Callable[[str, int], None]
    """Configure acquisition mode and frame count.
    
    Parameters:
    - mode: Acquisition mode ('SINGLE', 'SEQUENCE', 'CONTINUOUS')
    - nframes: Number of frames to acquire (for SEQUENCE mode)
    """

    start_acquisition: Callable[[], None]
    """Start the configured acquisition."""

    stop_acquisition: Callable[[], None]
    """Stop the current acquisition."""

    clear_acquisition: Callable[[], None]
    """Clear any pending acquisition and free resources."""

    get_trigger_time: Callable[[], float]
    """Get the trigger response time in seconds."""

    set_trigger_mode: Callable[[str], None]
    """Set the trigger mode ('INTERNAL', 'EXTERNAL', 'SOFTWARE', etc.)."""

    set_shutter_mode: Callable[[str], None]
    """Set the shutter mode ('GLOBAL', 'ROLLING', etc.)."""

    get_shutter_mode: Callable[[], str]
    """Get the current shutter mode."""

    get_roi: Callable[[], Tuple[int, int, int, int]]
    """Get the current ROI as (x_min, x_max, y_min, y_max)."""

    set_roi: Callable[[int, int, int, int], None]
    """Set the ROI using (x_min, x_max, y_min, y_max)."""

    get_hardware_binning: Callable[[], Tuple[int, int]]
    """Get the current hardware binning factors (horizontal, vertical)."""

    set_hardware_binning: Callable[[int, int], None]
    """Set hardware binning factors (horizontal, vertical)."""

    get_frame_shape: Callable[[], Tuple[int, int]]
    """Get the current frame dimensions (width, height) in pixels."""

    set_frame_shape: Callable[[int, int], None]
    """Set the frame dimensions (width, height) in pixels."""

    update_data_size: Callable[[], Tuple[int, int]]
    """Update and return the current data dimensions after settings changes."""

    get_data_size: Callable[[], Tuple[int, int]]
    """Get the current data dimensions (width, height) in pixels."""

    get_exposure_time: Callable[[], float]
    """Get the current exposure time in seconds."""

    set_exposure_time: Callable[[float], None]
    """Set the exposure time in seconds."""

    wait_for_frame: Callable[[], None]
    """Wait for the next frame to be available."""

    start_video: Callable[[], None]
    """Start continuous video acquisition."""

    stop_video: Callable[[], None]
    """Stop continuous video acquisition."""


@runtime_checkable
class RFSourceProtocol(Protocol):
    """Methods required for RF source functionality.

    This protocol defines the interface for RF signal generators used in
    spectroscopy and control applications. RF sources must provide methods
    for controlling frequency, power, output state, and frequency sweeps.

    Implementations must support:
    - Single frequency generation
    - Frequency sweeps and lists
    - Power control
    - Output state control
    - Triggering options
    - Modulation capabilities
    """

    set_freq: Callable[[float], None]
    """Set the output frequency in MHz."""

    get_freq: Callable[[], float]
    """Get the current output frequency in MHz."""

    set_power: Callable[[float], None]
    """Set the output power in dBm."""

    get_power: Callable[[], float]
    """Get the current output power in dBm."""

    set_state: Callable[[bool], None]
    """Set the RF output state (on/off)."""

    get_state: Callable[[], bool]
    """Get the current RF output state (on/off)."""

    set_freq_list: Callable[[Sequence[float], float], Sequence[float]]
    """Configure a frequency list for stepped sweeps.
    
    Parameters:
    - frequencies: Sequence of frequencies in MHz
    - step_time: Dwell time at each frequency in seconds
    
    Returns:
    - Sequence of actual frequencies set (may differ slightly from requested)
    """

    set_f_table: Callable[[Sequence[float], Sequence[float]], None]
    """Configure a frequency table with corresponding power levels.
    
    Parameters:
    - frequencies: Sequence of frequencies in MHz
    - powers: Sequence of power levels in dBm
    """

    reconnect: Callable[[], None]
    """Attempt to reconnect to the device if connection was lost."""

    start_fm_mod: Callable[[], None]
    """Start frequency modulation."""

    stop_fm_mod: Callable[[], None]
    """Stop frequency modulation."""

    set_trigger: Callable[[str], None]
    """Set the trigger mode for frequency sweeps."""

    start_sweep: Callable[[], None]
    """Start a frequency sweep using the configured frequency list."""

    reset_sweep: Callable[[], None]
    """Reset the sweep to the first frequency in the list."""

    stop_sweep: Callable[[], None]
    """Stop the current frequency sweep."""


@runtime_checkable
class SeqGenProtocol(Protocol):
    """Methods required for sequence generator functionality.

    This protocol defines the interface for sequence generators used for
    timing and pulse control in quantum experiments. Sequence generators
    must provide methods for loading, starting, stopping, and resetting
    pulse sequences.

    Sequence generators typically control:
    - Precise timing of experimental events
    - Digital output patterns
    - Trigger signals for other instruments
    - Synchronization of multiple devices

    Examples of sequence generators include:
    - SpinCore PulseBlaster
    - Arbitrary waveform generators
    - FPGA-based timing controllers
    """

    load_seq: Callable[[str, ...], None]
    """Load a named sequence with optional parameters.
    
    Parameters:
    - seq_name: Name of the sequence to load
    - Additional keyword arguments specific to the sequence
    """

    start: Callable[[], None]
    """Start the loaded sequence."""

    stop: Callable[[], None]
    """Stop the currently running sequence."""

    reset: Callable[[], None]
    """Reset the sequence generator to its initial state."""
