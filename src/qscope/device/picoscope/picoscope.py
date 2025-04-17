import ctypes
import time
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from numpy import dtype, floating, ndarray

from qscope.device.device import Device


class WaveType(Enum):
    """Wave types supported by signal generator."""

    SINE = "SINE"
    SQUARE = "SQUARE"
    TRIANGLE = "TRIANGLE"
    DC = "DC"


@dataclass
class SignalConfig:
    """Configuration for signal generator output.

    Attributes
    ----------
    frequency : float
        Signal frequency in Hz
    amplitude : float
        Peak-to-peak amplitude in volts
    offset : float
        DC offset in volts
    duty_cycle : float
        Duty cycle for square waves (0-100%)
    """

    frequency: float = 1.0
    amplitude: float = 3.3
    offset: float = 0.0

    def validate(self) -> None:
        """Validate signal configuration."""
        if self.frequency <= 0:
            raise ValueError("Frequency must be positive")
        if self.amplitude <= 0:
            raise ValueError("Amplitude must be positive")


# Lazy import of PicoSDK
try:
    from picosdk.functions import adc2mV, assert_pico_ok
    from picosdk.PicoDeviceEnums import picoEnum

    PICOSDK_AVAILABLE = True
except ImportError:
    logger.warning("PicoSDK not available - Picoscope functionality will be limited")
    PICOSDK_AVAILABLE = False


class PicoSDKImportError(Exception):
    """Raised when required device libraries cannot be imported."""

    pass


class Picoscope(Device):
    # Subclasses must define their voltage ranges
    VOLTAGE_RANGES: Dict[float, str] = {}
    """Base class for Picoscope digitizers.
    
    Handles common functionality across all Picoscope models.
    Specific models should inherit from this and implement model-specific details.
    
    Attributes:
        _chandle: Handle to the device
        _ps: Reference to the picosdk module for this device
        _status: Dict tracking device status codes
        _sample_interval: Sampling interval in seconds
        _resolution: ADC resolution in bits
        _channel_range: Voltage range setting
    """

    def __init__(self, **config_kwargs):
        super().__init__(**config_kwargs)
        self._chandle: Optional[ctypes.c_int16] = None
        self._status: Dict[str, Any] = {}

        # Configuration parameters
        self._sample_interval_s: Optional[float] = None  # seconds
        self._resolution: Optional[int] = None  # bits
        self._channel_range: Optional[str] = None
        self._is_streaming: bool = False

        # Buffer management
        self._buffers: Dict[int, np.ndarray] = {}  # One buffer per channel
        self._buffer_max: Dict[int, np.ndarray] = {}  # Current segment buffer
        self._buffer_complete: Dict[int, np.ndarray] = {}  # Complete capture buffer
        self._samples_collected: int = 0
        self._buffer_size: int = 0
        self._enabled_channels: List[int] = []
        self._channel_ranges: Dict[int, str] = {}
        self._downsampling_ratio = 1

        if not hasattr(self, "_downsampling_mode"):
            raise ValueError("Downsampling mode not set in subclass.")
        if not hasattr(self, "_ps"):
            raise ValueError("Picosdk module not set in subclass.")

    # required methods
    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _set_channel(
        self,
        channel_id: int,
        enabled: bool,
        coupling: int,
        range_id: int,
        offset: float,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def _get_channel_range(self, channel: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def _read_max_value(self):
        raise NotImplementedError

    @abstractmethod
    def _set_simple_trigger(
        self, ch: int, threshold: int, direction: str, delay: float, autotrigger: float
    ):
        raise NotImplementedError

    @abstractmethod
    def _stop(self):
        raise NotImplementedError

    @abstractmethod
    def _run_streaming(
        self,
        sample_interval_s: float,
        buffer_size: int,
        num_buffers: int,
        trigger_enabled: bool = False,
    ):
        raise NotImplementedError

    @abstractmethod
    def _get_info(self):
        raise NotImplementedError

    @staticmethod
    def assert_ok(status):
        """Check if a PicoScope status code indicates success.

        Parameters
        ----------
        status : int
            Status code to check

        Raises
        ------
        RuntimeError
            If status indicates an error
        """
        assert_pico_ok(status)

    def set_sample_interval(self, interval: float) -> None:
        """Set sampling interval in seconds.

        Parameters
        ----------
        interval : float
            Sampling interval in seconds
        """
        self._sample_interval_s = interval

    def set_resolution(self, resolution: int) -> None:
        """Set ADC resolution in bits.

        Parameters
        ----------
        resolution : int
            Resolution in bits (typically 8, 12, 14, 15, or 16)
        """
        if resolution not in {8, 12, 14, 15, 16}:
            raise ValueError(
                f"Invalid resolution: {resolution}. Must be one of: 8, 12, 14, 15, 16"
            )
        self._resolution = resolution

    def _adc_to_v(self, buffer: np.ndarray, channel: int) -> np.ndarray:
        """Convert ADC counts to volts for a given channel.

        Parameters
        ----------
        buffer : np.ndarray
            Buffer containing ADC counts
        channel : int
            Channel number

        Returns
        -------
        np.ndarray
            Buffer converted to volts
        """
        _, _, maxADC = self._read_max_value()

        # Get the voltage range for this channel
        try:
            range_str = self._channel_ranges[channel]
        except KeyError:
            raise ValueError(f"Channel {channel} not configured, does not have range.")
        channel_range = self._get_channel_range(channel)
        logger.trace("Converting channel {} data using range {}", channel, range_str)

        # Convert to float array first
        buffer = buffer.astype(np.float64)

        # Mark overflow data as NaN
        buffer[buffer == -32768] = np.nan

        # adc2mV returns millivolts, convert to volts by dividing by 1000
        return np.array(adc2mV(buffer, channel_range, maxADC)) / 1000.0

    def close(self) -> None:
        """Clean shutdown of picoscope."""
        if self._chandle and self._ps:
            if self._is_streaming:
                logger.trace("Stopping streaming before close")
                self._stop()
            logger.info("Closing Picoscope device")
            self._ps._CloseUnit(self._chandle)
            self._chandle = None
            # Clean up buffers and configuration
            self._buffers.clear()
            self._enabled_channels.clear()
            self._channel_ranges.clear()
            self._status.clear()

    def configure_channels(
        self,
        channels: List[int],
        ranges: List[Union[float, str]],
        coupling: Optional[List[str]] = None,
    ) -> None:
        """Configure input channels."""
        if coupling is None:
            coupling = ["DC"] * len(channels)

        if not (len(channels) == len(ranges) == len(coupling)):
            raise ValueError(
                "channels, ranges, and coupling lists must have same length"
            )

        # Clear previous configuration
        self._enabled_channels = []
        self._channel_ranges.clear()

        for ch, ch_range, cp in zip(channels, ranges, coupling):
            # Get range string
            range_str = (
                self.VOLTAGE_RANGES[ch_range]
                if isinstance(ch_range, float)
                else ch_range
            )
            if isinstance(ch_range, float) and ch_range not in self.VOLTAGE_RANGES:
                raise ValueError(
                    f"Invalid voltage range: {ch_range}V. "
                    f"Valid ranges: {sorted(self.VOLTAGE_RANGES.keys())}V"
                )

            status = self._set_channel(
                ch,
                True,  # enabled
                cp,
                range_str,
                0.0,  # analog offset
            )
            self.assert_ok(status)

            # Store configuration
            self._enabled_channels.append(ch)
            self._channel_ranges[ch] = range_str

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
        """Start streaming data acquisition."""
        min_interval = 1e-6  # 1 μs minimum
        if sample_interval_s < min_interval:
            raise ValueError(f"Sample interval must be >= {min_interval} seconds")

        logger.trace(
            "Starting streaming capture with interval={}, buffer_size={}, num_buffers={}",
            sample_interval_s,
            buffer_size,
            num_buffers,
        )
        self._buffer_size = buffer_size
        self._num_buffers = num_buffers
        self._next_sample = 0
        self._auto_stop_outer = False
        self._was_called_back = False
        self._sample_interval_s = sample_interval_s

        if not self._enabled_channels:
            logger.warning(
                "No channels configured before streaming. Using default channels [0, 1]"
            )
            self._enabled_channels = [0, 1]
        logger.trace(
            "Starting streaming with enabled channels: {}", self._enabled_channels
        )
        self._setup_buffers_streaming(self._enabled_channels)
        self._register_callback()

        self._start_streaming(
            sample_interval_s=sample_interval_s,
            buffer_size=buffer_size,
            num_buffers=num_buffers,
            trigger_enabled=trigger_enabled,
            trigger_channel=trigger_channel,
            trigger_threshold=trigger_threshold,
            trigger_direction=trigger_direction,
            trigger_delay=trigger_delay,
        )

    def _start_streaming(
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
            Size of each buffer in samples
        num_buffers : Optional[int]
            Number of buffers to capture, None for continuous
        trigger_enabled : bool
            Enable trigger for streaming
        trigger_channel : Optional[int]
            Channel to trigger on (required if trigger_enabled)
        trigger_threshold : Optional[float]
            Trigger threshold in volts (required if trigger_enabled)
        trigger_direction : str
            Trigger direction ('RISING' or 'FALLING')
        trigger_delay : float
            Post-trigger delay in seconds
        """
        if trigger_enabled:
            if trigger_channel is None or trigger_threshold is None:
                raise ValueError(
                    "trigger_channel and trigger_threshold required when trigger_enabled"
                )
            self.set_trigger(
                trigger_channel, trigger_threshold, trigger_direction, trigger_delay
            )

        status = self._run_streaming(
            sample_interval_s, buffer_size, num_buffers, trigger_enabled
        )
        self.assert_ok(status)
        self._status["runStreaming"] = status
        self._is_streaming = True

    def stop_streaming(self) -> None:
        """Stop streaming data acquisition."""
        if self._is_streaming:
            status = self._stop()
            self.assert_ok(status)
            self._status["stop"] = status
            self._is_streaming = False

    def set_trigger(
        self,
        channel: int,
        threshold: float,
        direction: str = "RISING",
        delay: float = 0.0,
        autotrigger: float = 10.0,
    ) -> None:
        """Configure triggering.

        Parameters
        ----------
        channel : int
            Channel to trigger on
        threshold : float
            Trigger threshold voltage
        direction : str
            Trigger direction ('RISING', 'FALLING', etc) [RISING]
        delay : float
            Trigger delay in seconds [0.0]
        autotrigger : float
            Time in seconds before autotrigger [100ms]
        """
        logger.trace(
            "Setting trigger: channel={}, threshold={}V, direction={}, delay={}s, autotrigger={}s",
            channel,
            threshold,
            direction,
            delay,
            autotrigger,
        )

        # Convert threshold to ADC counts
        _, maxADC_value, maxADC = self._read_max_value()

        # Get the voltage range for the trigger channel
        try:
            range_str = self._channel_ranges[channel]
        except KeyError:
            raise ValueError(f"Channel {channel} not configured, cannot set trigger")

        # Extract the voltage range value from the range string
        # Range strings are like "PS5000A_50MV" or "PS5000A_5V"
        range_value = None
        for v, s in self.VOLTAGE_RANGES.items():
            if s == range_str:
                range_value = v
                break

        if range_value is None:
            raise ValueError(f"Could not determine voltage range from {range_str}")

        # Calculate threshold as a fraction of the full range
        # maxADC_value is the maximum ADC count (e.g., 32767)
        # range_value is the full voltage range (e.g., 5.0V)
        threshold_fraction = threshold / range_value
        threshold_counts = int(threshold_fraction * maxADC_value)

        logger.trace(
            "Trigger threshold calculation: {}V in {}V range = {} counts (of {} max)",
            threshold,
            range_value,
            threshold_counts,
            maxADC_value,
        )

        # Set the trigger
        self._set_simple_trigger(
            channel, threshold_counts, direction, delay, autotrigger
        )

    @staticmethod
    def _resolution_to_enum(resolution: int) -> int:
        """Convert resolution in bits to picosdk enum.

        Parameters
        ----------
        resolution : int
            Resolution in bits

        Returns
        -------
        int
            PicoSDK resolution enum value
        """
        resolution_map = {
            8: picoEnum.PICO_DEVICE_RESOLUTION["PICO_DR_8BIT"],
            12: picoEnum.PICO_DEVICE_RESOLUTION["PICO_DR_12BIT"],
            14: picoEnum.PICO_DEVICE_RESOLUTION["PICO_DR_14BIT"],
            15: picoEnum.PICO_DEVICE_RESOLUTION["PICO_DR_15BIT"],
            16: picoEnum.PICO_DEVICE_RESOLUTION["PICO_DR_16BIT"],
        }
        return resolution_map[resolution]

    def is_responding(self) -> bool:
        """Check if device is connected and responding.

        Returns
        -------
        bool
            True if device is connected and responding
        """
        if not self._chandle:
            return False
        try:
            # Try to read max ADC value as a simple check
            status, _, _ = self._read_max_value()
            return status == 0  # PICO_OK
        except Exception as e:
            logger.error("Error checking device response: {}", str(e))
            return False

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information and capabilities.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing device information
        """
        if not self._chandle:
            raise RuntimeError("Device not initialized")

        info = {}
        try:
            self._get_info()

            # Get current settings
            info["resolution"] = self._resolution
            info["enabled_channels"] = self._enabled_channels.copy()
            info["channel_ranges"] = self._channel_ranges.copy()
            info["is_streaming"] = self._is_streaming

            return info

        except Exception as e:
            logger.error("Error getting device info: {}", str(e))
            raise

    def config_signal_generator(self, config: SignalConfig) -> None:
        """Configure the signal generator/AWG output without starting it.

        Parameters
        ----------
        config : SignalConfig
            Signal configuration object
        """
        config.validate()
        self._config_signal_generator(config)

    def start_signal_generator(self) -> None:
        """Start the previously configured signal generator."""
        self._start_signal_generator()

    def stop_signal_generator(self) -> None:
        """Start the previously configured signal generator."""
        self._stop_signal_generator()

    def config_start_signal_generator(self, config: SignalConfig) -> None:
        """Configure the signal generator/AWG output and start it immediately.

        Parameters
        ----------
        config : SignalConfig
            Signal configuration object
        """
        config.validate()
        self._config_start_signal_generator(config)

    @abstractmethod
    def _config_signal_generator(self, config: SignalConfig) -> None:
        """Implementation-specific signal generator configuration."""
        raise NotImplementedError

    @abstractmethod
    def _start_signal_generator(self) -> None:
        """Implementation-specific signal generator start."""
        raise NotImplementedError

    @abstractmethod
    def _config_start_signal_generator(self, config: SignalConfig) -> None:
        """Implementation-specific signal generator configuration and start."""
        raise NotImplementedError

    @abstractmethod
    def get_timebase(
        self, desired_interval: float, num_samples: int
    ) -> tuple[int, float]:
        """Calculate optimal timebase for desired sample interval.

        Parameters
        ----------
        desired_interval : float
            Desired time between samples in seconds
        num_samples : int
            Number of samples to capture

        Returns
        -------
        tuple[int, float]
            Tuple containing:
            - Timebase index
            - Actual sampling interval in seconds
        """
        raise NotImplementedError

    def _start_block_capture(
        self,
        sample_interval: float,
        num_samples: int,
        num_blocks: int = 1,
        trigger_enabled: bool = False,
        trigger_channel: Optional[int] = None,
        trigger_threshold: Optional[float] = None,
        trigger_direction: str = "RISING",
        trigger_delay: float = 0.0,
        pre_trigger_samples: int = 0,
    ) -> float:
        """Start block-mode data acquisition."""
        # Get appropriate timebase
        timebase, actual_interval = self.get_timebase(sample_interval, num_samples)

        self._setup_memory_segments(num_blocks)

        self._set_num_caps(num_blocks)

        # Configure trigger if enabled
        if trigger_enabled:
            self.set_trigger(
                trigger_channel, trigger_threshold, trigger_direction, trigger_delay
            )

        # Start the block capture
        self._run_block(num_samples, timebase, pre_trigger_samples)

        return actual_interval

    @abstractmethod
    def _run_block(self, num_samples: int, timebase: int, pre_trigger_samples: int = 0):
        raise NotImplementedError

    @abstractmethod
    def _set_num_caps(self, num_blocks: int):
        raise NotImplementedError

    @abstractmethod
    def _setup_memory_segments(self, num_blocks: int):
        raise NotImplementedError

    def start_block_capture(
        self,
        sample_interval: float,
        num_samples: int,
        num_blocks: int = 1,
        trigger_enabled: bool = False,
        trigger_channel: Optional[int] = None,
        trigger_threshold: Optional[float] = None,
        trigger_direction: str = "RISING",
        trigger_delay: float = 0.0,
        pre_trigger_samples: int = 0,
    ) -> float:
        """Start block-mode data acquisition."""
        logger.trace(
            "Starting block capture with interval={}, samples={}, blocks={}",
            sample_interval,
            num_samples,
            num_blocks,
        )

        # Set up buffers
        if not self._enabled_channels:
            logger.warning("No channels configured. Using default channels [0, 1]")
            self._enabled_channels = [0, 1]
        self._setup_block_buffers(self._enabled_channels, num_samples, num_blocks)

        # Start capture using parent class method
        actual_interval = self._start_block_capture(
            sample_interval=sample_interval,
            num_samples=num_samples,
            num_blocks=num_blocks,
            trigger_enabled=trigger_enabled,
            trigger_channel=trigger_channel,
            trigger_threshold=trigger_threshold,
            trigger_direction=trigger_direction,
            trigger_delay=trigger_delay,
            pre_trigger_samples=pre_trigger_samples,
        )

        self._sample_interval_s = actual_interval
        return actual_interval

    @abstractmethod
    def set_downsampling(self, mode: str, downsample_ratio: int) -> None:
        """Configure hardware downsampling.

        Parameters
        ----------
        mode : str
            Downsampling mode ('NONE', 'AVERAGE', 'DECIMATE', 'MIN_MAX')
        downsample_ratio : int
            Number of samples to combine
        """
        raise NotImplementedError

    def get_block_data(
        self, timeout_s=1
    ) -> tuple[ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[Any]]]:
        """Get block-mode captured data with hardware downsampling."""
        self._wait_for_capture(timeout_s)
        captures = self._get_num_caps()

        # Calculate samples after downsampling
        samples_per_segment = (
            len(self._buffers[self._enabled_channels[0]]) // captures.value
        )
        # had a `// self._downsampling_ratio` here, but that cropped data...
        downsampled_samples = samples_per_segment

        # Set up buffers for each segment
        for ch in self._enabled_channels:
            for segment in range(captures.value):
                self._set_data_buffer(
                    ch, downsampled_samples, samples_per_segment, segment
                )

        self._get_values_bulk(captures, downsampled_samples)

        # Convert data for all channels
        voltage_data = tuple(
            self._adc_to_v(self._buffers[ch], ch) for ch in self._enabled_channels
        )
        # Create time array for all blocks
        total_samples = len(voltage_data[0])
        time_data = np.linspace(
            0,
            (total_samples - 1) * self._sample_interval_s * self._downsampling_ratio,
            total_samples,
        )

        return np.array(time_data), np.array(voltage_data)

    @abstractmethod
    def _get_values_bulk(self, captures, downsampled_samples):
        raise NotImplementedError

    @abstractmethod
    def _set_data_buffer(self, ch, downsampled_samples, samples_per_segment, segment):
        raise NotImplementedError

    @abstractmethod
    def _get_num_caps(self):
        raise NotImplementedError

    @abstractmethod
    def _wait_for_capture(self, timeout_s):
        raise NotImplementedError

    def _handle_power_status(self, status: int) -> None:
        """Handle power-related status codes from device initialization.

        Parameters
        ----------
        status : int
            Status code from device open call

        Raises
        ------
        RuntimeError
            If device initialization fails
        """
        if status == 0:  # PICO_OK
            return

        # Power supply status codes
        PICO_POWER_SUPPLY_NOT_CONNECTED = 286
        PICO_USB3_0_DEVICE_NON_USB3_0_PORT = 282

        if status in {
            PICO_POWER_SUPPLY_NOT_CONNECTED,
            PICO_USB3_0_DEVICE_NON_USB3_0_PORT,
        }:
            # For PS5000A, we can continue without the power supply
            # The device will operate with reduced functionality
            logger.warning(
                "Device opened with limited power supply. Some features may be restricted."
            )
            return
        else:
            raise RuntimeError(f"Failed to initialize device. Status: {status}")

    def get_data(
        self, timeout: float = 10.0
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
        """Get acquired data from all enabled channels.

        Parameters
        ----------
        timeout : float
            Maximum time to wait for data in seconds

        Returns
        -------
        Tuple[np.ndarray, Tuple[np.ndarray, ...]]
            Tuple containing:
            - Time array in seconds
            - Tuple of arrays containing voltage data for each channel

        Raises
        ------
        TimeoutError
            If data collection exceeds timeout
        """
        total_samples = self._buffer_size
        if self._num_buffers is not None:
            total_samples *= self._num_buffers

        self._get_stream(timeout, total_samples)

        logger.info("Data collection complete: {} samples collected", self._next_sample)

        # Create time array
        time_data = np.linspace(
            0, (total_samples - 1) * self._sample_interval_s, total_samples
        )

        # Convert all channels to Volts
        voltage_data = tuple(
            self._adc_to_v(self._buffers[ch], ch) for ch in self._enabled_channels
        )

        return time_data, voltage_data

    @abstractmethod
    def _get_stream(self, timeout, total_samples):
        raise NotImplementedError

    def check_overflow(self) -> Dict[str, bool]:
        """Check if any channels have experienced overflow.

        Returns
        -------
        Dict[str, bool]
            Dictionary mapping channel letters to overflow status
        """
        return {
            k[11]: v for k, v in self._status.items() if k.startswith("overflow_ch")
        }

    def _register_callback(self) -> None:
        """Convert the Python callback into a C function pointer."""
        logger.trace("Registering streaming callback")
        self._c_callback = self._ps.StreamingReadyType(self._streaming_callback)

    @abstractmethod
    def _setup_buffers_streaming(self, channels: list[int]) -> None:
        """Set up data buffers for streaming.

        Parameters
        ----------
        channels : list[int]
            List of channel numbers to set up buffers for
        """
        raise NotImplementedError

    @abstractmethod
    def _setup_block_buffers(
        self, channels: list[int], num_samples: int, num_blocks: int
    ) -> None:
        """Set up data buffers for block mode capture.

        Parameters
        ----------
        channels : list[int]
            List of channel numbers to set up buffers for
        num_samples : int
            Number of samples per block
        num_blocks : int
            Number of blocks to capture
        """
        raise NotImplementedError

    def _streaming_callback(
        self,
        handle: int,
        num_samples: int,
        start_index: int,
        overflow: int,
        trigger_at: int,
        triggered: int,
        auto_stop: int,
        param: Any,
    ) -> None:
        """Callback function for streaming data collection.

        Called by the driver when data is ready.

        Parameters
        ----------
        handle : int
            Device handle
        num_samples : int
            Number of new samples
        start_index : int
            Starting index in buffer
        overflow : int
            Bit field indicating which channels have overflowed
        trigger_at : int
            Index of trigger point
        triggered : int
            Whether trigger has occurred
        auto_stop : int
            Whether auto stop has occurred
        param : Any
            User parameter (unused)
        """
        self._was_called_back = True

        # Check for buffer overflow
        if overflow:
            channels = []
            for ch in range(2):  # FIXME assuming 2 channels only -> fixup
                if overflow & (1 << ch):
                    channels.append(chr(65 + ch))
            overflow_msg = f"Buffer overflow on channels: {', '.join(channels)}"
            logger.warning(overflow_msg)

            # Mark the overflowed data in the buffer
            for ch_letter in channels:
                ch_num = ord(ch_letter) - ord("A")
                if ch_num in self._buffer_max:
                    # Set overflow flag in status
                    self._status[f"overflow_ch{ch_letter}"] = True
                    # Mark overflow data with min int16 value
                    self._buffer_max[ch_num][
                        start_index : start_index + num_samples
                    ] = -32768

        try:
            dest_end = self._next_sample + num_samples
            first_channel = min(self._buffer_max.keys())
            if dest_end > len(self._buffers[first_channel]):
                logger.error(
                    "Buffer overrun: trying to write {} samples when only {} remaining",
                    num_samples,
                    len(self._buffers[first_channel]) - self._next_sample,
                )
                raise BufferError("Buffer overrun in streaming callback")
            source_end = start_index + num_samples

            # Copy data from driver buffers to our complete buffers
            for ch in self._buffer_max.keys():
                self._buffers[ch][self._next_sample : dest_end] = self._buffer_max[ch][
                    start_index:source_end
                ]

            self._next_sample += num_samples
        except Exception as e:
            logger.error("Error in streaming callback: {}", str(e))
            raise
        if auto_stop:
            logger.info("Auto stop triggered after {} samples", self._next_sample)
            self._auto_stop_outer = True
