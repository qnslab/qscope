import ctypes
import importlib
import time
from typing import Tuple

import numpy as np
from loguru import logger

from .picoscope import (
    PICOSDK_AVAILABLE,
    Picoscope,
    PicoSDKImportError,
    SignalConfig,
    WaveType,
)

if PICOSDK_AVAILABLE:
    from picosdk.PicoDeviceEnums import picoEnum


class Picoscope5000a(Picoscope):
    """Implementation for PicoScope 5000a series devices."""

    # Valid voltage ranges for 5000a series
    VOLTAGE_RANGES = {
        0.01: "PS5000A_10MV",
        0.02: "PS5000A_20MV",
        0.05: "PS5000A_50MV",
        0.1: "PS5000A_100MV",
        0.2: "PS5000A_200MV",
        0.5: "PS5000A_500MV",
        1.0: "PS5000A_1V",
        2.0: "PS5000A_2V",
        5.0: "PS5000A_5V",
        10.0: "PS5000A_10V",
        20.0: "PS5000A_20V",
    }

    def __init__(self, **config_kwargs):
        """Initialize Picoscope 5000a series device.

        Raises
        ------
        PicoSDKImportError
            If PicoSDK cannot be imported or is not properly installed
        """
        if not PICOSDK_AVAILABLE:
            raise PicoSDKImportError(
                "PicoSDK not available - cannot initialize Picoscope5000a"
            )

        try:
            # First try importing the base picosdk
            import picosdk

            # Some versions may not have __version__
            version = getattr(picosdk, "__version__", "unknown")
            logger.debug("Found PicoSDK (version: {})", version)

            # Then import the specific model's module
            ps_module = importlib.import_module("picosdk.ps5000a")
            logger.debug("Successfully imported ps5000a module")

            # Try accessing the ps5000a attribute
            if not hasattr(ps_module, "ps5000a"):
                raise ImportError("ps5000a module does not contain ps5000a attribute")

            self._ps = ps_module.ps5000a
            logger.debug("Successfully initialized ps5000a interface")

        except ImportError as e:
            raise PicoSDKImportError(
                f"Failed to properly import PicoSDK: {str(e)}\n"
                "Please ensure PicoSDK is installed correctly:\n"
                "1. pip install picosdk\n"
                "2. Install Pico Technology SDK from https://www.picotech.com/downloads"
            ) from e

        # Set downsampling mode before calling parent constructor
        self._downsampling_mode = self._ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"]

        # Call parent constructor
        super().__init__(**config_kwargs)

    def open(self) -> None:
        """Initialize connection to 5000a series scope."""
        if not self._resolution:
            raise ValueError("Resolution must be set before opening device")

        self._chandle = ctypes.c_int16()
        res = self._resolution_to_enum(self._resolution)

        status = self._ps.ps5000aOpenUnit(ctypes.byref(self._chandle), None, res)
        self._handle_power_status(status)

    def _set_channel(
        self,
        channel_id: int,
        enabled: bool,
        coupling: int | str,
        rang: str | float,
        offset: float,
    ) -> None:
        """Set up input channel.

        Parameters
        ----------
        channel_id : int
            Channel number (0=A, 1=B, etc)
        enabled : bool
            Enable/disable channel
        coupling : str
            Coupling type ('AC' or 'DC')
        rang : str | float
            Voltage range identifier or value in volts
        offset : float
            Analog offset in volts
        """
        # Convert range to string if float provided
        range_str = self.VOLTAGE_RANGES[rang] if isinstance(rang, float) else rang

        # Ensure coupling is uppercase
        coupling = coupling.upper()

        status = self._ps.ps5000aSetChannel(
            self._chandle,
            self._get_channel(channel_id),
            int(enabled),
            self._ps.PS5000A_COUPLING[f"PS5000A_{coupling}"],
            self._ps.PS5000A_RANGE[range_str],
            offset,
        )
        return status  # Make sure we return the status

    def _get_channel_range(self, channel: int) -> str:
        try:
            range_str = self._channel_ranges[channel]
        except KeyError:
            raise ValueError(f"Channel {channel} not configured, does not have range.")
        return self._ps.PS5000A_RANGE[range_str]

    def _get_channel(self, ch: int) -> int:
        return self._ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{chr(65 + ch)}"]

    def _read_max_value(self) -> Tuple[int, int, ctypes.c_int16]:
        maxADC = ctypes.c_int16()
        status = self._ps.ps5000aMaximumValue(self._chandle, ctypes.byref(maxADC))
        value = maxADC.value
        return status, value, maxADC

    def _set_simple_trigger(
        self, ch: int, threshold: int, direction: str, delay: float, autotrigger: float
    ):
        status = self._ps.ps5000aSetSimpleTrigger(
            self._chandle,
            1,  # enable
            self._get_channel(ch),
            threshold,
            self._ps.PS5000A_THRESHOLD_DIRECTION[f"PS5000A_{direction}"],
            int(delay),  # delay -->>> see docs for calculation, using timebase
            int(autotrigger * 1e3),  # autotrigger in milliseconds
        )
        self.assert_ok(status)
        self._status["setTrigger"] = status

    def _stop(self):
        self._ps.ps5000aStop(self._chandle)

    def _run_streaming(
        self,
        sample_interval_s: float,
        buffer_size: int,
        num_buffers: int,
        trigger_enabled: bool = False,
    ):
        # Convert sample interval to microseconds
        interval_us = int(sample_interval_s * 1e6)
        interval = ctypes.c_int32(interval_us)

        total_samples = buffer_size
        if num_buffers is not None:
            total_samples *= num_buffers

        status = self._ps.ps5000aRunStreaming(
            self._chandle,
            ctypes.byref(interval),
            self._ps.PS5000A_TIME_UNITS["PS5000A_US"],
            buffer_size if trigger_enabled else 0,  # maxPreTriggerSamples
            total_samples,
            1 if num_buffers else 0,  # autoStopOn
            1,  # downsampleRatio
            self._ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"],
            buffer_size,
        )
        return status

    def _get_info(self):
        variant = ctypes.create_string_buffer(8)
        serial = ctypes.create_string_buffer(16)
        hw_version = ctypes.c_int16()

        status = self._ps.ps5000aGetUnitInfo(
            self._chandle, variant, 8, None, self._ps.PICO_INFO["PICO_VARIANT_INFO"]
        )
        self.assert_ok(status)
        status = self._ps.ps5000aGetUnitInfo(
            self._chandle,
            serial,
            16,
            None,
            self._ps.PICO_INFO["PICO_BATCH_AND_SERIAL"],
        )
        self.assert_ok(status)
        status = self._ps.ps5000aGetUnitInfo(
            self._chandle,
            hw_version,
            2,
            None,
            self._ps.PICO_INFO["PICO_HARDWARE_VERSION"],
        )
        self.assert_ok(status)

        return variant.value.decode(), serial.value.decode(), hw_version.value

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
        # Query device for available timebases
        timebase_val = 0
        interval = ctypes.c_float()
        max_samples = ctypes.c_int32()

        while True:
            timebase = ctypes.c_uint32(timebase_val)
            status = self._ps.ps5000aGetTimebase2(
                self._chandle,
                timebase,
                ctypes.c_int32(num_samples),
                ctypes.byref(interval),
                ctypes.byref(max_samples),  # maxSamples
                ctypes.c_uint32(0),  # segmentIndex
            )

            if status == 0:  # PICO_OK
                actual_interval = interval.value * 1e-9  # Convert from nanoseconds
                if actual_interval >= desired_interval:
                    return timebase_val, actual_interval

            timebase_val += 1

    def _setup_memory_segments(self, num_blocks: int):
        # Set up memory segments
        status = self._ps.ps5000aMemorySegments(
            self._chandle,
            num_blocks,
            ctypes.byref(ctypes.c_int32()),  # max_samples
        )
        self.assert_ok(status)

    def _run_block(self, num_samples: int, timebase: int, pre_trigger_samples: int = 0):
        timeIndisposedMs = ctypes.c_int32()
        status = self._ps.ps5000aRunBlock(
            self._chandle,
            pre_trigger_samples,  # pre-trigger samples
            num_samples - pre_trigger_samples,  # post-trigger samples
            ctypes.c_uint32(timebase),  # timebase needs to stay uint32
            ctypes.byref(timeIndisposedMs),  # time indisposed ms
            0,  # segment index
            None,  # lpReady callback
            None,  # pParameter
        )
        self.assert_ok(status)

    def _set_num_caps(self, num_blocks: int):
        # Set number of captures
        status = self._ps.ps5000aSetNoOfCaptures(self._chandle, num_blocks)
        self.assert_ok(status)

    def set_downsampling(self, mode: str, downsample_ratio: int) -> None:
        """Configure hardware downsampling.

        Parameters
        ----------
        mode : str
            Downsampling mode ('NONE', 'AVERAGE', 'DECIMATE', 'MIN_MAX')
        downsample_ratio : int
            Number of samples to combine
        """
        mode_map = {
            "NONE": "PS5000A_RATIO_MODE_NONE",
            "AVERAGE": "PS5000A_RATIO_MODE_AVERAGE",
            "DECIMATE": "PS5000A_RATIO_MODE_DECIMATE",
            "MIN_MAX": "PS5000A_RATIO_MODE_AGGREGATE",
        }

        if mode not in mode_map:
            raise ValueError(
                f"Invalid downsampling mode. Must be one of {list(mode_map.keys())}"
            )

        self._downsampling_mode = self._ps.PS5000A_RATIO_MODE[mode_map[mode]]
        self._downsampling_ratio = downsample_ratio

    def _wait_for_capture(self, timeout_s):
        ready = ctypes.c_int16(0)
        check = 0
        while ready.value == 0:
            status = self._ps.ps5000aIsReady(self._chandle, ctypes.byref(ready))
            time.sleep(0.01)
            check += 1
            if check >= timeout_s / 0.01:  # Timeout after ~1s default second
                raise TimeoutError("Block capture timeout")

    def _get_num_caps(self):
        # Get number of captures completed
        captures = ctypes.c_uint32()
        status = self._ps.ps5000aGetNoOfCaptures(self._chandle, ctypes.byref(captures))
        self.assert_ok(status)
        return captures

    def _set_data_buffer(self, ch, downsampled_samples, samples_per_segment, segment):
        status = self._ps.ps5000aSetDataBuffer(
            self._chandle,
            self._ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{chr(65 + ch)}"],
            self._buffers[ch][
                segment * samples_per_segment : (segment + 1) * samples_per_segment
            ].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            downsampled_samples,
            segment,
            self._downsampling_mode,
        )
        self.assert_ok(status)

    def _get_values_bulk(self, captures, downsampled_samples):
        # Get captured data using rapid block mode with downsampling
        max_samples = ctypes.c_uint32(downsampled_samples)
        overflow = ctypes.c_int16()
        status = self._ps.ps5000aGetValuesBulk(
            self._chandle,
            ctypes.byref(max_samples),
            0,  # fromSegmentIndex
            captures.value - 1,  # toSegmentIndex
            ctypes.c_uint32(self._downsampling_ratio),
            self._downsampling_mode,
            ctypes.byref(overflow),
        )
        self.assert_ok(status)

    def _get_stream(self, timeout, total_samples):
        start_time = time.time()
        while self._next_sample < total_samples and not self._auto_stop_outer:
            if time.time() - start_time > timeout:
                logger.error("Data acquisition timed out after {} seconds", timeout)
                raise TimeoutError(
                    f"Data acquisition timed out after {timeout} seconds"
                )

            self._was_called_back = False
            status = self._ps.ps5000aGetStreamingLatestValues(
                self._chandle, self._c_callback, None
            )
            if not self._was_called_back:
                # If no data ready, wait briefly before trying again
                time.sleep(0.01)

    def _setup_buffers_streaming(self, channels: list[int]) -> None:
        """Set up data buffers for streaming.

        Parameters
        ----------
        channels : list[int]
            List of channel numbers to set up buffers for
        """
        logger.trace("Setting up buffers for channels: {}", channels)
        total_samples = self._buffer_size
        if self._num_buffers is not None:
            total_samples *= self._num_buffers

        # Create buffers for each channel
        for ch in channels:
            # Buffer for current segment
            self._buffer_max[ch] = np.zeros(shape=self._buffer_size, dtype=np.int16)
            # Buffer for complete capture
            self._buffers[ch] = np.zeros(shape=total_samples, dtype=np.int16)

            logger.trace(
                "Allocating buffer for channel {}: size={}", ch, self._buffer_size
            )
            # Register buffers with driver
            status = self._ps.ps5000aSetDataBuffers(
                self._chandle,
                self._ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{chr(65 + ch)}"],
                self._buffer_max[ch].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                None,
                self._buffer_size,
                0,  # segment index
                self._ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"],
            )
            self.assert_ok(status)
            self._status[f"setDataBuffers{chr(65 + ch)}"] = status

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
        logger.trace("Setting up block mode buffers for channels: {}", channels)

        # Calculate downsampled buffer size
        downsampled_samples = num_samples // self._downsampling_ratio

        # Create buffers for each channel
        for ch in channels:
            # Buffer for complete capture
            self._buffers[ch] = np.zeros(
                shape=downsampled_samples * num_blocks, dtype=np.int16
            )

            # Register buffer with driver
            status = self._ps.ps5000aSetDataBuffer(
                self._chandle,
                self._ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{chr(65 + ch)}"],
                self._buffers[ch].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                downsampled_samples,
                0,  # segment index
                self._downsampling_mode,  # Use configured downsampling mode
            )
            self.assert_ok(status)

    def _config_signal_generator(self, config: SignalConfig) -> None:
        """Configure built-in signal generator without starting it.

        Parameters
        ----------
        config : SignalConfig
            Signal configuration object

        Raises
        ------
        ValueError
            If frequency is outside valid range for the selected wave type
        """
        # Define frequency limits
        MIN_FREQUENCY = 0.03  # Hz
        MAX_FREQUENCY = 20_000_000  # 20 MHz

        # square wave (see enPS5000AWaveType), DO NOT USE `picoEnum.PICO_WAVE_TYPE`
        wave_type = ctypes.c_int32(1)

        if not MIN_FREQUENCY <= config.frequency <= MAX_FREQUENCY:
            raise ValueError(
                f"Frequency {config.frequency} Hz is outside valid range for this wave: "
                f"({MIN_FREQUENCY} Hz to {MAX_FREQUENCY} Hz)"
            )

        # Convert voltages to microvolts and ensure they're integers
        offset_mv = int(config.offset * 1_000_000)  # Plain integer
        amplitude_mv = int(config.amplitude * 1_000_000)  # Plain integer

        # Configure with software trigger
        status = self._ps.ps5000aSetSigGenBuiltInV2(
            self._chandle,
            offset_mv,
            amplitude_mv,
            wave_type,
            config.frequency,
            config.frequency,
            0.0,
            0.0,
            picoEnum.PICO_SWEEP_TYPE["PICO_UP"],
            0,
            0,
            0,
            picoEnum.PICO_SIGGEN_TRIG_TYPE["PICO_SIGGEN_GATE_HIGH"],
            picoEnum.PICO_SIGGEN_TRIG_SOURCE["PICO_SIGGEN_SOFT_TRIG"],
            0,
        )
        self.assert_ok(status)

        logger.debug("Signal generator configured with software trigger")

    def _start_signal_generator(self) -> None:
        """Start the previously configured signal generator using software trigger."""
        # Use ps5000aSigGenSoftwareControl to trigger the signal generator
        status = self._ps.ps5000aSigGenSoftwareControl(
            self._chandle,
            1,  # Set to 1 to start the signal generator
        )
        self.assert_ok(status)
        logger.debug("Signal generator started via software trigger")

    def _stop_signal_generator(self) -> None:
        """Stop running sig gen (gate off)."""
        # Use ps5000aSigGenSoftwareControl to trigger the signal generator
        status = self._ps.ps5000aSigGenSoftwareControl(
            self._chandle,
            0,  # Set to 0 to stop the signal generator
        )
        self.assert_ok(status)
        logger.debug("Signal generator stopped via software trigger")

    def _config_start_signal_generator(self, config: SignalConfig) -> None:
        """Configure and immediately start the signal generator.

        Parameters
        ----------
        config : SignalConfig
            Signal configuration object

        Raises
        ------
        ValueError
            If frequency is outside valid range for the selected wave type
        """
        self._config_signal_generator(config)
        self._start_signal_generator()
