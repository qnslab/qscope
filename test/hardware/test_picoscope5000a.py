import time

import numpy as np
import pytest
from loguru import logger

import qscope.util
from qscope.device import Picoscope5000a
from qscope.util import TEST_LOGLEVEL


class TestPicoscope5000aLocal:
    @pytest.fixture()
    def client_log(self):
        qscope.util.start_client_log(log_to_file=True, log_level=TEST_LOGLEVEL)
        yield
        qscope.util.shutdown_client_log()

    @pytest.mark.usefixtures("client_log")
    def test_streaming_capture(self):
        """Hardware test for streaming data capture from Picoscope 5000a series."""
        logger.info("Starting Picoscope 5000a streaming test")

        # Initialize scope
        scope = Picoscope5000a()
        scope.set_resolution(12)  # 12-bit mode
        scope.open()

        try:
            # Print available voltage ranges for reference
            logger.info(
                f"Available voltage ranges (V): {sorted(Picoscope5000a.VOLTAGE_RANGES.keys())}"
            )

            # Configure channels
            scope.configure_channels(
                channels=[0, 1], ranges=[2.0, 2.0], coupling=["DC", "DC"]
            )
            logger.info("Channels configured")

            # Streaming parameters
            buffer_size = 500
            num_buffers = 10
            sample_interval = 250e-6  # 250 μs
            expected_samples = buffer_size * num_buffers

            # Start streaming
            logger.info("Starting data streaming")
            scope.start_streaming(
                sample_interval=sample_interval,
                buffer_size=buffer_size,
                num_buffers=num_buffers,
            )

            # Get data
            logger.info("Getting data")
            time_data, channel_data = scope.get_data()

            # Verify data properties
            assert len(time_data) == expected_samples, (
                "Incorrect number of time samples"
            )
            assert len(channel_data) == 2, "Expected data from 2 channels"
            assert len(channel_data[0]) == expected_samples, (
                "Incorrect number of voltage samples"
            )

            # Verify time array properties
            assert np.allclose(time_data[1] - time_data[0], sample_interval), (
                "Incorrect time step"
            )
            assert np.allclose(
                time_data[-1], (expected_samples - 1) * sample_interval
            ), "Incorrect total time"

            logger.info("Data verification complete")

        finally:
            logger.info("Closing scope")
            scope.close()

    @pytest.mark.usefixtures("client_log")
    def test_triggered_streaming(self):
        """Hardware test for triggered streaming data capture from Picoscope 5000a series."""
        logger.info("Starting Picoscope 5000a triggered streaming test")

        # Initialize scope
        scope = Picoscope5000a()
        scope.set_resolution(12)  # 12-bit mode
        scope.open()

        try:
            # Configure channels
            scope.configure_channels(
                channels=[0, 1], ranges=[2.0, 2.0], coupling=["DC", "DC"]
            )
            logger.info("Channels configured")

            # Streaming parameters
            buffer_size = 500
            num_buffers = 10
            sample_interval = 250e-6  # 250 μs
            expected_samples = buffer_size * num_buffers

            # Start triggered streaming
            logger.info("Starting triggered data streaming")
            scope.start_streaming(
                sample_interval=sample_interval,
                buffer_size=buffer_size,
                num_buffers=num_buffers,
                trigger_enabled=True,
                trigger_channel=0,
                trigger_threshold=0.001,  # Trigger set point
                trigger_direction="RISING",
                trigger_delay=0.0,
            )

            # Get data
            logger.info("Getting data")
            time_data, channel_data = scope.get_data()

            # Verify basic data properties
            assert len(time_data) == expected_samples, (
                "Incorrect number of time samples"
            )
            assert len(channel_data) == 2, "Expected data from 2 channels"
            assert len(channel_data[0]) == expected_samples, (
                "Incorrect channel data length"
            )

            # Verify timing
            assert np.allclose(time_data[1] - time_data[0], sample_interval), (
                "Incorrect time step"
            )
            assert np.allclose(
                time_data[-1], (expected_samples - 1) * sample_interval
            ), "Incorrect total time"

            # Verify data is numeric and within range
            for ch_data in channel_data:
                assert not np.any(np.isnan(ch_data)), "NaN values in channel data"
                assert np.all(np.abs(ch_data) <= 2000), (
                    "Values outside ±2V range (in mV)"
                )

            logger.info("Triggered streaming test complete")

        finally:
            logger.info("Closing scope")
            scope.close()

    @pytest.mark.usefixtures("client_log")
    def test_block_capture(self):
        """Hardware test for block-mode data capture."""
        logger.info("Starting Picoscope 5000a block capture test")

        scope = Picoscope5000a()
        scope.set_resolution(12)
        scope.open()

        try:
            # Configure channels
            scope.configure_channels(
                channels=[0, 1], ranges=[2.0, 2.0], coupling=["DC", "DC"]
            )
            logger.info("Channels configured")

            # Block capture parameters
            sample_interval = 1e-6  # 1 μs
            num_samples = 1000
            num_blocks = 5

            # Start block capture
            logger.info("Starting block capture")
            actual_interval = scope.start_block_capture(
                sample_interval=sample_interval,
                num_samples=num_samples,
                num_blocks=num_blocks,
                trigger_enabled=True,
                trigger_channel=0,
                trigger_threshold=0.5,
                trigger_direction="RISING",
            )

            # Get and verify data
            logger.info("Getting block data")
            time_data, channel_data = scope.get_block_data()

            # Verify basic data properties
            assert len(time_data) == num_samples * num_blocks, (
                "Incorrect number of samples"
            )
            assert len(channel_data) == 2, "Expected data from 2 channels"
            assert len(channel_data[0]) == num_samples * num_blocks, (
                "Incorrect channel data length"
            )

            # Verify timing
            assert actual_interval >= sample_interval, (
                "Actual interval shorter than requested"
            )
            assert np.allclose(np.diff(time_data), actual_interval), (
                "Incorrect time step"
            )

            # Verify data is numeric and within range
            for ch_data in channel_data:
                assert not np.any(np.isnan(ch_data)), "NaN values in channel data"
                assert np.all(np.abs(ch_data) <= 2000), (
                    "Values outside ±2V range (in mV)"
                )

            logger.info("Block capture test complete")

        finally:
            logger.info("Closing scope")
            scope.close()

    @pytest.mark.hardware
    @pytest.mark.usefixtures("client_log")
    def test_rapid_start_stop(self):
        """Test rapid starting and stopping of streaming."""
        scope = Picoscope5000a()
        scope.set_resolution(12)
        scope.open()

        try:
            scope.configure_channels([0], [2.0])

            for _ in range(3):  # Test multiple start/stops
                scope.start_streaming(sample_interval=1e-3, buffer_size=100)
                time.sleep(0.1)
                scope.stop_streaming()
                time.sleep(0.1)  # Brief pause between cycles

            assert not scope._is_streaming

        finally:
            scope.close()

    @pytest.mark.hardware
    @pytest.mark.usefixtures("client_log")
    def test_multiple_channel_ranges(self):
        """Test different voltage ranges on different channels."""
        scope = Picoscope5000a()
        scope.set_resolution(12)
        scope.open()

        try:
            # Configure different ranges
            scope.configure_channels(
                channels=[0, 1],
                ranges=[2.0, 0.2],  # 2V and 200mV ranges
                coupling=["DC", "DC"],
            )

            scope.start_streaming(sample_interval=1e-3, buffer_size=100, num_buffers=2)

            time_data, channel_data = scope.get_data()

            # Verify channel scaling
            assert np.max(np.abs(channel_data[0])) <= 2000  # ±2V in mV
            assert np.max(np.abs(channel_data[1])) <= 200  # ±200mV in mV

        finally:
            scope.close()

    @pytest.mark.usefixtures("client_log")
    def test_error_handling(self):
        """Test error handling scenarios."""
        scope = Picoscope5000a()

        # Test operations before opening
        with pytest.raises(RuntimeError):
            scope.get_device_info()

        # Test invalid trigger configuration
        scope.set_resolution(12)
        scope.open()
        try:
            with pytest.raises(ValueError):
                scope.start_streaming(
                    sample_interval=1e-3,
                    buffer_size=100,
                    trigger_enabled=True,  # Missing required trigger params
                )
        finally:
            scope.close()


if __name__ == "__main__":
    # Setup logging when run directly
    qscope.util.start_client_log(log_to_file=True, log_level=TEST_LOGLEVEL)
    test = TestPicoscope5000aLocal()
    test.test_streaming_capture()
    test.test_triggered_streaming()
    test.test_block_capture()
