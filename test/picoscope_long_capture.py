import os
import time
from pathlib import Path

import numpy as np
from loguru import logger

import qscope.util
from qscope.device import Picoscope5000a
from qscope.util import TEST_LOGLEVEL


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    output_dir = Path("mock_output/picoscope")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def long_block_capture(duration_s, sample_rate_hz):
    """Long duration block capture."""
    import matplotlib.pyplot as plt

    scope = Picoscope5000a()
    scope.set_resolution(12)
    scope.open()

    try:
        # Configure channels
        scope.configure_channels(
            channels=[0, 1], ranges=[2.0, 2.0], coupling=["DC", "DC"]
        )

        # Calculate capture parameters
        sample_interval = 1.0 / sample_rate_hz
        num_samples = int(duration_s * sample_rate_hz)

        logger.info(f"Starting {duration_s}s capture at {sample_rate_hz}Hz")
        logger.info(f"Sample interval: {sample_interval * 1e6:.1f}μs")
        logger.info(f"Total samples: {num_samples}")

        # Start block capture
        actual_interval = scope.start_block_capture(
            sample_interval=sample_interval,
            num_samples=num_samples,
            num_blocks=1,
            trigger_enabled=False,
        )

        logger.info(f"Actual sample interval: {actual_interval * 1e6:.1f}μs")
        logger.info(f"Actual sample rate: {1.0 / actual_interval:.1f}Hz")

        # Get the captured data
        logger.info("Capturing data...")
        time_data, channel_data = scope.get_block_data(timeout_s=duration_s + 1)

        # Optional: Save data for analysis
        output_dir = ensure_output_dir()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = output_dir / f"picoscope_test_capture_{timestamp}.npz"
        np.savez(
            filename,
            time=time_data,
            channelA=channel_data[0],
            channelB=channel_data[1],
            sample_rate=1.0 / actual_interval,
        )
        logger.info(f"Test data saved to {filename}")

        # Plot the captured data
        plt.figure(figsize=(12, 6))
        plt.plot(time_data, channel_data[0], label="Channel A")
        plt.plot(time_data, channel_data[1], label="Channel B")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.title(f"Picoscope {duration_s}s Capture @ {sample_rate_hz / 1e3:.1f} kHz")
        plt.grid(True)
        plt.legend()
        plt.show()

    finally:
        scope.close()


if __name__ == "__main__":
    # Setup logging when run directly
    qscope.util.start_client_log(log_to_file=True, log_level=TEST_LOGLEVEL)
    sample_spacing = 1e-6  # 1us
    long_block_capture(10.0, 1 / sample_spacing)
    qscope.util.shutdown_client_log()
