import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
from click_option_group import optgroup
from loguru import logger

from qscope.device.picoscope import SignalConfig, WaveType
from qscope.device.SMU2450 import TriggerConfig, TriggerEdge
from qscope.system import System
from qscope.util.save import NumpyEncoder, _get_base_path

from .base import (
    managed_devices,
    setup_logging,
)


@click.command(name="trigtime")
@click.option(
    "--system-name",
    "-n",
    type=str,
    required=True,
    help="Name of the system configuration",
)
@click.option(
    "--project-name",
    "-p",
    type=str,
    default="timing",
    help="Project name for saving results",
)
@optgroup.group("Signal Parameters")
@optgroup.option(
    "--frequency",
    "-f",
    type=float,
    default=1.0,
    help="Square wave frequency in Hz (default: 1.0)",
)
@optgroup.option(
    "--duration",
    "-d",
    type=float,
    default=1.0,
    help="Test duration in seconds (default: 1.0)",
)
@optgroup.option(
    "--current",
    "-c",
    type=float,
    default=0.1,
    help="Test current in amps (default: 0.1A)",
)
@optgroup.group("Electromagnet Configuration")
@optgroup.option(
    "--source-mode",
    type=click.Choice(["current", "voltage"]),
    default="voltage",
    help="SMU source mode (default: voltage)",
)
@optgroup.option(
    "--coil-resistance",
    "-r",
    type=float,
    default=27.0,
    help="Electromagnet coil resistance in ohms (default: 27.0)",
)
@optgroup.option(
    "--additional-resistance",
    "-ar",
    type=float,
    default=0.0,
    help="Additional resistance in series with coil in ohms (default: 0.0)",
)
@optgroup.option(
    "--voltage-limit",
    "-vl",
    type=float,
    default=None,
    help="Maximum voltage to apply (default: calculated from current and resistance)",
)
@optgroup.option(
    "--force-sourcing",
    is_flag=True,
    default=False,
    help="Force operation even when approaching SMU limits (use with caution)",
)
@optgroup.group("Hardware Configuration")
@optgroup.option(
    "--sample-rate",
    "-s",
    type=float,
    default=1e6,
    help="Digitizer sample rate in Hz (default: 1e6)",
)
@optgroup.option(
    "--downsample-ratio",
    "-dr",
    type=int,
    default=1,
    help="Hardware downsampling ratio (must be >=1, default: 1)",
)
@optgroup.option(
    "--smu-address", type=str, default=None, help="Optional VISA address of SMU"
)
@optgroup.group("Diagnostic Options")
@optgroup.option(
    "--diagnostic",
    "-diag",
    is_flag=True,
    default=False,
    help="Run in diagnostic mode to check trigger signals",
)
@optgroup.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging"
)
@optgroup.option(
    "--check-config-lists",
    "-ccl",
    type=bool,
    default=False,
    help="Check the TRIGGER configuration lists",
)
def trigtime(
    system_name: str,
    frequency: float,
    duration: float,
    current: float,
    source_mode: str,
    coil_resistance: float,
    additional_resistance: float,
    voltage_limit: Optional[float],
    force_sourcing: bool,
    sample_rate: float,
    downsample_ratio: int,
    smu_address: str,
    project_name: str,
    diagnostic: bool,
    verbose: bool,
    check_config_lists: bool,
):
    """Test timing synchronization between AWG trigger and SMU response.

    Physical Connections:
    AWG Out ─┬─> Channel A (monitor clock)
             └─> SMU Digital Input (pin 3)

    SMU Out ──> Channel B (monitor current)

    Verifies:

    1. AWG signal generation
    2. SMU trigger detection and response time
    3. Current switching timing/delays

    Usage
    `qscope mpl trigtime -n zyla -p mpl_cds -c 0.45 -f 0.25 -d 10 -s 1e7 -dr 5000`
    """
    setup_logging(log_level="TRACE" if verbose else "INFO")
    logger.info("Starting timing synchronization test")
    logger.info(f"Parameters: {frequency=}Hz, {duration=}s, {current=}A")

    # Pre-measurement validation
    if downsample_ratio < 1:
        raise click.BadParameter("Downsample ratio must be >= 1")

    # Calculate measurement parameters
    total_samples = int(duration * sample_rate)
    if total_samples // downsample_ratio > 1e8:  # Arbitrary limit
        raise click.BadParameter("Requested duration/sample rate too large")

    with managed_devices(
        smu_address=smu_address,
        current=current,
        coil_resistance=coil_resistance,
        additional_resistance=additional_resistance,
        voltage_limit=voltage_limit,
        source_mode=source_mode,
        force_sourcing=force_sourcing,
    ) as (smu, scope):
        # Add diagnostic mode to check trigger signals
        if diagnostic:
            run_diagnostic_mode(scope, frequency, duration, sample_rate)
            return  # Exit after diagnostic

        # -----------------------------------------------------------------------------------------
        # Main Timing Test
        # -----------------------------------------------------------------------------------------

        # Configure hardware downsampling
        if downsample_ratio > 1:
            scope.set_downsampling(mode="AVERAGE", downsample_ratio=downsample_ratio)

        # Calculate actual sample rate after downsampling
        actual_rate = sample_rate / downsample_ratio
        logger.info(
            f"Sampling rates - measurement: {sample_rate:.2e} Hz, "
            f"after downsampling: {actual_rate:.2e} Hz"
        )

        # Configure scope channels
        logger.info("Configuring scope channels")
        scope.configure_channels(
            channels=[0, 1],  # A and B
            ranges=[10.0, 20.0],
            coupling=["DC", "DC"],
        )

        # Stop any running trigger model first
        logger.info("Stopping any running trigger model")
        smu.stop_trigger_model()

        # Configure SMU triggering - use RISING edge only
        logger.info("Configuring SMU trigger")
        # Calculate pre-trigger samples as 10% of one clock cycle
        clock_period = 1.0 / (
            frequency * 2
        )  # Period of the clock signal (not the field switching)
        pre_trigger_time = 0.1 * clock_period  # 10% of one clock period
        actual_pre_trigger_samples = int(pre_trigger_time * sample_rate)
        logger.info(
            f"Using {actual_pre_trigger_samples} pre-trigger samples (10% of one clock period)"
        )

        # Adjust count to account for pre-trigger time
        effective_duration = duration - pre_trigger_time

        # Configure trigger based on source mode
        if source_mode == "current":
            trigger_config = TriggerConfig(
                pin=3,
                edge=TriggerEdge.RISING,  # Only trigger on rising edges
                count=int(effective_duration * frequency * 2),
                high_value=current,
                low_value=0.0,
            )
        else:
            # Calculate voltage for voltage sourcing mode
            total_resistance = coil_resistance + additional_resistance
            required_voltage = current * total_resistance

            # Add diagnostic information
            logger.info(f"SMU mode before trigger: {smu.get_mode()}")
            logger.info(
                f"Calculated voltage: {required_voltage:.3f}V for {current:.3f}A with {total_resistance:.2f}Ω"
            )
            compliance = smu.get_compliance_limit()
            logger.info(f"SMU compliance limit: {compliance:.3f}A")

            # Test direct voltage setting
            logger.info("Testing direct voltage setting...")
            smu.set_voltage(required_voltage)
            time.sleep(0.5)
            measured_current = smu.get_current()
            logger.info(
                f"Test: Applied {required_voltage:.3f}V, measured {measured_current:.3f}A"
            )

            # Check if measured current is significantly different from expected
            if abs(measured_current - current) > 0.05 * current:
                logger.warning(
                    f"Measured current ({measured_current:.3f}A) differs from expected ({current:.3f}A)"
                )
                logger.warning("This may affect edge detection in voltage mode")

            trigger_config = TriggerConfig(
                pin=3,
                edge=TriggerEdge.RISING,  # Only trigger on rising edges
                count=int(effective_duration * frequency * 2),
                high_value=required_voltage,
                low_value=0.0,
                resistance=total_resistance,  # Pass the actual resistance for current calculations
            )

        smu.configure_trigger(trigger_config)

        # Check trigger model state
        logger.info(
            f"Trigger model state after configuration: {smu.get_trigger_state()}"
        )

        # Verify digital I/O configuration
        smu.configure_digital_io(3, mode="trigger")
        logger.info("Verified digital I/O pin 3 is configured for triggering")

        # Configure the AWG output - double frequency for consistent phase synchronization
        logger.info("Configuring AWG output")
        signal_config = SignalConfig(
            frequency=frequency * 2,  # Double the frequency
            amplitude=1.5,
            offset=0.5,
        )
        scope.config_signal_generator(signal_config)

        # Start scope capture with triggering on Channel A (clock signal)
        logger.info("Starting scope capture")
        scope.start_block_capture(
            sample_interval=1.0 / sample_rate,
            num_samples=int(duration * sample_rate),
            trigger_enabled=True,
            trigger_channel=0,  # Channel A (clock)
            trigger_threshold=2.5,  # Mid-point of clock signal
            trigger_direction="RISING",
            pre_trigger_samples=actual_pre_trigger_samples,
        )

        # Start SMU trigger model
        logger.info("Starting SMU trigger model")
        smu.start_trigger_model()

        # Check trigger model state after starting
        logger.info(f"Trigger model state after starting: {smu.get_trigger_state()}")

        # Check for compliance issues
        if smu.check_compliance():
            logger.error("SMU compliance limit reached before starting measurement")
            raise click.Abort(
                "Compliance limit reached. Check connections and current settings."
            )

        if check_config_lists:
            # Check if the configuration list was created properly
            logger.warning("|> Config list check begin")
            try:
                # First check the source mode
                source_mode = smu._communicate(":SOUR:FUNC?", query=True)
                logger.info(f"Current source mode: {source_mode}")

                # Check if we're in the expected mode
                if smu._mode == "current":
                    if "CURR" not in source_mode:
                        logger.error(
                            f"SMU should be in current mode but is in {source_mode} mode"
                        )
                        # Try to force current mode
                        smu._communicate(":SOUR:FUNC CURR")
                        logger.info("Attempted to force current mode")
                        # Verify again
                        source_mode = smu._communicate(":SOUR:FUNC?", query=True)
                        logger.info(f"Source mode after correction: {source_mode}")

                # Check configuration list
                list_info = smu._communicate("SOUR:CONF:LIST:CAT?", query=True)
                if not list_info:
                    logger.error("No configuration list found")
                else:
                    logger.info(f"Configuration list: {list_info}")

                # Check the contents of the sourceList
                if "sourceList" in list_info:
                    list_size = smu._communicate(
                        'SOUR:CONF:LIST:SIZE? "sourceList"', query=True
                    )
                    logger.info(f"sourceList size: {list_size}")

                    # Check the values in the list
                    low_val = smu._communicate(
                        'SOUR:CONF:LIST:QUER? "sourceList", 1', query=True
                    )
                    high_val = smu._communicate(
                        'SOUR:CONF:LIST:QUER? "sourceList", 2', query=True
                    )
                    logger.info(f"Low state config: {low_val}")
                    logger.info(f"High state config: {high_val}")

                    # Check current source values directly
                    if "CURR" in source_mode:
                        curr_val = smu._communicate(":SOUR:CURR?", query=True)
                        logger.info(f"Current source value: {curr_val}A")
                    else:
                        volt_val = smu._communicate(":SOUR:VOLT?", query=True)
                        logger.info(f"Voltage source value: {volt_val}V")
            except Exception as e:
                logger.warning(f"Error checking configuration lists: {e}")
            logger.warning("<| Config list check end")

        # Start the signal generator
        logger.info("Starting signal generator")
        scope.start_signal_generator()

        # Get scope data
        logger.info("Collecting scope data")
        time_data, (clock_data, smu_data) = scope.get_block_data(
            timeout_s=max(5.0, duration * 2.5)
        )

        # Shift time array so trigger point is at t=0
        if actual_pre_trigger_samples > 0:
            # Adjust pre-trigger samples for downsampling
            adjusted_pre_trigger = actual_pre_trigger_samples // downsample_ratio

            # Make sure the index is valid
            if adjusted_pre_trigger >= len(time_data):
                logger.warning(
                    f"Pre-trigger index ({adjusted_pre_trigger}) exceeds data length ({len(time_data)})"
                )
                adjusted_pre_trigger = 0  # Use first sample as fallback

            trigger_time = time_data[adjusted_pre_trigger]
            time_data = time_data - trigger_time
            logger.info(
                f"Shifted time data so trigger point (sample {adjusted_pre_trigger}) is at t=0"
            )

        # Check for compliance after measurement
        compliance_reached = smu.check_compliance()
        if compliance_reached:
            logger.warning("SMU compliance limit reached during measurement")

        # Stop SMU trigger model
        logger.info("Stopping SMU trigger model")
        smu.stop_trigger_model()

        # Analyze and plot results
        analyze_and_plot_results(
            time_data,
            clock_data,
            smu_data,
            coil_resistance,
            additional_resistance,
            current,
            source_mode,
            sample_rate,
            downsample_ratio,
            system_name,
            project_name,
            frequency,
            voltage_limit,
            compliance_reached,
            no_show=False,
        )


def run_diagnostic_mode(scope, frequency, duration, sample_rate):
    """Run diagnostic mode to check trigger signals."""
    logger.info("Running in diagnostic mode to check trigger signals")

    # Configure scope to capture the trigger signal
    scope.configure_channels(
        channels=[0],  # Just channel A for the clock/trigger
        ranges=[10.0],
        coupling=["DC"],
    )

    # Configure and start AWG output
    signal_config = SignalConfig(
        frequency=frequency,
        amplitude=1.5,
        offset=0.5,
    )
    scope.config_signal_generator(signal_config)

    # Calculate pre-trigger samples (10% of one clock cycle)
    pre_trigger_samples = int(0.1 * sample_rate / frequency)
    logger.info(f"Using {pre_trigger_samples} pre-trigger samples for diagnostic")

    # Capture the signal
    scope.start_block_capture(
        sample_interval=1.0 / sample_rate,
        num_samples=int(duration * sample_rate),
        trigger_enabled=True,
        trigger_channel=0,
        trigger_threshold=2.5,
        trigger_direction="RISING",
        pre_trigger_samples=pre_trigger_samples,
    )

    # Start the signal generator
    scope.start_signal_generator()

    # Get scope data
    time_data, (clock_data,) = scope.get_block_data(timeout_s=max(5.0, duration * 2.5))

    # Shift time array so trigger point is at t=0
    if pre_trigger_samples > 0:
        # Make sure we don't exceed array bounds
        trigger_index = min(pre_trigger_samples, len(time_data) - 1)
        trigger_time = time_data[trigger_index]
        time_data = time_data - trigger_time
        logger.info(f"Shifted time data so trigger point is at t=0")

    # Plot the trigger signal
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, clock_data, "b-", label="Trigger Signal")
    plt.axhline(y=1.65, color="k", linestyle="--", label="Trigger Threshold")
    plt.axhline(y=0.7, color="r", linestyle="dotted", label="Max logic Off")
    plt.axhline(y=3.3, color="g", linestyle="dotted", label="Min logic On")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Trigger Signal Diagnostic")
    plt.ylim(bottom=0)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    # Count edges and verify frequency
    edges = np.where(np.diff(clock_data > 1.65))[0]
    measured_freq = len(edges) / (
        2 * duration
    )  # Divide by 2 because each cycle has 2 edges

    logger.info(f"Detected {len(edges)} edges in {duration}s")
    logger.info(
        f"Measured frequency: {measured_freq:.2f}Hz (expected: {frequency:.2f}Hz)"
    )

    # Check signal levels
    min_v = np.min(clock_data)
    max_v = np.max(clock_data)
    logger.info(f"Signal range: {min_v:.2f}V to {max_v:.2f}V")


def get_command_string() -> str:
    """Get the original command string that was used to run this script.

    Returns
    -------
    str
        The full command string including all arguments
    """
    return " ".join(sys.argv)


def save_results(
    results: Dict[str, Any],
    system_name: str,
    project_name: str,
    frequency: float,
    duration: float,
    sample_rate: float,
    downsample_ratio: int,
    coil_resistance: float,
    additional_resistance: float,
    current: float,
    source_mode: str,
    voltage_limit: Optional[float],
    compliance_reached: bool,
    fig: Optional[plt.Figure] = None,
    no_show: bool = False,
) -> None:
    """Save measurement results to file.

    Parameters
    ----------
    results : Dict[str, Any]
        Processed measurement results
    system_name : str
        System configuration name
    project_name : str
        Project name for save directory
    frequency : float
        Signal frequency in Hz
    duration : float
        Measurement duration in seconds
    sample_rate : float
        Sample rate in Hz
    downsample_ratio : int
        Hardware downsampling ratio
    coil_resistance : float
        Coil resistance in ohms
    current : float
        Applied current in amps
    compliance_reached : bool
        Whether SMU compliance was reached during measurement
    fig : plt.Figure, optional
        Figure to save, by default None
    no_show : bool, optional
        Whether to suppress plot display, by default False
    """
    # Create system object and get base path
    system = System(system_name)
    base_path = _get_base_path(system, project_name, "time_trig")

    # Clean up results to remove non-serializable objects
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {
                k: clean_for_json(v)
                for k, v in obj.items()
                if not callable(v) and k != "function"
            }
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj if not callable(item)]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    # Prepare metadata dictionary
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "command": get_command_string(),
        "parameters": {
            "frequency": frequency,
            "duration": duration,
            "current": current,
            "sample_rate": sample_rate,
            "downsample_ratio": downsample_ratio,
            "coil_resistance": coil_resistance,
            "additional_resistance": additional_resistance,
            "total_resistance": coil_resistance + additional_resistance,
            "source_mode": source_mode,
            "voltage_limit": voltage_limit,
        },
        "analysis": {
            "mean_delay_ms": float(results.get("mean_delay", 0)),
            "std_delay_ms": float(results.get("std_delay", 0)),
            "num_clock_edges": len(results.get("edges_clock", [])),
            "num_effective_edges": len(results.get("effective_clock_edges", [])),
            "num_smu_edges": len(results.get("edges_smu", [])),
            "compliance_reached": compliance_reached,
        },
    }

    # Save data and figure in parallel
    def save_data():
        # Save metadata as JSON
        with open(f"{base_path}.json", "w") as f:
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)

        # Save raw data as compressed NumPy file
        np.savez_compressed(
            f"{base_path}_raw.npz",
            time_data=results.get("time_data", []),
            clock_data=results.get("clock_data", []),
            smu_data=results.get("smu_data", []),
            smu_current=results.get("smu_current", []),
            edges_clock=results.get("edges_clock", []),
            edges_smu=results.get("edges_smu", []),
            effective_clock_edges=results.get("effective_clock_edges", []),
            delays=results.get("delays", []),
        )

        logger.debug(f"Raw data saved to {base_path}_raw.npz")

    def save_figure():
        if fig is not None:
            fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
            plt.close(fig) if no_show else None

    with ThreadPoolExecutor(max_workers=2) as executor:
        data_future = executor.submit(save_data)
        fig_future = executor.submit(save_figure)

        # Wait for both to complete and check for errors
        data_future.result()
        fig_future.result()

    logger.info(f"Metadata saved to {base_path}.json")
    logger.info(f"Raw data saved to {base_path}_raw.npz")
    if fig is not None:
        logger.info(f"Figure saved to {base_path}.png")


def analyze_and_plot_results(
    time_data: np.ndarray,
    clock_data: np.ndarray,
    smu_data: np.ndarray,
    coil_resistance: float,
    additional_resistance: float,
    current: float,
    source_mode: str,
    sample_rate: float,
    downsample_ratio: int,
    system_name: str,
    project_name: str,
    frequency: float,
    voltage_limit: Optional[float],
    compliance_reached: bool,
    no_show: bool = False,
) -> None:
    """Analyze timing relationships and plot results.

    Parameters
    ----------
    time_data : np.ndarray
        Time data points
    clock_data : np.ndarray
        Clock signal data
    smu_data : np.ndarray
        SMU data (voltage in current mode, current in voltage mode)
    coil_resistance : float
        Coil resistance in ohms
    additional_resistance : float
        Additional resistance in series with coil in ohms
    current : float
        Applied current in amps
    source_mode : str
        SMU source mode ("voltage" or "current")
    sample_rate : float
        Sample rate in Hz
    downsample_ratio : int
        Hardware downsampling ratio
    system_name : str
        System configuration name
    project_name : str
        Project name for save directory
    frequency : float
        Signal frequency in Hz
    voltage_limit : float, optional
        Maximum voltage to apply
    compliance_reached : bool
        Whether SMU compliance was reached during measurement
    no_show : bool, optional
        Whether to suppress plot display, by default False
    """
    # Calculate total resistance
    total_resistance = coil_resistance + additional_resistance

    # Convert SMU data to current based on source mode
    if source_mode == "current":
        # SMU data is voltage, convert to current
        smu_current = smu_data / total_resistance
    else:
        # SMU data is voltage across the coil, convert to current
        smu_current = smu_data / total_resistance

    # Analyze timing relationships
    edges_clock = np.where(np.diff(clock_data > 3))[0]  # Clock threshold

    if source_mode == "current":
        # In current mode, use a threshold based on the expected current
        smu_threshold = 3 * current / 4
        logger.info(f"Using current-based threshold: {smu_threshold:.3f}A")
        edges_smu = np.where(np.diff(smu_current > smu_threshold))[0]  # SMU threshold
    else:
        # In voltage mode, use a more adaptive threshold based on the actual data range
        smu_min = np.min(smu_current)
        smu_max = np.max(smu_current)
        smu_threshold = smu_min + (smu_max - smu_min) * 0.5  # Use 50% of the range

        logger.info(
            f"Adaptive threshold for SMU current: {smu_threshold:.3f}A (min: {smu_min:.3f}A, max: {smu_max:.3f}A)"
        )
        edges_smu = np.where(np.diff(smu_current > smu_threshold))[
            0
        ]  # Adaptive threshold

    if len(edges_clock) == 0 or len(edges_smu) == 0:
        logger.warning("No edges detected in signals. Check connections and settings.")
        delays = []
        mean_delay = 0
        std_delay = 0
        effective_clock_edges = np.array([])
    else:
        # Filter clock edges to only include those that should trigger current changes
        # Note: Data acquisition starts with field OFF due to pre-trigger capture
        # First rising edge in data = field ON, second rising edge = field OFF, etc.
        field_on_edges = edges_clock[::2]  # Every other rising edge (0, 2, 4...)
        field_off_edges = edges_clock[1::2]  # Every other rising edge (1, 3, 5...)

        logger.info(
            f"Detected {len(field_on_edges)} field ON edges and {len(field_off_edges)} field OFF edges"
        )

        # Use field ON edges as the effective clock edges for delay calculation
        effective_clock_edges = field_on_edges

        # Now calculate delays using the effective clock edges
        delays = []
        for clock_edge in effective_clock_edges:
            # Find closest SMU edge
            if len(edges_smu) > 0:
                closest_smu = edges_smu[np.argmin(np.abs(edges_smu - clock_edge))]
                delay = (
                    (closest_smu - clock_edge) / (sample_rate / downsample_ratio) * 1e3
                )  # Convert to ms
                delays.append(delay)

        mean_delay = np.mean(delays) if delays else 0
        std_delay = np.std(delays) if delays else 0

    logger.info(f"Mean delay: {mean_delay:.2f}ms ± {std_delay:.2f}ms")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plt.title(f"Timing Test: {frequency:.1f} Hz switching, AWG: {frequency * 2:.1f} Hz")

    # Plot clock signal in volts
    ax1.plot(time_data, clock_data, "b-", label="Clock (Ch A)")
    ax1.set_ylabel("Voltage (V)")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax1.axhline(y=0.7, color="xkcd:grey", linestyle="dotted", label="Max logic Off")
    ax1.axhline(y=3.3, color="xkcd:dark grey", linestyle="dotted", label="Min logic On")

    # Plot SMU output in mamps
    ax1_twin = ax1.twinx()  # Create a second y-axis
    ax1_twin.plot(
        time_data, 1e3 * smu_current, "r-", label="SMU Current (Ch B)", alpha=0.7
    )
    ax1_twin.set_ylabel("Current (mA)")

    # Add threshold line to help visualize edge detection
    if "smu_threshold" in locals():
        ax1_twin.axhline(
            y=1e3 * smu_threshold,
            color="r",
            linestyle="--",
            label=f"Threshold: {smu_threshold * 1e3:.1f} mA",
        )

    ax1_twin.legend(loc="upper right")

    ax1.set_ylim(bottom=0)
    ax1_twin.set_ylim(bottom=0)

    # Plot delays if we have any
    if len(delays) > 0:
        # Use effective_clock_edges for plotting
        plot_indices = min(len(delays), len(effective_clock_edges))
        ax2.plot(
            time_data[effective_clock_edges[:plot_indices]], delays[:plot_indices], "b."
        )

        # Find corresponding SMU edges for plotting
        smu_plot_edges = []
        for i, clock_edge in enumerate(effective_clock_edges[:plot_indices]):
            if i < len(delays):
                closest_smu = edges_smu[np.argmin(np.abs(edges_smu - clock_edge))]
                smu_plot_edges.append(closest_smu)

        ax2.plot(time_data[smu_plot_edges], delays[: len(smu_plot_edges)], "rs")
        ax2.set_ylabel("Delay (ms)")
        ax2.set_xlabel("Time (s)")
        ax2.grid(True)
        ax2.axhline(
            mean_delay, color="k", linestyle="--", label=f"Mean: {mean_delay:.1f}ms"
        )
        ax2.fill_between(
            time_data[[0, -1]],
            [mean_delay - std_delay] * 2,
            [mean_delay + std_delay] * 2,
            color="k",
            alpha=0.2,
            label=f"Std: {std_delay:.1f}ms",
        )
        ax2.legend(loc="upper right")
    else:
        ax2.text(
            0.5,
            0.5,
            "No timing data available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax2.transAxes,
        )
        ax2.set_ylabel("Delay (ms)")
        ax2.set_xlabel("Time (s)")

    plt.tight_layout()
    if not no_show:
        plt.show()

    # Prepare results dictionary for saving
    results_dict = {
        "time_data": time_data,
        "clock_data": clock_data,
        "smu_data": smu_data,
        "smu_current": smu_current,
        "edges_clock": edges_clock,
        "edges_smu": edges_smu,
        "effective_clock_edges": effective_clock_edges,
        "delays": delays,
        "mean_delay": mean_delay,
        "std_delay": std_delay,
    }

    # Save results
    save_results(
        results=results_dict,
        system_name=system_name,
        project_name=project_name,
        frequency=frequency,
        duration=len(time_data)
        / (sample_rate / downsample_ratio),  # Calculate duration from data
        sample_rate=sample_rate,
        downsample_ratio=downsample_ratio,
        coil_resistance=coil_resistance,
        additional_resistance=additional_resistance,
        current=current,
        source_mode=source_mode,
        voltage_limit=voltage_limit,
        compliance_reached=compliance_reached,
        fig=fig,
        no_show=no_show,
    )
