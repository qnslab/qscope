"""Magnetophotoluminescence (MPL) triggered time trace measurements.

This module provides tools for measuring PL response to triggered magnetic field changes.
The measurement uses the Picoscope's AWG to generate a clock signal that triggers
the SMU current switching while simultaneously measuring both the clock and PL signals.

Key Capabilities
--------------
- Precise timing synchronization between field switching and measurement
- Hardware-triggered current switching for consistent timing
- Configurable pre-trigger capture for baseline measurements
- Support for averaged and real-time measurements
- Automated data collection and analysis

Hardware Requirements
------------------
- Keithley 2450 SMU: Precise current control for electromagnet
- Picoscope: High-speed data acquisition with AWG output
- Photodiode: PL detection with DC/AC coupling
- Electromagnet: Field generation (typical R=27Ω)
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import matplotlib.pyplot as plt
import numpy as np
from click_option_group import optgroup
from loguru import logger
from tqdm.auto import trange

from qscope.cli.mpl.plots import plot_mpl_results
from qscope.device.picoscope import SignalConfig, WaveType
from qscope.device.SMU2450 import TriggerConfig, TriggerEdge
from qscope.fitting.mpl import MPLFitter
from qscope.system import System
from qscope.util import DEFAULT_LOGLEVEL

from qscope.cli.mpl.base import analyze_mpl_response, managed_devices, setup_logging

def detrend_pl_signal(time_data, pl_data, clock_data, trigger_threshold):
    """Remove linear trend from PL signal.

    Parameters
    ----------
    time_data : np.ndarray
        Time points
    pl_data : np.ndarray
        PL signal data
    clock_data : np.ndarray
        Clock signal data
    trigger_threshold : float
        Threshold for clock signal edge detection

    Returns
    -------
    np.ndarray
        Detrended PL signal
    dict
        Trend information including slope and intercept
    """
    # Find baseline regions (when field is OFF)
    clock_high = clock_data > trigger_threshold
    rising_edges = np.where(np.diff(clock_high.astype(int)) > 0)[0]

    # Create mask for baseline points (when field is OFF)
    baseline_mask = np.ones_like(time_data, dtype=bool)

    # Toggle state at each rising edge
    current_on = False
    for edge_idx in rising_edges:
        current_on = not current_on
        if edge_idx + 1 < len(baseline_mask):
            # When current is ON, exclude these points from baseline
            if current_on:
                baseline_mask[edge_idx + 1 :] = False
            else:
                baseline_mask[edge_idx + 1 :] = True

    # Fit linear trend to baseline points only
    if np.sum(baseline_mask) > 10:  # Ensure we have enough points
        # Use polyfit to get linear trend
        coeffs = np.polyfit(time_data[baseline_mask], pl_data[baseline_mask], 1)
        slope, intercept = coeffs

        # Calculate trend line
        trend = slope * time_data + intercept

        # Subtract trend from data
        detrended_pl = pl_data - trend + pl_data[0]  # Keep the initial value

        trend_info = {
            "slope": slope,
            "intercept": intercept,
            "trend": trend,
            "method": "baseline_linear",
        }
    else:
        # Fallback to using all points if we can't identify enough baseline points
        coeffs = np.polyfit(time_data, pl_data, 1)
        slope, intercept = coeffs

        # Calculate trend line
        trend = slope * time_data + intercept

        # Subtract trend from data
        detrended_pl = pl_data - trend + pl_data[0]  # Keep the initial value

        trend_info = {
            "slope": slope,
            "intercept": intercept,
            "trend": trend,
            "method": "full_linear",
        }

    return detrended_pl, trend_info


def process_results(
    raw_data: Union[Dict[str, Any], List[Dict[str, Any]]],
    coil_resistance: float,
    additional_resistance: float = 0.0,
    current: float = 0.1,
    source_mode: str = "voltage",
    trigger_threshold: float = 2.5,
    fit_exponential: bool = False,
    plot_pulses: bool = False,
    fit_type: str = "single",
    detrend: bool = False,
) -> Dict[str, Any]:
    """Process raw measurement data.

    Parameters
    ----------
    raw_data : Dict[str, Any] | List[Dict[str, Any]]
        Raw measurement data, either:
        - Single measurement dict with time_data, clock_data, pl_data
        - List of measurement dicts for averaging
    coil_resistance : float
        Electromagnet coil resistance in ohms
    current : float, optional
        Applied current in amps, by default 0.1
    trigger_threshold : float, optional
        Threshold for clock signal edge detection, by default 2.5
    fit_exponential : bool, optional
        Whether to fit exponential curves to transitions, by default False
    plot_pulses : bool, optional
        Whether to plot detailed pulse analysis, by default False
    fit_type : str, optional
        Type of exponential fit to use, by default "single"
    detrend : bool, optional
        Whether to remove linear trend from PL signal, by default False

    Returns
    -------
    Dict[str, Any]
        Processed results including:
        - mean_time: Average time points
        - mean_clock: Average clock signal
        - mean_pl: Average PL signal
        - std_pl: PL standard deviation
        - analysis: Analysis results
        - time_data: Raw time data (if multiple measurements)
        - clock_data: Raw clock data (if multiple measurements)
        - pl_data: Raw PL data (if multiple measurements)
    """
    # Process data arrays more efficiently
    if isinstance(raw_data, list):
        # Stack arrays for efficient operations
        time_data = np.stack([d["time_data"] for d in raw_data])
        clock_data = np.stack([d["clock_data"] for d in raw_data])
        pl_data = np.stack([d["pl_data"] for d in raw_data])

        # Calculate means and std in one pass
        mean_time = np.mean(time_data, axis=0)
        mean_clock = np.mean(clock_data, axis=0)
        mean_pl = np.mean(pl_data, axis=0)
        std_pl = np.std(pl_data, axis=0, ddof=1)  # Use sample standard deviation
    else:
        # Single measurement - no need for averaging
        mean_time = np.array(raw_data["time_data"])
        mean_clock = np.array(raw_data["clock_data"])
        mean_pl = np.array(raw_data["pl_data"])
        std_pl = np.zeros_like(mean_pl)  # No std dev for single measurement

        # Reshape for consistency
        time_data = mean_time[np.newaxis, :]
        clock_data = mean_clock[np.newaxis, :]
        pl_data = mean_pl[np.newaxis, :]

    # Derive current from clock signal using rising edge detection
    # Find rising edges in the clock signal
    rising_edges = np.where(np.diff(mean_clock > trigger_threshold) > 0)[0]

    # Initialize current array (starts OFF because we capture with pre-trigger)
    mean_current = np.zeros_like(mean_clock)

    # Set initial state (OFF)
    current_state = False

    # Process regions between rising edges
    for i in range(len(rising_edges)):
        start_idx = 0 if i == 0 else rising_edges[i - 1]
        end_idx = rising_edges[i]

        # Set current value for this region
        if current_state:
            mean_current[start_idx:end_idx] = current

        # Toggle state for next region
        current_state = not current_state

    # Handle the final region after the last rising edge
    if rising_edges.size > 0:
        mean_current[rising_edges[-1] :] = 0 if current_state else current

    # If no rising edges were detected, assume all ON
    if rising_edges.size == 0:
        mean_current[:] = current

    # Prepare results dictionary with raw data
    results = {
        "mean_time": mean_time,
        "mean_clock": mean_clock,
        "mean_pl": mean_pl,
        "std_pl": std_pl,
        "mean_current": mean_current,
        "time_data": time_data,
        "clock_data": clock_data,
        "pl_data": pl_data,
    }

    # Apply detrending if requested - BEFORE analysis
    trend_info = None
    if detrend:
        detrended_pl, trend_info = detrend_pl_signal(
            mean_time, mean_pl, mean_clock, trigger_threshold
        )
        # Store both raw and detrended data
        results["mean_pl_raw"] = mean_pl.copy()
        results["mean_pl"] = detrended_pl
        results["trend_info"] = trend_info

        # Also detrend individual traces if available
        if isinstance(raw_data, list) and len(raw_data) > 1:
            detrended_pl_data = np.zeros_like(results["pl_data"])
            for i in range(len(raw_data)):
                detrended, _ = detrend_pl_signal(
                    results["time_data"][i],
                    results["pl_data"][i],
                    results["clock_data"][i],
                    trigger_threshold,
                )
                detrended_pl_data[i] = detrended

            # Store raw and detrended individual traces
            results["pl_data_raw"] = results["pl_data"].copy()
            results["pl_data"] = detrended_pl_data

            # Recalculate standard deviation with detrended data
            results["std_pl"] = np.std(detrended_pl_data, axis=0, ddof=1)

        logger.info("Detrending applied - analysis will be performed on detrended data")

    # Run appropriate analysis based on fit_exponential flag
    if fit_exponential:
        transition_analysis = MPLFitter.analyze_and_fit(
            mean_time,
            results["mean_pl"],
            mean_clock,
            trigger_threshold,
            plot_pulses,
            use_cache=True,
            fit_type=fit_type,
        )
    else:
        transition_analysis = MPLFitter.analyze_transitions(
            mean_time,
            results["mean_pl"],
            mean_clock,
            trigger_threshold,
            plot_pulses,
            use_cache=True,
        )

    # Combine with basic response analysis
    basic_analysis = analyze_mpl_response(results["mean_pl"], mean_current)
    results["analysis"] = {**basic_analysis, **transition_analysis}

    return results


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
    pl_range: float,
    pl_coupling: str,
    averages: int,
    coil_resistance: float,
    additional_resistance: float,
    current: float,
    source_mode: str,
    voltage_limit: Optional[float],
    downsample_ratio: int,
    trigger_threshold: float,
    detrend: bool = False,
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
    pl_range : float
        PL channel voltage range
    pl_coupling : str
        PL channel coupling
    averages : int
        Number of measurements averaged
    coil_resistance : float
        Coil resistance in ohms
    current : float
        Applied current in amps
    downsample_ratio : int
        Hardware downsampling ratio
    trigger_threshold : float
        Trigger threshold voltage
    no_show : bool, optional
        Whether to suppress plot display, by default False
    """
    from qscope.system import System
    from qscope.util.save import NumpyEncoder, _get_base_path

    # Create system object and get base path
    system = System(system_name)
    base_path = _get_base_path(system, project_name, "mpl_trig")

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

    # Clean up analysis results
    if "analysis" in results:
        # Handle fit parameters specifically
        for fit_key in ["rise_fit_params", "fall_fit_params"]:
            if fit_key in results["analysis"]:
                # Clean each fit parameter dict
                if isinstance(results["analysis"][fit_key], list):
                    results["analysis"][fit_key] = [
                        clean_for_json(param) for param in results["analysis"][fit_key]
                    ]

    # Prepare metadata dictionary (everything except raw data arrays)
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "command": get_command_string(),
        "parameters": {
            "frequency": frequency,
            "duration": duration,
            "sample_rate": sample_rate,
            "pl_range": pl_range,
            "pl_coupling": pl_coupling,
            "averages": averages,
            "coil_resistance": coil_resistance,
            "additional_resistance": additional_resistance,
            "total_resistance": coil_resistance + additional_resistance,
            "current": current,
            "source_mode": source_mode,
            "voltage_limit": voltage_limit,
            "trigger_threshold": trigger_threshold,
            "downsample_ratio": downsample_ratio,
            "detrend": detrend,
        },
        "analysis": clean_for_json(results["analysis"]),
        # Include processed results but not the raw data arrays
        "results": clean_for_json(
            {
                k: v
                for k, v in results.items()
                if k
                not in ["time_data", "clock_data", "pl_data", "pl_data_raw", "analysis"]
                and not isinstance(v, np.ndarray)
            }
        ),
    }

    # Add trend information if available
    if "trend_info" in results:
        metadata["detrending"] = clean_for_json(
            {
                k: v
                for k, v in results["trend_info"].items()
                if k != "trend"  # Don't include the full trend array in metadata
            }
        )

    # Save data and figure in parallel
    def save_data():
        # Save metadata as JSON
        with open(f"{base_path}.json", "w") as f:
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)

        # Save raw data as compressed NumPy file
        save_data = {
            "time_data": results["time_data"],
            "clock_data": results["clock_data"],
            "mean_time": results["mean_time"],
            "mean_clock": results["mean_clock"],
            "std_pl": results["std_pl"],
        }

        # Handle raw and detrended PL data
        if "mean_pl_raw" in results:
            save_data["mean_pl_raw"] = results["mean_pl_raw"]
            save_data["mean_pl"] = results["mean_pl"]
            if "trend_info" in results and "trend" in results["trend_info"]:
                save_data["trend"] = results["trend_info"]["trend"]
        else:
            save_data["mean_pl"] = results["mean_pl"]

        # Handle individual traces
        if "pl_data_raw" in results:
            save_data["pl_data_raw"] = results["pl_data_raw"]
            save_data["pl_data"] = results["pl_data"]
        else:
            save_data["pl_data"] = results["pl_data"]

        np.savez_compressed(f"{base_path}_raw.npz", **save_data)

        logger.debug(f"Raw data saved to {base_path}_raw.npz")

    def save_figure():
        fig, _ = plot_mpl_results(
            results=results,
            sample_rate=sample_rate,
            downsample_ratio=downsample_ratio,
            current=current,
            frequency=frequency,
            averages=averages,
            coil_resistance=coil_resistance,
            trigger_threshold=trigger_threshold,
            show_individual=True,
            no_show=True,
        )
        fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    with ThreadPoolExecutor(max_workers=2) as executor:
        data_future = executor.submit(save_data)
        fig_future = executor.submit(save_figure)

        # Wait for both to complete and check for errors
        data_future.result()
        fig_future.result()

    logger.info(f"Metadata saved to {base_path}.json")
    logger.info(f"Raw data saved to {base_path}_raw.npz")
    logger.info(f"Figure saved to {base_path}.png")

def trigtrace(
    system_name: str,
    current: float,
    frequency: float,
    duration: float,
    averages: int,
    sample_rate: float,
    pl_range: float,
    pl_coupling: str,
    trigger_threshold: float,
    downsample_ratio: int,
    trigger_source: str,
    source_mode: str,
    coil_resistance: float,
    additional_resistance: float,
    voltage_limit: Optional[float],
    force_sourcing: bool,
    smu_address: str,
    project_name: str,
    fit: bool = False,
    fit_type: str = "single",
    plot_pulses: bool = False,
    detrend: bool = False,
    save: bool = True,
    show_individual: bool = True,
    log_to_file: bool = True,
    log_to_stdout: bool = True,
    log_path: str = "",
    clear_prev_log: bool = True,
    log_level: str = DEFAULT_LOGLEVEL,
) -> None:
    """Measure PL response to triggered magnetic field changes.

    Physical Connections:
    AWG Out ─┬─> Channel A (monitor clock)
             └─> SMU Digital Input (pin 3)

    PL In ────> Channel B (photodetector)

    Pre-measurement checks:

    - Validates all device connections
    - Checks signal levels are within range
    - Verifies trigger configuration

    \b
    Parameter Groups:
      Measurement:     Current, frequency, duration, averages
      Hardware:        Sample rate, ranges, coupling, trigger settings
      Data Collection: Save options, trace display
      Logging:         File/console logging, paths, levels
    """
    # Setup logging
    setup_logging(
        log_path=log_path,
        clear_prev_log=clear_prev_log,
        log_to_file=log_to_file,
        log_to_stdout=log_to_stdout,
        log_level=log_level,
    )

    logger.info("Starting triggered MPL measurement")
    if duration < 0:
        duration = 1 / frequency
    logger.info(f"Parameters: {frequency=}Hz, {duration=}s, {current=}A, {averages=}")

    # Pre-measurement validation
    if downsample_ratio < 1:
        raise click.BadParameter("Downsample ratio must be >= 1")

    # Calculate measurement parameters
    total_samples = int(duration * sample_rate)
    if total_samples // downsample_ratio > 1e8:  # Arbitrary limit
        raise click.BadParameter("Requested duration/sample rate too large")

    # Calculate actual sample rate after downsampling
    actual_rate = sample_rate / downsample_ratio
    logger.info(
        f"Sampling rates - measurement: {sample_rate:.2e} Hz, "
        f"after downsampling: {actual_rate:.2e} Hz"
    )

    try:
        with managed_devices(
            smu_address=smu_address,
            current=current,
            coil_resistance=coil_resistance,
            additional_resistance=additional_resistance,
            voltage_limit=voltage_limit,
            source_mode=source_mode,
            force_sourcing=force_sourcing,
        ) as (smu, scope):
            # Validate device connections
            if not scope.is_responding():
                raise click.ClickException("Picoscope not responding")
            if not smu.is_responding():
                raise click.ClickException("SMU not responding")

            # Configure hardware downsampling
            scope.set_downsampling(
                mode="AVERAGE" if downsample_ratio > 1 else "NONE",
                downsample_ratio=downsample_ratio,
            )
            # Configure AWG output (but don't start yet)
            logger.debug("Configuring AWG (but not starting yet)")
            signal_config = SignalConfig(
                frequency=frequency * 2,  # Double the frequency
                amplitude=1.5,
                offset=0.5,
            )
            scope.config_signal_generator(signal_config)

            # Configure scope channels
            logger.debug("Configuring scope channels")
            scope.configure_channels(
                channels=[0, 1],  # A and B
                ranges=[10.0, pl_range],
                coupling=["DC", pl_coupling],
            )

            # Calculate pre-trigger samples as 10% of one clock cycle
            clock_period = 1.0 / (
                frequency * 2
            )  # Period of the clock signal (not the field switching)
            pre_trigger_time = 0.1 * clock_period  # 10% of one clock period
            actual_pre_trigger_samples = int(pre_trigger_time * sample_rate)
            logger.info(
                f"Using {actual_pre_trigger_samples} pre-trigger samples (10% of one clock period)"
            )

            # Configure SMU triggering
            logger.debug(f"Configuring SMU trigger")
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
                trigger_config = TriggerConfig(
                    pin=3,
                    edge=TriggerEdge.RISING,  # Only trigger on rising edges
                    count=int(effective_duration * frequency * 2),
                    high_value=required_voltage,
                    low_value=0.0,
                    resistance=total_resistance,  # Pass the actual resistance for current calculations
                )

            # Storage for raw data
            raw_data = []

            # Perform measurements
            logger.debug(f"Starting {averages} measurements")
            with trange(averages, unit="avg") as pbar:
                for avg_idx in pbar:
                    # Make sure any previous trigger model is stopped
                    smu.stop_trigger_model()
                    smu.configure_trigger(trigger_config)

                    scope.start_block_capture(
                        sample_interval=1.0 / sample_rate,
                        num_samples=int(duration * sample_rate),
                        trigger_enabled=trigger_source == "CLOCK",
                        trigger_channel=0 if trigger_source == "CLOCK" else None,
                        trigger_threshold=trigger_threshold,
                        trigger_direction="RISING",
                        pre_trigger_samples=actual_pre_trigger_samples,
                    )
                    # Start SMU trigger model
                    logger.debug(
                        f"Starting SMU trigger model for measurement {avg_idx + 1}"
                    )
                    smu.start_trigger_model()
                    # Now start the signal generator to begin the measurement
                    logger.debug("Starting signal generator")
                    scope.start_signal_generator()

                    try:
                        # Get scope data with timeout
                        time_data, (clock_data, pl_data) = scope.get_block_data(
                            timeout_s=max(5.0, duration * 2.5)
                        )

                        # Shift time array so trigger point is at t=0
                        if actual_pre_trigger_samples > 0:
                            trigger_time = time_data[
                                actual_pre_trigger_samples // downsample_ratio
                            ]
                            time_data = time_data - trigger_time
                            logger.debug(
                                f"Shifted time data so trigger point (sample {actual_pre_trigger_samples}) is at t=0"
                            )

                        # Stop the trigger model after data collection
                        logger.debug(f"Stopping SMU trigger model")
                        smu.stop_trigger_model()

                        scope.stop_signal_generator()

                        # Check for SMU compliance
                        if smu.check_compliance():
                            logger.warning(
                                f"SMU compliance reached during measurement {avg_idx + 1}"
                            )
                            if click.confirm("Continue measurement?", default=True):
                                continue
                            else:
                                break

                        # Store data for this measurement
                        raw_data.append(
                            {
                                "time_data": time_data,
                                "clock_data": clock_data,
                                "pl_data": pl_data,
                            }
                        )

                        # Update progress description
                        pbar.set_description(f"Avg #{avg_idx + 1}")

                    except Exception as e:
                        logger.exception(f"Error in measurement {avg_idx + 1}")
                        if not click.confirm("Retry measurement?", default=False):
                            break

            # Ensure trigger model is stopped
            logger.debug("Ensuring trigger model is stopped")
            smu.stop_trigger_model()

            # Process results
            logger.debug("Processing results")
            results = process_results(
                raw_data=raw_data,
                coil_resistance=coil_resistance,
                additional_resistance=additional_resistance,
                current=current,
                source_mode=source_mode,
                trigger_threshold=trigger_threshold,
                fit_exponential=fit,
                plot_pulses=plot_pulses,
                fit_type=fit_type,
                detrend=detrend,
            )
            if "analysis" in results:
                click.echo("\nResults:")
                # Extract only the key parameters from the analysis dictionary
                key_params = {
                    "mean_rise_time": "Field ON response time",
                    "mean_fall_time": "Field OFF response time",
                    "mean_contrast": "Contrast",
                    "mean_rise_tau": "Field ON time constant",
                    "mean_fall_tau": "Field OFF time constant",
                }

                for key, label in key_params.items():
                    if key in results["analysis"] and not np.isnan(
                        results["analysis"][key]
                    ):
                        # Format time values in milliseconds
                        if "time" in key or "tau" in key:
                            value = f"{results['analysis'][key] * 1000:.2f} ms"
                        # Format contrast as percentage
                        elif key == "mean_contrast":
                            value = f"{results['analysis'][key]:.2f}%"
                        else:
                            value = results["analysis"][key]
                        click.echo(f"  {label}: {value}")

            # Plot results
            logger.debug("Plotting results")
            plot_mpl_results(
                results=results,
                sample_rate=sample_rate,
                downsample_ratio=downsample_ratio,
                current=current,
                frequency=frequency,
                averages=averages,
                coil_resistance=coil_resistance,
                trigger_threshold=trigger_threshold,
                show_individual=show_individual,
                detrend=detrend,
            )

            # Save if requested
            if save:
                logger.debug("Saving results")
                save_results(
                    results=results,
                    system_name=system_name,
                    project_name=project_name,
                    frequency=frequency,
                    duration=duration,
                    sample_rate=sample_rate,
                    pl_range=pl_range,
                    pl_coupling=pl_coupling,
                    averages=averages,
                    coil_resistance=coil_resistance,
                    additional_resistance=additional_resistance,
                    current=current,
                    source_mode=source_mode,
                    voltage_limit=voltage_limit,
                    downsample_ratio=downsample_ratio,
                    trigger_threshold=trigger_threshold,
                    detrend=detrend,
                )
    except Exception as e:
        click.echo(f"Error during measurement: {e}", err=True)
        logger.exception("Measurement failed")



if __name__ == "__main__":


    trigtrace(
        system_name = "Zyla",
        current = 0.45, # Amps
        frequency = 10, # Hz
        duration= 10, # seconds
        averages = 10, # Number of measurements to average
        sample_rate = 1e6, # Hz
        pl_range = 0.5, # V
        pl_coupling = "DC", # Coupling type for PL channel
        trigger_threshold = 2.5, # V
        downsample_ratio = 1000, # Downsample ratio for hardware downsampling [int]
        trigger_source = "FREE", # Trigger source for the measurement [str] [FREE|CLOCK]
        source_mode = "voltage", # Source mode for the SMU [str] [current|voltage]
        coil_resistance = 27.0, # Resistance of the coil [float] [Ohm]
        additional_resistance = 0.0, # Additional resistance in series with the coil [float] [Ohm]
        force_sourcing = False, # Force sourcing mode for the SMU [bool]
        smu_address = "GPIB0::24::INSTR", # Address of the SMU [str]
        project_name = "mpl_CQDs_", # Project name for saving results [str]
        fit = False,
        fit_type = "single",
        plot_pulses = False,
        detrend  = False,
        save  = True,
        show_individual = True,
        log_to_file = True,
        log_to_stdout = True,
        log_path = "",
        clear_prev_log = True,
        log_level = DEFAULT_LOGLEVEL,
    )