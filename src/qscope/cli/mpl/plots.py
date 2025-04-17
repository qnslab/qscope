"""Plotting utilities for MPL data visualization."""

from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib import axes

from qscope.fitting.mpl import MPLFitter


# Define global exponential functions for backward compatibility
def global_rise_exp(t, a, tau, c, t0):
    """Exponential rise function with absolute time reference."""
    return a * (1 - np.exp(-(t - t0) / tau)) + c


def global_fall_exp(t, a, tau, c, t0):
    """Exponential decay function with absolute time reference."""
    return a * np.exp(-(t - t0) / tau) + c


def plot_mpl_results(
    results: Dict[str, Any],
    sample_rate: float,
    downsample_ratio: int,
    current: float,
    frequency: float,
    averages: int,
    coil_resistance: float,
    trigger_threshold: float,
    show_individual: bool = True,
    detrend: bool = False,
    no_show: bool = False,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot measurement results.

    Parameters
    ----------
    results : Dict[str, Any]
        Processed measurement results
    sample_rate : float
        Original sample rate in Hz
    downsample_ratio : int
        Hardware downsampling ratio
    current : float
        Applied current in amps
    frequency : float
        Signal frequency in Hz
    averages : int
        Number of measurements averaged
    coil_resistance : float
        Coil resistance in ohms
    trigger_threshold : float
        Trigger threshold voltage
    show_individual : bool, optional
        Whether to show individual traces, by default True
    no_show : bool, optional
        Whether to suppress plot display, by default False

    Returns
    -------
    Tuple[plt.Figure, List[plt.Axes]]
        Figure and axes objects
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Add measurement parameters as a title
    title = (
        f"Triggered MPL Measurement\n"
        f"Measurement rate: {sample_rate:.2e} Hz → {sample_rate / downsample_ratio:.2e} Hz "
        f"(÷{downsample_ratio})\n"
        f"Coil current: {current * 1000:.0f} mA, {frequency:.2f} Hz switching, "
        f"AWG: {frequency * 2:.2e} Hz, {averages} averages, R={coil_resistance:.2f} Ω"
    )
    fig.suptitle(title, y=0.98)

    # Pre-compute slices for plotting
    window_size = 5
    pad = window_size // 2
    valid_slice = slice(pad, -pad if pad > 0 else None)

    plot_time = results["mean_time"][valid_slice]
    plot_pl = results["mean_pl"][valid_slice]
    plot_clock = results["mean_clock"][valid_slice]

    # Detect rising edges in clock signal to determine current state
    # First create a binary signal where clock is above threshold
    clock_high = plot_clock > trigger_threshold

    # Find rising edges (where signal transitions from low to high)
    rising_edges = np.where(np.diff(clock_high.astype(int)) > 0)[0]

    # Create effective current signal (starts OFF due to pre-trigger capture)
    current_state = np.zeros_like(plot_clock)
    current_on = False  # Start with current OFF due to pre-trigger

    # Toggle state at each rising edge
    for edge_idx in rising_edges:
        current_on = not current_on  # Toggle state at each rising edge
        if edge_idx + 1 < len(current_state):
            current_state[edge_idx + 1 :] = current if current_on else 0

    # Create regions for shading
    current_regions = []
    region_start = None
    for i in range(len(current_state)):
        if i > 0:
            if current_state[i] > 0 and current_state[i - 1] == 0:
                # Current just turned on
                region_start = plot_time[i]
            elif current_state[i] == 0 and current_state[i - 1] > 0:
                # Current just turned off
                if region_start is not None:
                    current_regions.append((region_start, plot_time[i]))
                    region_start = None

    # Add final region if measurement ends with current on
    if region_start is not None:
        current_regions.append((region_start, plot_time[-1]))

    # Plot 1: Clock Signal
    ax1.plot(plot_time, plot_clock, "b-", label="Clock Signal")
    ax1.axhline(
        y=trigger_threshold,
        color="k",
        linestyle="--",
        label=f"Trigger ({trigger_threshold}V)",
    )
    ax1.axhline(y=0.7, color="xkcd:grey", linestyle="dotted", label="Max logic Off")
    ax1.axhline(y=3.3, color="xkcd:dark grey", linestyle="dotted", label="Min logic On")

    # Shade regions where current is on
    for start, end in current_regions:
        ax1.axvspan(start, end, color="g", alpha=0.1)

    # Annotate all detected edges in the clock signal
    # First create a binary signal where clock is above threshold
    clock_binary = plot_clock > trigger_threshold

    # Find all edges (transitions in the binary signal)
    edges = np.where(np.diff(clock_binary.astype(int)) != 0)[0]

    # Separate rising and falling edges
    rising_edges = np.where(np.diff(clock_binary.astype(int)) > 0)[0]
    falling_edges = np.where(np.diff(clock_binary.astype(int)) < 0)[0]

    # Identify magnetic field transitions
    field_on_edges = (
        rising_edges[::2] if len(rising_edges) > 0 else []
    )  # Every other rising edge (0, 2, 4...)
    field_off_edges = (
        rising_edges[1::2] if len(rising_edges) > 1 else []
    )  # Every other rising edge (1, 3, 5...)

    # Mark field ON edges with green triangles
    for edge in field_on_edges:
        if edge < len(plot_time) - 1:  # Ensure we don't go out of bounds
            ax1.plot(
                plot_time[edge],
                plot_clock[edge],
                "g^",
                markersize=8,
                label="Field ON" if edge == field_on_edges[0] else "",
            )

    # Mark field OFF edges with red triangles
    for edge in field_off_edges:
        if edge < len(plot_time) - 1:  # Ensure we don't go out of bounds
            ax1.plot(
                plot_time[edge],
                plot_clock[edge],
                "rv",
                markersize=8,
                label="Field OFF"
                if len(field_off_edges) > 0 and edge == field_off_edges[0]
                else "",
            )

    if averages > 1 and show_individual:
        # Plot first and last clock traces
        ax1.plot(
            results["time_data"][0, valid_slice],
            results["clock_data"][0, valid_slice],
            "xkcd:sky blue",
            alpha=0.75,
            lw=0.75,
            ls="dashed",
            label="First trace",
        )
        ax1.plot(
            results["time_data"][-1, valid_slice],
            results["clock_data"][-1, valid_slice],
            "xkcd:azure",
            alpha=0.75,
            lw=0.75,
            ls="dotted",
            label="Last trace",
        )
    ax1.grid(True)
    ax1.set_ylabel("Clock (V)")
    ax1.legend(loc="upper right")

    # Plot 2: PL Signal
    if "mean_pl_raw" in results and "trend_info" in results:
        # Plot raw data with trend line
        raw_pl = results["mean_pl_raw"][valid_slice]
        trend = results["trend_info"]["trend"][valid_slice]

        ax2.plot(plot_time, raw_pl, "xkcd:grey", alpha=0.5, label="<Raw PL Signal>")
        ax2.plot(
            plot_time,
            trend,
            "xkcd:grey",
            linestyle="--",
            alpha=0.7,
            label="Bleaching Trend",
        )
        ax2.plot(plot_time, plot_pl, "r-", label="Detrended PL Signal")

        # Add trend slope to plot title
        slope = results["trend_info"]["slope"]
        method = results["trend_info"]["method"]
        method_text = (
            "baseline points only" if method == "baseline_linear" else "all points"
        )
        slope_text = f"Trend: {slope:.2e} V/s ({method_text})"
        ax2.text(
            0.02,
            0.02,
            slope_text,
            transform=ax2.transAxes,
            bbox=dict(facecolor="white", alpha=0.7),
        )
    else:
        ax2.plot(plot_time, plot_pl, "r-", label="Average PL Signal")

    # Shade regions where current is on
    for start, end in current_regions:
        ax2.axvspan(
            start,
            end,
            color="g",
            alpha=0.1,
            label="Current On" if start == current_regions[0][0] else "",
        )

    if averages > 1:
        plot_std = results["std_pl"][valid_slice]
        ax2.fill_between(
            plot_time,
            plot_pl - plot_std,
            plot_pl + plot_std,
            color="xkcd:goldenrod",
            lw=0,
            alpha=0.2,
            label="±1σ",
        )

        if show_individual:
            # Plot individual traces more efficiently
            if "pl_data_raw" in results:
                # If we have raw data, plot it with lower opacity
                ax2.plot(
                    results["time_data"][0, valid_slice],
                    results["pl_data_raw"][0, valid_slice],
                    "xkcd:green",
                    alpha=0.4,
                    lw=0.75,
                    linestyle=":",
                    label="First trace (raw)",
                )
                ax2.plot(
                    results["time_data"][-1, valid_slice],
                    results["pl_data_raw"][-1, valid_slice],
                    "xkcd:hot pink",
                    alpha=0.4,
                    lw=0.75,
                    linestyle=":",
                    label="Last trace (raw)",
                )

            # Plot detrended individual traces
            ax2.plot(
                results["time_data"][0, valid_slice],
                results["pl_data"][0, valid_slice],
                "xkcd:forest green",
                alpha=0.6,
                lw=0.75,
                label="First trace"
                if "pl_data_raw" not in results
                else "First trace (detrended)",
            )
            ax2.plot(
                results["time_data"][-1, valid_slice],
                results["pl_data"][-1, valid_slice],
                "xkcd:purple",
                alpha=0.6,
                lw=0.75,
                label="Last trace"
                if "pl_data_raw" not in results
                else "Last trace (detrended)",
            )
    ax2.grid(True)
    ax2.set_ylabel("PL (V)")
    ax2.legend(loc="upper right")

    # Plot 3: Normalized Signals with Current State
    # Calculate PL contrast using the exact field ON edge points for baseline
    if len(field_on_edges) > 0:
        # Use the point right before the first field ON edge
        baseline_idx = max(0, field_on_edges[0] - 1)
        pl_baseline = plot_pl[baseline_idx - pad]  # Adjust for valid_slice offset
    else:
        # Fallback to first point
        pl_baseline = plot_pl[0]

    contrast = 100 * plot_pl / pl_baseline

    # Add transition analysis results if available
    if "analysis" in results and "mean_rise_time" in results["analysis"]:
        analysis = results["analysis"]
        if not np.isnan(analysis.get("mean_rise_time", np.nan)):
            # Add text box with analysis results
            analysis_text = (
                f"Field ON response time (10-90%): {analysis['mean_rise_time'] * 1000:.2f} ms\n"
                f"Field OFF response time (10-90%): {analysis['mean_fall_time'] * 1000:.2f} ms\n"
                f"Contrast: {analysis['mean_contrast']:.3f}%"
            )

            # Add exponential fit results if available
            if "fit_type" in analysis and analysis["fit_type"] == "double":
                # Double exponential fit results
                if "mean_rise_tau1" in analysis and not np.isnan(
                    analysis["mean_rise_tau1"]
                ):
                    analysis_text += f"\nField ON response: τ1={analysis['mean_rise_tau1'] * 1000:.2f} ms, τ2={analysis['mean_rise_tau2'] * 1000:.2f} ms\n"
                    analysis_text += f"Field OFF response: τ1={analysis['mean_fall_tau1'] * 1000:.2f} ms, τ2={analysis['mean_fall_tau2'] * 1000:.2f} ms"
            elif "mean_rise_tau" in analysis and not np.isnan(
                analysis["mean_rise_tau"]
            ):
                # Single exponential fit results
                analysis_text += f"\nField ON response τ: {analysis['mean_rise_tau'] * 1000:.2f} ms\n"
                analysis_text += (
                    f"Field OFF response τ: {analysis['mean_fall_tau'] * 1000:.2f} ms"
                )
            ax3.text(
                0.02,
                0.98,
                analysis_text,
                transform=ax3.transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Annotate 10/90% points on the PL curve in ax2
            if "rise_10_indices" in analysis and analysis["rise_10_indices"]:
                for r10, r90 in zip(
                    analysis["rise_10_indices"], analysis["rise_90_indices"]
                ):
                    # Adjust indices to account for the valid_slice offset
                    adjusted_r10 = r10 - pad
                    adjusted_r90 = r90 - pad

                    if 0 <= adjusted_r10 < len(plot_time) and 0 <= adjusted_r90 < len(
                        plot_time
                    ):
                        # Get the actual PL values at these points
                        pl_10 = plot_pl[adjusted_r10]
                        pl_90 = plot_pl[adjusted_r90]

                        # Mark 10% field ON response point
                        ax2.plot(
                            plot_time[adjusted_r10],
                            pl_10,
                            "o",
                            color="blue",
                            markersize=4,
                            label="10% Field ON Response"
                            if r10 == analysis["rise_10_indices"][0]
                            else "",
                        )
                        # Mark 90% field ON response point
                        ax2.plot(
                            plot_time[adjusted_r90],
                            pl_90,
                            "o",
                            color="green",
                            markersize=4,
                            label="90% Field ON Response"
                            if r90 == analysis["rise_90_indices"][0]
                            else "",
                        )
                        # Draw a line connecting the points
                        ax2.plot(
                            [plot_time[adjusted_r10], plot_time[adjusted_r90]],
                            [pl_10, pl_90],
                            "-",
                            color="green",
                            alpha=0.5,
                            linewidth=1,
                        )

            if "fall_10_indices" in analysis and analysis["fall_10_indices"]:
                for f90, f10 in zip(
                    analysis["fall_90_indices"], analysis["fall_10_indices"]
                ):
                    # Adjust indices to account for the valid_slice offset
                    adjusted_f90 = f90 - pad
                    adjusted_f10 = f10 - pad

                    if 0 <= adjusted_f10 < len(plot_time) and 0 <= adjusted_f90 < len(
                        plot_time
                    ):
                        # Get the actual PL values at these points
                        pl_90 = plot_pl[adjusted_f90]
                        pl_10 = plot_pl[adjusted_f10]

                        # Mark 90% field OFF response point (first crossed when field turns off)
                        ax2.plot(
                            plot_time[adjusted_f90],
                            pl_90,
                            "o",
                            color="orange",
                            markersize=4,
                            label="90% Field OFF Response"
                            if f90 == analysis["fall_90_indices"][0]
                            else "",
                        )
                        # Mark 10% field OFF response point
                        ax2.plot(
                            plot_time[adjusted_f10],
                            pl_10,
                            "o",
                            color="purple",
                            markersize=4,
                            label="10% Field OFF Response"
                            if f10 == analysis["fall_10_indices"][0]
                            else "",
                        )
                        # Draw a line connecting the points
                        ax2.plot(
                            [plot_time[adjusted_f90], plot_time[adjusted_f10]],
                            [pl_90, pl_10],
                            "-",
                            color="purple",
                            alpha=0.5,
                            linewidth=1,
                        )

    # Add exponential fit curves if available
    if "analysis" in results and "rise_fit_params" in results["analysis"]:
        analysis = results["analysis"]

        # Define exponential functions for plotting
        def rise_exp(t, a, tau, c):
            return a * (1 - np.exp(-t / tau)) + c

        def fall_exp(t, a, tau, c):
            return a * np.exp(-t / tau) + c

        # Plot rise fits
        if "rise_fit_params" in analysis and "rise_indices" in analysis:
            for i, (params, start_idx) in enumerate(
                zip(analysis["rise_fit_params"], analysis["rise_indices"])
            ):
                if i < len(analysis.get("rise_10_indices", [])) and i < len(
                    analysis.get("rise_90_indices", [])
                ):
                    # Get segment time range
                    r10_idx = analysis["rise_10_indices"][i] - pad
                    r90_idx = analysis["rise_90_indices"][i] - pad

                    if 0 <= r10_idx < len(plot_time) and 0 <= r90_idx < len(plot_time):
                        # Create time array for fit curve - use the full segment width
                        # Find the corresponding segment in the analysis data
                        on_segments = analysis.get("on_segments", [])
                        if i < len(on_segments):
                            # Get the segment boundaries (adjusted for valid_slice offset)
                            on_edge, off_edge = on_segments[i]
                            segment_start = max(0, on_edge - pad)
                            segment_end = min(len(plot_time) - 1, off_edge - pad)
                        else:
                            # Fallback to using the 10% and 90% indices with some extension
                            segment_width = abs(r90_idx - r10_idx)
                            segment_start = max(0, r10_idx - segment_width)
                            segment_end = min(
                                len(plot_time) - 1, r90_idx + segment_width
                            )

                        # Ensure indices are within valid range
                        segment_start = max(0, min(segment_start, len(plot_time) - 1))
                        segment_end = max(0, min(segment_end, len(plot_time) - 1))

                        # Create time array spanning the full segment
                        t_start = plot_time[segment_start]
                        t_end = plot_time[segment_end]
                        t_fit = np.linspace(t_start, t_end, 100)

                        # Check if we have the complete parameter set
                        if isinstance(params, dict) and "function" in params:
                            # Use the stored function directly
                            y_fit = params["function"](t_fit)

                            # Get fit type and R-squared if available
                            fit_type = params.get("fit_type", "single")
                            r_squared = params.get("r_squared", None)

                            # Create appropriate label
                            if fit_type == "double":
                                label = f"Double Exp. ON Fit" if i == 0 else ""
                                if r_squared is not None:
                                    label += f" (R²={r_squared:.3f})" if i == 0 else ""
                            else:
                                label = f"Field ON Response Fit" if i == 0 else ""
                                if r_squared is not None:
                                    label += f" (R²={r_squared:.3f})" if i == 0 else ""
                        elif (
                            len(params) >= 4
                        ):  # We have all parameters including time offset
                            if isinstance(params, dict) and "is_rise" in params:
                                # Use the is_rise flag to determine which function to use
                                fit_type = params.get("fit_type", "single")

                                if fit_type == "double":
                                    if params["is_rise"]:
                                        y_fit = MPLFitter.global_double_rise_exp(
                                            t_fit, *params["params"]
                                        )
                                    else:
                                        y_fit = MPLFitter.global_double_fall_exp(
                                            t_fit, *params["params"]
                                        )
                                else:  # single exponential
                                    if params["is_rise"]:
                                        y_fit = MPLFitter.global_rise_exp(
                                            t_fit, *params["params"]
                                        )
                                    else:
                                        y_fit = MPLFitter.global_fall_exp(
                                            t_fit, *params["params"]
                                        )

                                label = "Field ON Response Fit" if i == 0 else ""
                                if "r_squared" in params:
                                    label += (
                                        f" (R²={params['r_squared']:.3f})"
                                        if i == 0
                                        else ""
                                    )
                                else:
                                    y_fit = global_fall_exp(t_fit, *params["params"])
                                label = "Field ON Response Fit" if i == 0 else ""
                            elif isinstance(params, (list, tuple)):
                                # Check if we have function type information
                                if len(params) > 4 and params[3] == 1:
                                    # This is a rise exponential
                                    y_fit = global_rise_exp(
                                        t_fit,
                                        params[0],
                                        params[1],
                                        params[2],
                                        params[4],
                                    )
                                else:
                                    # This is a fall exponential
                                    y_fit = global_fall_exp(
                                        t_fit,
                                        params[0],
                                        params[1],
                                        params[2],
                                        params[4],
                                    )
                                label = "Field ON Response Fit" if i == 0 else ""
                        else:
                            # Fallback for backward compatibility
                            logger.warning(
                                "Incomplete fit parameters, using fallback method"
                            )
                            if len(params) > 3 and params[3] == 1:
                                # This is a rise exponential
                                y_fit = rise_exp(t_fit - t_fit[0], *params[:3])
                            else:
                                # This is a fall exponential
                                y_fit = fall_exp(t_fit - t_fit[0], *params[:3])
                            label = "Field ON Response Fit" if i == 0 else ""

                        # Plot fit curve
                        ax2.plot(
                            t_fit,
                            y_fit,
                            ":",
                            color="k",
                            alpha=0.7,
                            linewidth=1.5,
                            label=label,
                        )

        # Plot fall fits
        if "fall_fit_params" in analysis and "fall_indices" in analysis:
            for i, (params, start_idx) in enumerate(
                zip(analysis["fall_fit_params"], analysis["fall_indices"])
            ):
                if i < len(analysis.get("fall_10_indices", [])) and i < len(
                    analysis.get("fall_90_indices", [])
                ):
                    # Get segment time range
                    f90_idx = analysis["fall_90_indices"][i] - pad
                    f10_idx = analysis["fall_10_indices"][i] - pad

                    if 0 <= f90_idx < len(plot_time) and 0 <= f10_idx < len(plot_time):
                        # Create time array for fit curve - use the full segment width
                        # Find the corresponding segment in the analysis data
                        off_segments = analysis.get("off_segments", [])
                        if i < len(off_segments):
                            # Get the segment boundaries (adjusted for valid_slice offset)
                            off_edge, next_on_edge = off_segments[i]
                            segment_start = max(0, off_edge - pad)
                            segment_end = min(len(plot_time) - 1, next_on_edge - pad)
                        else:
                            # Fallback to using the 10% and 90% indices with some extension
                            segment_width = abs(f10_idx - f90_idx)
                            segment_start = max(0, f90_idx - segment_width)
                            segment_end = min(
                                len(plot_time) - 1, f10_idx + segment_width
                            )

                        # Ensure indices are within valid range
                        segment_start = max(0, min(segment_start, len(plot_time) - 1))
                        segment_end = max(0, min(segment_end, len(plot_time) - 1))

                        # Create time array spanning the full segment
                        t_start = plot_time[segment_start]
                        t_end = plot_time[segment_end]
                        t_fit = np.linspace(t_start, t_end, 100)

                        # Check if we have the complete parameter set
                        if isinstance(params, dict) and "function" in params:
                            # Use the stored function directly
                            y_fit = params["function"](t_fit)

                            # Get fit type and R-squared if available
                            fit_type = params.get("fit_type", "single")
                            r_squared = params.get("r_squared", None)

                            # Create appropriate label
                            if fit_type == "double":
                                label = (
                                    f"Double Exp. OFF Fit"
                                    if i == 0 and "rise_fit_params" not in analysis
                                    else ""
                                )
                                if r_squared is not None:
                                    label += (
                                        f" (R²={r_squared:.3f})"
                                        if i == 0 and "rise_fit_params" not in analysis
                                        else ""
                                    )
                            else:
                                label = (
                                    f"Field OFF Response Fit"
                                    if i == 0 and "rise_fit_params" not in analysis
                                    else ""
                                )
                                if r_squared is not None:
                                    label += (
                                        f" (R²={r_squared:.3f})"
                                        if i == 0 and "rise_fit_params" not in analysis
                                        else ""
                                    )
                        elif (
                            len(params) >= 4
                        ):  # We have all parameters including time offset
                            if isinstance(params, dict) and "is_rise" in params:
                                # Use the is_rise flag to determine which function to use
                                fit_type = params.get("fit_type", "single")

                                if fit_type == "double":
                                    if params["is_rise"]:
                                        y_fit = MPLFitter.global_double_rise_exp(
                                            t_fit, *params["params"]
                                        )
                                    else:
                                        y_fit = MPLFitter.global_double_fall_exp(
                                            t_fit, *params["params"]
                                        )
                                else:  # single exponential
                                    if params["is_rise"]:
                                        y_fit = MPLFitter.global_rise_exp(
                                            t_fit, *params["params"]
                                        )
                                    else:
                                        y_fit = MPLFitter.global_fall_exp(
                                            t_fit, *params["params"]
                                        )

                                label = (
                                    "Field OFF Response Fit"
                                    if i == 0 and "rise_fit_params" not in analysis
                                    else ""
                                )
                                if "r_squared" in params:
                                    label += (
                                        f" (R²={params['r_squared']:.3f})"
                                        if i == 0 and "rise_fit_params" not in analysis
                                        else ""
                                    )
                                else:
                                    y_fit = global_fall_exp(t_fit, *params["params"])
                                label = (
                                    "Field OFF Response Fit"
                                    if i == 0 and "rise_fit_params" not in analysis
                                    else ""
                                )
                            elif isinstance(params, (list, tuple)):
                                # Check if we have function type information
                                if len(params) > 4 and params[3] == 1:
                                    # This is a rise exponential
                                    y_fit = global_rise_exp(
                                        t_fit,
                                        params[0],
                                        params[1],
                                        params[2],
                                        params[4],
                                    )
                                else:
                                    # This is a fall exponential
                                    y_fit = global_fall_exp(
                                        t_fit,
                                        params[0],
                                        params[1],
                                        params[2],
                                        params[4],
                                    )
                                label = (
                                    "Field OFF Response Fit"
                                    if i == 0 and "rise_fit_params" not in analysis
                                    else ""
                                )
                        else:
                            # Fallback for backward compatibility
                            logger.warning(
                                "Incomplete fit parameters, using fallback method"
                            )
                            if len(params) > 3 and params[3] == 1:
                                # This is a rise exponential
                                y_fit = rise_exp(t_fit - t_fit[0], *params[:3])
                            else:
                                # This is a fall exponential
                                y_fit = fall_exp(t_fit - t_fit[0], *params[:3])
                            label = (
                                "Field OFF Response Fit"
                                if i == 0 and "rise_fit_params" not in analysis
                                else ""
                            )

                        # Plot fit curve
                        ax2.plot(
                            t_fit,
                            y_fit,
                            ":",
                            color="k",
                            alpha=0.7,
                            linewidth=1.5,
                            label=label,
                        )

    # Plot current state as step function
    ax3.step(
        plot_time,
        current_state / current,
        "g-",
        where="post",
        label="Current State",
        alpha=0.8,
    )

    # Shade regions where current is on
    for start, end in current_regions:
        ax3.axvspan(start, end, color="g", alpha=0.1)

    # Add secondary y-axis for PL contrast
    ax3_pl = ax3.twinx()
    contrast_min, contrast_max = np.min(contrast), np.max(contrast)
    contrast_range = contrast_max - contrast_min
    padding = 0.05 * contrast_range

    # Plot the contrast line
    ax3_pl.plot(plot_time, contrast, "r-", label="PL Contrast")

    # Add fill_between to highlight the contrast
    baseline = 100  # Since we normalized to 100%
    for start, end in current_regions:
        # Find indices within this region
        region_mask = (plot_time >= start) & (plot_time <= end)
        if np.any(region_mask):
            ax3_pl.fill_between(
                plot_time[region_mask],
                baseline,
                contrast[region_mask],
                color="r",
                alpha=0.2,
                label="Contrast Change" if start == current_regions[0][0] else "",
            )

    ax3_pl.set_ylabel("Contrast (%)")
    ax3_pl.set_ylim(contrast_min - padding, contrast_max + padding)

    # Set primary y-axis limits and labels
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_ylabel("Gate")
    ax3.set_xlabel("Time (s)")

    # Create combined legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_pl.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()

    if not no_show:
        plt.show()

    return fig, [ax1, ax2, ax3]
