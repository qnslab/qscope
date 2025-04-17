"""Fitting utilities for magnetophotoluminescence (MPL) data analysis.

This module provides tools for analyzing transitions in MPL data, including:
- 10-90% rise/fall time analysis
- Exponential curve fitting for time constants
- Combined analysis workflows

The main class (MPLFitter) handles all fitting operations with both
static methods for individual operations and combined workflows.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.optimize import curve_fit


class MPLFitter:
    """Handles fitting operations for MPL transition data.

    This class provides methods for analyzing transitions in MPL data,
    including 10-90% rise/fall time analysis and exponential curve fitting.

    Methods
    -------
    analyze_transitions
        Analyze PL transitions using clock signal for edge detection
    fit_exponential_curves
        Fit exponential curves to rise and fall transitions
    analyze_and_fit
        Perform both transition analysis and exponential fitting in one operation

    Attributes
    ----------
    _cache : Dict[str, Any]
        Cache for storing analysis results to avoid redundant calculations
    """

    # Class-level cache for storing analysis results
    _cache = {}

    @staticmethod
    def rise_exp(
        t: NDArray[np.float64], a: float, tau: float, c: float
    ) -> NDArray[np.float64]:
        """Exponential rise function: f(t) = a * (1 - exp(-t/tau)) + c

        Parameters
        ----------
        t : np.ndarray
            Time points
        a : float
            Amplitude
        tau : float
            Time constant
        c : float
            Offset

        Returns
        -------
        np.ndarray
            Function values
        """
        return a * (1 - np.exp(-t / tau)) + c

    @staticmethod
    def fall_exp(
        t: NDArray[np.float64], a: float, tau: float, c: float
    ) -> NDArray[np.float64]:
        """Exponential decay function: f(t) = a * exp(-t/tau) + c

        Parameters
        ----------
        t : np.ndarray
            Time points
        a : float
            Amplitude
        tau : float
            Time constant
        c : float
            Offset

        Returns
        -------
        np.ndarray
            Function values
        """
        return a * np.exp(-t / tau) + c

    @classmethod
    def _detect_clock_edges(
        cls, clock_data: NDArray[np.float64], trigger_threshold: float = 2.5
    ) -> Dict[str, Any]:
        """Detect edges in clock signal for transition analysis.

        Parameters
        ----------
        clock_data : NDArray[np.float64]
            Clock signal data
        trigger_threshold : float, optional
            Threshold for clock signal edge detection, by default 2.5

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - edges: All detected edges
            - rising_edges: Rising edge indices
            - falling_edges: Falling edge indices
            - field_on_edges: Field ON edge indices
            - field_off_edges: Field OFF edge indices
            - on_segments: List of (start, end) indices for ON segments
            - off_segments: List of (start, end) indices for OFF segments
        """
        # Create binary signal where clock is above threshold
        clock_binary = clock_data > trigger_threshold

        # Find all edges (transitions in the binary signal)
        edges = np.where(np.diff(clock_binary.astype(int)) != 0)[0]

        # Separate rising and falling edges
        rising_edges = np.where(np.diff(clock_binary.astype(int)) > 0)[0]
        falling_edges = np.where(np.diff(clock_binary.astype(int)) < 0)[0]

        # Log debug info about detected edges
        logger.debug(f"Clock signal analysis:")
        logger.debug(f"  Total edges detected: {len(edges)}")
        logger.debug(
            f"  Rising edges detected: {len(rising_edges)} at indices {rising_edges}"
        )
        logger.debug(
            f"  Falling edges detected: {len(falling_edges)} at indices {falling_edges}"
        )

        # Identify magnetic field transitions based on clock edges
        # Note: Data acquisition starts with field OFF due to pre-trigger capture
        # First rising edge in data = field ON, second rising edge = field OFF, etc.
        field_on_edges = rising_edges[::2]  # Every other rising edge (0, 2, 4...)
        field_off_edges = rising_edges[1::2]  # Every other rising edge (1, 3, 5...)

        logger.debug(
            f"  Field ON edges detected: {len(field_on_edges)} at indices {field_on_edges}"
        )
        logger.debug(
            f"  Field OFF edges detected: {len(field_off_edges)} at indices {field_off_edges}"
        )

        # Identify full pulse segments
        # For each ON edge, find the corresponding OFF edge
        on_segments = []
        for i, on_edge in enumerate(field_on_edges):
            if i < len(field_off_edges):
                off_edge = field_off_edges[i]
                on_segments.append((on_edge, off_edge))

        # For each OFF edge, find the next ON edge or use the end of the data
        off_segments = []
        for i, off_edge in enumerate(field_off_edges):
            if i + 1 < len(field_on_edges):
                next_on_edge = field_on_edges[i + 1]
            else:
                # Use the end of the data as the end of the last OFF segment
                next_on_edge = len(clock_data) - 1
            off_segments.append((off_edge, next_on_edge))

        logger.debug(
            f"  Identified {len(on_segments)} ON segments and {len(off_segments)} OFF segments"
        )

        return {
            "edges": edges,
            "rising_edges": rising_edges,
            "falling_edges": falling_edges,
            "field_on_edges": field_on_edges,
            "field_off_edges": field_off_edges,
            "on_segments": on_segments,
            "off_segments": off_segments,
        }

    @classmethod
    def _analyze_pulse_segment(
        cls,
        time_data: NDArray[np.float64],
        pl_data: NDArray[np.float64],
        start_idx: int,
        end_idx: int,
        is_on_segment: bool,
        plot_pulses: bool = False,
    ) -> Tuple[float, float, int, int]:
        """Analyze a single pulse segment for transition timing and contrast.

        Parameters
        ----------
        time_data : NDArray[np.float64]
            Time points
        pl_data : NDArray[np.float64]
            PL signal data
        start_idx : int
            Start index of the pulse segment
        end_idx : int
            End index of the pulse segment
        is_on_segment : bool
            Whether this is an ON segment (True) or OFF segment (False)
        plot_pulses : bool, optional
            Whether to plot detailed pulse analysis, by default False

        Returns
        -------
        Tuple[float, float, int, int]
            (transition_time, contrast, 10% index, 90% index)
        """
        # Extract the full segment
        segment_time = time_data[start_idx:end_idx]
        segment_pl = pl_data[start_idx:end_idx]

        if len(segment_time) < 10:
            logger.debug(
                f"    Segment too small ({len(segment_time)} points), skipping analysis"
            )
            return np.nan, np.nan, -1, -1

        # Calculate baseline and steady-state levels
        # Use first 0.1% and last 0.1% of the segment
        n_points = len(segment_pl)
        n_baseline = max(int(n_points * 0.001), 1)
        n_steady = max(int(n_points * 0.001), 1)

        baseline_level = np.mean(segment_pl[:n_baseline])
        steady_state = np.mean(segment_pl[-n_steady:])

        # Calculate contrast
        contrast_value = (
            100 * (steady_state - baseline_level) / baseline_level
            if baseline_level != 0
            else np.nan
        )

        # Determine if contrast is positive or negative
        is_positive_contrast = steady_state > baseline_level

        logger.debug(
            f"  {'ON' if is_on_segment else 'OFF'} segment from {start_idx} to {end_idx}:"
        )
        logger.debug(
            f"    Baseline level: {baseline_level:.6f}, Steady state: {steady_state:.6f}"
        )
        logger.debug(
            f"    Contrast: {contrast_value:.2f}%, {'Positive' if is_positive_contrast else 'Negative'}"
        )

        # Calculate 10% and 90% levels
        level_change = steady_state - baseline_level

        if abs(level_change) < 1e-9:
            logger.warning(
                f"    Level change too small ({abs(level_change):.9f}), skipping transition analysis"
            )
            return np.nan, contrast_value, -1, -1

        if is_positive_contrast:
            level_10 = baseline_level + 0.1 * level_change
            level_90 = baseline_level + 0.9 * level_change
        else:
            level_10 = baseline_level + 0.9 * level_change
            level_90 = baseline_level + 0.1 * level_change

        logger.debug(f"    10% level: {level_10:.6f}, 90% level: {level_90:.6f}")

        # Find indices where signal crosses these levels - vectorized operation
        if is_positive_contrast:
            idx_10_candidates = np.where(segment_pl > level_10)[0]
            idx_90_candidates = np.where(segment_pl > level_90)[0]
        else:
            idx_10_candidates = np.where(segment_pl < level_10)[0]
            idx_90_candidates = np.where(segment_pl < level_90)[0]

        logger.debug(
            f"    Found {len(idx_10_candidates)} candidates for 10% crossing, {len(idx_90_candidates)} for 90% crossing"
        )

        # Get the first valid crossing points
        idx_10 = idx_10_candidates[0] if len(idx_10_candidates) > 0 else -1
        idx_90 = idx_90_candidates[0] if len(idx_90_candidates) > 0 else -1

        # Ensure correct ordering of crossings
        if idx_10 >= 0 and idx_90 >= 0:
            if is_on_segment and idx_10 > idx_90:
                # For rising edges, 10% should come before 90%
                logger.warning(
                    f"    10% crossing ({idx_10}) after 90% crossing ({idx_90}), swapping"
                )
                idx_10, idx_90 = idx_90, idx_10
            elif not is_on_segment and idx_90 > idx_10:
                # For falling edges, 90% should come before 10%
                logger.warning(
                    f"    90% crossing ({idx_90}) after 10% crossing ({idx_10}), swapping"
                )
                idx_90, idx_10 = idx_10, idx_90

        if plot_pulses:
            cls._plot_pulse_segment(
                segment_time,
                segment_pl,
                n_baseline,
                n_steady,
                baseline_level,
                steady_state,
                level_10,
                level_90,
                idx_10,
                idx_90,
                is_on_segment,
                is_positive_contrast,
            )

        # Calculate transition time if both points are valid
        if idx_10 >= 0 and idx_90 >= 0:
            # Calculate transition time
            transition_time = abs(segment_time[idx_90] - segment_time[idx_10])
            logger.debug(f"    Transition time: {transition_time * 1000:.3f} ms")

            # Convert to global indices for plotting
            global_idx_10 = start_idx + idx_10
            global_idx_90 = start_idx + idx_90

            logger.debug(
                f"    Global indices - 10%: {global_idx_10}, 90%: {global_idx_90}"
            )
            logger.debug(
                f"    Time points - 10%: {segment_time[idx_10]:.6f}s, 90%: {segment_time[idx_90]:.6f}s"
            )

            return transition_time, contrast_value, global_idx_10, global_idx_90

        # If we get here, something went wrong
        return np.nan, contrast_value, -1, -1

    @classmethod
    def _plot_pulse_segment(
        cls,
        segment_time: NDArray[np.float64],
        segment_pl: NDArray[np.float64],
        n_baseline: int,
        n_steady: int,
        baseline_level: float,
        steady_state: float,
        level_10: float,
        level_90: float,
        idx_10: int,
        idx_90: int,
        is_on_segment: bool,
        is_positive_contrast: bool,
    ) -> None:
        """Plot detailed pulse segment analysis.

        Parameters
        ----------
        segment_time : NDArray[np.float64]
            Time points for the segment
        segment_pl : NDArray[np.float64]
            PL data for the segment
        n_baseline : int
            Number of points in baseline region
        n_steady : int
            Number of points in steady-state region
        baseline_level : float
            Baseline level
        steady_state : float
            Steady-state level
        level_10 : float
            10% level
        level_90 : float
            90% level
        idx_10 : int
            Index of 10% crossing
        idx_90 : int
            Index of 90% crossing
        is_on_segment : bool
            Whether this is an ON segment
        is_positive_contrast : bool
            Whether contrast is positive
        """
        # Create debug plot for this transition
        plt.figure(figsize=(12, 8))

        # Plot the segment data
        plt.title(f"{'ON' if is_on_segment else 'OFF'} Segment Analysis")
        plt.plot(segment_time, segment_pl, "b-", label="Segment Data")

        # Mark the baseline and steady-state regions
        plt.axvspan(
            segment_time[0],
            segment_time[n_baseline - 1],
            color="g",
            alpha=0.2,
            label="Baseline Region",
        )
        plt.axvspan(
            segment_time[-n_steady],
            segment_time[-1],
            color="r",
            alpha=0.2,
            label="Steady-state Region",
        )

        # Mark the baseline and steady-state levels
        plt.axhline(
            y=baseline_level,
            color="g",
            linestyle="-",
            label=f"Baseline: {baseline_level:.6f}",
        )
        plt.axhline(
            y=steady_state,
            color="r",
            linestyle="-",
            label=f"Steady-state: {steady_state:.6f}",
        )

        # Mark the 10% and 90% levels
        plt.axhline(
            y=level_10, color="g", linestyle="--", label=f"10% Level: {level_10:.6f}"
        )
        plt.axhline(
            y=level_90, color="r", linestyle="--", label=f"90% Level: {level_90:.6f}"
        )

        # Mark the detected crossing points if valid
        if idx_10 >= 0:
            plt.plot(
                segment_time[idx_10],
                segment_pl[idx_10],
                "go",
                markersize=8,
                label=f"10% Crossing at {idx_10}",
            )
        if idx_90 >= 0:
            plt.plot(
                segment_time[idx_90],
                segment_pl[idx_90],
                "ro",
                markersize=8,
                label=f"90% Crossing at {idx_90}",
            )

        # Add exponential fit if valid crossing points are found
        if idx_10 >= 0 and idx_90 >= 0:
            cls._plot_exponential_fit(
                segment_time,
                segment_pl,
                idx_10,
                idx_90,
                baseline_level,
                steady_state,
                is_on_segment,
                is_positive_contrast,
            )

        plt.grid(True)
        plt.legend()

        # Calculate contrast value for the info text
        contrast_value = (
            100 * (steady_state - baseline_level) / baseline_level
            if baseline_level != 0
            else np.nan
        )

        # Add information about contrast and timing
        info_text = (
            f"Baseline: {baseline_level:.6f}\n"
            f"Steady state: {steady_state:.6f}\n"
            f"Contrast: {contrast_value:.2f}%\n"
            f"Is positive contrast: {is_positive_contrast}\n"
        )

        if idx_10 >= 0 and idx_90 >= 0:
            transition_time = abs(segment_time[idx_90] - segment_time[idx_10])
            info_text += f"Transition time: {transition_time * 1000:.3f} ms"

        plt.figtext(
            0.02, 0.02, info_text, fontsize=10, bbox=dict(facecolor="white", alpha=0.8)
        )

        plt.tight_layout()
        plt.show()

    @classmethod
    def _plot_exponential_fit(
        cls,
        segment_time: NDArray[np.float64],
        segment_pl: NDArray[np.float64],
        idx_10: int,
        idx_90: int,
        baseline_level: float,
        steady_state: float,
        is_on_segment: bool,
        is_positive_contrast: bool,
    ) -> None:
        """Plot exponential fit for a pulse segment.

        Parameters
        ----------
        segment_time : NDArray[np.float64]
            Time points for the segment
        segment_pl : NDArray[np.float64]
            PL data for the segment
        idx_10 : int
            Index of 10% crossing
        idx_90 : int
            Index of 90% crossing
        baseline_level : float
            Baseline level
        steady_state : float
            Steady-state level
        is_on_segment : bool
            Whether this is an ON segment
        is_positive_contrast : bool
            Whether contrast is positive
        """
        from scipy.optimize import curve_fit

        # Define exponential functions for fitting
        def rise_exp(t, a, tau, c):
            """Exponential rise function: f(t) = a * (1 - exp(-t/tau)) + c"""
            return a * (1 - np.exp(-t / tau)) + c

        def fall_exp(t, a, tau, c):
            """Exponential decay function: f(t) = a * exp(-t/tau) + c"""
            return a * np.exp(-t / tau) + c

        # Use the full segment for fitting
        fit_time = segment_time
        fit_pl = segment_pl

        # Check if we have enough data points for fitting
        if len(fit_time) == 0:
            logger.error(f"    No data points in fit region")
            return

        # Normalize time to start at 0 for fitting
        t_norm = fit_time - fit_time[0]

        try:
            # Initial parameter guesses
            if is_on_segment:  # Field ON response
                # Choose appropriate function based on contrast sign
                if is_positive_contrast:
                    # PL increases: use exponential rise function
                    fit_func = rise_exp
                    p0 = [
                        abs(steady_state - baseline_level),  # amplitude
                        abs(segment_time[idx_90] - segment_time[idx_10])
                        / 2.2,  # time constant
                        baseline_level,  # baseline
                    ]
                else:
                    # PL decreases: use exponential decay function
                    fit_func = fall_exp
                    p0 = [
                        abs(baseline_level - steady_state),  # amplitude
                        abs(segment_time[idx_90] - segment_time[idx_10])
                        / 2.2,  # time constant
                        steady_state,  # final level
                    ]

                # Fit appropriate exponential function
                popt, _ = curve_fit(fit_func, t_norm, fit_pl, p0=p0, maxfev=5000)

                # Plot fit curve
                t_fit = np.linspace(0, max(t_norm), 100)
                y_fit = fit_func(t_fit, *popt)
                plt.plot(
                    t_fit + fit_time[0],
                    y_fit,
                    "k--",
                    label=f"Field ON Response Fit: τ = {popt[1] * 1000:.2f} ms",
                )
            else:  # Field OFF response
                # Calculate initial and final values for the segment
                n_initial = max(
                    int(len(segment_pl) * 0.05), 5
                )  # Use first 5% for initial value
                n_final = max(
                    int(len(segment_pl) * 0.05), 5
                )  # Use last 5% for final value

                initial_value = np.mean(segment_pl[:n_initial])
                final_value = np.mean(segment_pl[-n_final:])

                # Choose appropriate function based on contrast sign
                if is_positive_contrast:
                    # PL was higher during field ON, now decreasing: use exponential decay
                    fit_func = fall_exp
                    p0 = [
                        abs(initial_value - final_value),  # amplitude
                        abs(segment_time[idx_10] - segment_time[idx_90])
                        / 2.2,  # time constant
                        final_value,  # final level
                    ]
                else:
                    # PL was lower during field ON, now increasing: use exponential rise
                    fit_func = rise_exp
                    p0 = [
                        abs(final_value - initial_value),  # amplitude
                        abs(segment_time[idx_10] - segment_time[idx_90])
                        / 2.2,  # time constant
                        initial_value,  # baseline
                    ]

                # Fit appropriate exponential function
                popt, _ = curve_fit(fit_func, t_norm, fit_pl, p0=p0, maxfev=5000)

                # Plot fit curve
                t_fit = np.linspace(0, max(t_norm), 100)
                y_fit = fit_func(t_fit, *popt)
                plt.plot(
                    t_fit + fit_time[0],
                    y_fit,
                    "k--",
                    label=f"Field OFF Response Fit: τ = {popt[1] * 1000:.2f} ms",
                )
        except Exception as e:
            logger.exception(f"    Error fitting exponential: {e}")

    @classmethod
    def analyze_transitions(
        cls,
        time_data: NDArray[np.float64],
        pl_data: NDArray[np.float64],
        clock_data: NDArray[np.float64],
        trigger_threshold: float = 2.5,
        plot_pulses: bool = False,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Analyze PL transitions using clock signal for edge detection with full pulse width analysis.

        This function always calculates 10-90% rise/fall times.

        Parameters
        ----------
        time_data : np.ndarray
            Time points
        pl_data : np.ndarray
            PL signal data
        clock_data : np.ndarray
            Clock signal data
        trigger_threshold : float, optional
            Threshold for clock signal edge detection, by default 2.5
        plot_pulses : bool, optional
            Whether to plot detailed pulse analysis, by default False

        Returns
        -------
        dict
            Analysis results including:
            - rise_times: List of 10-90% rise times
            - fall_times: List of 10-90% fall times
            - contrasts: List of contrast values for each transition
            - rise_indices: Indices of rise transitions
            - fall_indices: Indices of fall transitions
            - rise_10_indices: Indices of 10% rise points
            - rise_90_indices: Indices of 90% rise points
            - fall_10_indices: Indices of 10% fall points
            - fall_90_indices: Indices of 90% fall points
            - on_segments: List of (start, end) indices for ON segments
            - off_segments: List of (start, end) indices for OFF segments
        """
        # Generate a cache key based on input data
        if use_cache:
            # Use hash of data arrays for cache key
            cache_key = (
                hash(time_data.tobytes()),
                hash(pl_data.tobytes()),
                hash(clock_data.tobytes()),
                trigger_threshold,
            )

            # Check if we have cached results
            if cache_key in cls._cache:
                logger.debug("Using cached analysis results")
                # Return cached results but respect the plot_pulses parameter
                cached_results = cls._cache[cache_key].copy()
                if plot_pulses:
                    # If plotting is requested, we need to reanalyze the segments
                    # but we can reuse the edge detection results
                    edge_data = {
                        "on_segments": cached_results.get("on_segments", []),
                        "off_segments": cached_results.get("off_segments", []),
                        "field_on_edges": cached_results.get("rise_indices", []),
                        "field_off_edges": cached_results.get("fall_indices", []),
                    }
                else:
                    return cached_results

        # Detect clock edges and segments
        edge_data = cls._detect_clock_edges(clock_data, trigger_threshold)

        # Ensure we have at least one edge
        if len(edge_data["edges"]) == 0:
            logger.warning("  No edges detected!")
            return {
                "rise_times": [],
                "fall_times": [],
                "contrasts": [],
                "rise_indices": [],
                "fall_indices": [],
                "rise_10_indices": [],
                "rise_90_indices": [],
                "fall_10_indices": [],
                "fall_90_indices": [],
                "mean_rise_time": np.nan,
                "mean_fall_time": np.nan,
                "mean_contrast": np.nan,
            }

        # Extract data from edge detection
        field_on_edges = edge_data["field_on_edges"]
        field_off_edges = edge_data["field_off_edges"]
        on_segments = edge_data["on_segments"]
        off_segments = edge_data["off_segments"]

        # Storage for analysis results
        rise_times = []
        fall_times = []
        contrasts = []
        rise_indices = field_on_edges.tolist()
        fall_indices = field_off_edges.tolist()
        rise_10_indices = []
        rise_90_indices = []
        fall_10_indices = []
        fall_90_indices = []

        # Function to analyze a full pulse segment
        def analyze_pulse_segment(start_idx, end_idx, is_on_segment):
            """Analyze a full pulse segment.

            Parameters
            ----------
            start_idx : int
                Start index of the pulse segment
            end_idx : int
                End index of the pulse segment
            is_on_segment : bool
                Whether this is an ON segment (True) or OFF segment (False)

            Returns
            -------
            tuple
                (transition_time, contrast, 10% index, 90% index)
            """
            # Extract the full segment
            segment_time = time_data[start_idx:end_idx]
            segment_pl = pl_data[start_idx:end_idx]

            if len(segment_time) < 10:
                logger.debug(
                    f"    Segment too small ({len(segment_time)} points), skipping analysis"
                )
                return np.nan, np.nan, -1, -1

            # Calculate baseline and steady-state levels
            # Use first 0.1% and last 0.1% of the segment
            n_points = len(segment_pl)
            n_baseline = max(int(n_points * 0.001), 1)
            n_steady = max(int(n_points * 0.001), 1)

            baseline_level = np.mean(segment_pl[:n_baseline])
            steady_state = np.mean(segment_pl[-n_steady:])

            # Calculate contrast
            contrast_value = (
                100 * (steady_state - baseline_level) / baseline_level
                if baseline_level != 0
                else np.nan
            )

            # Determine if contrast is positive or negative
            is_positive_contrast = steady_state > baseline_level

            logger.debug(
                f"  {'ON' if is_on_segment else 'OFF'} segment from {start_idx} to {end_idx}:"
            )
            logger.debug(
                f"    Baseline level: {baseline_level:.6f}, Steady state: {steady_state:.6f}"
            )
            logger.debug(
                f"    Contrast: {contrast_value:.2f}%, {'Positive' if is_positive_contrast else 'Negative'}"
            )

            # Calculate 10% and 90% levels
            level_change = steady_state - baseline_level

            if abs(level_change) < 1e-9:
                logger.warning(
                    f"    Level change too small ({abs(level_change):.9f}), skipping transition analysis"
                )
                return np.nan, contrast_value, -1, -1

            if is_positive_contrast:
                level_10 = baseline_level + 0.1 * level_change
                level_90 = baseline_level + 0.9 * level_change
            else:
                level_10 = baseline_level + 0.9 * level_change
                level_90 = baseline_level + 0.1 * level_change

            logger.debug(f"    10% level: {level_10:.6f}, 90% level: {level_90:.6f}")

            # Find indices where signal crosses these levels
            if is_positive_contrast:
                idx_10_candidates = np.where(segment_pl > level_10)[0]
                idx_90_candidates = np.where(segment_pl > level_90)[0]
            else:
                idx_10_candidates = np.where(segment_pl < level_10)[0]
                idx_90_candidates = np.where(segment_pl < level_90)[0]

            logger.debug(
                f"    Found {len(idx_10_candidates)} candidates for 10% crossing, {len(idx_90_candidates)} for 90% crossing"
            )

            # Get the first valid crossing points
            idx_10 = idx_10_candidates[0] if len(idx_10_candidates) > 0 else -1
            idx_90 = idx_90_candidates[0] if len(idx_90_candidates) > 0 else -1

            # Ensure correct ordering of crossings
            if idx_10 >= 0 and idx_90 >= 0:
                if is_on_segment and idx_10 > idx_90:
                    # For rising edges, 10% should come before 90%
                    logger.warning(
                        f"    10% crossing ({idx_10}) after 90% crossing ({idx_90}), swapping"
                    )
                    idx_10, idx_90 = idx_90, idx_10
                elif not is_on_segment and idx_90 > idx_10:
                    # For falling edges, 90% should come before 10%
                    logger.warning(
                        f"    90% crossing ({idx_90}) after 10% crossing ({idx_10}), swapping"
                    )
                    idx_90, idx_10 = idx_10, idx_90

            if plot_pulses:
                # Create debug plot for this transition
                plt.figure(figsize=(12, 8))

                # Plot the segment data
                plt.title(f"{'ON' if is_on_segment else 'OFF'} Segment Analysis")
                plt.plot(segment_time, segment_pl, "b-", label="Segment Data")

                # Mark the baseline and steady-state regions
                plt.axvspan(
                    segment_time[0],
                    segment_time[n_baseline - 1],
                    color="g",
                    alpha=0.2,
                    label="Baseline Region",
                )
                plt.axvspan(
                    segment_time[-n_steady],
                    segment_time[-1],
                    color="r",
                    alpha=0.2,
                    label="Steady-state Region",
                )

                # Mark the baseline and steady-state levels
                plt.axhline(
                    y=baseline_level,
                    color="g",
                    linestyle="-",
                    label=f"Baseline: {baseline_level:.6f}",
                )
                plt.axhline(
                    y=steady_state,
                    color="r",
                    linestyle="-",
                    label=f"Steady-state: {steady_state:.6f}",
                )

                # Mark the 10% and 90% levels
                plt.axhline(
                    y=level_10,
                    color="g",
                    linestyle="--",
                    label=f"10% Level: {level_10:.6f}",
                )
                plt.axhline(
                    y=level_90,
                    color="r",
                    linestyle="--",
                    label=f"90% Level: {level_90:.6f}",
                )

                # Mark the detected crossing points if valid
                if idx_10 >= 0:
                    plt.plot(
                        segment_time[idx_10],
                        segment_pl[idx_10],
                        "go",
                        markersize=8,
                        label=f"10% Crossing at {idx_10}",
                    )
                if idx_90 >= 0:
                    plt.plot(
                        segment_time[idx_90],
                        segment_pl[idx_90],
                        "ro",
                        markersize=8,
                        label=f"90% Crossing at {idx_90}",
                    )

                # Add exponential fit if valid crossing points are found
                if idx_10 >= 0 and idx_90 >= 0:
                    # Define exponential functions for fitting
                    def rise_exp(t, a, tau, c):
                        """Exponential rise function: f(t) = a * (1 - exp(-t/tau)) + c"""
                        return a * (1 - np.exp(-t / tau)) + c

                    def fall_exp(t, a, tau, c):
                        """Exponential decay function: f(t) = a * exp(-t/tau) + c"""
                        return a * np.exp(-t / tau) + c

                    # Use the full segment for fitting
                    fit_time = segment_time
                    fit_pl = segment_pl

                    # Check if we have enough data points for fitting
                    if len(fit_time) == 0:
                        logger.error(f"    No data points in fit region")
                        # Skip the rest of this iteration
                        return np.nan, contrast_value, -1, -1

                    # Normalize time to start at 0 for fitting
                    t_norm = fit_time - fit_time[0]

                    try:
                        # Determine if PL increases or decreases during this segment
                        is_positive_contrast = steady_state > baseline_level

                        # Initial parameter guesses
                        if is_on_segment:  # Field ON response
                            # Choose appropriate function based on contrast sign
                            if is_positive_contrast:
                                # PL increases: use exponential rise function
                                fit_func = rise_exp
                                p0 = [
                                    abs(steady_state - baseline_level),  # amplitude
                                    abs(segment_time[idx_90] - segment_time[idx_10])
                                    / 2.2,  # time constant
                                    baseline_level,  # baseline
                                ]
                            else:
                                # PL decreases: use exponential decay function
                                fit_func = fall_exp
                                p0 = [
                                    abs(baseline_level - steady_state),  # amplitude
                                    abs(segment_time[idx_90] - segment_time[idx_10])
                                    / 2.2,  # time constant
                                    steady_state,  # final level
                                ]

                            # Fit appropriate exponential function
                            popt, _ = curve_fit(
                                fit_func, t_norm, fit_pl, p0=p0, maxfev=5000
                            )

                            # Plot fit curve
                            t_fit = np.linspace(0, max(t_norm), 100)
                            y_fit = fit_func(t_fit, *popt)
                            plt.plot(
                                t_fit + fit_time[0],
                                y_fit,
                                "k--",
                                label=f"Field ON Response Fit: τ = {popt[1] * 1000:.2f} ms",
                            )
                        else:  # Field OFF response
                            # Calculate initial and final values for the segment
                            n_initial = max(
                                int(len(segment_pl) * 0.05), 5
                            )  # Use first 5% for initial value
                            n_final = max(
                                int(len(segment_pl) * 0.05), 5
                            )  # Use last 5% for final value

                            initial_value = np.mean(segment_pl[:n_initial])
                            final_value = np.mean(segment_pl[-n_final:])

                            # Choose appropriate function based on contrast sign
                            if is_positive_contrast:
                                # PL was higher during field ON, now decreasing: use exponential decay
                                fit_func = fall_exp
                                p0 = [
                                    abs(initial_value - final_value),  # amplitude
                                    abs(segment_time[idx_10] - segment_time[idx_90])
                                    / 2.2,  # time constant
                                    final_value,  # final level
                                ]
                            else:
                                # PL was lower during field ON, now increasing: use exponential rise
                                fit_func = rise_exp
                                p0 = [
                                    abs(final_value - initial_value),  # amplitude
                                    abs(segment_time[idx_10] - segment_time[idx_90])
                                    / 2.2,  # time constant
                                    initial_value,  # baseline
                                ]

                            # Fit appropriate exponential function
                            popt, _ = curve_fit(
                                fit_func, t_norm, fit_pl, p0=p0, maxfev=5000
                            )

                            # Plot fit curve
                            t_fit = np.linspace(0, max(t_norm), 100)
                            y_fit = fit_func(t_fit, *popt)
                            plt.plot(
                                t_fit + fit_time[0],
                                y_fit,
                                "k--",
                                label=f"Field OFF Response Fit: τ = {popt[1] * 1000:.2f} ms",
                            )
                    except Exception as e:
                        logger.exception(f"    Error fitting exponential: {e}")

                plt.grid(True)
                plt.legend()

                # Add information about contrast and timing
                info_text = (
                    f"Baseline: {baseline_level:.6f}\n"
                    f"Steady state: {steady_state:.6f}\n"
                    f"Contrast: {contrast_value:.2f}%\n"
                    f"Is positive contrast: {is_positive_contrast}\n"
                )

                if idx_10 >= 0 and idx_90 >= 0:
                    transition_time = abs(segment_time[idx_90] - segment_time[idx_10])
                    info_text += f"Transition time: {transition_time * 1000:.3f} ms"

                plt.figtext(
                    0.02,
                    0.02,
                    info_text,
                    fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.8),
                )

                plt.tight_layout()
                plt.show()

            # Calculate transition time if both points are valid
            if idx_10 >= 0 and idx_90 >= 0:
                # Calculate transition time
                transition_time = abs(segment_time[idx_90] - segment_time[idx_10])
                logger.debug(f"    Transition time: {transition_time * 1000:.3f} ms")

                # Convert to global indices for plotting
                global_idx_10 = start_idx + idx_10
                global_idx_90 = start_idx + idx_90

                logger.debug(
                    f"    Global indices - 10%: {global_idx_10}, 90%: {global_idx_90}"
                )
                logger.debug(
                    f"    Time points - 10%: {segment_time[idx_10]:.6f}s, 90%: {segment_time[idx_90]:.6f}s"
                )

                return transition_time, contrast_value, global_idx_10, global_idx_90

            # If we get here, something went wrong
            return np.nan, contrast_value, -1, -1

        # Analyze ON segments (field turning ON)
        for i, (on_edge, off_edge) in enumerate(on_segments):
            logger.debug(f"\nAnalyzing ON segment {i + 1}: {on_edge} to {off_edge}")

            # Check if segment is valid (has enough points)
            if off_edge - on_edge < 10:
                logger.debug(
                    f"  Segment too small ({off_edge - on_edge} points), skipping analysis"
                )
                continue

            rise_time, contrast, idx_10, idx_90 = analyze_pulse_segment(
                on_edge, off_edge, True
            )

            if not np.isnan(rise_time) and not np.isnan(contrast):
                rise_times.append(rise_time)
                contrasts.append(contrast)
                rise_10_indices.append(idx_10)
                rise_90_indices.append(idx_90)

        # Analyze OFF segments (field turning OFF)
        for i, (off_edge, next_on_edge) in enumerate(off_segments):
            logger.debug(
                f"\nAnalyzing OFF segment {i + 1}: {off_edge} to {next_on_edge}"
            )

            # Check if segment is valid (has enough points)
            if next_on_edge - off_edge < 10:
                logger.debug(
                    f"  Segment too small ({next_on_edge - off_edge} points), skipping analysis"
                )
                continue

            fall_time, _, idx_10, idx_90 = analyze_pulse_segment(
                off_edge, next_on_edge, False
            )

            if not np.isnan(fall_time):
                fall_times.append(fall_time)
                fall_10_indices.append(idx_10)
                fall_90_indices.append(idx_90)

        # Calculate mean values - use numpy for efficiency
        mean_rise_time = np.mean(rise_times) if rise_times else np.nan
        mean_fall_time = np.mean(fall_times) if fall_times else np.nan
        mean_contrast = np.mean(contrasts) if contrasts else np.nan

        # Log only key summary information
        if not np.isnan(mean_contrast):
            logger.info(f"  Mean contrast: {mean_contrast:.2f}%")

        # Prepare results dictionary
        results = {
            "rise_times": rise_times,  # Field ON response times
            "fall_times": fall_times,  # Field OFF response times
            "contrasts": contrasts,
            "rise_indices": rise_indices,  # Field ON edges
            "fall_indices": fall_indices,  # Field OFF edges
            "rise_10_indices": rise_10_indices,  # 10% points for field ON response
            "rise_90_indices": rise_90_indices,  # 90% points for field ON response
            "fall_10_indices": fall_10_indices,  # 10% points for field OFF response
            "fall_90_indices": fall_90_indices,  # 90% points for field OFF response
            "mean_rise_time": mean_rise_time,  # Mean field ON response time
            "mean_fall_time": mean_fall_time,  # Mean field OFF response time
            "mean_contrast": mean_contrast,
            # Add segment data for potential exponential fitting
            "on_segments": on_segments,  # Field ON segments
            "off_segments": off_segments,  # Field OFF segments
        }

        # Cache the results if caching is enabled
        if use_cache:
            cache_key = (
                hash(time_data.tobytes()),
                hash(pl_data.tobytes()),
                hash(clock_data.tobytes()),
                trigger_threshold,
            )
            cls._cache[cache_key] = results.copy()

            # Limit cache size to prevent memory issues
            if len(cls._cache) > 10:
                # Remove oldest entry (first key)
                oldest_key = next(iter(cls._cache))
                del cls._cache[oldest_key]

        return results

    @staticmethod
    def global_rise_exp(
        t: NDArray[np.float64], a: float, tau: float, c: float, t0: float
    ) -> NDArray[np.float64]:
        """Exponential rise function with absolute time reference: f(t) = a * (1 - exp(-(t-t0)/tau)) + c

        Parameters
        ----------
        t : np.ndarray
            Time points (absolute time)
        a : float
            Amplitude
        tau : float
            Time constant
        c : float
            Offset
        t0 : float
            Time offset (start time)

        Returns
        -------
        np.ndarray
            Function values
        """
        return a * (1 - np.exp(-(t - t0) / tau)) + c

    @staticmethod
    def global_fall_exp(
        t: NDArray[np.float64], a: float, tau: float, c: float, t0: float
    ) -> NDArray[np.float64]:
        """Exponential decay function with absolute time reference: f(t) = a * exp(-(t-t0)/tau) + c

        Parameters
        ----------
        t : np.ndarray
            Time points (absolute time)
        a : float
            Amplitude
        tau : float
            Time constant
        c : float
            Offset
        t0 : float
            Time offset (start time)

        Returns
        -------
        np.ndarray
            Function values
        """
        return a * np.exp(-(t - t0) / tau) + c

    @staticmethod
    def double_rise_exp(
        t: NDArray[np.float64], a1: float, tau1: float, a2: float, tau2: float, c: float
    ) -> NDArray[np.float64]:
        """Double exponential rise function: f(t) = a1*(1-exp(-t/tau1)) + a2*(1-exp(-t/tau2)) + c

        Parameters
        ----------
        t : np.ndarray
            Time points
        a1 : float
            Amplitude of first component
        tau1 : float
            Time constant of first component
        a2 : float
            Amplitude of second component
        tau2 : float
            Time constant of second component
        c : float
            Offset

        Returns
        -------
        np.ndarray
            Function values
        """
        return a1 * (1 - np.exp(-t / tau1)) + a2 * (1 - np.exp(-t / tau2)) + c

    @staticmethod
    def double_fall_exp(
        t: NDArray[np.float64], a1: float, tau1: float, a2: float, tau2: float, c: float
    ) -> NDArray[np.float64]:
        """Double exponential decay function: f(t) = a1*exp(-t/tau1) + a2*exp(-t/tau2) + c

        Parameters
        ----------
        t : np.ndarray
            Time points
        a1 : float
            Amplitude of first component
        tau1 : float
            Time constant of first component
        a2 : float
            Amplitude of second component
        tau2 : float
            Time constant of second component
        c : float
            Offset

        Returns
        -------
        np.ndarray
            Function values
        """
        return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + c

    @staticmethod
    def global_double_rise_exp(
        t: NDArray[np.float64],
        a1: float,
        tau1: float,
        a2: float,
        tau2: float,
        c: float,
        t0: float,
    ) -> NDArray[np.float64]:
        """Double exponential rise function with absolute time reference

        Parameters
        ----------
        t : np.ndarray
            Time points (absolute time)
        a1 : float
            Amplitude of first component
        tau1 : float
            Time constant of first component
        a2 : float
            Amplitude of second component
        tau2 : float
            Time constant of second component
        c : float
            Offset
        t0 : float
            Time offset (start time)

        Returns
        -------
        np.ndarray
            Function values
        """
        return (
            a1 * (1 - np.exp(-(t - t0) / tau1))
            + a2 * (1 - np.exp(-(t - t0) / tau2))
            + c
        )

    @staticmethod
    def global_double_fall_exp(
        t: NDArray[np.float64],
        a1: float,
        tau1: float,
        a2: float,
        tau2: float,
        c: float,
        t0: float,
    ) -> NDArray[np.float64]:
        """Double exponential decay function with absolute time reference

        Parameters
        ----------
        t : np.ndarray
            Time points (absolute time)
        a1 : float
            Amplitude of first component
        tau1 : float
            Time constant of first component
        a2 : float
            Amplitude of second component
        tau2 : float
            Time constant of second component
        c : float
            Offset
        t0 : float
            Time offset (start time)

        Returns
        -------
        np.ndarray
            Function values
        """
        return a1 * np.exp(-(t - t0) / tau1) + a2 * np.exp(-(t - t0) / tau2) + c

    @classmethod
    def _fit_segment_exponential(
        cls,
        segment_time: NDArray[np.float64],
        segment_pl: NDArray[np.float64],
        is_on_segment: bool,
        rise_time: Optional[float] = None,
        fit_type: str = "single",
    ) -> Tuple[Dict[str, Any], bool]:
        """Fit exponential curve to a segment.

        Parameters
        ----------
        segment_time : NDArray[np.float64]
            Time points for the segment
        segment_pl : NDArray[np.float64]
            PL data for the segment
        is_on_segment : bool
            Whether this is an ON segment
        rise_time : Optional[float], optional
            Rise time for initial guess, by default None
        fit_type : str, optional
            Type of exponential fit to use ("single" or "double"), by default "single"

        Returns
        -------
        Tuple[Dict[str, Any], bool]
            (fit parameters, is_positive_contrast)
        """
        # Calculate baseline and steady state for better initial guesses
        n_points = len(segment_pl)
        n_baseline = max(int(n_points * 0.05), 5)  # Use first 5% for baseline
        n_steady = max(int(n_points * 0.05), 5)  # Use last 5% for steady state

        baseline_level = np.mean(segment_pl[:n_baseline])
        steady_state = np.mean(segment_pl[-n_steady:])

        # Determine if PL increases or decreases
        is_positive_contrast = steady_state > baseline_level

        # For field OFF segments, we need to check if PL is increasing or decreasing
        # This is different from the contrast calculation
        is_decreasing = False
        if not is_on_segment:
            # For field OFF segments, check the actual trend
            is_decreasing = baseline_level > steady_state

        # Initial time constant guess
        time_constant_guess = rise_time / 2.2 if rise_time is not None else 0.1

        # Get the time offset (start time of the segment)
        t0 = segment_time[0]

        try:
            # Determine if we're using single or double exponential fit
            if fit_type == "single":
                if (is_on_segment and is_positive_contrast) or (
                    not is_on_segment and not is_decreasing
                ):
                    # PL increases: use exponential rise function with global time reference
                    p0 = [
                        abs(steady_state - baseline_level),  # amplitude
                        time_constant_guess,  # time constant
                        baseline_level,  # baseline
                        t0,  # time offset
                    ]

                    # Define a wrapper function for curve_fit that uses the global time
                    def fit_func(t, a, tau, c):
                        return cls.global_rise_exp(t, a, tau, c, t0)

                    popt, pcov = curve_fit(
                        fit_func, segment_time, segment_pl, p0=p0[:3], maxfev=5000
                    )

                    # Add the time offset to the parameters
                    popt = list(popt) + [t0]

                    # Create a function that can be used to evaluate the fit at any time
                    def fit_function(t):
                        return cls.global_rise_exp(t, *popt)

                    # Store the function type
                    is_rise = True
                else:
                    # PL decreases: use exponential decay function with global time reference
                    p0 = [
                        abs(baseline_level - steady_state),  # amplitude
                        time_constant_guess,  # time constant
                        steady_state,  # final level
                        t0,  # time offset
                    ]

                    # Define a wrapper function for curve_fit that uses the global time
                    def fit_func(t, a, tau, c):
                        return cls.global_fall_exp(t, a, tau, c, t0)

                    popt, pcov = curve_fit(
                        fit_func, segment_time, segment_pl, p0=p0[:3], maxfev=5000
                    )

                    # Add the time offset to the parameters
                    popt = list(popt) + [t0]

                    # Create a function that can be used to evaluate the fit at any time
                    def fit_function(t):
                        return cls.global_fall_exp(t, *popt)

                    # Store the function type
                    is_rise = False

                # Calculate residuals and R-squared for goodness of fit
                residuals = segment_pl - fit_function(segment_time)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((segment_pl - np.mean(segment_pl)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                # Store all the necessary information for plotting
                fit_info = {
                    "params": popt,
                    "is_rise": is_rise,
                    "function": fit_function,
                    "baseline": baseline_level,
                    "steady_state": steady_state,
                    "is_positive_contrast": is_positive_contrast,
                    "fit_type": "single",
                    "r_squared": r_squared,
                }

            else:  # Double exponential fit
                if (is_on_segment and is_positive_contrast) or (
                    not is_on_segment and not is_decreasing
                ):
                    # PL increases: use double exponential rise function
                    # For initial guess, split the amplitude in two parts with different time constants
                    amplitude = abs(steady_state - baseline_level)
                    p0 = [
                        amplitude * 0.7,  # amplitude of first component (70%)
                        time_constant_guess * 0.5,  # faster time constant
                        amplitude * 0.3,  # amplitude of second component (30%)
                        time_constant_guess * 3.0,  # slower time constant
                        baseline_level,  # baseline
                        t0,  # time offset
                    ]

                    # Define a wrapper function for curve_fit that uses the global time
                    def fit_func(t, a1, tau1, a2, tau2, c):
                        return cls.global_double_rise_exp(t, a1, tau1, a2, tau2, c, t0)

                    # Try to fit with double exponential, but handle potential failures
                    try:
                        popt, pcov = curve_fit(
                            fit_func,
                            segment_time,
                            segment_pl,
                            p0=p0[:5],
                            maxfev=10000,
                            bounds=(
                                [0, 0, 0, 0, -np.inf],
                                [np.inf, np.inf, np.inf, np.inf, np.inf],
                            ),
                        )
                    except RuntimeError:
                        # If double exponential fit fails, try with more relaxed bounds
                        logger.warning(
                            "Double exponential fit failed with initial bounds, trying with relaxed bounds"
                        )
                        popt, pcov = curve_fit(
                            fit_func, segment_time, segment_pl, p0=p0[:5], maxfev=10000
                        )

                    # Add the time offset to the parameters
                    popt = list(popt) + [t0]

                    # Create a function that can be used to evaluate the fit at any time
                    def fit_function(t):
                        return cls.global_double_rise_exp(t, *popt)

                    # Store the function type
                    is_rise = True
                else:
                    # PL decreases: use double exponential decay function
                    amplitude = abs(baseline_level - steady_state)
                    p0 = [
                        amplitude * 0.7,  # amplitude of first component (70%)
                        time_constant_guess * 0.5,  # faster time constant
                        amplitude * 0.3,  # amplitude of second component (30%)
                        time_constant_guess * 3.0,  # slower time constant
                        steady_state,  # final level
                        t0,  # time offset
                    ]

                    # Define a wrapper function for curve_fit that uses the global time
                    def fit_func(t, a1, tau1, a2, tau2, c):
                        return cls.global_double_fall_exp(t, a1, tau1, a2, tau2, c, t0)

                    # Try to fit with double exponential, but handle potential failures
                    try:
                        popt, pcov = curve_fit(
                            fit_func,
                            segment_time,
                            segment_pl,
                            p0=p0[:5],
                            maxfev=10000,
                            bounds=(
                                [0, 0, 0, 0, -np.inf],
                                [np.inf, np.inf, np.inf, np.inf, np.inf],
                            ),
                        )
                    except RuntimeError:
                        # If double exponential fit fails, try with more relaxed bounds
                        logger.warning(
                            "Double exponential fit failed with initial bounds, trying with relaxed bounds"
                        )
                        popt, pcov = curve_fit(
                            fit_func, segment_time, segment_pl, p0=p0[:5], maxfev=10000
                        )

                    # Add the time offset to the parameters
                    popt = list(popt) + [t0]

                    # Create a function that can be used to evaluate the fit at any time
                    def fit_function(t):
                        return cls.global_double_fall_exp(t, *popt)

                    # Store the function type
                    is_rise = False

                # Calculate residuals and R-squared for goodness of fit
                residuals = segment_pl - fit_function(segment_time)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((segment_pl - np.mean(segment_pl)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                # Sort time constants to have the faster one first
                if popt[1] > popt[3]:
                    # Swap time constants and amplitudes
                    a1, tau1, a2, tau2 = popt[0], popt[1], popt[2], popt[3]
                    popt[0], popt[1], popt[2], popt[3] = a2, tau2, a1, tau1

                # Store all the necessary information for plotting
                fit_info = {
                    "params": popt,
                    "is_rise": is_rise,
                    "function": fit_function,
                    "baseline": baseline_level,
                    "steady_state": steady_state,
                    "is_positive_contrast": is_positive_contrast,
                    "fit_type": "double",
                    "r_squared": r_squared,
                }

            return fit_info, is_positive_contrast
        except Exception as e:
            logger.exception(f"  Error fitting exponential: {e}")
            return {"fit_type": "failed"}, is_positive_contrast

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the analysis results cache.

        This method should be called when memory usage is a concern
        or when fresh analysis is required regardless of input data.
        """
        cls._cache.clear()
        logger.debug("MPLFitter cache cleared")

    @classmethod
    def fit_exponential_curves(
        cls,
        time_data: NDArray[np.float64],
        pl_data: NDArray[np.float64],
        transition_results: Dict[str, Any],
        fit_type: str = "single",
    ) -> Dict[str, Any]:
        """Fit exponential curves to the rise and fall transitions.

        Parameters
        ----------
        time_data : np.ndarray
            Time points
        pl_data : np.ndarray
            PL signal data
        transition_results : dict
            Results from analyze_transitions function
        fit_type : str, optional
            Type of exponential fit to use ("single" or "double"), by default "single"

        Returns
        -------
        dict
            Exponential fit results including:
            - rise_tau: List of rise time constants
            - fall_tau: List of fall time constants
            - rise_fit_params: List of all fit parameters for rise transitions
            - fall_fit_params: List of all fit parameters for fall transitions
            - mean_rise_tau: Mean rise time constant
            - mean_fall_tau: Mean fall time constant
            - fit_type: Type of exponential fit used
        """
        # Get segments from transition results
        on_segments = transition_results.get("on_segments", [])
        off_segments = transition_results.get("off_segments", [])

        # Get 10-90% times for initial guesses
        rise_times = transition_results.get("rise_times", [])
        fall_times = transition_results.get("fall_times", [])

        # Storage for fit results
        rise_tau = []
        fall_tau = []
        rise_fit_params = []
        fall_fit_params = []

        # For double exponential, we'll store both time constants
        if fit_type == "double":
            rise_tau1 = []
            rise_tau2 = []
            fall_tau1 = []
            fall_tau2 = []

        # Fit ON segments (field ON response)
        for i, (on_edge, off_edge) in enumerate(on_segments):
            if i < len(rise_times):
                # Extract segment data
                segment_time = time_data[on_edge:off_edge]
                segment_pl = pl_data[on_edge:off_edge]

                # Fit exponential
                fit_info, _ = cls._fit_segment_exponential(
                    segment_time, segment_pl, True, rise_times[i], fit_type
                )

                # Check if fit_info is a dictionary with valid parameters
                if isinstance(fit_info, dict) and "params" in fit_info:
                    if fit_type == "single" and len(fit_info["params"]) > 1:
                        rise_tau.append(fit_info["params"][1])  # tau
                        rise_fit_params.append(fit_info)
                    elif fit_type == "double" and len(fit_info["params"]) > 3:
                        # For double exponential, store both time constants
                        rise_tau1.append(fit_info["params"][1])  # tau1
                        rise_tau2.append(fit_info["params"][3])  # tau2
                        rise_fit_params.append(fit_info)
                elif isinstance(fit_info, list) and len(fit_info) > 1:
                    # Handle old format for backward compatibility
                    rise_tau.append(fit_info[1])  # tau
                    rise_fit_params.append(fit_info)

        # Fit OFF segments (field OFF response)
        for i, (off_edge, next_on_edge) in enumerate(off_segments):
            if i < len(fall_times):
                # Extract segment data
                segment_time = time_data[off_edge:next_on_edge]
                segment_pl = pl_data[off_edge:next_on_edge]

                # Fit exponential
                fit_info, _ = cls._fit_segment_exponential(
                    segment_time, segment_pl, False, fall_times[i], fit_type
                )

                # Check if fit_info is a dictionary with valid parameters
                if isinstance(fit_info, dict) and "params" in fit_info:
                    if fit_type == "single" and len(fit_info["params"]) > 1:
                        fall_tau.append(fit_info["params"][1])  # tau
                        fall_fit_params.append(fit_info)
                    elif fit_type == "double" and len(fit_info["params"]) > 3:
                        # For double exponential, store both time constants
                        fall_tau1.append(fit_info["params"][1])  # tau1
                        fall_tau2.append(fit_info["params"][3])  # tau2
                        fall_fit_params.append(fit_info)
                elif isinstance(fit_info, list) and len(fit_info) > 1:
                    # Handle old format for backward compatibility
                    fall_tau.append(fit_info[1])  # tau
                    fall_fit_params.append(fit_info)

        # Calculate mean values using numpy for efficiency
        if fit_type == "single":
            mean_rise_tau = np.mean(rise_tau) if rise_tau else np.nan
            mean_fall_tau = np.mean(fall_tau) if fall_tau else np.nan

            return {
                "rise_tau": rise_tau,
                "fall_tau": fall_tau,
                "rise_fit_params": rise_fit_params,
                "fall_fit_params": fall_fit_params,
                "mean_rise_tau": mean_rise_tau,
                "mean_fall_tau": mean_fall_tau,
                "fit_type": fit_type,
            }
        else:  # Double exponential
            mean_rise_tau1 = np.mean(rise_tau1) if rise_tau1 else np.nan
            mean_rise_tau2 = np.mean(rise_tau2) if rise_tau2 else np.nan
            mean_fall_tau1 = np.mean(fall_tau1) if fall_tau1 else np.nan
            mean_fall_tau2 = np.mean(fall_tau2) if fall_tau2 else np.nan

            return {
                "rise_tau1": rise_tau1,
                "rise_tau2": rise_tau2,
                "fall_tau1": fall_tau1,
                "fall_tau2": fall_tau2,
                "rise_fit_params": rise_fit_params,
                "fall_fit_params": fall_fit_params,
                "mean_rise_tau1": mean_rise_tau1,
                "mean_rise_tau2": mean_rise_tau2,
                "mean_fall_tau1": mean_fall_tau1,
                "mean_fall_tau2": mean_fall_tau2,
                "fit_type": fit_type,
            }

    @classmethod
    def analyze_and_fit(
        cls,
        time_data: NDArray[np.float64],
        pl_data: NDArray[np.float64],
        clock_data: NDArray[np.float64],
        trigger_threshold: float = 2.5,
        plot_pulses: bool = False,
        use_cache: bool = True,
        fit_type: str = "single",
    ) -> Dict[str, Any]:
        """Perform both transition analysis and exponential fitting in one operation.

        This method combines analyze_transitions and fit_exponential_curves for efficiency.

        Parameters
        ----------
        time_data : np.ndarray
            Time points
        pl_data : np.ndarray
            PL signal data
        clock_data : np.ndarray
            Clock signal data
        trigger_threshold : float, optional
            Threshold for clock signal edge detection, by default 2.5
        plot_pulses : bool, optional
            Whether to plot detailed pulse analysis, by default False
        use_cache : bool, optional
            Whether to use cached results, by default True
        fit_type : str, optional
            Type of exponential fit to use ("single" or "double"), by default "single"

        Returns
        -------
        dict
            Combined analysis results from both transition analysis and exponential fitting
        """
        # First perform transition analysis
        transition_results = cls.analyze_transitions(
            time_data, pl_data, clock_data, trigger_threshold, plot_pulses, use_cache
        )

        # Check if we have valid transitions to fit
        if (
            len(transition_results.get("rise_times", [])) > 0
            or len(transition_results.get("fall_times", [])) > 0
        ):
            # Perform exponential fitting
            fit_results = cls.fit_exponential_curves(
                time_data, pl_data, transition_results, fit_type
            )

            # Merge results
            transition_results.update(fit_results)

        return transition_results
