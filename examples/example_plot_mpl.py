"""Example script for loading and plotting magnetophotoluminescence (MPL) data.

This script demonstrates how to:
1. Load MPL data saved by the qscope square command
2. Analyze the PL response characteristics
3. Create publication-quality plots with customized styling
4. Calculate additional metrics like rise times and noise levels

Usage:
    python example_plot_mpl.py path/to/data.json

The script expects data saved by the qscope mpl mpl square command.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
from loguru import logger

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal


def set_mpl_rcparams(options):
    """Reads matplotlib-relevant parameters in options and used to define matplotlib rcParams"""
    from warnings import warn
    for optn, val in options.items():
        if isinstance(val, (list, tuple)):
            val = tuple(val)
        try:
            mpl.rcParams[optn] = val
        except KeyError:
            warn(
                f"mpl rcparams key '{optn}' not recognised as a valid rc parameter."
            )


def load_mpl_data(filepath: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load MPL measurement data from JSON file.
    
    Parameters
    ----------
    filepath : str
        Path to JSON file containing MPL data
        
    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        Tuple containing (parameters, results)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['parameters'], data['results'], data['analysis']


def get_readable_time_unit(max_time: float) -> Tuple[float, str]:
    """Get appropriate time scaling factor and unit.
    
    Parameters
    ----------
    max_time : float
        Maximum time value in seconds
        
    Returns
    -------
    scale : float
        Scale factor to multiply time by
    unit : str
        Unit string (ns, μs, ms, s)
    """
    if max_time < 1e-6:
        return 1e9, 'ns'
    elif max_time < 1e-3:
        return 1e6, 'μs'
    elif max_time < 1:
        return 1e3, 'ms'
    else:
        return 1, 's'

def get_readable_time(t: float) -> Tuple[float, str]:
    """Get time in better units, with unit str"""
    if t < 1e-6:
        return t*1e9, 'ns'
    elif t < 1e-3:
        return t*1e6, 'μs'
    elif t < 1:
        return t*1e3, 'ms'
    else:
        return t, 's'

def calculate_response_times(time: np.ndarray, pl: np.ndarray, 
                           current: np.ndarray) -> Dict[str, float]:
    """Calculate PL response times to field switching."""
    # Normalize signals
    current_norm = current / np.max(np.abs(current))
    
    # Use threshold crossing to find edges
    threshold = 0.5
    rising_edges = np.where((current_norm[1:] > threshold) & 
                           (current_norm[:-1] <= threshold))[0]
    falling_edges = np.where((current_norm[1:] < threshold) & 
                            (current_norm[:-1] >= threshold))[0]
    
    if len(rising_edges) == 0 and len(falling_edges) == 0:
        logger.warning("No edges found in current signal")
        return {
            'rise_time': np.nan,
            'fall_time': np.nan,
            'rise_time_std': np.nan,
            'fall_time_std': np.nan
        }
    
    # Calculate response times for each edge
    rise_times = []
    fall_times = []
    window_points = 100
    
    def process_edge(edge_idx: int, is_rising: bool):
        window = slice(max(0, edge_idx-window_points), 
                      min(len(time), edge_idx+window_points))
        t = time[window]
        p = pl[window]
        
        # Normalize PL in window
        p_norm = (p - np.min(p)) / (np.max(p) - np.min(p))
        
        # Determine if PL increases or decreases with current
        pl_start = np.mean(p[:10])
        pl_end = np.mean(p[-10:])
        pl_increases = pl_end > pl_start
        
        # Find 10-90% transition times (or 90-10% if PL decreases)
        if pl_increases:
            t10 = t[np.argmin(np.abs(p_norm - 0.1))]
            t90 = t[np.argmin(np.abs(p_norm - 0.9))]
        else:
            t10 = t[np.argmin(np.abs(p_norm - 0.9))]
            t90 = t[np.argmin(np.abs(p_norm - 0.1))]
        
        return abs(t90 - t10)
    
    # Process all edges
    for edge in rising_edges:
        rise_times.append(process_edge(edge, True))
    
    for edge in falling_edges:
        fall_times.append(process_edge(edge, False))
    
    return {
        'rise_time': np.mean(rise_times) if rise_times else np.nan,
        'fall_time': np.mean(fall_times) if fall_times else np.nan,
        'rise_time_std': np.std(rise_times) if len(rise_times) > 1 else np.nan,
        'fall_time_std': np.std(fall_times) if len(fall_times) > 1 else np.nan
    }


def plot_mpl_data(time: np.ndarray, pl: np.ndarray, current: np.ndarray, 
                  params: Dict[str, Any], analysis: Dict[str, float]) -> None:
    """Create publication-quality plot of MPL data."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), height_ratios=[2, 1])
    
    # Get appropriate time scaling
    time_scale, time_unit = get_readable_time_unit(np.max(time))
    
    # Plot PL and current
    ax1.plot(time*time_scale, pl, 'b-', label='PL Signal')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time*time_scale, current, 'r-', label='Current', alpha=0.7)
    
    # Add labels and legend
    ax1.set_xlabel(f'Time ({time_unit})')
    ax1.set_ylabel('PL Signal (V)')
    ax1_twin.set_ylabel('Current (mA)')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    rise_time, rise_unit = get_readable_time(analysis['rise_time'])
    fall_time, fall_unit = get_readable_time(analysis['fall_time'])
    rts, rtu = get_readable_time(analysis['rise_time_std'])
    fts, ftu = get_readable_time(analysis['fall_time_std'])

    
    # Add measurement parameters as text
    param_text = (
        f"Measurement Parameters:\n"
        f"Sample rate: {params['sample_rate']/1e6:.1f} MHz\n"
        f"Frequency: {params['frequency']:.0f} Hz\n"
        f"Averages: {params['averages']}\n"
        f"Coil R: {params['coil_resistance']:.1f} Ω\n\n"
        f"Analysis Results:\n"
        f"Contrast: {analysis['contrast']*100:.1f}%\n"
        f"Rise time: {rise_time:.1f} {rise_unit} ± {rts:.1f} {rtu}\n"
        f"Fall time: {fall_time:.1f} {fall_unit} ± {fts:.1f} {ftu}"
    )
    ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes, 
             verticalalignment='top', fontsize=8, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot zoomed region of a single transition
    # Find a rising edge for zoom plot
    current_norm = current / np.max(np.abs(current))
    threshold = 0.5
    rising_edges = np.where((current_norm[1:] > threshold) & 
                           (current_norm[:-1] <= threshold))[0]
    
    if len(rising_edges) == 0:
        logger.warning("No rising edges found for zoom plot")
        edge = len(time) // 2  # Default to middle of dataset
    else:
        edge = rising_edges[0]
    
    # Extract and plot window around edge
    window = slice(edge-100, edge+100)
    t_zoom = time[window]
    pl_zoom = pl[window]
    current_zoom = current[window]
    
    ax2.plot(t_zoom - t_zoom[0], pl_zoom, 'b-', label='PL Signal')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(t_zoom - t_zoom[0], current_zoom, 'r-',
                  label='Current', alpha=0.7)
    
    ax2.set_xlabel(f'Time ({time_unit})')
    ax2.set_ylabel('PL Signal (V)')
    ax2_twin.set_ylabel('Current (mA)')
    
    # Add title to zoom plot
    ax2.set_title('Zoomed View of Rising Edge', fontsize=10)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Set plotting style
    set_mpl_rcparams({
        "figure.constrained_layout.use": True,
        "figure.dpi": 90,
        "figure.figsize": (6 / 2.55, 6 / 2.55),
        "font.family": ("Roboto", "sans-serif"),
        "font.size": 8,
        "legend.fontsize": "x-small",
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.6,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 1.2,
        "lines.linewidth": 1.8,
        "xtick.labelsize": 8,
        "xtick.major.size": 3,
        "xtick.direction": "out",
        "ytick.labelsize": 8,
        "ytick.direction": "in",
        "ytick.major.size": 3,
        "axes.formatter.useoffset": False,
        "axes.formatter.use_mathtext": True,
        "errorbar.capsize": 3.0,
        "axes.linewidth": 1.4,
        "xtick.major.width": 1.4,
        "xtick.minor.width": 1.1,
        "ytick.major.width": 1.4,
        "ytick.minor.width": 1.1,
        "axes.labelsize": 8,
    })

    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python example_plot_mpl.py path/to/data.json")
        sys.exit(1)

    # Load data
    data_path = Path(sys.argv[1])
    if not data_path.exists():
        print(f"Error: File {data_path} not found")
        sys.exit(1)

    params, results, analysis = load_mpl_data(data_path)

    # Convert data arrays from lists to numpy arrays
    time = np.array(results['mean_time'])
    pl = np.array(results['mean_pl'])
    current = np.array(results['mean_current'])

    # Calculate response times
    timing_analysis = calculate_response_times(time, pl, current)

    # Combine original analysis with timing analysis
    analysis = {**analysis, **timing_analysis}

    # Create plot
    fig = plot_mpl_data(time, pl, current, params, analysis)

    # Save plot
    output_path = data_path.with_suffix('.analysis.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved to: {output_path}")

    rise_time, rise_unit = get_readable_time(analysis['rise_time'])
    fall_time, fall_unit = get_readable_time(analysis['fall_time'])
    rts, rtu = get_readable_time(analysis['rise_time_std'])
    fts, ftu = get_readable_time(analysis['fall_time_std'])
    # Display analysis results
    print("\nAnalysis Results:")
    print(f"Contrast: {analysis['contrast']*100:.1f}%")
    print(f"Rise time: {rise_time:.1f} {rise_unit} ± {rts:.1f} {rtu}")
    print(f"Fall time: {fall_time:.1f} {fall_unit} ± {fts:.1f} {ftu}")

    plt.show()
