"""Analysis tools for MPL triggered trace measurements."""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from qscope.cli.mpl.plots import plot_mpl_results as plot_results
from qscope.fitting.mpl import MPLFitter


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option(
    "--show-individual/--no-show-individual",
    default=True,
    help="Plot individual traces along with average (default: True)",
)
@click.option(
    "--fit/--no-fit",
    default=False,
    help="Fit exponential curves to transitions (default: disabled)",
)
@click.option(
    "--fit-type",
    type=click.Choice(["single", "double"]),
    default="single",
    help="Type of exponential fit to use (default: single)",
)
@click.option(
    "--save/--no-save",
    default=False,
    help="Save replotted figure to file (default: False)",
)
@click.option(
    "--output-path",
    "-o",
    type=str,
    default=None,
    help="Custom path for saving replotted figure (default: same as input with _replot suffix)",
)
@click.option(
    "--plot-pulses/--no-plot-pulses",
    default=False,
    help="Plot detailed pulse analysis for each transition (default: False)",
)
def analyze(
    filename: str,
    show_individual: bool = True,
    fit: bool = False,
    fit_type: str = "single",
    save: bool = False,
    output_path: Optional[str] = None,
    plot_pulses: bool = False,
) -> None:
    """Analyze and replot saved MPL triggered trace measurement data.

    FILENAME: Path to saved measurement JSON file and associated NPZ data file

    This command loads previously saved MPL triggered trace data and allows
    replotting with different visualization options. It can also recalculate
    analysis metrics from the raw data.

    Example usage:
    qscope mpl analyze path/to/measurement.json
    """
    try:
        # Load metadata from JSON file
        with open(filename, "r") as f:
            metadata = json.load(f)

        # Determine path to raw data NPZ file
        npz_path = str(filename).replace(".json", "_raw.npz")
        if not Path(npz_path).exists():
            raise FileNotFoundError(f"Raw data file not found: {npz_path}")

        # Load raw data from NPZ file
        raw_data = np.load(npz_path)

        # Extract parameters from metadata
        params = metadata["parameters"]
        sample_rate = params["sample_rate"]
        downsample_ratio = params.get("downsample_ratio", 1)
        current = params["current"]
        frequency = params["frequency"]
        averages = params["averages"]
        coil_resistance = params["coil_resistance"]
        trigger_threshold = params.get("trigger_threshold", 2.5)

        # Reconstruct results dictionary
        results = {
            "mean_time": raw_data["mean_time"],
            "mean_clock": raw_data["mean_clock"],
            "mean_pl": raw_data["mean_pl"],
            "std_pl": raw_data["std_pl"],
            "analysis": metadata["analysis"],
            "time_data": raw_data["time_data"],
            "clock_data": raw_data["clock_data"],
            "pl_data": raw_data["pl_data"],
        }

        # Always perform transition analysis if not already in results
        if "mean_rise_time" not in results["analysis"]:
            # Run transition analysis
            click.echo("\nPerforming transition analysis...")
            transition_analysis = MPLFitter.analyze_transitions(
                results["mean_time"],
                results["mean_pl"],
                results["mean_clock"],
                trigger_threshold,
                plot_pulses=plot_pulses,
                use_cache=True,
            )

            # Update analysis results
            results["analysis"].update(transition_analysis)

        # Perform exponential fitting if requested
        if fit and (
            "mean_rise_tau" not in results["analysis"]
            or (fit_type == "double" and "mean_rise_tau1" not in results["analysis"])
        ):
            # Use detrended data if available and detrending was applied in the original measurement
            analysis_pl = results["mean_pl"]
            if "mean_pl_raw" in raw_data and "detrend" in params and params["detrend"]:
                logger.info("Using detrended data for exponential fitting")

            click.echo(f"\nFitting {fit_type} exponential curves to transitions...")
            exponential_fits = MPLFitter.fit_exponential_curves(
                results["mean_time"],
                analysis_pl,
                results["analysis"],
                fit_type=fit_type,
            )

            # Clean up function objects before updating analysis results
            def clean_for_json(obj):
                if isinstance(obj, dict):
                    return {
                        k: clean_for_json(v)
                        for k, v in obj.items()
                        if not callable(v) and k != "function"
                    }
                elif isinstance(obj, list):
                    return [clean_for_json(item) for item in obj if not callable(item)]
                else:
                    return obj

            # Clean fit parameters
            for fit_key in ["rise_fit_params", "fall_fit_params"]:
                if fit_key in exponential_fits:
                    if isinstance(exponential_fits[fit_key], list):
                        exponential_fits[fit_key] = [
                            clean_for_json(param) for param in exponential_fits[fit_key]
                        ]

            # Update analysis results
            results["analysis"].update(exponential_fits)

        # Print only key analysis results
        if "mean_rise_time" in results["analysis"] and not np.isnan(
            results["analysis"]["mean_rise_time"]
        ):
            click.echo(
                f"Field ON: {results['analysis']['mean_rise_time'] * 1000:.2f} ms, "
                + f"Field OFF: {results['analysis']['mean_fall_time'] * 1000:.2f} ms, "
                + f"Contrast: {abs(results['analysis']['mean_contrast']):.2f}%"
            )

            # Add time constants if available
            if fit:
                if (
                    fit_type == "single"
                    and "mean_rise_tau" in results["analysis"]
                    and not np.isnan(results["analysis"]["mean_rise_tau"])
                ):
                    click.echo(
                        f"Time constants - ON: {results['analysis']['mean_rise_tau'] * 1000:.2f} ms, "
                        + f"OFF: {results['analysis']['mean_fall_tau'] * 1000:.2f} ms"
                    )
                elif fit_type == "double" and "mean_rise_tau1" in results["analysis"]:
                    click.echo(f"Double exponential time constants:")
                    click.echo(
                        f"  ON: τ1={results['analysis']['mean_rise_tau1'] * 1000:.2f} ms, "
                        + f"τ2={results['analysis']['mean_rise_tau2'] * 1000:.2f} ms"
                    )
                    click.echo(
                        f"  OFF: τ1={results['analysis']['mean_fall_tau1'] * 1000:.2f} ms, "
                        + f"τ2={results['analysis']['mean_fall_tau2'] * 1000:.2f} ms"
                    )

        # Calculate mean current if not in the raw data
        if "mean_current" not in raw_data:
            # Derive current from clock signal using rising edge detection
            # Find rising edges in the clock signal
            clock_threshold = trigger_threshold
            clock_high = results["mean_clock"] > clock_threshold
            rising_edges = np.where(np.diff(clock_high.astype(int)) > 0)[0]

            # Initialize current array (starts OFF due to pre-trigger capture)
            mean_current = np.zeros_like(results["mean_clock"])
            current_on = False  # Start with current OFF

            # Toggle state at each rising edge
            for edge_idx in rising_edges:
                current_on = not current_on  # Toggle state at each rising edge
                if edge_idx + 1 < len(mean_current):
                    mean_current[edge_idx + 1 :] = current if current_on else 0

            results["mean_current"] = mean_current
        else:
            results["mean_current"] = raw_data["mean_current"]

        # Plot results
        fig, axes = plot_results(
            results=results,
            sample_rate=sample_rate,
            downsample_ratio=downsample_ratio,
            current=current,
            frequency=frequency,
            averages=averages,
            coil_resistance=coil_resistance,
            trigger_threshold=trigger_threshold,
            show_individual=show_individual,
            no_show=False,
        )

        # Print analysis results
        click.echo("\nMeasurement Analysis:")
        click.echo(f"Timestamp: {metadata.get('timestamp', 'Not available')}")
        if "command" in metadata:
            click.echo(f"Original command: {metadata['command']}")
        click.echo("\nParameters:")
        for key, value in params.items():
            click.echo(f"  {key}: {value}")

        if "analysis" in metadata:
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
                if key in metadata["analysis"] and not np.isnan(
                    metadata["analysis"][key]
                ):
                    # Format time values in milliseconds
                    if "time" in key or "tau" in key:
                        value = f"{metadata['analysis'][key] * 1000:.2f} ms"
                    # Format contrast as percentage
                    elif key == "mean_contrast":
                        value = f"{abs(metadata['analysis'][key]):.2f}%"
                    else:
                        value = metadata["analysis"][key]
                    click.echo(f"  {label}: {value}")

        # Save replotted figure if requested
        if save:
            if output_path is None:
                output_path = str(filename).replace(".json", "_replot.png")
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            click.echo(f"\nReplotted figure saved to {output_path}")

            # Save analysis metadata with both original and analysis commands
            analysis_metadata_path = str(filename).replace(".json", "_reanalysis.json")
            with open(analysis_metadata_path, "w") as f:
                analysis_metadata = {
                    "original_command": metadata.get("command", "Not available"),
                    "analysis_command": " ".join(sys.argv),
                    "timestamp": datetime.now().isoformat(),
                    "parameters": params,
                    "analysis": results["analysis"],
                }
                json.dump(analysis_metadata, f, indent=4)
            click.echo(f"Analysis metadata saved to {analysis_metadata_path}")

        plt.show()

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
    except json.JSONDecodeError:
        click.echo(f"Error: Invalid JSON format in {filename}", err=True)
    except Exception as e:
        logger.exception(f"Error analyzing file: {e}")
        click.echo(f"Error analyzing file: {e}", err=True)
