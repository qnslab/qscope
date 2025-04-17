# -*- coding: utf-8 -*-
"""Utilities for saving measurement data and results.

This module provides functions for saving various types of measurement data,
including sweeps, images, and time traces. It handles:

- Directory structure and organization
- File naming and numbering
- Data formats and serialization
- Plot generation and saving
- Metadata collection and storage

Directory Structure
-----------------
Data is saved in a hierarchical structure:
<save_dir>/<YYYY>/<YYYY-MM>/<YYYY-MM-DD>_<project_name>/

Directory Selection Logic
-----------------------
- Empty project_name: Always creates/uses directory for current day
- Named project: Reuses most recent matching directory to keep related measurements together

File Naming
----------
Files within directories use 4-digit counters (0000-9999):
<counter>_<measurement_type>.<extension>

Example: 0001_mpl.json, 0002_snapshot.npy
"""

from __future__ import annotations

import sys
import typing

from qscope.gui.util.cbar import add_colorbar

from .normalisation import norm

if typing.TYPE_CHECKING:
    from qscope.meas import Measurement
    from qscope.system import System
import glob
import os
import time
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import simplejson as json
from cmap import Colormap
from loguru import logger
from matplotlib.colors import to_rgba
from reportlab.lib.units import cm, inch
from reportlab.pdfgen.canvas import Canvas

matplotlib.use("qtagg")  # for headless operation

# =============================================================================
# Helper functions
# =============================================================================


def get_command_string() -> str:
    """Get the original command string that was used to run this script.

    Returns
    -------
    str
        The full command string including all arguments
    """
    return " ".join(sys.argv)


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    # o = an object to be encoded
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def _get_latest_directory(root_dir: str, project_name: str) -> str:
    """Get the most recent directory matching our date format and project name.

    Searches for directories matching pattern:
    <root>/<YYYY>/<YYYY-MM>/<YYYY-MM-DD>_<project_name>

    Returns the most recently created matching directory, regardless of date.
    This allows related measurements to be grouped together even across days.

    Parameters
    ----------
    root_dir : str
        Root directory to search in
    project_name : str
        Project name to match

    Returns
    -------
    str
        Path to latest matching directory, or empty string if none found

    Notes
    -----
    Directory selection is based on creation time, not the date in the path.
    This means measurements can continue in a previous day's directory if it
    was the last one used for that project name.
    """
    # Use glob to find all directories matching our specific date pattern
    # TODO need to do a unix/windows check here??
    date_pattern = os.path.join(
        root_dir,
        "[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9]-[0-9][0-9]/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_"
        + project_name,
    )
    directories = glob.glob(date_pattern)

    if not directories:
        return ""  # won't match to anything

    # Sort directories by creation time and return most recent
    latest_directory = max(directories, key=os.path.getctime)

    return latest_directory


def _get_dir(system: System, project_name: str = "") -> str:
    """Get directory for saving data.

    Determines the appropriate directory for saving data based on the project name:

    - Empty project_name: Always creates/uses directory for current day
    - Named project: First tries to reuse most recent matching directory,
      creates new dated directory if none found

    Directory structure:
    <save_dir>/<YYYY>/<YYYY-MM>/<YYYY-MM-DD>_<project_name>/

    Parameters
    ----------
    system : System
        System object containing save_dir configuration
    project_name : str, optional
        Project name to use in directory path. If empty, will create/use a directory
        for the current day.

    Returns
    -------
    str
        Path to directory for saving data

    Notes
    -----
    This function implements the core directory selection logic:
    - Empty project names always get a fresh directory for today
    - Named projects reuse their most recent directory to keep related measurements together
    - New directories are created with today's date when needed
    """
    data_root = os.path.abspath(system.save_dir)
    if data_root[-1] != "/":
        data_root = data_root + "/"

    # For empty project name, always use today's date
    if not project_name:
        date = time.strftime("%Y/%Y-%m/%Y-%m-%d_")
        return os.path.normpath(data_root + date + project_name + "")

    # For named projects, try to reuse latest directory
    latest_dir = _get_latest_directory(data_root, project_name)
    if latest_dir:
        return latest_dir + "/"

    # If no matching directory found, create new one
    date = time.strftime("%Y/%Y-%m/%Y-%m-%d_")
    return os.path.normpath(data_root + date + project_name + "")


def _get_path(dr: str, meas_type: str) -> str:
    """Generate a numbered path for a new measurement file.

    Creates paths with format: <dir>/<counter>_<meas_type>
    where counter is a 4-digit number (0000-9999) that hasn't been used yet.

    Parameters
    ----------
    dr : str
        Directory to save in
    meas_type : str
        Type of measurement (used in filename)

    Returns
    -------
    str
        Full path with next available number

    Raises
    ------
    ValueError
        If directory already contains 9999 files

    Examples
    --------
    >>> _get_path("/data/today", "mpl")
    '/data/today/0000_mpl'
    >>> _get_path("/data/today", "snapshot")
    '/data/today/0001_snapshot'
    """
    counter = 0
    file_list = os.listdir(dr)
    while True:
        check_str = f"{counter:04}"  # counter:04 means 4 digits, leading zeros e.g. 0001, 0002, ...
        if not any(filter(lambda x: check_str in x, file_list)):
            break
        counter += 1
        if counter > 9999:
            raise ValueError("Too many files in directory")
    return f"{dr}//{counter:04}_{meas_type}"


def _get_base_path(system: System, project_name: str, measurement: Measurement | str):
    """Get the base path for saving data.

    Creates the full path for saving measurement data by:
    1. Getting/creating appropriate directory based on project name
    2. Generating next available numbered filename for measurement type

    Parameters
    ----------
    system : System
        System object containing save configuration
    project_name : str
        Project name for directory organization
    measurement : Measurement | str
        Either a Measurement object or measurement type string

    Returns
    -------
    str
        Full path (without extension) for saving measurement files

    Notes
    -----
    This is the main entry point for getting paths to save measurement data.
    It handles both the directory structure and file numbering.
    """
    dr = _get_dir(system, project_name)
    if not os.path.exists(dr):
        os.makedirs(dr)
    meas_name = (
        measurement
        if isinstance(measurement, str)
        else measurement.get_meas_save_name()
    )
    return _get_path(dr, meas_name)


def _get_x_multiplier(measurement: Measurement, x_data: np.ndarray) -> float:
    # Check if the measurement x-data is in time rather than frequency
    meas_type = measurement.get_meas_type_name()
    if "ESR" in meas_type:
        return 1

    max_time = np.max(x_data)
    if max_time > 1e-3:
        x_multiplier = 1e3
    elif max_time > 1e-6:
        x_multiplier = 1e6
    elif max_time > 1e-9:
        x_multiplier = 1e9
    elif max_time > 1e-12:
        x_multiplier = 1e12
    elif max_time > 1e-15:
        x_multiplier = 1e15
    else:
        x_multiplier = 1

    return int(x_multiplier)


def _get_colormap(name: str):
    # check if the color map is in the cmap package
    try:
        cmap = Colormap(name).to_mpl()
    except:
        logger.warning(f"Color map {name} not found, using default 'gray'")
        cmap = Colormap("seaborn:mako").to_mpl()
    return cmap


# =============================================================================
# Save functions
# =============================================================================


def _save_metadata(system: System, measurement: Measurement, base_path: str):
    with open(base_path + "_metadata.json", "w") as f:
        json.dump(
            {
                "meas_metadata": measurement.get_metadata(),
                "sys_metadata": system.get_metadata(),
            },
            f,
            cls=NumpyEncoder,
            indent=4,
            allow_nan=True,
        )


def _save_sweep(
    system: System,
    measurement: Measurement,
    base_path: str,
) -> tuple[str, str]:
    # save the sweep plot
    fig = plt.figure(figsize=(8, 4))
    sweep_data = measurement.get_sweep()

    x = sweep_data[0]
    # adjust the unit of the x axis for time plots according to the x multi value
    if measurement.x_label == "Time (s)":
        try:
            x_multi = _get_x_multiplier(measurement, x)
        except:
            x_multi = 1
        if x_multi == 1e3:
            measurement.x_label = "Time (ms)"
        elif x_multi == 1e6:
            measurement.x_label = "Time (us)"
        elif x_multi == 1e9:
            measurement.x_label = "Time (ns)"
        elif x_multi == 1e12:
            measurement.x_label = "Time (ps)"
        elif x_multi == 1e15:
            measurement.x_label = "Time (fs)"
    else:
        x_multi = 1

    x = x * x_multi  # needs to be a numpy array for multiplication

    y_sig = sweep_data[1]
    y_ref = sweep_data[2]

    edge_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    marker_color = to_rgba(edge_color, alpha=0.5)
    plt.plot(
        x,
        y_sig,
        "-o",
        markerfacecolor=marker_color,
        markeredgecolor=edge_color,
        label="signal",
    )
    edge_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
    marker_color = to_rgba(edge_color, alpha=0.5)
    plt.plot(
        x,
        y_ref,
        "-o",
        markerfacecolor=marker_color,
        markeredgecolor=edge_color,
        label="reference",
    )
    plt.xlim(np.min(x), np.max(x))
    plt.legend()

    plt.xlabel(measurement.x_label)
    plt.ylabel(measurement.y_label)
    spec_img_path = base_path + "_sweep.png"
    fig.savefig(spec_img_path)
    # save the data
    data_path = base_path + "_sweep.npy"
    np.save(data_path, sweep_data)
    plt.close(fig)

    return spec_img_path, data_path


def _save_sweep_w_fit(
    system: System,
    measurement: Measurement,
    base_path: str,
    fit_data: dict,
    comparison_data: dict = None,
    comparison_label: str = None,
):
    # Extract the data
    x = np.array(fit_data["x"])  # needs to be a numpy array for multiplication
    y = fit_data["y"]

    # save the sweep plot
    fig = plt.figure(figsize=(8, 4))
    # using the defaul colours adjust the facecolor of the markers to be a little transparent
    edge_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    marker_color = to_rgba(edge_color, alpha=0.5)

    try:
        x_multi = _get_x_multiplier(measurement, x)
    except:
        x_multi = 1

    plt.plot(
        x * x_multi,
        y,
        "-o",
        markerfacecolor=marker_color,
        markeredgecolor=edge_color,
        label="data",
    )
    if fit_data["x_fit"] is not None and fit_data["y_fit"] is not None:
        x_fit = (
            np.array(fit_data["x_fit"]) * x_multi
        )  # needs to be a numpy array for multiplication
        y_fit = fit_data["y_fit"]
        plt.plot(x_fit, y_fit, "-", linewidth=1, label="fit")

    # check if the comparison data is a numpy array
    if comparison_data["x"] is not None:
        
        logger.info(comparison_data["x"])
        logger.info(comparison_data["y"])

        x_comp = (
            np.array(comparison_data["x"]) * x_multi
        )
        y_comp = comparison_data["y"]
        comp_plt_options = {
                "c": "xkcd:dark gray",
                "mfc": (0, 0, 0, 0),
                "mec": "xkcd:dark gray",
                "mew": 2,
                "ms": 4,
            }
        plt.plot(
            x_comp,
            y_comp,
            "-o",
            label=comparison_label,
            zorder=-10,
            **comp_plt_options,
        )

    plt.xlim(np.min(x * x_multi), np.max(x * x_multi))
    plt.legend()

    # adjust the unit of the z axis for time plots according to the x multi value
    if measurement.x_label == "Time (s)":
        if x_multi == 1e3:
            measurement.x_label = "Time (ms)"
        elif x_multi == 1e6:
            measurement.x_label = "Time (us)"
        elif x_multi == 1e9:
            measurement.x_label = "Time (ns)"
        elif x_multi == 1e12:
            measurement.x_label = "Time (ps)"
        elif x_multi == 1e15:
            measurement.x_label = "Time (fs)"

    plt.xlabel(measurement.x_label)
    plt.ylabel(measurement.y_label)
    filename = base_path + "_sweep_w_fit.png"
    fig.savefig(filename)

    # save the data and fit as a json
    with open(base_path + "_sweep_w_fit.json", "w") as f:
        json.dump(
            fit_data,
            f,
            cls=NumpyEncoder,
            indent=4,
            allow_nan=True,
        )
    plt.close(fig)

    return filename


def _save_overview(
    measurement_type: str,
    base_path: str = None,
    spec_path: str = None,
    spec_fit_path: str = None,
    image_path: str = None,
    image_aoi_path: str = None,
    meas_params: dict = None,
    fit_results: str = None,
):
    # create a test pdf
    pdf_path = base_path + ".pdf"

    canvas = Canvas(pdf_path, pagesize=(33 * cm, 19 * cm))

    # Add a title with a big font
    titleobject = canvas.beginText(1 * cm, 18 * cm)
    titleobject.setFont("Helvetica", 20)
    titleobject.textLine(measurement_type)
    canvas.drawText(titleobject)

    # add the meausuremnet line plot to the top
    if spec_path is not None:
        canvas.drawImage(spec_path, 8 * cm, 10.5 * cm, 1.0 * 18 * cm, 1.0 * 9 * cm)

    # add the norm plot to the bottom
    if spec_fit_path is not None:
        canvas.drawImage(spec_fit_path, 8 * cm, 0.5 * cm, 1.0 * 18 * cm, 1.0 * 9 * cm)

    # add the meausuremnet imshow image to the top right
    if image_path is not None:
        canvas.drawImage(image_path, 25 * cm, 11 * cm, 8 * cm, 8 * cm)

    if image_aoi_path is not None:
        canvas.drawImage(image_aoi_path, 25 * cm, 1 * cm, 8 * cm, 8 * cm)

    # add some text
    if meas_params is not None:
        # Add the save number to the save_name in the meas_params
        meas_params["save_number"] = base_path.split("_")[-2]

        # convert the meas_params dict to a string with newlines
        meas_params = "\n".join([f"{k}: {v}" for k, v in meas_params.items()])

        # remove lines that contain certain substrings
        meas_params = "\n".join(
            [
                line
                for line in meas_params.split("\n")
                if not any([s in line for s in ["sweep_x", "meas_type"]])
            ]
        )

        textobject = canvas.beginText(1 * cm, 17 * cm)
        textobject.setFont("Helvetica", 10)
        textobject.textLines(meas_params)
        canvas.drawText(textobject)

    # Add fit results
    if fit_results is not None:
        textobject = canvas.beginText(1 * cm, 5 * cm)
        textobject.setFont("Helvetica", 10)
        textobject.textLines(fit_results)
        canvas.drawText(textobject)

    canvas.save()


def _save_integrated_image(
    system: System,
    measurement: Measurement,
    base_path: str,
    color_map: str = "seaborn:mako",
):
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        image_data = measurement.get_integrated_image()
        im = ax.imshow(
            image_data,
            cmap=_get_colormap(color_map),
            origin="upper",
        )
        cbar = plt.colorbar(im)
        cbar.set_label("Intensity", rotation=270, labelpad=20)
        aoi = measurement.get_aoi()
        if aoi is not None:
            rect = plt.Rectangle(
                (aoi[1], aoi[0]),
                aoi[3] - aoi[1],
                aoi[2] - aoi[0],
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)
        filename = base_path + "_img_sum.png"
        fig.savefig(filename)
        plt.close(fig)
        return filename
    except NotImplementedError:
        logger.info(
            f"No get_integrated_image method for {measurement.meas_type}, skipping image save",
        )
        plt.close(fig)
        pass


def _save_aoi_integrated_image(
    system: System,
    measurement: Measurement,
    base_path: str,
    color_map: str = "seaborn:mako",
):
    aoi = measurement.get_aoi()
    if aoi is None:
        aoi_slice = [slice(None), slice(None)]
    else:
        aoi_slice = [slice(aoi[0], aoi[2]), slice(aoi[1], aoi[3])]

    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        image_data = measurement.get_integrated_image()
        image_data = image_data[aoi_slice[0], aoi_slice[1]]

        if aoi is not None:
            im = ax.imshow(
                image_data,
                cmap=_get_colormap(color_map),
                origin="upper",
                extent=[aoi[1], aoi[3], aoi[2], aoi[0]],
            )
        else:
            im = ax.imshow(
                image_data,
                cmap=_get_colormap(color_map),
                origin="upper",
            )
        cbar = plt.colorbar(im)
        cbar.set_label("Intensity", rotation=270, labelpad=20)
        filename = base_path + "_img_sum_aoi.png"
        fig.savefig(filename)
        plt.close(fig)
        return filename
    except:
        logger.info(
            f"No get_integrated_image method for {measurement.meas_type}, skipping image save",
        )
        plt.close(fig)
        return None


def _save_full_data(measurement: Measurement, base_path: str):
    full_y_sig, full_y_ref = measurement.get_full_data()
    path = base_path + ".npz"
    np.savez_compressed(path, x=measurement.sweep_x, y_sig=full_y_sig, y_ref=full_y_ref)
    return path


def _save_rolling_avg(
    measurement: Measurement, base_path: str, norm_type="None"
) -> str:
    ravg_idxs, ravg_ys, ravg_yr = measurement.get_rolling_avg()
    if not ravg_ys:
        logger.debug("No data to save in _save_rolling_avg, skipping.")
        return ""  # no data to save.
    path = base_path + "_rolling_avg"
    np.savez(path, idxs=ravg_idxs, ys=ravg_ys, yr=ravg_yr)

    ravg_normd = norm(ravg_ys, ravg_yr, norm_type)

    x = measurement.sweep_x

    fig = plt.figure(figsize=(8, 4))
    plt.pcolormesh(
        x,
        ravg_idxs,
        ravg_normd,
        cmap=Colormap("seaborn:mako").to_mpl(),
        shading="nearest",
    )
    plt.tick_params(axis="y", labelsize=8, direction="in")
    plt.tick_params(axis="x", labelsize=8, direction="in")
    plt.xlabel(measurement.x_label)
    # plt.xticks([])
    fig.savefig(base_path + "_rolling_avg.png")
    plt.close(fig)
    return path


def _save_notes(base_path: str, notes: str = "") -> str:
    if notes:
        path = base_path + ".md"
        with open(path, "w") as f:
            f.write(notes)
        return path


# =============================================================================
# Public API
# =============================================================================


def save_sweep(
    system: System,
    measurement: Measurement,
    project_name: str,
    notes: str = "",
    norm_type: str = "None",
) -> str:
    base_path = _get_base_path(system, project_name, measurement)
    _save_notes(base_path, notes)
    _save_metadata(system, measurement, base_path)
    spec_path, data_path = _save_sweep(system, measurement, base_path)
    image_path = _save_integrated_image(system, measurement, base_path)
    _save_overview(
        measurement_type=measurement.get_meas_type_name(),
        base_path=base_path,
        spec_path=spec_path,
        image_path=image_path,
        meas_params=measurement.get_config_dict(),
    )
    _save_rolling_avg(measurement, base_path, norm_type=norm_type)
    return data_path


def save_sweep_w_fit(
    system: System,
    measurement: Measurement,
    project_name: str,
    x_data: typing.Sequence,
    y_data: typing.Sequence,
    x_fit: typing.Sequence,
    y_fit: typing.Sequence,
    fit_results: str = None,
    comparison_x: typing.Sequence = None,
    comparison_y: typing.Sequence = None,
    comparison_label: str = None,
    color_map: str = "gray",
    notes: str = "",
    norm_type: str = "None",
) -> str:
    base_path = _get_base_path(system, project_name, measurement)
    _save_notes(base_path, notes)
    _save_metadata(system, measurement, base_path)
    spec_path = _save_sweep(system, measurement, base_path)
    logger.info(f"comparison x: {comparison_x}")
    logger.info(f"comparison y: {comparison_y}")
    spec_fit_path = _save_sweep_w_fit(
        system,
        measurement,
        base_path,
        {"x": x_data, "y": y_data, "x_fit": x_fit, "y_fit": y_fit},
        {"x": comparison_x, "y": comparison_y},
        comparison_label
    )
    image_path = _save_integrated_image(
        system, measurement, base_path, color_map=color_map
    )
    image_aoi_path = _save_aoi_integrated_image(
        system, measurement, base_path, color_map=color_map
    )
    _save_overview(
        measurement_type=measurement.get_meas_type_name(),
        base_path=base_path,
        spec_path=spec_path[0],
        spec_fit_path=spec_fit_path,
        image_path=image_path,
        image_aoi_path=image_aoi_path,
        meas_params=measurement.get_config_dict(),
        fit_results=fit_results,
    )
    _save_rolling_avg(measurement, base_path, norm_type=norm_type)

    return base_path


def save_full_data(
    system: System,
    measurement: Measurement,
    project_name: str,
    notes: str = "",
    norm_type: str = "None",
) -> str:
    base_path = _get_base_path(system, project_name, measurement)

    _save_notes(base_path, notes)
    _save_metadata(system, measurement, base_path)
    spec_path, data_path = _save_sweep(system, measurement, base_path)
    image_path = _save_integrated_image(system, measurement, base_path)
    _save_overview(
        measurement_type=measurement.get_meas_type_name(),
        base_path=base_path,
        spec_path=spec_path,
        image_path=image_path,
        meas_params=measurement.get_config_dict(),
    )
    _save_rolling_avg(measurement, base_path, norm_type=norm_type)

    data_path = _save_full_data(measurement, base_path)
    return data_path


def save_snapshot(
    system: System, project_name: str, frame: np.ndarray, notes: str = ""
) -> str:
    base_path = _get_base_path(system, project_name, "snapshot")
    _save_notes(base_path, notes)
    np.save(base_path + ".npy", frame)
    with open(base_path + "_metadata.json", "w") as f:
        json.dump(
            {
                "command": get_command_string(),
                "timestamp": datetime.now().isoformat(),
                "sys_metadata": system.get_metadata(),
            },
            f,
            cls=NumpyEncoder,
            indent=4,
            allow_nan=True,
        )
    return base_path + ".npy"


def save_notes(system: System, project_name: str, notes: str) -> str:
    base_path = _get_base_path(system, project_name, "notes")
    _save_notes(base_path, notes)
    # Add metadata with command string
    with open(base_path + "_metadata.json", "w") as f:
        json.dump(
            {
                "command": get_command_string(),
                "timestamp": datetime.now().isoformat(),
            },
            f,
            cls=NumpyEncoder,
            indent=4,
            allow_nan=True,
        )
    return base_path


def save_latest_stream(
    system: System,
    project_name: str,
    stream_chunk: np.ndarray,
    stream_ttrace: list[float],
    color_map: str,
    notes: str = "",
) -> str:
    base_path = _get_base_path(system, project_name, "stream")
    _save_notes(base_path, notes)
    with open(base_path + "_metadata.json", "w") as f:
        metadata = system.get_metadata()
        json.dump(
            {
                "command": get_command_string(),
                "timestamp": datetime.now().isoformat(),
                "sys_metadata": metadata,
            },
            f,
            cls=NumpyEncoder,
            indent=4,
            allow_nan=True,
        )
    np.save(base_path + "_streamm_chunk.npy", stream_chunk)
    np.save(base_path + "_stream_ttrace.npy", stream_ttrace)

    if stream_chunk.ndim == 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        plot = ax.imshow(stream_chunk, cmap=_get_colormap(color_map), origin="upper")
        # Add a colorbar
        cbar = add_colorbar(
            plot,
            fig,
            ax,
            aspect=25,
            orientation="horizontal",
            location="top",
        )
        cbar.ax.tick_params(axis="x", labelsize=8, direction="in")
        cbar.set_label("Intensity (cps)")

        fig.savefig(base_path + "_stream_chunk.png")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(stream_chunk)
        ax.set_xlabel("Sample #")
        ax.set_ylabel("PL (arb. units)")
        fig.savefig(base_path + "_stream_chunk.png")

    fig, ax = plt.subplots(figsize=(8, 4))

    # get the correct time for the ttrace (if we find camera)
    # Find the camera in the metadata
    cam_keys = [key for key in metadata.keys() if "camera" in key.lower()]
    if cam_keys:
        cam_key = cam_keys[0]
        exp_time = metadata[cam_key]["exposure_time"]
        x_time = np.arange(0, len(stream_ttrace)) * exp_time
        ax.set_xlabel("Time (s)")
    else:
        x_time = np.arange(0, len(stream_ttrace))
        ax.set_xlabel("Stream #")

    ax.plot(x_time, stream_ttrace)
    ax.set_ylabel("PL (arb. units)")
    fig.savefig(base_path + "_stream_ttrace.png")
    return base_path
