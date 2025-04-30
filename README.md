# Qscope

![Project Status: Beta](https://img.shields.io/badge/status-beta-yellow)
[![Docs](https://img.shields.io/badge/docs-v0.2-blue)](https://qnslab.github.io/qscope/qscope/index.html)

`Quantum Spin Control and Optics Programming Engine`

A (python) library for controlling physics experiments via script or GUI front-ends and a hardware/measurement server.
Currently implements spin defect measurements, in particular for widefield/camera-based experiments.

| ![Qscope GUI](./docs/images/qscope_gui.png) |
|:--:|
| *GUI for widefield defect microscopy* |

<!-- TODO: add scripting example (gif?) -->

## Pitch

Do you:

- Need some software to control a widefield defect microscope?
- Have long-running quantitative (camera) spectroscopy experiments?
- Want to borrow a framework for controlling your lab, with a server-client model?

Then this project might be of interest to you!

## Quick Start

### Installation

For the uv approach on this branch
- Clone the repo.
- Install uv to powershell using thier instructions https://docs.astral.sh/uv/getting-started/installation/#cargo
- Navigate to the repo folder in powershell
- Using the command 'uv add .' to install qscope
- uv doesn't install all of the packages so then us 'uv pip install pylablib'

From this point you can now use qscope through uv using the following:
'uv run qscope' followed by the qscope commands. 
e.g. 'uv run qscope gui -n mock'


Clone the repo & `pip install` it. For more info see [Installation](https://qnslab.github.io/qscope/qscope/docs/tutorials.html#installation).

### Starting the GUI

```bash
# Basic start
qscope gui

# Start with mock system started in subprocess
qscope gui -n mock
```

## Overview

Qscope features a server-client architecture designed for flexible control of experiments:
- **Server**: Manages hardware devices and oversees experiment logic. Runs it its own process.
- **Client**: API to control the system on a server process. Can be used directly in script or through the GUI wrapper.
- **Measurement**: Handles experiment logic with parameter sweeps. Each measurement runs in an async loop as a state machine.

Multiple clients (script or gui) can connect simultaneously to control the system, run measurements, and monitor data in real-time.

## Key Features

- **Distributed Architecture**: Server-client model allows multiple control points
- **Hardware Abstraction**: Unified interface for diverse scientific instruments (defined by their `Role` and `RoleInterface`)
- **Measurement Framework**: Standardized approach to common swept measurements (T1, ODMR, PL spectra, ...)
- **Real-time Data Visualization**: Live data monitoring and analysis, separated from experiment control logic
- **Scripting & GUI Interfaces**: Flexible control options for different needs

## Current Systems

For documentation on supported hardware see the relevant [docs page](https://qnslab.github.io/qscope/qscope/supportedhardware.html).

- **Widefield defect microscope (QDM)**: ESR, T1, Rabi measurements, etc.
- Ensemble NV measurements
- Photodiode-based systems: magnetophotoluminescence
- Any pulsed camera measurement

## Planned Systems

- Rastered microscopy systems (confocal, AFM)

## Documentation

QScope documentation is organized according to the Diátaxis framework:

- **[Tutorials](https://qnslab.github.io/qscope/qscope/docs/tutorials.html)**: Step-by-step guides for beginners.
- **[How-to Guides](https://qnslab.github.io/qscope/qscope/docs/howto.html)**: Task-oriented guides for specific use cases.
- **[Explanations](https://qnslab.github.io/qscope/qscope/docs/explanation.html)**: Conceptual documentation about the system.
- **[Reference](https://qnslab.github.io/qscope/qscope/index.html#header-submodules)**: Generated API documentation.

- **[Hardware Support](https://qnslab.github.io/qscope/qscope/docs/supportedhardware.html)**: Documentation for the hardware devices supported.
- **[MPL Documentation](https://qnslab.github.io/qscope/qscope/docs/mpl.html)**: The CLI MPL docs are separate for now, due to its different architecture.