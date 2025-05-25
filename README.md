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

## Programming a new measurement
In order to make a new measurement that is accessible through the client-server interface there is multiple locations in the code that need to be updated. 

1. You need to programming the new pulse sequence itself. See examples under devices/seqgen/pulseblaster. 
2. Once the sequence is programmed it need to be imported into the seqgen main script. For example with the pulseblaster, it needs to be imported into pulseblaster.py 
```
from .seq_cw_esr_long_exposure import seq_cw_esr_long_exposure
from .seq_cw_esr import seq_cw_esr
from .seq_p_esr import seq_p_esr
from .seq_rabi import seq_rabi
```
3. Next a measurement needs to be defined. If the measurement type is similar in terms of setup and collection we can make a subclass of an already programmed measurement and change the configuration reference and import the new config file from types. e.g. 
```
from qscope.types import (
    MAIN_CAMERA,
    PRIMARY_RF,
    SEQUENCE_GEN,
    MeasurementFrame,
    SGAndorCWESRConfig,
    SGAndorCWESRLongExpConfig,
    SGAndorPESRConfig,
)

@requires_hardware(
    SGCameraSystem,
    roles=(MAIN_CAMERA, SEQUENCE_GEN, PRIMARY_RF),
)
class SGAndorPESR(SGAndorESRBase):
    """Pulsed ESR measurement using camera detection."""

    _meas_config_type = SGAndorPESRConfig
    meas_config: SGAndorPESRConfig
```

4. Make a new configuration in types/config.py  which defines the input variable for the pulse sequence that will be passed by the client-server connection. e.g. 
```
@dataclass(kw_only=True, repr=False)
class SGAndorCWESRConfig(CameraConfig):
    """Configuration for CW ESR measurements."""

    meas_type: str = "SGAndorCWESR"
    save_name: str = "ESR"
    avg_per_point: int
    fmod_freq: float
    rf_pow: float
    laser_delay: float
```
Make sure you add the new config name to the __inti__.py imports and add it to the __all__ list of commands at the end of that file.

Now you should be able to run the measurement through a script. To add this to the gui itself requires more work to make sure the gui can also access all of the required files. 