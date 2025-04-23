"""# Tutorials

This section contains step-by-step tutorials to help you get started with QScope.

## Installation

1. Clone the repository and create a Python 3.11+ environment (`conda create --prefix ./conda_env python=3.11`)
2. Install with `pip install -e .`
3. Verify with `doit check_systems` and `doit test_logic` (`pip install doit` first if you want to verify)

## Hardware Dependencies

Many devices require specific drivers or libraries to function:

- Some libraries are required in the `proprietary_artefacts` folder
- For Andor cameras, install the Andor SDK or Solis software
- For Picoscope, install the Picoscope SDK
- Others must be installed separately (see []())

## Build System

Qscope uses the `doit` build system for various tasks:

```bash
# Run logic tests
doit test_logic
```

## Getting Started with QScope

This tutorial will guide you through your first steps with QScope.

### Starting the GUI

The simplest way to get started with QScope is to use the GUI:

```bash
# Basic start
qscope gui

# Start with mock system for testing
qscope gui -n mock

# Auto-connect to running server
qscope gui --auto_connect
```

### Your First Script

Here's a simple script to connect to a mock system and run a basic measurement:

```python
import matplotlib.pyplot as plt
import qscope.server
import qscope.system
import qscope.types
from qscope.scripting import meas_close_after_nsweeps

# Start logging
qscope.server.start_client_log()

# Create a connection manager
manager = qscope.server.ConnectionManager()

# Start a local server with a mock system
manager.start_local_server(
    qscope.system.SYSTEMS.MOCK,
    qscope.system.SYSTEM_TYPES.PULSED_CAMERA,
)

# Connect to the server and initialize hardware
manager.connect()
manager.startup()

# Run a test measurement
config = qscope.types.TESTING_MEAS_CONFIG
meas_id = manager.add_measurement(config)
manager.start_measurement_wait(meas_id)

# Collect data from 2 sweeps
sweep_data = meas_close_after_nsweeps(manager, meas_id, 2)

# Clean up
manager.stop_local_server()

# Plot the results
fig, ax = plt.subplots()
x, y_sig, y_ref = sweep_data
ax.plot(x, y_sig, "-o", label="Signal")
ax.plot(x, y_ref, "-o", label="Reference")
ax.legend()
plt.show()
```

### Understanding the Output

When you run the script above, you'll see log output similar to this:

```
# client.log
2024-11-12 20:45:54.581 | INFO     | qscope.server:start_bg:106 - Server started on 127.0.0.1:8850 @ pid=177053
2024-11-12 20:45:57.585 | INFO     | qscope.server.client:_open_connection:373 - Attempting full connection to server on 127.0.0.1:8850.
2024-11-12 20:45:57.587 | INFO     | qscope.server.client:open_connection:486 - Connection established on 127.0.0.1
2024-11-12 20:45:57.590 | INFO     | qscope.server.client:startup:769 - Device status: {'MockSeqGen_1': {'status': True, 'message': 'MockSeqGen opened'}, 'MockRFSource_1': {'status': True, 'message': 'MockRFSource opened'}, 'MockCamera_1': {'status': True, 'message': 'Connected to Camera: MockCamera'}}
2024-11-12 20:45:57.677 | INFO     | qscope.server.client:add_measurement:1099 - Measurement MockSGAndorESRConfig (52ff464b-102c-47bd-b008-e2d86fcad480) added.
2024-11-12 20:45:57.677 | INFO     | qscope.server.client:listen:1572 - Starting notification listener
```

This shows:

 1. The server starting
 2. The client connecting to the server
 3. Hardware devices initializing
 4. A measurement being added and started

## Working with Real Hardware

This tutorial explains how to connect to and control real hardware devices with QScope.

### Connecting to VISA Devices

Many lab instruments use the VISA protocol for communication. Here's how to connect to a VISA device:

```python
import qscope.server
import qscope.system
from qscope.util.check_hw import list_visa_devices

# List available VISA devices
devices = list_visa_devices()
for addr, info in devices.items():
    print(f"{addr}: {info['idn']}")

# Connect to a specific system with real hardware
manager = qscope.server.ConnectionManager()
manager.start_local_server(
    qscope.system.SYSTEMS.LAB1,  # Use your actual system name
)

manager.connect()
manager.startup()

# Now you can control devices through the manager
```

### Working with Cameras

To capture and process images:

```python
import numpy as np
import matplotlib.pyplot as plt
from qscope.server import ConnectionManager

# Connect to a system with a camera
manager = ConnectionManager()
manager.connect()  # Connect to running server

# Start video stream
manager.start_video()

# Capture a single frame
frame = manager.get_frame()

# Display the image
plt.figure(figsize=(8, 8))
plt.imshow(frame, cmap='viridis')
plt.colorbar(label='Counts')
plt.title('Camera Frame')
plt.show()

# Stop video stream
manager.stop_video()
```

## Creating a Complete Experiment

This tutorial walks you through setting up a more complex experiment with QScope.

### Defining a Custom Measurement Configuration

First, create a configuration for your experiment:

```python
from dataclasses import dataclass
from qscope.types.config import MeasurementConfig

@dataclass(kw_only=True, repr=False)
class MyExperimentConfig(MeasurementConfig):
    \"\"\"Configuration for my custom experiment.\"\"\"

    # Required parameters
    start_freq: float
    stop_freq: float
    num_points: int

    # Optional parameters with defaults
    integration_time: float = 1.0
    num_averages: int = 3
    power_level: float = -10.0  # dBm
```

### Running the Experiment

Now use this configuration to run your experiment:

```python
import qscope.server
import qscope.system
from my_module import MyExperimentConfig

# Connect to the server
manager = qscope.server.ConnectionManager()
manager.connect()

# Create your measurement configuration
config = MyExperimentConfig(
    start_freq=2.87e9,
    stop_freq=2.88e9,
    num_points=101,
    integration_time=0.5,
    power_level=-5.0
)

# Add and start the measurement
meas_id = manager.add_measurement(config)
manager.start_measurement(meas_id)

# Monitor progress
while True:
    status = manager.get_measurement_status(meas_id)
    if status['state'] == 'FINISHED':
        break
    print(f"Progress: {status['progress']:.1f}%")
    time.sleep(1)

# Get the final data
data = manager.get_measurement_data(meas_id)

# Plot results
plt.figure()
plt.plot(data['x_data'], data['y_data'])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Signal (a.u.)')
plt.title('Experiment Results')
plt.show()
```

### Saving and Loading Data

To save your experimental data:

```python
import json
import numpy as np
from datetime import datetime

# Get data from measurement
data = manager.get_measurement_data(meas_id)
config = manager.get_measurement_config(meas_id)

# Create a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save data as NumPy arrays
np.savez(
    f"experiment_data_{timestamp}.npz",
    x_data=data['x_data'],
    y_data=data['y_data'],
    config=json.dumps(config)
)

# To load the data later
loaded = np.load(f"experiment_data_{timestamp}.npz")
x_data = loaded['x_data']
y_data = loaded['y_data']
config = json.loads(loaded['config'])
```
"""
