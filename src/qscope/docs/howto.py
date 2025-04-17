"""# How-to Guides

This section provides task-oriented guides for specific use cases with QScope.

## How to Add a New Device

This guide explains how to add support for a new hardware device to QScope.

### 1. Identify the Device Role

First, determine which role(s) your device will fulfill:

- Camera
- RF Source
- Sequence Generator
- Digitizer
- etc.

### 2. Create a Device Class

Create a new Python file in the appropriate subdirectory of `src/qscope/device/`.

```python
from qscope.device.device import Device
from qscope.types.protocols import CameraProtocol

class MyNewCamera(Device):
    \"\"\"Implementation for MyNewCamera model.

    Parameters
    ----------
    device_id : str
        Identifier for the camera
    \"\"\"

    def __init__(self, device_id):
        super().__init__()
        self.device_id = device_id
        self.connected = False

    def connect(self):
        # Implementation for connecting to the camera
        self.connected = True
        return True

    def is_connected(self):
        return self.connected

    # Implement other methods required by CameraProtocol
```

### 3. Register the Device in Your System

Add your device to the appropriate system configuration:

```python
from qscope.types.roles import MainCamera
from mymodule.mynewcamera import MyNewCamera

# In your system configuration
system.add_device_with_role(MyNewCamera(device_id="cam1"), MainCamera)
```

## How to Create a Custom Measurement

This guide explains how to create a custom measurement type in QScope.

### 1. Define a Measurement Configuration

Create a configuration class that defines the parameters for your measurement:

```python
@dataclass(kw_only=True, repr=False)
class MyCustomMeasConfig(MeasurementConfig):
    \"\"\"Configuration for my custom measurement.\"\"\"

    # Required parameters
    start_freq: float
    stop_freq: float
    num_points: int

    # Optional parameters with defaults
    integration_time: float = 1.0
    num_averages: int = 1
```

### 2. Implement the Measurement Class

Create a new measurement class that implements your measurement logic:

```python
class MyCustomMeasurement(Measurement):
    \"\"\"Custom measurement implementation.\"\"\"

    def __init__(self, system, config: MyCustomMeasConfig):
        super().__init__(system, config)
        self.rf = system.get_device_by_role(PrimaryRFSource)

    def setup(self):
        \"\"\"Prepare for measurement.\"\"\"
        # Setup code here

    def sweep(self):
        \"\"\"Execute one sweep of the measurement.\"\"\"
        # Sweep implementation

    def cleanup(self):
        \"\"\"Clean up after measurement.\"\"\"
        # Cleanup code here
```

### 3. Register Your Measurement

Register your measurement type with the system:

```python
# In your system configuration
system.register_measurement_type(MyCustomMeasConfig, MyCustomMeasurement)
```

## How to Debug Common Issues

This guide provides solutions for common problems you might encounter when using QScope.

### Connection Issues

If you're having trouble connecting to the server:

1. **Check if the server is running**:

```bash
# List running QScope servers
qscope server --list
   ```

2. **Verify network settings**:

   - Ensure the server address and port are correct
   - Check for firewall restrictions

3. **Restart the server**:

```bash
# Kill any running servers
qscope server --kill

# Start a new server
qscope server -n mock
```

### Hardware Communication Problems

If devices aren't responding:

1. **Check device connections**:

    - Verify physical connections (USB, Ethernet, etc.)
    - Check that device drivers are installed

2. **Inspect device logs**:

    - Look for error messages in the server log
    - Enable verbose logging for more details:

```python
import qscope.server
qscope.server.start_client_log(level="DEBUG")
```

3. **Test devices individually**:

    - Use device-specific diagnostic tools
    - Try connecting with vendor software

### Measurement Issues

If measurements aren't working as expected:

1. **Verify configuration**:

    - Check all measurement parameters
    - Ensure required devices are available

2. **Monitor measurement state**:

```python
# Get measurement status
status = manager.get_measurement_status(meas_id)
print(status)
```

3. **Inspect data during measurement**:

```python
# Get current data
data = manager.get_measurement_data(meas_id)
```

## How to Extend the GUI

This guide explains how to add custom widgets and functionality to the QScope GUI.

### Adding a Custom Widget

1. **Create a widget class**:

```python
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton

class MyCustomWidget(QWidget):
   def __init__(self, parent=None):
       super().__init__(parent)
       self.layout = QVBoxLayout(self)

       self.button = QPushButton("Click Me")
       self.button.clicked.connect(self.on_button_click)

       self.layout.addWidget(self.button)

   def on_button_click(self):
       print("Button clicked!")
```

2. **Register the widget with the GUI**:

```python
# In your GUI initialization code
from my_module import MyCustomWidget

# Add to a specific dock area
self.add_widget_to_dock(MyCustomWidget(), "My Widget", area="right")
```

### Creating a Custom Tab

To add a new tab to the main interface:

```python
# Create your tab widget
tab_widget = QWidget()
tab_layout = QVBoxLayout(tab_widget)
# Add your controls to tab_layout

# Add to the main tab widget
main_tabs.addTab(tab_widget, "My Custom Tab")
```

### Build the documentation

View the full documentation by running:
```bash
pdoc3 --output-dir docs/ --html --template-dir docs/ --force --skip-errors ./src/qscope/
```

Then open `docs/qscope/index.html` in your browser.
Commit the above change to main to update the github pages docs website.

"""
