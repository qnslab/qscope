"""
Graphical user interface for Qscope.

This module provides a PyQt-based GUI for controlling Qscope experiments,
including:

- Connection management to servers
- Measurement configuration and control
- Real-time data visualization
- Camera viewing and control
- Hardware parameter adjustment

The GUI is designed to be user-friendly while providing access to all
the capabilities of the Qscope framework.

Examples
--------
Starting the GUI from Python:
```python
from qscope.gui import start_gui
start_gui()
```

Or from command line:
```bash
$ qscope gui -n mock
```

See Also
--------
qscope.server : Server-client communication
qscope.cli : Command-line interface
"""
