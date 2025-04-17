"""
Command-line interface for Qscope.

This module provides command-line tools for interacting with Qscope,
including:

- Starting the GUI
- Managing server instances
- Hardware diagnostics and testing
- Utility commands for system management

The CLI is built using the Click framework and provides a hierarchical
command structure with consistent help documentation.

Examples
--------
Starting the GUI with a mock system:
```bash
$ qscope gui -n mock
```

Listing available VISA devices:
```bash
$ qscope visa
```

See Also
--------
qscope.server : Server-client communication
qscope.gui : Graphical user interface


CLI Tree
--------

```
$ qscope --tree
cli
└── dev
    └── smu
        └── read
        └── set
        └── zero
└── gui
└── kill
└── list
└── mpl
    └── analyze
    └── trigtime
    └── trigtrace
└── ports
└── server
└── system
    └── check_hardware
    └── copy
    └── install
    └── install-all
    └── list
└── visa
```

CLI Help
--------
```
$ qscope --help
Usage: qscope [OPTIONS] COMMAND [ARGS]...

  QScope - Quantum Diamond Microscope (QDM) control software.

  A comprehensive control system for quantum diamond microscopes, providing:

  - Server-based hardware control and coordination

  - GUI interface for microscope operation and data visualization

  - Command-line tools for system management, hardware charac etc.

Options:
  --tree  Show command tree from this point
  --help  Show this message and exit.

Commands:
  dev     Hardware device control tools.
  gui     Start the QScope GUI.
  kill    Kill all running QScope servers.
  list    List all running QScope servers.
  mpl     Magnetophotoluminescence (MPL) time trace measurement tools.
  ports   List all available COM ports.
  server  Start the QScope server.
  system  Manage system configurations.
  visa    List all available VISA devices.
```

"""

import click

from .base import cli, tree_option
from .mpl import mpl

# Register subcommands directly under cli
cli.add_command(mpl)

# Import and register device commands
from .dev import dev

cli.add_command(dev)
