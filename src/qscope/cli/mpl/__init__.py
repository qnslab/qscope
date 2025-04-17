"""Magnetophotoluminescence (MPL) time trace measurement tools.

This module provides command-line tools for working with time-resolved
magnetophotoluminescence measurements. It includes functionality for:

- Capturing time-resolved PL traces with trigger synchronization
- Analyzing PL transitions and dynamics
- Fitting exponential models to rise/fall characteristics
- Processing and visualizing MPL time trace data

The commands in this module are designed to work with various hardware
configurations and provide robust analysis capabilities for quantum
optical experiments.

[See MPL docs](../../docs/mpl.html) for more.
"""

import click

from qscope.cli.base import tree_option


@click.group()
@tree_option
def mpl():
    """Magnetophotoluminescence (MPL) time trace measurement tools."""
    pass


# Import commands after group definition
from .analyze import analyze
from .trigtime import trigtime
from .trigtrace import trigtrace

# Register commands
mpl.add_command(trigtime)
mpl.add_command(trigtrace)
mpl.add_command(analyze)

__all__ = ["mpl"]
