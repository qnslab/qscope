"""
QScope Documentation Module

This module contains documentation organized according to the Diátaxis framework:

- Tutorials: Step-by-step guides for beginners
- How-to Guides: Task-oriented guides for specific use cases
- Explanations: Conceptual documentation about the system

As well as:
- [Hardware Support](docs/supportedhardware.html): Documentation for the hardware devices supported.
- [MPL documentation](docs/mpl.html): The CLI MPL docs are separate for now, due to its different architecture.

These documentation modules are not meant to be imported or used in code.
They exist solely to provide structured documentation through pdoc3.
"""

from . import explanation, howto, tutorials, supportedhardware, mpl
