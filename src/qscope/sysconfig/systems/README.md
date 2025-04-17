# QScope System Configurations

This directory contains the default system configurations for QScope.
These configurations are version controlled and serve as templates and defaults.

Users can copy and customize these configurations to their local `~/.qscope/systems.ini` file.
User configurations take precedence over these package defaults.

## Available Configurations

- `mock.ini`: Mock system for testing and development
- `chroma.ini`: Chroma microscope configuration
- `hqdm.ini`: HQDM system configuration
- `attodry.ini`: AttoDry system configuration
- `gmx.ini`: GMX system configuration

## Usage

These configurations can be:
1. Used directly (read-only)
2. Copied to user space for customization:
   ```bash
   qscope system copy mock custom_mock
   ```
3. Used as templates for new system configurations
