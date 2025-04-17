# Installation Guide for Qscope

This document provides detailed instructions for installing and setting up Qscope on different operating systems.

## Prerequisites

Before installing Qscope, ensure you have the following:

- Python 3.11 or later
- Git
- Conda or Miniconda (recommended for environment management)
- Hardware drivers for your specific devices (see [DEVDOCS.md](./DEVDOCS.md))

## Development Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/qscope.git
cd qscope
```

### Step 2: Create a Conda Environment

From the root directory of the project:

```bash
# Create a new environment in the project directory
conda create --prefix ./conda_env python=3.11

# Activate the environment
conda activate ./conda_env
```

### Step 3: Install Dependencies

```bash
# Install doit build tool
pip install doit

# Install the package in development mode
pip install -e .
```

### Step 4: Verify Installation

To verify that Qscope is installed correctly:

```bash
# Check system compatibility
doit check_systems

# Run logic tests
doit test_logic
```

## Hardware Dependencies

Many devices require specific drivers or libraries to function:

- Some libraries are required in the `proprietary_artefacts` folder
- Others must be installed separately (see [DEVDOCS.md](./DEVDOCS.md))
- For Andor cameras, install the Andor SDK or Solis software
- For Picoscope, install the Picoscope SDK

## Build System

Qscope uses the `doit` build system for various tasks:

```bash
# Install the package (currently not the primary method)
doit install

# Run logic tests
doit test_logic

# Check system compatibility
doit check_systems
```