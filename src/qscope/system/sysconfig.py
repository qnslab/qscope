"""System configuration handling for QScope.

This module provides functionality for managing QScope system configurations through INI files.

Configuration versions:
- v1: Legacy code-based configurations (pre-INI format)
- v2: Initial INI format with basic system and device settings
The configuration system allows:

- Loading and validating system configurations
- Creating and copying system configurations
- Migrating from code-based sysconfig to INI files
- Managing system configurations via CLI

The INI file format uses sections for each system, with the following structure:

[system_name]
# Basic settings
system_type = SGCameraSystem
save_dir = ./output/

# Objectives configuration
objective.20x = 1e-6
objective.40x = 0.5e-6

# Device configurations
device.camera.type = MockCamera
device.camera.role = MAIN_CAMERA
device.camera.exposure = 0.1

device.seqgen.type = MockSeqGen
device.seqgen.role = SEQUENCE_GEN
device.seqgen.board = 0

See Also
--------
qscope.system.config : Code-based system configurations
qscope.system.system : System class implementations
qscope.types : Device role definitions
"""

from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
from typing import Type

from loguru import logger

from qscope.system.base_config import ConfigVersion, SystemConfig
from qscope.types import PREFIX_TO_ROLE, get_valid_device_types

# Forward declare system types, will be populated when needed
VALID_SYSTEM_TYPES = {}


def validate_system_config(config: ConfigParser, section: str) -> tuple[bool, str]:
    """Validate system configuration section.

    Parameters
    ----------
    config : ConfigParser
        ConfigParser instance containing the configuration
    section : str
        Name of the section to validate

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)
    """
    # Check required fields
    if "system_type" not in config[section] or "save_dir" not in config[section]:
        return False, "Missing required fields: system_type and save_dir"

    # Import needed types and populate VALID_SYSTEM_TYPES if empty
    if not VALID_SYSTEM_TYPES:
        from qscope.system.system import SGCameraSystem, SGSystem, System

        VALID_SYSTEM_TYPES.update(
            {"SGCameraSystem": SGCameraSystem, "SGSystem": SGSystem, "System": System}
        )
    from qscope.device import MockCamera, MockRFSource, MockSeqGen
    from qscope.types import MAIN_CAMERA, PREFIX_TO_ROLE, PRIMARY_RF, SEQUENCE_GEN

    # Validate system type
    system_type = config[section]["system_type"]
    if system_type not in VALID_SYSTEM_TYPES:
        return False, f"Invalid system type: {system_type}"

    # Collect device roles from config
    device_roles = set()
    for key in config[section]:
        if key.startswith("device.") and key.endswith(".type"):
            prefix = key[len("device.") : -(len(".type"))]
            if prefix not in PREFIX_TO_ROLE:
                return False, f"Invalid device prefix: {prefix}"
            device_roles.add(PREFIX_TO_ROLE[prefix])

            # Validate device type
            dev_type = config[section][key]
            valid_types = get_valid_device_types()
            if dev_type not in valid_types:
                return False, f"Invalid device type: {dev_type}"

    # Check system role requirements
    if system_type == "SGCameraSystem":
        required_roles = {SEQUENCE_GEN, PRIMARY_RF, MAIN_CAMERA}
        missing = required_roles - device_roles
        if missing:
            return (
                False,
                f"Missing required roles: {', '.join(str(r) for r in missing)}",
            )

    return True, ""


def load_system_config(system_name: str) -> SystemConfig:
    """Load system configuration from INI file.

    Checks both package defaults and user sysconfig (~/.qscope/systems.ini).
    User sysconfig take precedence over package defaults.

    Parameters
    ----------
    system_name : str
        Name of the system configuration to load

    Returns
    -------
    SystemConfig
        Loaded and validated system configuration object

    Notes
    -----
    Search order:
    1. ~/.qscope/systems.ini
    2. package/sysconfig/systems/<system_name>.ini
    """
    # Check user config first
    user_config_dir = Path.home() / ".qscope"
    user_systems_file = user_config_dir / "systems.ini"

    # Check package defaults
    import qscope

    package_config_dir = Path(qscope.__file__).parent / "sysconfig" / "systems"
    package_system_file = package_config_dir / f"{system_name.lower()}.ini"

    config = ConfigParser()

    # Try user config first
    if user_systems_file.exists():
        config.read(user_systems_file)
        # Case-insensitive section lookup
        for section in config.sections():
            if section.lower() == system_name.lower():
                return _create_system_config(config, section)

    # Fall back to package defaults
    if package_system_file.exists():
        config.read(package_system_file)
        # Case-insensitive section lookup
        for section in config.sections():
            if section.lower() == system_name.lower():
                return _create_system_config(config, section)

    raise ValueError(
        f"System '{system_name}' not found in:\n"
        f"- User config: {user_systems_file}\n"
        f"- Package config: {package_system_file}"
    )


def create_default_systems_file(file_path: Path) -> None:
    """Create default systems.ini file with example configurations."""
    from loguru import logger

    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Creating default systems file at {file_path}")

    config = ConfigParser()
    config.read_dict(
        {
            "DEFAULT": {
                "# QScope System Configurations": "",
                "# ---------------------------": "",
                "#": "",
                "# This file contains system configurations for QScope.": "",
                "# Configuration format version: v2": "",
                "version": ConfigVersion.CURRENT,
                "# Each section represents a different system configuration.": "",
                "#": "",
                "# Required fields for each system:": "",
                "# - system_type: Type of system (e.g., SGCameraSystem)": "",
                "# - save_dir: Directory for saving data": "",
                "#": "",
                "# Device configuration format:": "",
                "# device.<name>.type: Device class name": "",
                "# device.<name>.role: Device role in system": "",
                "# device.<name>.<parameter>: Device specific parameters": "",
            }
        }
    )

    # Add Mock system as example
    config["Mock"] = {
        "system_type": "SGCameraSystem",
        "save_dir": "./mock_output/",
        "use_save_dir_only": "true",
        "objective.20x": "nan",
        "device.seqgen.type": "MockSeqGen",
        "device.seqgen.role": "SEQUENCE_GEN",
        "device.rf.type": "MockRFSource",
        "device.rf.role": "PRIMARY_RF",
        "device.camera.type": "MockCamera",
        "device.camera.role": "MAIN_CAMERA",
    }

    # Read existing config if it exists
    if file_path.exists():
        logger.debug("Found existing config file")
        existing_config = ConfigParser()
        existing_config.read(file_path)

        logger.debug(f"Existing sections: {existing_config.sections()}")
        logger.debug(f"Existing DEFAULT settings: {dict(existing_config['DEFAULT'])}")

        # Debug dump of existing config
        logger.debug("Full existing config contents:")
        for section in existing_config.sections():
            logger.debug(f"[{section}]")
            for key, value in existing_config[section].items():
                logger.debug(f"{key} = {value}")

        # Copy existing sections to new config
        for section in existing_config.sections():
            if section not in config.sections():
                logger.debug(f"Preserving existing section: {section}")
                # Create new section and copy items individually
                config[section] = {}
                for key, value in existing_config[section].items():
                    config[section][key] = value
                    logger.debug(f"Copied {section}.{key} = {value}")
            else:
                logger.debug(f"Section {section} already exists in new config")
                # Merge items from existing section
                for key, value in existing_config[section].items():
                    if key not in config[section]:
                        config[section][key] = value
                        logger.debug(f"Merged {section}.{key} = {value}")

        # Copy DEFAULT section settings
        if "DEFAULT" in existing_config:
            for key, value in existing_config["DEFAULT"].items():
                if key not in config["DEFAULT"]:
                    logger.debug(f"Preserving DEFAULT setting: {key}={value}")
                    config["DEFAULT"][key] = value
                else:
                    logger.debug(f"DEFAULT setting {key} already exists in new config")

    # Create parent directories if needed
    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug("Writing final config:")
    for section in config.sections():
        logger.debug(f"Section {section}: {dict(config[section])}")
    logger.debug(f"Final DEFAULT settings: {dict(config['DEFAULT'])}")

    with file_path.open("w") as f:
        config.write(f)


def list_available_systems() -> dict[str, str]:
    """List all available system configurations.

    Returns
    -------
    dict[str, str]
        Dictionary mapping system names to their source ('user' or 'package')

    Notes
    -----
    - Checks both package defaults and user configurations
    - User configurations take precedence over package defaults
    - Does not validate configurations

    Examples
    --------
    >>> systems = list_available_systems()
    >>> print(systems)
    {'Mock': 'package', 'CustomMock': 'user'}
    """
    systems = {}

    # Check package defaults
    import qscope

    package_config_dir = Path(qscope.__file__).parent / "sysconfig" / "systems"
    if package_config_dir.exists():
        for file in package_config_dir.glob("*.ini"):
            config = ConfigParser()
            config.read(file)
            for section in config.sections():
                systems[section] = "package"

    # Check user sysconfig (overrides package defaults)
    user_systems_file = Path.home() / ".qscope" / "systems.ini"
    if user_systems_file.exists():
        config = ConfigParser()
        config.read(user_systems_file)
        for section in config.sections():
            systems[section] = "user"

    return systems


def copy_system_config(source: str, dest: str, to_user: bool = True) -> None:
    """Copy an existing system configuration to create a new one.

    Creates a new system configuration by duplicating an existing one.
    Useful for creating variations of existing configurations.

    Parameters
    ----------
    source : str
        Name of the existing system configuration to copy
    dest : str
        Name for the new system configuration

    Raises
    ------
    FileNotFoundError
        If systems.ini configuration file doesn't exist
    ValueError
        If source system doesn't exist or destination name already in use

    Notes
    -----
    - Copies all settings including:
        - System type
        - Save directory
        - Device configurations
        - Objective settings
    - New configuration can be modified after copying

    Examples
    --------
    >>> copy_system_config('Mock', 'CustomMock')
    # Creates new 'CustomMock' configuration identical to 'Mock'
    """
    config_file = Path.home() / ".qscope" / "systems.ini"
    if not config_file.exists():
        raise FileNotFoundError("No systems configuration file found")

    config = ConfigParser()
    config.read(config_file)

    if source not in config.sections():
        raise ValueError(f"Source system '{source}' not found")
    if dest in config.sections():
        raise ValueError(f"Destination system '{dest}' already exists")

    config[dest] = config[source]

    with config_file.open("w") as f:
        config.write(f)


def install_system_config(name: str) -> None:
    """Install a package system configuration to user directory.

    Parameters
    ----------
    name : str
        Name of the system configuration to install

    Raises
    ------
    FileNotFoundError
        If package configuration doesn't exist
    ValueError
        If system name is invalid or already exists in user config
    """
    import qscope

    package_config_dir = Path(qscope.__file__).parent / "sysconfig" / "systems"
    package_system_file = package_config_dir / f"{name.lower()}.ini"

    if not package_system_file.exists():
        raise FileNotFoundError(f"Package configuration '{name}' not found")

    # Read package config
    config = ConfigParser()
    config.read(package_system_file)

    # Verify system exists in package config
    if name not in config.sections():
        raise ValueError(f"System '{name}' not found in package configuration")

    # Create user config directory if needed
    user_config_dir = Path.home() / ".qscope"
    user_config_dir.mkdir(parents=True, exist_ok=True)

    user_systems_file = user_config_dir / "systems.ini"

    # Read existing user config if it exists
    if user_systems_file.exists():
        user_config = ConfigParser()
        user_config.read(user_systems_file)
        if name in user_config.sections():
            raise ValueError(f"System '{name}' already exists in user configuration")

    # Copy system config to user file
    if user_systems_file.exists():
        user_config = ConfigParser()
        user_config.read(user_systems_file)
    else:
        user_config = ConfigParser()
        user_config["DEFAULT"] = {"version": ConfigVersion.CURRENT.value}

    user_config[name] = config[name]

    with user_systems_file.open("w") as f:
        user_config.write(f)


def install_all_system_configs() -> list[str]:
    """Install all package system configurations to user directory.

    Returns
    -------
    list[str]
        Names of installed configurations

    Notes
    -----
    - Skips configurations that already exist in user config
    - Creates user config directory if needed
    """
    import qscope

    package_config_dir = Path(qscope.__file__).parent / "sysconfig" / "systems"

    installed = []

    for ini_file in package_config_dir.glob("*.ini"):
        if ini_file.name == "README.md":
            continue

        config = ConfigParser()
        config.read(ini_file)

        for section in config.sections():
            try:
                install_system_config(section)
                installed.append(section)
            except ValueError:
                # Skip if already exists in user config
                pass

    return installed


def get_config_version(config: ConfigParser | Type[SystemConfig]) -> ConfigVersion:
    """Get the version of a configuration.

    Parameters
    ----------
    config : ConfigParser | Type[SystemConfig]
        Configuration to check version of, either INI or code-based

    Returns
    -------
    ConfigVersion
        Version of the configuration

    Notes
    -----
    - INI sysconfig store version in DEFAULT section
    - Code sysconfig have version as class attribute
    - Returns LEGACY for code sysconfig without version
    """
    if isinstance(config, ConfigParser):
        return ConfigVersion(
            config.get("DEFAULT", "version", fallback=ConfigVersion.LEGACY)
        )
    else:
        return getattr(config, "config_version", ConfigVersion.LEGACY)


def migrate_from_code_config(system_config: SystemConfig, file_path: Path) -> None:
    """Migrate a code-based system configuration to INI file format.

    Converts a system configuration defined in Python code to an INI file entry.
    This facilitates migration from legacy code-based sysconfig to the new INI format.

    Parameters
    ----------
    system_config : SystemConfig
        System configuration instance to migrate
    file_path : Path
        Path where the INI file should be saved

    Raises
    ------
    ValueError
        If system configuration has incompatible version

    Notes
    -----
    - Preserves all configuration parameters including:
        - System type and basic settings
        - Device configurations and roles
        - Objective settings
        - Device-specific parameters
    - Adds warning about deprecated code-based configurations
    - Creates parent directories if needed

    Examples
    --------
    >>> from qscope.system.config import Mock
    >>> mock_config = Mock()
    >>> migrate_from_code_config(mock_config, Path('~/.qscope/systems.ini'))
    # Migrates Mock system configuration to INI file
    """
    # Check version compatibility
    version = get_config_version(system_config.__class__)
    if version != ConfigVersion.LEGACY:
        raise ValueError(
            f"Can only migrate LEGACY (v1) configurations. {system_config.__class__.__name__} "
            f"has version {version}"
        )

    # Read existing config if it exists
    config = ConfigParser()
    if file_path.exists():
        config.read(file_path)

    # Set version in DEFAULT section
    config["DEFAULT"] = {"version": ConfigVersion.CURRENT.value}

    sys_instance = system_config

    section = sys_instance.system_name
    config[section] = {}

    # Basic settings
    config[section]["system_type"] = sys_instance.system_type.__name__
    config[section]["save_dir"] = sys_instance.save_dir

    # Objectives
    for obj, size in sys_instance.objective_pixel_size.items():
        config[section][f"objective.{obj}"] = str(size)

    # Devices
    for role, (dev_class, params) in sys_instance.devices_config.items():
        # Use full role name in lowercase for device config keys
        dev_name = str(role).lower()
        config[section][f"device.{dev_name}.type"] = dev_class.__name__
        config[section][f"device.{dev_name}.role"] = str(role)

        # Device specific parameters
        for param_name, param_value in params.items():
            if isinstance(param_value, dict):
                for sub_name, sub_value in param_value.items():
                    if isinstance(sub_value, dict):
                        # Handle nested dictionaries (e.g. sequence_params)
                        for sub_sub_name, sub_sub_value in sub_value.items():
                            config[section][
                                f"device.{dev_name}.{param_name}.{sub_name}.{sub_sub_name}"
                            ] = str(sub_sub_value)
                    else:
                        config[section][
                            f"device.{dev_name}.{param_name}.{sub_name}"
                        ] = str(sub_value)
            else:
                config[section][f"device.{dev_name}.{param_name}"] = str(param_value)

    with file_path.open("w") as f:
        config.write(f)


def _create_system_config(config: ConfigParser, system_name: str) -> SystemConfig:
    """Create a SystemConfig instance from a ConfigParser section.

    Parameters
    ----------
    config : ConfigParser
        Configuration parser containing the system section
    system_name : str
        Name of the system section to load

    Returns
    -------
    SystemConfig
        Initialized system configuration

    Raises
    ------
    ValueError
        If configuration is invalid
    """
    # Validate configuration
    is_valid, error_msg = validate_system_config(config, system_name)
    if not is_valid:
        raise ValueError(error_msg)

    # Get section and create new SystemConfig instance
    section = config[system_name]

    system_type_str = section["system_type"]
    if system_type_str not in VALID_SYSTEM_TYPES:
        raise ValueError(f"Invalid system type: {system_type_str}")

    system_type = VALID_SYSTEM_TYPES[system_type_str]

    system_config = SystemConfig(system_name=system_name, system_type=system_type)

    # Load basic settings
    system_config.system_name = system_name
    system_config.system_type = system_type
    system_config.save_dir = section["save_dir"]

    # Load objective pixel sizes
    system_config.objective_pixel_size = {}
    for key in section:
        if key.startswith("objective."):
            obj_name = key.split(".")[1]
            system_config.objective_pixel_size[obj_name] = float(section[key])

    # Load devices configuration
    system_config.devices_config = {}

    # Group device settings
    devices = {}
    for key in section:
        if key.startswith("device."):
            _, dev_name, param = key.split(".", 2)
            if dev_name not in devices:
                devices[dev_name] = {}
            devices[dev_name][param] = section[key]

    # Import roles
    from qscope.types import (
        MAIN_CAMERA,
        PREFIX_TO_ROLE,
        PRIMARY_RF,
        SECONDARY_RF,
        SEQUENCE_GEN,
        get_valid_device_types,
    )

    # Get centralized device type mapping
    device_types = get_valid_device_types()

    # Get centralized role mapping
    role_map = {str(role): role for role in PREFIX_TO_ROLE.values()}

    # Process each device
    for dev_name, dev_params in devices.items():
        # Get role from params or infer from device name
        role = role_map.get(dev_params.get("role")) or PREFIX_TO_ROLE.get(dev_name)
        if role is None:
            logger.warning(f"No role specified for device {dev_name}, skipping")
            continue
        device_class = device_types[dev_params["type"]]

        # Build device parameters
        params = {}

        # Extract parameters excluding type and role
        for key, value in dev_params.items():
            if key not in ["type", "role"]:
                parts = key.split(".")
                if len(parts) == 1:
                    # Simple parameter
                    params[key] = value
                else:
                    # Nested parameter (e.g. sequence_params.laser_delay)
                    current = params
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    # Convert numeric values, but preserve control flag strings
                    if "ch_defs" in parts:
                        current[parts[-1]] = value
                    else:
                        try:
                            current[parts[-1]] = float(value)
                        except ValueError:
                            current[parts[-1]] = value

        system_config.devices_config[role] = (device_class, params)

    return system_config
