"""Tests for system configuration handling."""

from configparser import ConfigParser
from pathlib import Path

import pytest

from qscope.system.base_config import ConfigVersion, SystemConfig
from qscope.system.sysconfig import (
    ConfigVersion,
    copy_system_config,
    create_default_systems_file,
    list_available_systems,
    load_system_config,
    migrate_from_code_config,
    validate_system_config,
)
from qscope.types import MAIN_CAMERA, PRIMARY_RF, SEQUENCE_GEN


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary .qscope directory."""
    config_dir = tmp_path / ".qscope"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def mock_systems_file(temp_config_dir):
    """Create a mock systems.ini file."""
    systems_file = temp_config_dir / "systems.ini"
    config = ConfigParser()

    # Add mock system configuration
    config["Mock"] = {
        "system_type": "SGCameraSystem",
        "save_dir": "./mock_output/",
        "objective.20x": "nan",
        "device.sequence_gen.type": "MockSeqGen",
        "device.sequence_gen.role": "SEQUENCE_GEN",
        "device.sequence_gen.board_num": "0",
        "device.sequence_gen.ch_defs.laser": "00000001",
        "device.sequence_gen.ch_defs.rf_trig": "00001000",
        "device.sequence_gen.ch_defs.camera": "00000100",
        "device.sequence_gen.sequence_params.laser_delay": "1.5e-06",
        "device.sequence_gen.sequence_params.rf_delay": "1.2e-08",
        "device.primary_rf.type": "MockRFSource",
        "device.primary_rf.role": "PRIMARY_RF",
        "device.primary_rf.frequency": "2.87e9",
        "device.primary_rf.power": "10.0",
        "device.main_camera.type": "MockCamera",
        "device.main_camera.role": "MAIN_CAMERA",
        "device.main_camera.exposure_time": "0.1",
    }

    with systems_file.open("w") as f:
        config.write(f)

    return systems_file


def test_validate_system_config(mock_systems_file):
    """Test system configuration validation."""
    config = ConfigParser()
    config.read(mock_systems_file)

    # Test valid configuration
    is_valid, error_msg = validate_system_config(config, "Mock")
    assert is_valid, f"Valid configuration was marked as invalid: {error_msg}"
    assert error_msg == "", f"Expected empty error message but got: {error_msg}"

    # Test missing required field
    del config["Mock"]["system_type"]
    is_valid, error_msg = validate_system_config(config, "Mock")
    assert not is_valid, f"Configuration should be invalid but was valid"
    assert "Missing required fields" in error_msg, (
        f"Unexpected error message: {error_msg}"
    )


def test_validate_system_config_errors(mock_systems_file):
    """Test system configuration validation error cases."""
    # Test invalid device type
    config = ConfigParser()
    config.read(mock_systems_file)
    config["Mock"]["device.main_camera.type"] = "NonExistentCamera"
    is_valid, error_msg = validate_system_config(config, "Mock")
    assert not is_valid, f"Configuration with invalid device type should be invalid"
    assert "Invalid device type" in error_msg, f"Unexpected error message: {error_msg}"

    # Test missing required device role (separate test with fresh config)
    config = ConfigParser()
    config.read(mock_systems_file)
    del config["Mock"]["device.sequence_gen.type"]
    del config["Mock"]["device.sequence_gen.role"]
    is_valid, error_msg = validate_system_config(config, "Mock")
    assert not is_valid, f"Configuration with missing required role should be invalid"
    assert "Missing required roles" in error_msg, (
        f"Unexpected error message: {error_msg}"
    )


def test_validate_system_type(mock_systems_file):
    """Test system type validation."""
    config = ConfigParser()
    config.read(mock_systems_file)

    # Test valid system type
    is_valid, error_msg = validate_system_config(config, "Mock")
    assert is_valid
    assert error_msg == ""

    # Test invalid system type
    config["Mock"]["system_type"] = "NonExistentSystem"
    is_valid, error_msg = validate_system_config(config, "Mock")
    assert not is_valid, f"Configuration with invalid system type should be invalid"
    assert "Invalid system type" in error_msg, f"Unexpected error message: {error_msg}"

    # Test system type requirements
    config["Mock"]["system_type"] = "SGCameraSystem"
    del config["Mock"]["device.main_camera.type"]
    del config["Mock"]["device.main_camera.role"]
    is_valid, error_msg = validate_system_config(config, "Mock")
    assert not is_valid, (
        f"Configuration missing required device roles should be invalid"
    )
    assert "Missing required roles" in error_msg, (
        f"Unexpected error message: {error_msg}"
    )


def test_system_specific_config(mock_systems_file):
    """Test system-specific configuration handling."""
    config = ConfigParser()
    config.read(mock_systems_file)

    # Test sequence generator parameters
    config["Mock"]["device.sequence_gen.sequence_params.laser_delay"] = "3.4e-07"
    config["Mock"]["device.sequence_gen.ch_defs.laser"] = "00000001"
    is_valid, error_msg = validate_system_config(config, "Mock")
    assert is_valid
    assert error_msg == ""

    # Test camera parameters
    config["Mock"]["device.main_camera.exposure_time"] = "0.1"
    config["Mock"]["device.main_camera.binning"] = "2x2"
    is_valid, error_msg = validate_system_config(config, "Mock")
    assert is_valid
    assert error_msg == ""

    # Test RF source parameters
    config["Mock"]["device.primary_rf.frequency"] = "2.87e9"
    config["Mock"]["device.primary_rf.power"] = "10.0"
    is_valid, error_msg = validate_system_config(config, "Mock")
    assert is_valid
    assert error_msg == ""


def test_load_system_config(mock_systems_file, monkeypatch):
    """Test loading system configuration."""
    # Mock home directory to use temp directory
    monkeypatch.setattr(Path, "home", lambda: mock_systems_file.parent.parent)

    # Test loading valid configuration
    config = load_system_config("Mock")
    assert isinstance(config, SystemConfig)
    assert config.system_name == "Mock"
    assert config.save_dir == "./mock_output/"

    # Test loading non-existent configuration
    with pytest.raises(
        ValueError,
        match=r"System 'NonExistent' not found in:\n- User config:.*\n- Package config:.*",
    ):
        load_system_config("NonExistent")

    # Test loading truly non-existent system name
    with pytest.raises(
        ValueError,
        match=r"System 'truly_nonexistent_system' not found in:\n- User config:.*\n- Package config:.*",
    ):
        load_system_config("truly_nonexistent_system")


def test_create_default_systems_file(temp_config_dir):
    """Test creation of default systems.ini file."""
    systems_file = temp_config_dir / "systems.ini"

    # Create initial config with custom system
    initial_config = ConfigParser()
    initial_config["CustomSystem"] = {
        "system_type": "SGCameraSystem",
        "save_dir": "./custom_output/",
        "use_save_dir_only": "true",
        "device.main_camera.type": "MockCamera",
        "device.main_camera.role": "MAIN_CAMERA",
    }
    with systems_file.open("w") as f:
        initial_config.write(f)

    # Create default systems file
    create_default_systems_file(systems_file)

    # Verify both default and custom systems exist
    config = ConfigParser()
    config.read(systems_file)
    assert "Mock" in config.sections(), "Default Mock system not created"
    assert "CustomSystem" in config.sections(), "Custom system not preserved"

    # Verify custom system settings preserved
    custom_section = config["CustomSystem"]
    assert custom_section["save_dir"] == "./custom_output/"
    assert custom_section["device.main_camera.type"] == "MockCamera"


def test_list_available_systems(mock_systems_file, monkeypatch):
    """Test listing available system configurations."""
    # Mock home directory to use temp directory
    monkeypatch.setattr(Path, "home", lambda: mock_systems_file.parent.parent)

    systems = list_available_systems()
    assert "Mock" in systems


def test_copy_system_config(mock_systems_file, monkeypatch):
    """Test copying system configurations."""
    # Mock home directory to use temp directory
    monkeypatch.setattr(Path, "home", lambda: mock_systems_file.parent.parent)

    # Test successful copy
    copy_system_config("Mock", "NewMock")
    systems = list_available_systems()
    assert "Mock" in systems
    assert "NewMock" in systems

    # Test copying non-existent system
    with pytest.raises(ValueError, match="Source system 'NonExistent' not found"):
        copy_system_config("NonExistent", "NewSystem")

    # Test copying to existing system
    with pytest.raises(ValueError, match="Destination system 'Mock' already exists"):
        copy_system_config("Mock", "Mock")


def test_migrate_from_code_config(temp_config_dir):
    """Test migrating code-based configuration to INI file."""
    systems_file = temp_config_dir / "systems.ini"

    # Create initial config with custom system
    initial_config = ConfigParser()
    initial_config["CustomSystem"] = {
        "system_type": "SGCameraSystem",
        "save_dir": "./custom_output/",
        "use_save_dir_only": "true",
        "device.main_camera.type": "MockCamera",
        "device.main_camera.role": "MAIN_CAMERA",
    }
    with systems_file.open("w") as f:
        initial_config.write(f)

    # Test migration of mock system configuration
    from qscope.device import MockCamera, MockRFSource, MockSeqGen
    from qscope.system.system import SGCameraSystem
    from qscope.types import MAIN_CAMERA, PRIMARY_RF, SEQUENCE_GEN

    mock_config = SystemConfig(
        system_name="Mock",
        system_type=SGCameraSystem,
        save_dir="./mock_output/",
        objective_pixel_size={"20x": float("nan")},
        devices_config={
            SEQUENCE_GEN: (MockSeqGen, {}),
            PRIMARY_RF: (MockRFSource, {}),
            MAIN_CAMERA: (MockCamera, {}),
        },
    )
    migrate_from_code_config(mock_config, systems_file)

    # Verify version was set correctly
    config = ConfigParser()
    config.read(systems_file)
    assert config["DEFAULT"]["version"] == ConfigVersion.CURRENT.value

    # Verify both custom and migrated systems exist
    assert "CustomSystem" in config.sections(), "Custom system not preserved"
    assert "Mock" in config.sections(), "Mock system not migrated"

    # Verify custom system settings preserved
    custom_section = config["CustomSystem"]
    assert custom_section["save_dir"] == "./custom_output/"
    assert custom_section["device.main_camera.type"] == "MockCamera"

    # Verify migrated configuration
    mock_section = config["Mock"]
    assert mock_section["system_type"] == "SGCameraSystem"
    assert mock_section["save_dir"] == "./mock_output/"
    assert mock_section["objective.20x"] == "nan"

    # Verify device configurations
    assert mock_section["device.sequence_gen.type"] == "MockSeqGen"
    assert mock_section["device.primary_rf.type"] == "MockRFSource"
    assert mock_section["device.main_camera.type"] == "MockCamera"


def test_system_role_requirements(mock_systems_file, monkeypatch):
    """Test system role requirements validation."""
    # Mock home directory
    monkeypatch.setattr(Path, "home", lambda: mock_systems_file.parent.parent)

    config = ConfigParser()
    config.read(mock_systems_file)

    # Remove required SEQUENCE_GEN role
    del config["Mock"]["device.sequence_gen.type"]
    del config["Mock"]["device.sequence_gen.role"]

    with mock_systems_file.open("w") as f:
        config.write(f)

    # Should raise error when loading config missing required role
    with pytest.raises(ValueError, match="Missing required roles: SEQUENCE_GEN"):
        load_system_config("Mock")


def test_objective_handling(mock_systems_file, monkeypatch):
    """Test objective configuration handling."""
    monkeypatch.setattr(Path, "home", lambda: mock_systems_file.parent.parent)

    config = ConfigParser()
    config.read(mock_systems_file)

    # Add multiple objectives
    config["Mock"]["objective.10x"] = "400e-9"
    config["Mock"]["objective.40x"] = "100e-9"

    with mock_systems_file.open("w") as f:
        config.write(f)

    # Load config and verify objectives
    system_config = load_system_config("Mock")
    assert len(system_config.objective_pixel_size) == 3
    assert system_config.objective_pixel_size["10x"] == 400e-9
    assert system_config.objective_pixel_size["40x"] == 100e-9


def test_device_specific_params(mock_systems_file, monkeypatch):
    """Test handling of device-specific parameters."""
    monkeypatch.setattr(Path, "home", lambda: mock_systems_file.parent.parent)

    config = ConfigParser()
    config.read(mock_systems_file)

    # Add device-specific parameters
    config["Mock"]["device.sequence_gen.board_num"] = "0"
    config["Mock"]["device.sequence_gen.sequence_params.laser_delay"] = "345e-9"
    config["Mock"]["device.sequence_gen.ch_defs.laser"] = "00000001"

    with mock_systems_file.open("w") as f:
        config.write(f)

    # Load and verify device parameters
    system_config = load_system_config("Mock")
    seqgen_params = system_config.devices_config[SEQUENCE_GEN][1]
    assert seqgen_params["board_num"] == "0"
    assert seqgen_params["sequence_params"]["laser_delay"] == 345e-9
    assert seqgen_params["ch_defs"]["laser"] == "00000001"
