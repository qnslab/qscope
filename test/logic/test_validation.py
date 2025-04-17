"""Tests for device role and hardware validation.

This module test the validation of:
1. Device role mappings - ensuring required device types match their roles
2. Device states - verifying devices are properly connected
3. System requirements - validating system configurations meet their requirements
4. Hardware requirements - checking measurements have their needed device types
"""

import asyncio

import numpy as np
import pytest
from loguru import logger

from qscope.device import Device, MockCamera, MockRFSource, MockSeqGen
from qscope.meas import Measurement, SGCameraMeasurement
from qscope.meas.decorators import requires_hardware
from qscope.system import SGCameraSystem, System, system_requirements
from qscope.types import (
    MAIN_CAMERA,
    PRIMARY_RF,
    SECONDARY_CAMERA,
    SECONDARY_RF,
    SEQUENCE_GEN,
    TESTING_MEAS_CONFIG,
    DeviceRole,
    MeasurementConfig,
    ValidationError,
)
from qscope.types.roles import MAIN_DIGITIZER
from qscope.types.validation import validate_device_role_mapping, validate_device_states

MEASCONFIG = MeasurementConfig(
    meas_type="any", ref_mode="", sweep_x=np.array([]), save_name="test"
)
QUEUE = asyncio.Queue()


def test_device_role_validation():
    """Test validation of device roles against device types."""
    # Valid mapping
    roles = {MAIN_CAMERA}
    device_config = {MAIN_CAMERA: (MockCamera, "")}
    is_valid, msg = validate_device_role_mapping(roles, device_config)
    assert is_valid
    assert msg == ""

    # Invalid mapping
    roles = {PRIMARY_RF}
    device_config = {PRIMARY_RF: (MockCamera, "")}
    logger.critical(PRIMARY_RF.required_type)
    is_valid, msg = validate_device_role_mapping(roles, device_config)
    assert not is_valid
    assert msg.startswith(
        "Role PRIMARY_RF requires device implementing RFSourceProtocol"
    )

    # Multiple roles
    roles = {MAIN_CAMERA, PRIMARY_RF, SEQUENCE_GEN}
    device_config = {
        MAIN_CAMERA: (MockCamera, ""),
        PRIMARY_RF: (MockRFSource, ""),
        SEQUENCE_GEN: (MockSeqGen, ""),
    }
    is_valid, msg = validate_device_role_mapping(roles, device_config)
    assert is_valid
    assert msg == ""

    # Missing one of multiple required roles
    roles = {MAIN_CAMERA, PRIMARY_RF}
    device_config = {MAIN_CAMERA: (MockCamera, "")}
    is_valid, msg = validate_device_role_mapping(roles, device_config)
    assert not is_valid
    assert "Missing required role: PRIMARY_RF" in msg


def test_device_state_validation():
    """Test validation of device states."""

    class MockDevice(Device):
        def __init__(self, connected=True):
            self._connected = connected

        def is_connected(self):
            return self._connected

    # Device connected
    device = MockDevice(True)
    is_valid, msg = validate_device_states(device)
    assert is_valid
    assert msg == ""

    # Device disconnected
    device = MockDevice(False)
    is_valid, msg = validate_device_states(device)
    assert not is_valid
    assert "is not connected" in msg


def test_system_requirements_decorator():
    """Test system requirements decorator validation."""

    # Valid system configuration
    @system_requirements(
        required_roles={MAIN_CAMERA},
        optional_roles={PRIMARY_RF, SEQUENCE_GEN},
    )
    class ValidSystem(SGCameraSystem):
        pass

    _ = ValidSystem("mock")

    # Should raise error for missing required role
    with pytest.raises(ValueError) as exc_info:

        @system_requirements(required_roles={MAIN_DIGITIZER}, optional_roles=set())
        class InvalidSystem(SGCameraSystem):
            pass

        _ = InvalidSystem("mock")  # This should raise error

    assert "Missing required device roles" in str(exc_info.value)

    # Test optional roles
    @system_requirements(
        required_roles={MAIN_CAMERA, SEQUENCE_GEN},
        optional_roles={PRIMARY_RF, SECONDARY_RF},
    )
    class OptionalSystem(SGCameraSystem):
        pass

    # System should work with or without optional roles
    system = OptionalSystem("mock")


def test_measurement_hardware_requirements():
    """Test measurement hardware requirements validation."""

    # Valid measurement requirements
    @requires_hardware(System, roles=())
    class ValidMeasurement(Measurement):
        pass

    # Should work with valid system
    system = System("Empty")
    measurement = ValidMeasurement(system, MEASCONFIG, QUEUE)

    # Should raise error with wrong system type
    with pytest.raises(TypeError) as exc_info:

        @requires_hardware(System, roles=(MAIN_DIGITIZER,))
        class InvalidMeasurement(Measurement):
            pass

        measurement = InvalidMeasurement(
            system, MEASCONFIG, QUEUE
        )  # This should raise error

    assert "requires device roles" in str(exc_info.value)

    # Test multiple device requirements
    @requires_hardware(
        SGCameraSystem, roles=(MAIN_CAMERA, SEQUENCE_GEN, MAIN_DIGITIZER)
    )
    class ComplexMeasurement(Measurement):
        pass

    with pytest.raises(TypeError) as exc_info:
        measurement = ComplexMeasurement(SGCameraSystem("mock"))
    assert "requires device roles" in str(exc_info.value)


def test_complex_validation_scenarios():
    """Test more complex validation scenarios."""

    # Test system with multiple required and optional roles
    @system_requirements(
        required_roles={
            MAIN_CAMERA,
            SEQUENCE_GEN,
            PRIMARY_RF,
        },
        optional_roles={SECONDARY_RF},
    )
    class ComplexSystem(SGCameraSystem):
        pass

    # Should work with just required roles
    system = ComplexSystem("mock")

    # Test measurement with conditional device requirements
    @requires_hardware(System, roles=())
    class ConditionalMeasurement(Measurement):
        def __init__(self, system, meas_config, queue, use_tc=False):
            super().__init__(system, meas_config, queue)
            self.use_rf = use_tc
            required_roles = [MAIN_CAMERA, SEQUENCE_GEN]
            if use_tc:
                required_roles.append(MAIN_DIGITIZER)

            @requires_hardware(SGCameraSystem, roles=tuple(required_roles))
            class DynamicMeasurement(Measurement):
                pass

            # should override self instead?? I guess we aren't using it..
            self.measurement = DynamicMeasurement(system, meas_config, queue)

    # Should work without TC
    measurement = ConditionalMeasurement(system, MEASCONFIG, QUEUE, use_tc=False)

    # Should fail with TC requirement
    with pytest.raises(TypeError) as exc_info:
        measurement = ConditionalMeasurement(system, MEASCONFIG, QUEUE, use_tc=True)
    assert "requires device roles" in str(exc_info.value)
