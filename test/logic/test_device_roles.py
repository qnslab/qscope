import pytest
from loguru import logger

from qscope.device import MockCamera, MockRFSource
from qscope.system import SGCameraSystem, System
from qscope.types import MAIN_CAMERA, PRIMARY_RF, SECONDARY_CAMERA


class TestDeviceRoles:
    def test_unique_role_assignment(self):
        """Test that roles can only be assigned to one device."""
        # Create a minimal system configuration
        from qscope.system.base_config import SystemConfig

        sys_config = SystemConfig(
            system_name="test", system_type="SGCameraSystem"
        )  # Create instance with required args
        sys_config.system_name = "test"
        sys_config.system_type = SGCameraSystem  # Use actual class instead of string
        sys_config.devices_config = {}  # Initialize empty devices config
        sys_config.save_dir = "./test_output/"  # Add required save_dir

        # Create fresh system with minimal config
        system = System(sys_config)

        # Create two camera devices
        camera1 = MockCamera()
        camera2 = MockCamera()

        # First assignment should work
        system.add_device_with_role(camera1, MAIN_CAMERA)

        assert system.get_device_by_role(MAIN_CAMERA) is camera1
        assert system.get_device_by_role(MAIN_CAMERA) is not camera2

        # Second assignment of same role should fail
        with pytest.raises(ValueError) as exc_info:
            system.add_device_with_role(camera2, MAIN_CAMERA)

        assert "Role MainCamera already assigned" in str(exc_info.value)

        # But camera2 can take a different camera role
        system.add_device_with_role(camera2, SECONDARY_CAMERA)

    def test_role_reassignment_to_same_device(self):
        """Test that a role can be reassigned to the same device."""
        # Create a minimal system configuration
        from qscope.system.base_config import SystemConfig

        sys_config = SystemConfig(
            system_name="test", system_type="SGCameraSystem"
        )  # Create instance with required args
        sys_config.devices_config = {}  # Initialize empty devices config
        sys_config.save_dir = "./test_output/"  # Add required save_dir
        sys_config.use_save_dir_only = True  # Add required use_save_dir_only

        # Create fresh system with minimal config
        system = System(sys_config)
        camera = MockCamera()

        # First assignment
        system.add_device_with_role(camera, MAIN_CAMERA)

        # Reassigning same role to same device should work
        system.add_device_with_role(camera, MAIN_CAMERA)

        # Verify role is still assigned
        assert system.get_device_by_role(MAIN_CAMERA) is camera

    def test_device_role_type_safety(self):
        """Test that roles can only be assigned to compatible device types."""
        # Create a minimal system configuration
        from qscope.system.base_config import SystemConfig

        sys_config = SystemConfig(
            system_name="test", system_type="SGCameraSystem"
        )  # Create instance with required args
        sys_config.devices_config = {}  # Initialize empty devices config
        sys_config.save_dir = "./test_output/"  # Add required save_dir
        sys_config.use_save_dir_only = True  # Add required use_save_dir_only

        # Create fresh system with minimal config
        system = System(sys_config)

        camera = MockCamera()
        rf_source = MockRFSource()

        # Camera role to camera device should work
        system.add_device_with_role(camera, MAIN_CAMERA)

        # RF role to RF device should work
        system.add_device_with_role(rf_source, PRIMARY_RF)

        # Getting device should return correct type
        retrieved_camera = system.get_device_by_role(MAIN_CAMERA)
        assert isinstance(retrieved_camera, MockCamera)

        retrieved_rf = system.get_device_by_role(PRIMARY_RF)
        assert isinstance(retrieved_rf, MockRFSource)

    def test_role_lookup_missing_role(self):
        """Test that looking up a missing role raises appropriate error."""
        # Create a minimal system configuration
        from qscope.system.base_config import SystemConfig

        sys_config = SystemConfig(
            system_name="test", system_type="SGCameraSystem"
        )  # Create instance with required args
        sys_config.devices_config = {}  # Initialize empty devices config
        sys_config.save_dir = "./test_output/"  # Add required save_dir
        sys_config.use_save_dir_only = True  # Add required use_save_dir_only

        # Create fresh system with minimal config
        system = System(sys_config)

        with pytest.raises(ValueError) as exc_info:
            system.get_device_by_role(MAIN_CAMERA)

        assert "No device found for role" in str(exc_info.value)
