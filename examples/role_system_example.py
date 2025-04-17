"""
Example demonstrating the Qscope role system.

This example shows how to:
1. Create a new protocol
2. Create a new interface
3. Create a new role
4. Implement a device for the role
5. Use the device through the role interface

The role system allows for hardware abstraction, making it possible to
swap device implementations without changing measurement code.
"""

from typing import Protocol, runtime_checkable
from qscope.device import Device
from qscope.types.interfaces import RoleInterface
from qscope.types.roles import DeviceRole


# Step 1: Define a protocol
@runtime_checkable
class MotorControllerProtocol(Protocol):
    """Protocol for motor controller devices."""
    
    def move_to_position(self, position: float) -> None: ...
    def get_position(self) -> float: ...
    def home(self) -> None: ...


# Step 2: Create an interface
class MotorControllerInterface(RoleInterface):
    """Interface for motor controller devices."""
    
    def __init__(self, device: MotorControllerProtocol):
        super().__init__(device)
    
    def move_to_position(self, position: float) -> None:
        """Move motor to specified position.
        
        Parameters
        ----------
        position : float
            Target position in mm
        """
        return self._device.move_to_position(position)
    
    def get_position(self) -> float:
        """Get current motor position.
        
        Returns
        -------
        float
            Current position in mm
        """
        return self._device.get_position()
    
    def home(self) -> None:
        """Move motor to home position."""
        return self._device.home()


# Step 3: Define a role
class MotorController(DeviceRole[MotorControllerProtocol]):
    """Role for motor controller devices."""
    interface_class = MotorControllerInterface


# Create a singleton instance
MOTOR_CONTROLLER = MotorController()


# Step 4: Implement a device
class MockMotorController(Device):
    """Mock implementation of a motor controller."""
    
    def __init__(self, **config_kwargs):
        super().__init__(**config_kwargs)
        self._connected = False
        self._position = 0.0
    
    def open(self) -> tuple[bool, str]:
        self._connected = True
        return True, "Connected to mock motor controller"
    
    def close(self):
        self._connected = False
    
    def is_connected(self) -> bool:
        return self._connected
    
    # Implement MotorControllerProtocol methods
    def move_to_position(self, position: float) -> None:
        print(f"Moving to position {position} mm")
        self._position = position
    
    def get_position(self) -> float:
        return self._position
    
    def home(self) -> None:
        print("Homing motor")
        self._position = 0.0


# Step 5: Use the device through the role system
def main():
    # Import system classes
    from qscope.system import System
    
    # Create a simple system
    system = System()
    
    # Create device instance
    motor = MockMotorController()
    
    # Add device with role
    system.add_device_with_role(motor, MOTOR_CONTROLLER)
    
    # Get device by role (returns interface)
    motor_controller = system.get_device_by_role(MOTOR_CONTROLLER)
    
    # Use interface methods
    motor_controller.home()
    motor_controller.move_to_position(10.5)
    current_position = motor_controller.get_position()
    print(f"Current position: {current_position} mm")
    
    # This is the key benefit: measurements work with roles, not specific devices
    # We could replace MockMotorController with a real hardware implementation
    # without changing any of the code that uses it


if __name__ == "__main__":
    main()
