"""Device role definitions and base classes.

This module defines the role system that connects devices to their interfaces and
ensures they implement required protocols. The role system is the core of device
abstraction in QScope.

The Role System
--------------
The role system consists of three parts:

1. Protocols (protocols.py) - Define required methods for each role
2. Interfaces (interfaces.py) - Provide type-safe access to role functionality
3. Roles (this file) - Connect devices to interfaces and validate protocols

Example System Flow
-----------------
1. Device implements protocol methods:
   ```python
   class MyRFDevice(Device):
       def set_freq(self, freq: float) -> None: ...
       def set_power(self, power: float) -> None: ...
   ```

2. Role specifies required protocol:
   ```python
   class PrimaryRFSource(DeviceRole[RFSourceProtocol]):
       interface_class = RFSourceInterface
   ```

3. System validates and provides interface:
   ```python
   # Validates MyRFDevice implements RFSourceProtocol
   system.add_device_with_role(MyRFDevice(), PRIMARY_RF)

   # Returns RFSourceInterface wrapping MyRFDevice
   rf = system.get_device_by_role(PRIMARY_RF)
   ```

Benefits
--------
- Type Safety: Mypy checks protocol implementation
- Runtime Validation: System verifies devices implement required methods
- Clean API: Interfaces provide clear access to role functionality
- Flexibility: Devices can implement multiple roles
- Testability: Mock devices can easily implement protocols

See Also
--------
protocols.py : Protocol definitions
interfaces.py : Interface implementations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Type, TypeVar, get_args

from loguru import logger

from qscope.types.interfaces import (
    CameraInterface,
    DigitizerInterface,
    RFSourceInterface,
    RoleInterface,
    SeqGenInterface,
)
from qscope.types.protocols import (
    CameraProtocol,
    DigitizerProtocol,
    RFSourceProtocol,
    SeqGenProtocol,
)

if TYPE_CHECKING:
    from qscope.device import (
        Device,
        MockCamera,
        MockRFSource,
        MockSeqGen,
        PulseBlaster,
        SynthNV,
        Zyla42,
        Zyla55,
    )

D = TypeVar("D", bound="Device")

# Map of device type names to classes
# Populated on first access via get_valid_device_types()
VALID_DEVICE_TYPES = {}


def get_valid_device_types() -> dict[str, Type["Device"]]:
    """Get mapping of device type names to device classes.

    Returns
    -------
    dict[str, Type[Device]]
        Mapping of device type names to device classes

    Notes
    -----
    Lazily imports device classes to avoid circular imports.
    """
    global VALID_DEVICE_TYPES

    if not VALID_DEVICE_TYPES:
        # Import device classes
        from qscope.device import MockCamera, MockRFSource, MockSeqGen

        # Base set of mock devices that should always be available
        VALID_DEVICE_TYPES.update(
            {
                "MockCamera": MockCamera,
                "MockRFSource": MockRFSource,
                "MockSeqGen": MockSeqGen,
            }
        )

        # Try importing optional device types
        try:
            from qscope.device import PulseBlaster

            VALID_DEVICE_TYPES["PulseBlaster"] = PulseBlaster
        except ImportError:
            logger.error("PulseBlaster device class not found")

        try:
            from qscope.device import SynthNV

            VALID_DEVICE_TYPES["SynthNV"] = SynthNV
        except ImportError:
            logger.error("SynthNV device class not found")

        try:
            from qscope.device import Zyla42

            VALID_DEVICE_TYPES["Zyla42"] = Zyla42
        except ImportError:
            logger.error("Zyla42 device class not found")

        try:
            from qscope.device import Zyla55

            VALID_DEVICE_TYPES["Zyla55"] = Zyla55
        except ImportError:
            logger.error("Zyla55 device class not found")

    return VALID_DEVICE_TYPES


class DeviceRole(Generic[D]):
    """Base class for device roles.

    A DeviceRole defines an abstract capability that a device can fulfill.
    It connects concrete device implementations to abstract interfaces,
    allowing the system to work with devices through a consistent API
    regardless of the specific hardware.

    Each role:
    1. Specifies a protocol that devices must implement
    2. Provides an interface class for accessing device functionality
    3. Validates that devices can fulfill the role requirements

    Type Parameters
    --------------
    D : Type[Device]
        The device protocol type that can fulfill this role

    Examples
    --------
    Defining a new role:

    ```python
    class TemperatureController(DeviceRole[TempControlProtocol]):
        interface_class = TempControlInterface
    ```

    Using a role in a system:

    ```python
    # Add device with role
    system.add_device_with_role(my_temp_controller, TEMP_CONTROL)

    # Get interface for role
    temp = system.get_device_by_role(TEMP_CONTROL)
    temp.set_temperature(25.0)  # Use interface methods
    ```
    """

    interface_class: Type[RoleInterface] = None

    def __init__(self) -> None:
        # Get the type argument (will be a ForwardRef or actual type)
        self.required_type = get_args(self.__class__.__orig_bases__[0])[0]

    def get_interface(self, device: Device) -> RoleInterface:
        """Get the interface implementation for this role.

        Parameters
        ----------
        device : Device
            Device to wrap with interface

        Returns
        -------
        RoleInterface
            Role-specific interface implementation

        Raises
        ------
        NotImplementedError
            If role doesn't define an interface class
        """
        if not self.interface_class:
            raise NotImplementedError("Role must define interface_class")
        return self.interface_class(device)

    def validate_device_type(self, device_class: type["Device"]) -> tuple[bool, str]:
        """Validate if a device class can fulfill this role.

        This method checks if a device class implements all the methods required by
        the protocol associated with this role. It handles both runtime protocol
        checking and forward references to protocol classes.

        Parameters
        ----------
        device_class : type[Device]
            The device class to validate

        Returns
        -------
        tuple[bool, str]
            A tuple containing:
            - is_valid (bool): True if the device can fulfill this role
            - error_message (str): Empty string if valid, otherwise an error message

        Notes
        -----
        Validation works in two ways:
        1. For ForwardRef protocols: Checks if the protocol name appears in the device's class hierarchy
        2. For runtime protocols: Checks if the device implements all required methods
        """
        from typing import ForwardRef

        from qscope.types.validation import _get_base_classes

        # Handle ForwardRef
        if isinstance(self.required_type, ForwardRef):
            # Check if required type name appears in device's class hierarchy
            base_classes = _get_base_classes(device_class)
            if self.required_type.__forward_arg__ not in base_classes:
                return False, (
                    f"Role {self} requires device type {self.required_type.__forward_arg__}, "
                    f"got {device_class.__name__}"
                )
        else:
            # Check if device class implements required protocol methods
            missing_methods = []
            for method_name in self.required_type.__annotations__:
                if not hasattr(device_class, method_name):
                    missing_methods.append(method_name)

            if missing_methods:
                return False, (
                    f"Role {self} requires device implementing {self.required_type.__name__}, "
                    f"but {device_class.__name__} is missing methods: {', '.join(missing_methods)}"
                )
        return True, ""

    def __str__(self) -> str:
        # Find the singleton instance name by looking through module globals
        for name, value in globals().items():
            if isinstance(value, DeviceRole) and value is self:
                return name
        # Fallback to class name if singleton not found
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeviceRole):
            return NotImplemented
        return type(self) == type(other)

    def __hash__(self) -> int:
        return hash(type(self))


# Concrete role classes
class MainCamera(DeviceRole[CameraProtocol]):
    """Primary camera role for main imaging and measurements."""

    interface_class = CameraInterface


class SecondaryCamera(DeviceRole[CameraProtocol]):
    """Secondary/auxiliary camera role for additional imaging tasks."""

    interface_class = CameraInterface


class PrimaryRFSource(DeviceRole[RFSourceProtocol]):
    """Primary RF source role for main RF control."""

    interface_class = RFSourceInterface


class SecondaryRFSource(DeviceRole[RFSourceProtocol]):
    """Secondary RF source role for additional RF tasks."""

    interface_class = RFSourceInterface


class SequenceGenerator(DeviceRole[SeqGenProtocol]):
    """Sequence generator role for timing and pulse control."""

    interface_class = SeqGenInterface


class MainDigitizer(DeviceRole[DigitizerProtocol]):
    """Digitizer role for data acquisition."""

    interface_class = DigitizerInterface


# Singleton instances (use these)
MAIN_CAMERA = MainCamera()
SECONDARY_CAMERA = SecondaryCamera()
PRIMARY_RF = PrimaryRFSource()
SECONDARY_RF = SecondaryRFSource()
SEQUENCE_GEN = SequenceGenerator()
MAIN_DIGITIZER = MainDigitizer()

# Map device prefix in config to role singleton
PREFIX_TO_ROLE = {
    "main_camera": MAIN_CAMERA,
    "secondary_camera": SECONDARY_CAMERA,
    "primary_rf": PRIMARY_RF,
    "secondary_rf": SECONDARY_RF,
    "sequence_gen": SEQUENCE_GEN,
    "main_digitizer": MAIN_DIGITIZER,
}

__all__ = [
    "DeviceRole",
    "MainCamera",
    "SecondaryCamera",
    "PrimaryRFSource",
    "SecondaryRFSource",
    "SequenceGenerator",
    "MainDigitizer",
    "MAIN_CAMERA",
    "SECONDARY_CAMERA",
    "PRIMARY_RF",
    "SECONDARY_RF",
    "SEQUENCE_GEN",
    "MAIN_DIGITIZER",
    "PREFIX_TO_ROLE",
]
