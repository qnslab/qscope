"""Device base class and hardware abstraction layer.

This module defines the Device base class, which is the foundation of QScope's
hardware abstraction layer. All hardware devices in QScope inherit from this class
and implement the methods required by their intended roles.

The Device class provides:
1. Configuration validation
2. Role management
3. Connection handling
4. Attribute access

Devices connect to the role system by implementing protocol methods, which are
validated when the device is added to a system with a specific role.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Set, Type, TypeVar

from loguru import logger

if TYPE_CHECKING:
    from qscope.types import DeviceRole

D = TypeVar("D", bound="Device")


class Device:
    """Base class for all hardware devices in QScope.

    The Device class provides the foundation for hardware abstraction in QScope.
    Specific device implementations inherit from this class and implement the
    methods required by their intended roles.

    A device can fulfill multiple roles if it implements all required methods
    for those roles. The system validates role compatibility at runtime.

    Device Architecture
    ------------------
    Devices are part of QScope's hardware abstraction layer:

    1. Base Device Class (this class)
       - Provides configuration validation
       - Manages role assignments
       - Handles connection state
       - Defines required methods

    2. Device Implementations
       - Inherit from Device
       - Implement protocol methods for specific roles
       - Provide hardware-specific functionality
       - Can implement multiple protocols

    3. System Integration
       - Devices are added to systems with specific roles
       - System validates that devices implement required protocols
       - Devices are accessed through role interfaces

    Required Methods
    --------------
    All device implementations must override these methods:

    - open(): Connect to the hardware
    - close(): Disconnect from the hardware
    - is_connected(): Check connection status

    In addition, devices must implement all methods required by the
    protocols of the roles they intend to fulfill.

    Attributes
    ----------
    required_config : dict[str, Type]
        Required configuration parameters and their types
    _roles : Set[DeviceRole[Device]]
        Set of roles this device fulfills

    Examples
    --------
    Creating a new device implementation:

    ```python
    class MyRFSource(Device):
        required_config = {
            "visa_addr": str,
            "max_power": float
        }

        def __init__(self, **config_kwargs):
            super().__init__(**config_kwargs)
            self.connected = False

        def open(self) -> tuple[bool, str]:
            # Implementation for connecting to hardware
            self.connected = True
            return True, "Connected successfully"

        def close(self):
            # Implementation for disconnecting
            self.connected = False

        def is_connected(self) -> bool:
            return self.connected

        # Implement methods required by RFSourceProtocol
        def set_freq(self, freq: float) -> None:
            # Implementation
            pass

        def set_power(self, power: float) -> None:
            # Implementation
            pass
    ```

    Using the device with a role:

    ```python
    # Create device instance
    rf = MyRFSource(visa_addr="TCPIP0::192.168.1.1::INSTR", max_power=20.0)

    # Add to system with role
    system.add_device_with_role(rf, PRIMARY_RF)
    ```

    See Also
    --------
    qscope.types.protocols : Protocol definitions
    qscope.types.roles : Role definitions
    qscope.types.interfaces : Interface implementations
    qscope.system : System implementation
    """

    required_config: dict[str, Type] = {}  # Required configuration keys

    _roles: Set[DeviceRole[Device]]

    def __init__(self, **config_kwargs):
        for key, value in config_kwargs.items():
            setattr(self, key, value)
        for key, value in self.required_config.items():
            if not hasattr(self, key):
                logger.error(
                    f"Device {self.__class__.__name__} missing required config key: "
                    + f"{key}"
                )
                raise ValueError(
                    f"Device {self.__class__.__name__} missing required config "
                    + f"key: {key}"
                )
            if not isinstance(getattr(self, key), value):
                logger.error(
                    f"Device {self.__class__.__name__} config key {key} "
                    + f"has wrong type: {type(getattr(self, key))} (expected {value})"
                )
                raise ValueError(
                    f"Device {self.__class__.__name__} config key {key} has "
                    + f"wrong type: {type(getattr(self, key))} (expected {value})"
                )
        self._roles: Set[DeviceRole[Device]] = set()  # Roles this device fulfills

    def _add_role(self, role: DeviceRole[D]) -> None:
        """Add a role that this device fulfills.

        Note: This should only be called by System.add_device_with_role()

        Parameters
        ----------
        role : DeviceRole[D]
            Role to add
        """
        self._roles.add(role)

    def has_role(self, role: DeviceRole[D]) -> bool:
        """Check if device fulfills a specific role.

        Parameters
        ----------
        role : DeviceRole[D]
            Role to check

        Returns
        -------
        bool
            True if device fulfills role
        """
        return role in self._roles

    def get_roles(self) -> Set[DeviceRole["Device"]]:
        """Get all roles this device fulfills.

        Returns
        -------
        Set[DeviceRole[Device]]
            Set of roles
        """
        return self._roles.copy()

    def open(self) -> tuple[bool, str]:
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def is_connected(self) -> bool:
        raise NotImplementedError()

    def get_all_attrs(self):
        """
        Function to return all of the managed attributes of the class
        Managed attributes are the ones that start with a underscore
        """
        attrs = {}
        for key, value in self.__dict__.items():
            # single underscore attr are managed
            if key[0] == "_" and not key.startswith(f"_{self.__class__.__name__}"):
                if isinstance(value, set):
                    value = [str(role) for role in value]
                attrs[key[1:]] = value
        return attrs

    def unroll_metadata(self):
        # get all attributes in object and return as dict
        return self.get_all_attrs()
