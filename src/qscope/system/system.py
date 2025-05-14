# -*- coding: utf-8 -*-
""" "
This is a class to define the experimental system.
It is used to define the equipment and handle passing the device classes.
"""

from __future__ import annotations  # not needed in py3.12?

import asyncio
import time
from functools import wraps
from inspect import signature
from typing import Any, Callable, ForwardRef, ParamSpec, Set, Type, TypeVar, overload

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

import qscope.system.config
from qscope.device import Device
from qscope.system.sysconfig import load_system_config
from qscope.types import (
    MAIN_CAMERA,
    PRIMARY_RF,
    SECONDARY_RF,
    SEQUENCE_GEN,
    DeviceRole,
    ValidationError,
    validate_device_role_mapping,
    validate_device_states,
)
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

# =============================================================================
# Helpful decorators to enforce/validate behaviour
# =============================================================================

P = ParamSpec("P")
T = TypeVar("T")


def system_requirements(
    required_roles: Set[DeviceRole], optional_roles: Set[DeviceRole] = set()
) -> Callable[[Type[T]], Type[T]]:
    """Decorator to specify required and optional device roles for a system type.

    Parameters
    ----------
    required_roles : Set[DeviceRole]
        Device roles that must be present in the system
    optional_roles : Set[DeviceRole]
        Device roles that may be present in the system

    Examples
    --------
    @system_requirements(
        required_roles={SEQUENCE_GEN, MAIN_CAMERA},
        optional_roles={PRIMARY_RF}
    )
    class SGCameraSystem(SGSystem):
        '''System with sequence generator and camera.'''
        pass

    Raises
    ------
    ValidationError
        If role requirements cannot be satisfied by available device types
    """

    def decorator(cls: Type[T]) -> Type[T]:
        cls._required_roles = required_roles
        cls._optional_roles = optional_roles

        # Add validation method to class
        def validate_roles(self, config_roles: Set[DeviceRole]) -> tuple[bool, str]:
            """Validate that a set of roles meets this system's requirements.

            Parameters
            ----------
            config_roles : Set[DeviceRole]
                The roles to validate

            Returns
            -------
            tuple[bool, str]
                (is_valid, error_message)
            """
            from qscope.types.validation import validate_system_roles

            is_valid, error_msg = validate_system_roles(
                self._required_roles, self._optional_roles, config_roles
            )

            if not is_valid:
                raise ValueError(
                    f"Invalid roles for {self.__class__.__name__}: {error_msg}"
                )
            return True, ""

        cls.validate_roles = validate_roles
        return cls

    return decorator


def requires_connected_devices(
    *required_roles: DeviceRole,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to check if required device roles are present and properly connected.

    Parameters
    ----------
    *required_roles : DeviceRole
        The roles required (e.g. MAIN_CAMERA, RF_SOURCE)

    Examples
    --------
    @requires_connected_devices(MAIN_CAMERA, RF_SOURCE)
    def take_snapshot(self):
        '''Take camera snapshot with RF enabled.'''
        camera = self.get_device_by_role(MAIN_CAMERA)
        return camera.take_snapshot()

    Raises
    ------
    ValidationError
        If hardware is not started or devices are not connected/ready
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(self: System, *args: P.args, **kwargs: P.kwargs) -> T:
            if not self.hardware_started_up:
                raise ValidationError(
                    f"Cannot call {func.__name__}: "
                    "Hardware not started up. Call startup() first."
                )

            for role in required_roles:
                try:
                    device = self.get_device_by_role(role)
                    is_valid, error_msg = validate_device_states(device)
                    if not is_valid:
                        raise ValidationError(
                            f"Cannot call {func.__name__}: {error_msg}"
                        )
                except ValueError as e:
                    raise ValidationError(f"Cannot call {func.__name__}: {str(e)}")

            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def check_role_is_connected() -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(self: System, *args: P.args, **kwargs: P.kwargs) -> T:
            if not self.hardware_started_up:
                raise ValidationError(
                    f"Cannot call {func.__name__}: "
                    "Hardware not started up. Call startup() first."
                )
            # Get function signature
            sig = signature(func)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            # Check each parameter
            for param_name, param_value in bound_args.arguments.items():
                if isinstance(param_value, DeviceRole):
                    try:
                        device = self.get_device_by_role(param_value)
                        if not device.is_connected():
                            raise ValidationError(
                                f"Cannot call {func.__name__}: "
                                f"Device for role {param_value} is not connected"
                            )
                    except ValueError as e:
                        raise ValidationError(f"Cannot call {func.__name__}: {str(e)}")
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# System class
# =============================================================================


class System(object):
    """Abstract class for system. Specific configurations are sub-classes, such as
    CameraSystem, APDSystem, etc., depending on what is most often used.
    This could be implemented fully dynamically, but that would get messy.
    """

    _devices: list[Device]  # ALL device objects for this system

    _bg_meas_tasks: dict[str, asyncio.Task]  # dict: meas_id -> bg measurement tasks

    streaming: bool = False
    hardware_started_up: bool = False

    # these are overriden from config.py (here for type checking & clarity)
    system_name: str
    save_dir: str
    use_save_dir_only: bool
    objective_pixel_size: dict[str, float]
    devices_config: dict[DeviceRole, tuple[Type[Device], dict[str, Any]]]
    device_status: dict[str, dict[str, bool | str]]

    def __init__(self, sys_config: str | Type[qscope.system.base_config.SystemConfig]):
        """Initialize system with configuration.

        Parameters
        ----------
        sys_config : str | Type[SystemConfig]
            Either a system name to load from config file, or a SystemConfig class

        Raises
        ------
        ValueError
            If system configuration is invalid or not found
        """
        self.device_status = dict()
        self._devices = []  # can't be set in attr above as they are mutable.
        self._bg_meas_tasks = dict()

        self.set_matplotlib_style()  # This is for saving the data on the system side.

        try:
            if isinstance(sys_config, str):
                # Load from config file
                logger.info(f"Loading system configuration '{sys_config}'")
                config = load_system_config(sys_config)
                self._use_config(config)
            else:
                # Use provided config class or instance
                logger.info(
                    f"Using system configuration of type {type(sys_config).__name__}"
                )
                if isinstance(sys_config, type):  # If it's a class
                    self._use_config(sys_config())
                else:  # If it's an instance
                    self._use_config(sys_config)

            try:
                self._init_dev_config(self.devices_config)
            except Exception as e:
                logger.exception("Error initialising devices.")
                raise
        except Exception as e:
            logger.exception("Error initializing system configuration.")
            raise

    def set_matplotlib_style(self):
        """
        Matplotlib style that is used for saving data
        """
        light_grey = "#F0F0F0"
        dark_grey = "#D0D0D0"
        plt.rcParams.update(
            {
                "lines.color": "black",
                "patch.edgecolor": "black",
                "text.color": "black",
                "axes.facecolor": "white",
                "axes.edgecolor": "black",
                "axes.labelcolor": "black",
                "lines.linewidth": 0.75,
                "axes.linewidth": 0.75,
                "font.size": 10,
                "axes.titlesize": 10,
                "axes.labelsize": 10,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "xtick.color": "black",
                "ytick.color": "black",
                "grid.color": dark_grey,
                # Set the legend font color
                "legend.labelcolor": "black",
                "legend.facecolor": "white",
                "legend.framealpha": 0,
                "figure.facecolor": "white",
                "figure.edgecolor": "white",
                "savefig.facecolor": "white",
                "savefig.edgecolor": "white",
            }
        )

    def add_device_with_role(self, device: Device, role: DeviceRole) -> None:
        """Add a device to the system and assign it a role.

        Parameters
        ----------
        device : Device
            Device to add and assign role to
        role : DeviceRole
            Role to assign

        Raises
        ------
        ValueError
            If role is already assigned to a different device
        TypeError
            If device type is incompatible with role
        """
        from qscope.types.validation import _get_base_classes

        # Check device type compatibility
        is_valid, error_msg = role.validate_device_type(type(device))
        if not is_valid:
            raise TypeError(f"Cannot assign role {role}: {error_msg}")

        # Check if role already assigned to a different device
        for existing_device in self._devices:
            if existing_device is not device and existing_device.has_role(role):
                raise ValueError(
                    f"Role {role.__class__.__name__} already assigned to device "
                    f"{existing_device.__class__.__name__}"
                )

        # Add device to system's device list if not already present
        if device not in self._devices:
            self._devices.append(device)

        # Assign the role using protected method
        device._add_role(role)

    def _init_dev_config(
        self, dev_config: dict[DeviceRole, tuple[Type[Device], dict[str, Any]]]
    ) -> None:
        """Initialize device configuration and validate roles.

        Parameters
        ----------
        dev_config : dict[DeviceRole, tuple[Type[Device], dict[str, Any]]]
            Mapping from roles to (device class, config dict) tuples
        """
        logger.info("Initialising devices.")

        # Get configured roles
        config_roles = set(dev_config.keys())

        # Validate roles if system has requirements
        if hasattr(self, "validate_roles"):
            is_valid, error_msg = self.validate_roles(config_roles)
            if not is_valid:
                raise ValueError(error_msg)

        try:
            # Initialize devices
            for role, (device_class, config_dict) in dev_config.items():
                try:
                    # Filter out metadata fields from config dict
                    device_params = {
                        k: v
                        for k, v in config_dict.items()
                        if not k.startswith("_") and k not in ["role", "type"]
                    }

                    # Create device instance with filtered params
                    device = device_class(**device_params)
                    self.add_device_with_role(device, role)
                    logger.info(f"Initialized {device_class.__name__} with role {role}")
                except Exception as e:
                    logger.error(f"Failed to initialize {device_class.__name__}: {e}")
                    raise

        except Exception as e:
            logger.exception("Error initializing devices")
            raise

        # Validate device types against required roles after initialization
        if hasattr(self, "_required_roles"):
            # Construct device config dict for validation
            available_devices = {
                role: (type(device), "")
                # Empty string as config str not needed for validation
                for device in self._devices
                for role in device.get_roles()
            }

            # First check required roles
            is_valid, error_msg = validate_device_role_mapping(
                self._required_roles, available_devices
            )
            if not is_valid:
                raise ValueError(
                    f"Missing required roles for {self.__class__.__name__}: {error_msg}"
                )
                # Then check optional roles if present
            if hasattr(self, "_optional_roles") and self._optional_roles:
                is_valid, _ = validate_device_role_mapping(
                    self._optional_roles, available_devices
                )
                # Optional roles validation doesn't raise error if not found

    def _use_config(self, sysconfig: qscope.system.config.SystemConfig):
        for key in sysconfig.__dict__:
            if key[0] != "_":
                setattr(self, key, getattr(sysconfig, key))

    def has_camera(self):
        for device in self._devices:
            if isinstance(device, CameraProtocol):
                logger.info("System has a camera.")
                return True
        return False

    def get_all_devices_attrs(self):
        """This function is used to get the value of an attribute from the system class."""
        param_dict = {}
        for i in self._devices:
            name = i.__class__.__name__ + "_1"
            while name in param_dict:
                name = name[:-1] + str(int(name[-1]) + 1)
            param_dict[name] = i.get_all_attrs()
        return param_dict

    def get_metadata(self):
        """This function is used to unroll the metadata from the system class.
        It returns a dictionary with all the metadata from the system class."""
        param_dict = {}
        for i in self._devices:
            name = i.__class__.__name__ + "_1"
            while name in param_dict:
                name = name[:-1] + str(int(name[-1]) + 1)
            metadata = i.unroll_metadata()
            # Convert role constants to role class names
            if "roles" in metadata:
                metadata["roles"] = [role.__class__.__name__ for role in i.get_roles()]
            param_dict[name] = metadata
        param_dict["system_name"] = self.system_name
        param_dict["system_type"] = self.__class__.__name__
        param_dict["save_dir"] = self.save_dir
        param_dict["objective_pixel_size"] = self.objective_pixel_size
        return param_dict

    def connect_devices(self) -> dict[str, dict[str, bool | str]]:
        dev_status: dict[str, dict[str, bool | str]] = dict()
        for device in self._devices:
            ok, msg = device.open()
            name = device.__class__.__name__ + "_1"
            while name in dev_status:
                name = name[:-1] + str(int(name[-1]) + 1)
            dev_status[name] = {"status": ok, "message": msg}
        self.device_status = dev_status
        return dev_status

    def disconnect_devices(self):
        for device in self._devices:
            device.close()

    def add_bg_meas_task(self, meas_id: str, task: asyncio.Task):
        self._bg_meas_tasks[meas_id] = task

    def cancel_tasks(self):
        for task in self._bg_meas_tasks.values():
            try:
                task.cancel()
            except Exception:
                logger.exception("Error cancelling task. Continuing")
        self._bg_meas_tasks = dict()

    def packdown(self):
        self.cancel_tasks()  # cancels measurement tasks running in bg
        self.disconnect_devices()
        self.hardware_started_up = False

    def startup(self) -> dict[str, dict[str, bool | str]]:
        ret = self.connect_devices()
        self.hardware_started_up = True
        return ret

    # these are just for type-checking
    @overload
    def get_device_by_role(
        self, role: DeviceRole[CameraProtocol]
    ) -> CameraInterface: ...

    @overload
    def get_device_by_role(
        self, role: DeviceRole[RFSourceProtocol]
    ) -> RFSourceInterface: ...

    @overload
    def get_device_by_role(
        self, role: DeviceRole[SeqGenProtocol]
    ) -> SeqGenInterface: ...

    @overload
    def get_device_by_role(
        self, role: DeviceRole[DigitizerProtocol]
    ) -> DigitizerInterface: ...

    def get_device_by_role(self, role: DeviceRole) -> RoleInterface:
        """Get first device that fulfills the specified role.

        Parameters
        ----------
        role : DeviceRole
            Required role

        Returns
        -------
        Device
            Device fulfilling the role

        Raises
        ------
        ValueError
            If no device fulfills the role
        """
        for device in self._devices:
            if device.has_role(role):
                return device
        raise ValueError(f"No device found for role {role}")

    def has_device_role(self, role: DeviceRole) -> bool:
        """Check if system has any device fulfilling the specified role.

        Parameters
        ----------
        role : DeviceRole
            Role to check for

        Returns
        -------
        bool
            True if role is fulfilled
        """
        return any(d.has_role(role) for d in self._devices)

    def get_device_roles(self) -> set[DeviceRole]:
        """Get all roles fulfilled by devices in the system.

        Returns
        -------
        set[DeviceRole]
            Set of roles
        """
        return {role for d in self._devices for role in d.get_roles()}

    @check_role_is_connected()
    def setup_rf_sweep(
        self,
        freq_list: np.ndarray,
        power: float,
        step_time: float = 0.1,  # Needs to be the camera trigger time for camera systems
        role: DeviceRole[RFSourceProtocol] = PRIMARY_RF,
    ) -> None:
        """Configure RF source for frequency sweep.

        Parameters
        ----------
        freq_list : np.ndarray
            List of frequencies to sweep
        power : float
            RF power in dBm
        role : DeviceRole[RFSource] = PRIMARY_RF
            Which RF source to configure
        """
        rfsource = self.get_device_by_role(role)
        rfsource.set_power(power)  # POWER MUST BE SET BEFORE FREQ LIST
        rfsource.set_freq_list(freq_list, step_time)

    @check_role_is_connected()  # checks that 'role' device is connected
    def setup_single_rf_freq(
        self,
        rf_freq: float,
        rf_pow: float,
        role: DeviceRole[RFSourceProtocol] = PRIMARY_RF,
    ):
        rfsource = self.get_device_by_role(role)
        rfsource.set_freq(rf_freq)
        rfsource.set_power(rf_pow)

    @check_role_is_connected()
    def set_rf_state(
        self, state: bool, role: DeviceRole[RFSourceProtocol] = PRIMARY_RF
    ):
        rfsource = self.get_device_by_role(role)
        rfsource.set_state(state)

    @check_role_is_connected()
    def set_rf_params(
        self, freq: float, power: float, role: DeviceRole[RFSourceProtocol] = PRIMARY_RF
    ):
        rfsource = self.get_device_by_role(role)
        rfsource.set_freq(freq)
        rfsource.set_power(power)

    @check_role_is_connected()
    def start_rf_sweep(self, role: DeviceRole[RFSourceProtocol] = PRIMARY_RF) -> None:
        """Start RF frequency sweep."""
        self.get_device_by_role(role).start_sweep()

    @check_role_is_connected()
    def reconnect_rf(self, role: DeviceRole[RFSourceProtocol] = PRIMARY_RF) -> None:
        """Reconnect RF source."""
        self.get_device_by_role(role).reconnect()


@system_requirements(
    required_roles={SEQUENCE_GEN},
    optional_roles={PRIMARY_RF, SECONDARY_RF},
)
class SGSystem(System):
    """All systems with a common clock (pulseblaster etc.) - so most systems."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # use `requires_connected_devices` decorator here, assume we only ever
    # have one seqgen in a system
    @requires_connected_devices(SEQUENCE_GEN)  # checks that a SEQGEN is connected
    def set_laser_output(self, state):
        self.get_device_by_role(SEQUENCE_GEN).laser_output(state)

    @requires_connected_devices(SEQUENCE_GEN)
    def load_sequence(self, seq_name: str, **seq_kwargs) -> None:
        """Load sequence into sequence generator.

        Parameters
        ----------
        seq_name : str
            Name of sequence to load
        **seq_kwargs : dict
            Sequence parameters
        """
        self.get_device_by_role(SEQUENCE_GEN).load_seq(seq_name, **seq_kwargs)

    @requires_connected_devices(SEQUENCE_GEN)
    def reset_sequence(self) -> None:
        self.get_device_by_role(SEQUENCE_GEN).reset()

    @requires_connected_devices(SEQUENCE_GEN)
    def start_sequence(self) -> None:
        self.get_device_by_role(SEQUENCE_GEN).start()

    @requires_connected_devices(SEQUENCE_GEN)
    def set_laser_output(self, state):
        self.get_device_by_role(SEQUENCE_GEN).laser_output(state)

    @check_role_is_connected()
    @requires_connected_devices(SEQUENCE_GEN)
    def set_laser_rf_output(self, role: DeviceRole[RFSourceProtocol], state):
        """Set laser & rf to CW on/off."""
        self.get_device_by_role(SEQUENCE_GEN).laser_rf_output(state)
        self.set_rf_state(role, state)

    @check_role_is_connected()
    @requires_connected_devices(SEQUENCE_GEN)
    def set_rf_output(self, role: DeviceRole[RFSourceProtocol], state):
        """Set rf to CW on/off."""
        logger.info(f"Setting RF output to {state}")
        self.get_device_by_role(SEQUENCE_GEN).rf_output(state)
        self.set_rf_state(state, role)
        self.get_device_by_role(role).set_state(state)


@system_requirements(
    required_roles={SEQUENCE_GEN, MAIN_CAMERA},
    optional_roles={PRIMARY_RF, SECONDARY_RF},
)
class SGCameraSystem(SGSystem):
    """Common clock + camera."""

    # NOTE
    #  would need to implement differently for basic CW odmr systems.
    #  if we do that, copy a bunch of this across

    objective: str | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.objective_pixel_size) == 1:
            self.set_objective(list(self.objective_pixel_size.keys())[0])

    def _calc_pixel_size(self) -> tuple[float, float]:
        if self.has_camera():
            hwb = self.get_device_by_role(MAIN_CAMERA).get_hardware_binning()
            pa, pb = self.objective_pixel_size[self.get_objective()]
            return hwb * pa, hwb * pb
        else:
            logger.error("No camera found in system, cannot calculate pixel size.")
            return -1.0, -1.0

    @check_role_is_connected()
    def get_cam_trig_time(
        self, role: DeviceRole[CameraProtocol] = MAIN_CAMERA
    ) -> float:
        return self.get_device_by_role(role).get_trigger_time()

    def get_pixel_size(self) -> tuple[float, float]:
        return self._calc_pixel_size()

    def set_objective(self, objective) -> None:
        self.objective = objective

    def get_objective(self) -> str:
        if self.objective is None:
            raise ValueError("Objective attribute not set/chosen.")
        else:
            return self.objective

    @check_role_is_connected()
    def get_frame_shape(
        self, role: DeviceRole[CameraProtocol] = MAIN_CAMERA
    ) -> tuple[int, int]:
        return self.get_device_by_role(role).get_frame_shape()

    @check_role_is_connected()
    def take_snapshot(
        self, role: DeviceRole[CameraProtocol] = MAIN_CAMERA
    ) -> np.ndarray:
        return self.get_device_by_role(role).take_snapshot()

    @check_role_is_connected()
    async def start_stream(
        self, connection, typ, role: DeviceRole[CameraProtocol] = MAIN_CAMERA, **params
    ):
        self.stop_streams()  # stop any previous streams
        # NOTE -> above line means you need to clear stop_stream events in below!
        #  (see start_video in mock)
        if typ == "video":
            if not self.has_camera():
                logger.error("No camera found in system, cannot start video.")
                raise RuntimeError("No camera found in system, cannot start video.")
            await self.get_device_by_role(role).start_video(connection)
        else:
            logger.error(
                "Unknown stream type {}, can currently handle 'video' only.", typ
            )
            raise RuntimeError(
                f"Unknown stream type {typ}, can currently handle 'video' only."
            )
        self.streaming = True

    # unsure about the logic here, but probably sufficent for now.
    def stop_video(self):
        for cam in self._devices:
            if isinstance(cam, CameraProtocol):
                cam.stop_video()

    def stop_streams(self):
        self.stop_video()
        self.streaming = False

    @check_role_is_connected()
    def set_camera_params(
        self,
        exp_t: float,
        image_size: tuple[int, int],
        binning: tuple[int, int],
        role: DeviceRole[CameraProtocol] = MAIN_CAMERA,
    ):
        logger.info(
            f"Setting camera params: {exp_t=}, {image_size=}, {binning=}"
        )
        self.get_device_by_role(role).set_exposure_time(exp_t)
        self.get_device_by_role(role).set_frame_shape(
            image_size
        )  # FIXME change to set_roi??
        self.get_device_by_role(role).set_hardware_binning(binning)

    @check_role_is_connected()
    def setup_camera_sequence(
        self,
        nframes: int,
        exposure_time: float,
        frame_shape: tuple[int, int],
        role: DeviceRole[CameraProtocol] = MAIN_CAMERA,
    ):
        """Set up camera for sequence acquisition.

        Parameters
        ----------
        nframes : int
            Number of frames to acquire
        exposure_time : float # FIXME UNUSED??
            Exposure time in seconds
        frame_shape : tuple[int, int]
            ROI shape (height, width)
        role: DeviceRole[Camera] = MAIN_CAMERA
            Which camera to configure
        """
        # self.system.camera.set_hardware_binning(self.meas_config.hardware_binning)
        # camera also need to _set_frame_format -> set to 'chunks' for meas
        # also want to set bit depth?
        self.get_device_by_role(role).set_roi(frame_shape)
        self.get_device_by_role(role).clear_acquisition()
        self.get_device_by_role(role).setup_acquisition(
            mode="sequence", nframes=nframes
        )
        self.get_device_by_role(role).set_trigger_mode("ext_exp")
        time.sleep(0.5)  # wait for camera to set up

    @check_role_is_connected()
    def restart_camera_acquisition(
        self, role: DeviceRole[CameraProtocol] = MAIN_CAMERA
    ) -> None:
        """Stop and restart camera acquisition."""
        self.get_device_by_role(role).stop_acquisition()
        self.get_device_by_role(role).start_acquisition()

    def stop_all_acquisition(self) -> None:
        """Stop all acquisition devices, and stop sequences & rf sweeps etc.."""
        for device in self._devices:
            if isinstance(device, CameraProtocol):
                device.stop_acquisition()
                device.clear_acquisition()
            if isinstance(device, SeqGenProtocol):
                device.stop()
            if isinstance(device, RFSourceProtocol):
                device.stop_sweep()


class SGAPDSystem(SGSystem):
    pass
