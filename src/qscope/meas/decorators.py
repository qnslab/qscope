from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Callable, Type, TypeVar

from qscope.types import DeviceRole

if TYPE_CHECKING:
    from qscope.device import Device
    from qscope.meas import Measurement
    from qscope.system import System

T = TypeVar("T", bound="Measurement")


@dataclass
class MeasurementRequirements:
    """Requirements for a measurement type.

    Parameters
    ----------
    system_types : tuple[Type[System], ...]
        The required system types
    device_roles : tuple[DeviceRole, ...]
        The required device roles
    """

    system_types: tuple[Type[System], ...]
    device_roles: tuple[DeviceRole, ...]


def requires_hardware(
    *system_types: Type[System], roles: tuple[DeviceRole, ...] = ()
) -> Callable[[Type[T]], Type[T]]:
    """Decorators for specifying measurement hardware requirements.

    Parameters
    ----------
    *system_types : Type[System]
        Required system types
    roles : tuple[DeviceRole, ...], optional
        Required device roles

    Example
    -------
    @requires_hardware(
        SGCameraSystem,
        roles=(MAIN_CAMERA, SEQUENCE_GEN, RF_SOURCE)
    )
    class ESRMeasurement(Measurement):
        '''ESR measurement requiring camera, sequence generator and RF source.'''
        pass
    """

    def decorator(cls: Type[T]) -> Type[T]:
        class Wrapper(cls):  # type: ignore
            @wraps(cls.__init__)
            def __init__(self, system: System, *args, **kwargs) -> None:
                # Check system type
                if not isinstance(system, system_types):
                    raise TypeError(
                        f"{cls.__name__} requires one of these system types: "
                        f"{[t.__name__ for t in system_types]}, "
                        f"got {type(system).__name__}"
                    )

                self._hardware_requirements = MeasurementRequirements(
                    system_types=system_types, device_roles=roles
                )

                # Check required device roles
                missing_roles = []
                for role in self._hardware_requirements.device_roles:
                    if not system.has_device_role(role):
                        missing_roles.append(role)

                if missing_roles:
                    raise TypeError(
                        f"{cls.__name__} "
                        + f"requires device roles: {', '.join(str(r) for r in missing_roles)}. "
                        + f"Found roles: {system.get_device_roles()}"
                    )

                super().__init__(system, *args, **kwargs)

            def get_hardware_requirements(self) -> MeasurementRequirements:
                """Get measurement hardware requirements.

                Returns
                -------
                MeasurementRequirements
                    The hardware requirements
                """
                return self._hardware_requirements

        return Wrapper

    return decorator
