"""Validation utilities for device roles and states.

This module provides validation functions to ensure:
1. Device roles map correctly to actual device types
2. All required devices are present and properly connected
3. System configurations meet their specified requirements

The validation system uses a role-based approach where:
- Each device type (Camera, RFSource etc) can fulfill specific roles
- Systems declare which roles they require and which are optional
- Validation ensures all required roles are fulfilled by appropriate devices
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, ForwardRef, Set, Type, TypeVar

from .messages import Response

if TYPE_CHECKING:
    from qscope.device import Device
    from qscope.system import System

    from .roles import DeviceRole

T = TypeVar("T", bound=Response)


@dataclass
class HandlerInfo:
    """Stores the mapping between a server handler and its client methods.

    Attributes:
        handler_func: The server handler function
        client_methods: List of client method names that use this handler
        command: The command string that identifies this handler
        system_types: Required system types
        required_roles: Required device roles
    """

    handler_func: Callable
    client_methods: list[str]
    command: str
    system_types: tuple[Type[System], ...] = ()
    required_roles: tuple[DeviceRole, ...] = ()


HANDLER_REGISTRY: dict[str, HandlerInfo] = {}
PENDING_COMMAND_VALIDATIONS: list[tuple[str, str]] = []


def _get_base_classes(cls: Type) -> list[str]:
    """Get names of class and all its base classes."""
    return [cls.__name__] + [base.__name__ for base in cls.__mro__[1:]]


class ValidationError(Exception):
    """Base exception for validation errors."""

    pass


def validate_device_role_mapping(
    roles: Set["DeviceRole"],
    device_config: dict["DeviceRole", tuple[Type["Device"], str]],
    allow_missing: bool = False,
) -> tuple[bool, str]:
    """Validate that device roles map to actual device types.

    Parameters
    ----------
    roles : Set[DeviceRole]
        The roles to validate
    device_config : dict[DeviceRole, tuple[Type[Device], str]]
        Mapping of roles to (device_class, config) tuples
    allow_missing : bool, optional
        If True, missing roles are allowed, by default False

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)
    """
    for role in roles:
        if role not in device_config:
            if not allow_missing:
                return False, f"Missing required role: {role}"
            continue

        device_class = device_config[role][0]
        is_valid, error_msg = role.validate_device_type(device_class)
        if not is_valid:
            return False, error_msg

    return True, ""


def validate_system_roles(
    required_roles: Set["DeviceRole"],
    optional_roles: Set["DeviceRole"],
    config_roles: Set["DeviceRole"],
) -> tuple[bool, str]:
    """Validate that a set of roles meets system requirements.

    Parameters
    ----------
    required_roles : Set[DeviceRole]
        Roles that must be present
    optional_roles : Set[DeviceRole]
        Roles that may be present
    config_roles : Set[DeviceRole]
        The actual roles to validate

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)
    """
    missing_required = required_roles - config_roles
    invalid_roles = config_roles - (required_roles | optional_roles)

    if missing_required:
        return False, "Missing required device roles for system type"
    if invalid_roles:
        return False, f"Invalid roles: {[r for r in invalid_roles]}"
    return True, ""


def validate_device_states(device: Device) -> tuple[bool, str]:
    """Validate device state.

    Parameters
    ----------
    device : Device
        Device to validate

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)
    """
    if not device.is_connected():
        return False, f"Device {device.__class__.__name__} is not connected"
    return True, ""


def validate_handler_client_correspondence() -> list[str]:
    """Validates the bidirectional correspondence between handlers and client methods.

    This function performs comprehensive validation of the protocol mapping between
    server handlers and client methods. It checks:

    1. All client commands (@command decorated) have matching handlers registered
    2. All handlers (@handler decorated) have at least one client method
    3. All declared client methods actually exist in the client module
    4. All client methods are properly decorated with @command
    5. Commands match between handlers and their client methods

    Returns:
        List of validation error messages, empty if all valid

    The validation happens in phases:
    1. First validates all pending commands against the handler registry
    2. Then checks all handlers have registered client methods
    3. Finally verifies client methods exist and are properly decorated

    This function is typically called during application startup after all
    modules are imported but before any commands are used.
    """
    errors = []

    # First validate all pending commands against handlers
    for command, func_name in PENDING_COMMAND_VALIDATIONS:
        if command not in HANDLER_REGISTRY:
            errors.append(
                f"Command {command} used by {func_name} not found in handler registry"
            )

    # Check all handlers have registered client methods
    for command, info in HANDLER_REGISTRY.items():
        if not info.client_methods:
            errors.append(
                f"Handler {info.handler_func.__name__} for command {command}"
                + " has no registered client methods"
            )

    # Check all client methods exist and are properly decorated
    import qscope.server.client as client

    for command, info in HANDLER_REGISTRY.items():
        for client_method in info.client_methods:
            if not hasattr(client, client_method):
                errors.append(
                    f"Client method {client_method} for command {command}"
                    + " not found in client module"
                )
                continue

            func = getattr(client, client_method)
            if not hasattr(func, "_is_client_method"):
                errors.append(
                    f"Client method {client_method} is not decorated with @command"
                )
            elif func._command != command:
                errors.append(
                    f"Client method {client_method} uses command {func._command}"
                    + " but handler registered it for {command}"
                )

    return errors


def assert_valid_handler_client_correspondence():
    """Validates handler-client correspondence and raises if invalid.

    Raises:
        AssertionError: If any validation errors are found

    This is the main entry point for protocol validation, typically
    called from test.
    """
    errors = validate_handler_client_correspondence()
    if errors:
        raise AssertionError(
            "Handler-client correspondence validation failed:\n"
            + "\n".join(f"- {err}" for err in errors)
        )
