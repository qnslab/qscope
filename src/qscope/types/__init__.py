"""
Protocol definitions, hardware abstraction, and client-server communication.

The qscope.types package forms the foundation of QScope's architecture by providing:

1. Hardware Abstraction Layer
    - Protocols define required methods for device roles
    - Interfaces provide type-safe access to device functionality
    - Roles connect devices to their interfaces
    - Runtime validation ensures devices implement required methods

2. Client-Server Communication
    - Message classes for requests and responses
    - Serialization using MessagePack over ZeroMQ
    - Type validation for communication safety

3. Configuration
    - Measurement configuration classes
    - System configuration definitions
    - Device configuration validation

The Role System Architecture
---------------------------
The role system is the core of QScope's hardware abstraction layer:

1. Protocols (protocols.py)
    - Define the methods a device must implement to fulfill a role.

2. Interfaces (interfaces.py)
    - Wrap devices and provide a clean, type-safe API for accessing functionality.

3. Roles (roles.py)
    - Connect devices to interfaces and validate protocol compliance.

4. Devices (in qscope.device)
    - Implement the methods required by protocols.

This architecture allows measurements to work with abstract roles rather than
specific hardware implementations, making it possible to swap hardware without
changing measurement code.

Flow Example:

1) Device implements protocol methods:
```python
class MyRFDevice(Device):
   def set_freq(self, freq: float) -> None: ...
   def set_power(self, power: float) -> None: ...
```

2) Role specifies required protocol:
```python
class PrimaryRFSource(DeviceRole[RFSourceProtocol]):
   interface_class = RFSourceInterface
```

3) System validates and provides interface:
```python
# Validates MyRFDevice implements RFSourceProtocol
system.add_device_with_role(MyRFDevice(), PRIMARY_RF)

# Returns RFSourceInterface wrapping MyRFDevice
rf = system.get_device_by_role(PRIMARY_RF)
rf.set_freq(2870.0)  # Use interface methods
```

Examples
--------
Creating a measurement configuration:
```python
from qscope.types import SGAndorCWESRConfig
config = SGAndorCWESRConfig(
    name="ESR Measurement",
    start_freq=2.7e9,
    stop_freq=3.0e9,
    num_points=101
)
```

Handling responses:
```python
from qscope.types import ErrorResponse
if isinstance(response, ErrorResponse):
    print(f"Error: {response.message}")
```

See Also
--------
qscope.server : Server-client communication module
qscope.types.messages : Message class definitions
qscope.types.roles : Device role definitions
qscope.types.protocols : Protocol definitions
qscope.types.interfaces : Interface implementations
"""

from __future__ import annotations

import asyncio
import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import zmq

from .commands import CONSTS
from .config import (
    TESTING_MEAS_CONFIG,
    CameraConfig,
    MeasurementConfig,
    MockSGAndorESRConfig,
    SGAndorCWESRConfig,
    SGAndorCWESRLongExpConfig,
    SGAndorPESRConfig,
    SGAndorRabiConfig,
    SGAndorRamseyConfig,
    SGAndorSpinEchoConfig,
    SGAndorT1Config,
)
from .messages import (
    ArrayResponse,
    ClientSyncResponse,
    DictResponse,
    ErrorResponse,
    MeasurementFrame,
    MeasurementUpdate,
    Message,
    MsgResponse,
    NewMeasurement,
    NewStream,
    Notification,
    PleaseWaitResponse,
    Request,
    Response,
    RollingAvgSweepUpdate,
    SaveFullComplete,
    Shape2DResponse,
    SweepUpdate,
    TupleResponse,
    ValueResponse,
    WarningResponse,
    get_all_subclasses_map,
)
from .roles import (
    MAIN_CAMERA,
    PREFIX_TO_ROLE,
    PRIMARY_RF,
    SECONDARY_CAMERA,
    SECONDARY_RF,
    SEQUENCE_GEN,
    DeviceRole,
    MainCamera,
    PrimaryRFSource,
    SecondaryCamera,
    SecondaryRFSource,
    SequenceGenerator,
    get_valid_device_types,
)
from .validation import (
    HANDLER_REGISTRY,
    PENDING_COMMAND_VALIDATIONS,
    HandlerInfo,
    ValidationError,
    assert_valid_handler_client_correspondence,
    validate_device_role_mapping,
    validate_device_states,
    validate_handler_client_correspondence,
)


@dataclass
class ClientConnection:
    """Client-side connection information."""

    context: zmq.Context
    msg_socket: zmq.Socket  # REQ socket (sync)
    notif_socket: zmq.Socket  # SUB socket for notifications
    stream_socket: zmq.Socket  # SUB socket for video/apd stream data
    host: str
    msg_port: int
    notif_port: int
    stream_port: int


@dataclass
class DeviceLock:
    """Information about a device lock."""

    owner: str  # e.g. "measurement_123", "video_stream"
    description: str  # Human readable description of what's using it
    timestamp: float


@dataclass
class ServerConnection:
    """Server-side connection information."""

    msg_socket: zmq.asyncio.Socket  # ROUTER socket
    notif_socket: zmq.asyncio.Socket  # PUB socket for notifications
    stream_socket: zmq.asyncio.Socket  # PUB socket for streams
    host: str
    msg_port: int
    notif_port: int
    stream_port: int
    notif_queue: asyncio.Queue
    device_locks: dict[DeviceRole, DeviceLock] = field(default_factory=dict)
    last_stream_message: Optional[np.ndarray] = field(default=None, init=False)
    stream_time_trace: list[float] = field(default_factory=list)

    def reset_stream(self) -> None:
        self.stream_time_trace = []
        self.last_stream_message = None

    def send_stream_chunk(self, msg: np.ndarray, header="video") -> None:
        """Send a stream message to the client."""
        if self.last_stream_message is not None:
            self.last_stream_message[:] = msg  # in-place op.
        else:
            self.last_stream_message = msg

        if msg.ndim > 1:  # video data -> mean the frame
            self.stream_time_trace.append(np.mean(msg))
        else:  # photodiode data, each element is a sample
            self.stream_time_trace.extend(msg)

        header_frame = header.encode("utf-8")
        message_frame = pickle.dumps(msg)
        self.stream_socket.send_multipart([header_frame, message_frame])

    def get_last_stream_chunk(self) -> Optional[np.ndarray]:
        return self.last_stream_message

    def get_stream_time_trace(self) -> list[float]:
        return self.stream_time_trace


# Exceptions
class PleaseWait(Exception):
    """Raised when an operation needs to wait before proceeding."""

    pass


class MeasurementStoppedError(Exception):
    """Raised when a measurement is stopped while waiting for notifications.
    
    Contains the latest notification of the requested type if available.
    """
    def __init__(self, message, latest_notification=None):
        super().__init__(message)
        self.latest_notification = latest_notification


class CommsError(Exception):
    """Base exception for communication errors."""

    pass


__all__ = [
    "ClientConnection",
    "ServerConnection",
    "DeviceLock",
    "Message",
    "Request",
    "Response",
    "get_all_subclasses_map",
    "ArrayResponse",
    "DictResponse",
    "ValueResponse",
    "Shape2DResponse",
    "TupleResponse",
    "ErrorResponse",
    "WarningResponse",
    "ClientSyncResponse",
    "PleaseWaitResponse",
    "MeasurementFrame",
    "SweepUpdate",
    "RollingAvgSweepUpdate",
    "MeasurementUpdate",
    "NewMeasurement",
    "NewStream",
    "ClientConnection",
    "Notification",
    "CONSTS",
    "MeasurementConfig",
    "CameraConfig",
    "MockSGAndorESRConfig",
    "SGAndorCWESRConfig",
    "SGAndorCWESRLongExpConfig",
    "SGAndorPESRConfig",
    "SGAndorRabiConfig",
    "SGAndorT1Config",
    "SGAndorRamseyConfig",
    "SGAndorSpinEchoConfig",
    "TESTING_MEAS_CONFIG",
    "DeviceRole",
    "ValidationError",
    "validate_device_role_mapping",
    "validate_device_states",
    "validate_handler_client_correspondence",
    "assert_valid_handler_client_correspondence",
    "HandlerInfo",
    "PleaseWait",
    "CommsError",
    "MainCamera",
    "SecondaryCamera",
    "PrimaryRFSource",
    "SecondaryRFSource",
    "SequenceGenerator",
    "MAIN_CAMERA",
    "SECONDARY_CAMERA",
    "PRIMARY_RF",
    "SECONDARY_RF",
    "SEQUENCE_GEN",
    "PREFIX_TO_ROLE",
    "PENDING_COMMAND_VALIDATIONS",
]
