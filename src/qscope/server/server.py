# -*- coding: utf-8 -*-
"""
Server implementation of the client-server interface.

The server uses a decorator-based framework to maintain correspondence with client methods:

1. Each handler method is decorated with @handler to specify which client methods it handles
2. The handler decorator adds the mapping to the central protocol registry
3. Handler methods receive the server connection, system state, and client request
4. Handlers use the _send_response helper to reply to clients
5. The request_router maps incoming requests to the appropriate handler

This framework ensures that:
- Every handler explicitly declares which client methods it supports
- The mapping between handlers and clients is maintained in a central registry
- Protocol violations are caught through validation
- Responses are properly routed back to clients

The protocol correspondence can be validated using:
assert_valid_handler_client_correspondence()

See types.py for the protocol definitions and client.py for the client side.
"""
# ============================================================================

__pdoc__ = {
    "qscope.server.sys_server.start_local_server": True,
}

# ============================================================================

import asyncio
import json
import os
import time
from dataclasses import astuple
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Awaitable, Callable, Optional, Type, Union

import zmq
import zmq.asyncio
from loguru import logger
from setproctitle import setproctitle

# ============================================================================
import qscope
import qscope.meas
import qscope.system
import qscope.util
from qscope.meas import MEAS_STATE, Measurement
from qscope.server.bg_killer import kill_qscope_servers
from qscope.system import SGCameraSystem, SGSystem, System
from qscope.types import (
    CONSTS,
    HANDLER_REGISTRY,
    MAIN_CAMERA,
    PRIMARY_RF,
    SEQUENCE_GEN,
    ArrayResponse,
    ClientSyncResponse,
    DeviceLock,
    DeviceRole,
    DictResponse,
    ErrorResponse,
    HandlerInfo,
    MsgResponse,
    NewStream,
    Notification,
    PleaseWait,
    PleaseWaitResponse,
    Request,
    Response,
    SaveFullComplete,
    ServerConnection,
    Shape2DResponse,
    TupleResponse,
    ValueResponse,
    get_all_subclasses_map,
)
from qscope.util import (
    DEFAULT_HOST_ADDR,
    DEFAULT_LOGLEVEL,
    DEFAULT_PORT,
    format_error_response,
)

# ============================================================================


def get_servers_dir() -> Path:
    """Get the directory for storing server PID files."""
    base_dir = Path.home() / ".qscope"
    servers_dir = base_dir / "running_servers"
    servers_dir.mkdir(parents=True, exist_ok=True)
    return servers_dir


def register_server(host: str, ports: tuple[int, int, int]) -> Path:
    """Register a running server in the PID directory."""
    pid = os.getpid()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    server_info = {
        "pid": pid,
        "timestamp": timestamp,
        "host": host,
        "ports": {"msg": ports[0], "notif": ports[1], "stream": ports[2]},
    }

    pid_file = get_servers_dir() / f"server_{pid}.json"
    with pid_file.open("w") as f:
        json.dump(server_info, f, indent=2)

    return pid_file


# ============================================================================


async def _send_response(
    server_connection: ServerConnection, req_identity: bytes, response: Response
):
    logger.debug("*RESPONSE* (server->): {}", response)
    await server_connection.msg_socket.send_multipart(
        [req_identity, b"", response.to_msgpack()]
    )


# ============================================================================


async def client_handler(
    server_connection: ServerConnection,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
):
    # Add shutdown flag if it doesn't exist
    if not hasattr(server_connection, "shutdown_requested"):
        server_connection.shutdown_requested = False

    while not server_connection.shutdown_requested:
        # bit hacky: would be nicer to have a round-robin between notifs & msgs...

        # clear some items from notif_queue
        chunk = 0
        while not server_connection.notif_queue.empty():
            chunk += 1
            if chunk > 10:
                break  # limit to 10 notifs per loop, also need to check for msgs
            notif: Notification = server_connection.notif_queue.get_nowait()
            try:
                # below is rather loquacious
                logger.trace("*NOTIF* (server->): {}", notif)
                await server_connection.notif_socket.send(notif.to_msgpack())
            except Exception:
                logger.exception("ERROR SENDING NOTIF {}.", notif)

        # check for requests on the msg socket
        try:
            (
                req_identity,
                empty,
                req,
            ) = await server_connection.msg_socket.recv_multipart(zmq.NOBLOCK)
        except zmq.error.Again:
            await asyncio.sleep(0)
            continue
        # if msg recv'd, convert bytes to object
        try:
            request = Request.from_msgpack(req)
        except Exception:
            logger.exception("Request unpacking error:")
            await _send_response(
                server_connection,
                req_identity,
                ErrorResponse(value=format_error_response()),
            )
            continue

        # now handle the request
        try:
            await request_router(
                server_connection, req_identity, system, measurements, request
            )
        except Exception:
            logger.exception("Uncaught error in request_router.")
            await _send_response(
                server_connection,
                req_identity,
                ErrorResponse(value=format_error_response()),
            )

    # After exiting the loop due to shutdown
    logger.info("Client handler exiting due to shutdown request")


# ============================================================================


async def start_server(
    system_name: str,
    host: str = DEFAULT_HOST_ADDR,
    msg_port: int = DEFAULT_PORT,
    notif_port: int = DEFAULT_PORT + 1,
    stream_port: int = DEFAULT_PORT + 2,
    log_to_file: bool = True,
    log_to_stdout: bool = False,
    log_path: str = "",
    clear_prev_log: bool = True,
    log_level: str = DEFAULT_LOGLEVEL,
):
    kill_qscope_servers()  # only one server per machine at a time!

    # Format: "Qscope Server (2024-01-20 15:30:45)"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    setproctitle(f"qscope-server_{timestamp}")

    pid_file = register_server(host, (msg_port, notif_port, stream_port))

    qscope.util.start_server_log(
        log_to_file=log_to_file,
        log_to_stdout=log_to_stdout,
        log_path=log_path,
        clear_prev=clear_prev_log,
        log_level=log_level,
    )

    logger.info("Starting msg server on {}:{}", host, msg_port)

    system_config = qscope.system.get_system_config(str(system_name).lower())
    if system_config is None:
        logger.error("System {} not found.", system_name.lower())
        raise ValueError(f"System {system_name.lower()} not found.")

    systyp = system_config.system_type  # Get system type from config

    logger.info(
        "Opening a system of type {} and system type {}",
        system_name,
        systyp.__class__.__name__,
    )
    system = systyp(system_config)

    try:
        context = zmq.asyncio.Context()
        msg_socket = context.socket(zmq.ROUTER)
        msg_socket.bind(f"tcp://{host}:{msg_port}")  # bind on server side
        notif_socket = context.socket(zmq.PUB)
        notif_socket.bind(f"tcp://{host}:{notif_port}")
        stream_socket = context.socket(zmq.PUB)
        stream_socket.bind(f"tcp://{host}:{stream_port}")
        server_connection = ServerConnection(
            msg_socket=msg_socket,
            notif_socket=notif_socket,
            stream_socket=stream_socket,
            host=host,
            msg_port=msg_port,
            notif_port=notif_port,
            stream_port=stream_port,
            notif_queue=asyncio.Queue(),
        )
    except Exception as e:
        logger.exception("Error opening server-side connection.")
        raise e
    try:
        await client_handler(server_connection, system, dict())
    finally:
        if pid_file and pid_file.exists():
            try:
                pid_file.unlink()
            except:
                pass


# ============================================================================


def get_router_map() -> dict[
    str,
    Callable[
        [
            ServerConnection,
            bytes,
            qscope.system.System,
            dict[str, qscope.meas.Measurement],
            qscope.meas.Measurement | None,
            Request,
        ],
        Awaitable[None],
    ],
]:
    return {
        CONSTS.COMMS.PING: handle_ping,
        CONSTS.COMMS.SHUTDOWN: handle_shutdown,
        CONSTS.COMMS.ECHO: handle_echo,
        CONSTS.COMMS.GET_SERVER_LOG_PATH: handle_get_server_log_path,
        CONSTS.COMMS.STARTUP: handle_startup,
        CONSTS.COMMS.PACKDOWN: handle_packdown,
        CONSTS.COMMS.GET_ALL_MEAS_INFO: handle_get_all_meas_info,
        CONSTS.COMMS.GET_OTHER_PORTS: handle_get_other_ports,
        CONSTS.COMMS.CLIENT_SYNC: handle_client_sync,
        CONSTS.COMMS.IS_STREAMING: handle_is_streaming,
        CONSTS.COMMS.GET_DEVICE_LOCKS: handle_get_device_locks,
        CONSTS.COMMS.SAVE_LATEST_STREAM: handle_save_latest_stream,
        CONSTS.COMMS.SAVE_NOTES: handle_save_notes,
        # MEAS
        CONSTS.MEAS.GET_STATE: handle_get_measurement_state,
        CONSTS.MEAS.GET_INFO: handle_get_meas_info,
        CONSTS.MEAS.GET_FRAME: handle_get_frame_measurement,
        CONSTS.MEAS.GET_SWEEP: handle_get_sweep_measurement,
        CONSTS.MEAS.ADD: handle_add_measurement,
        CONSTS.MEAS.START: handle_start_measurement,
        CONSTS.MEAS.STOP: handle_stop_measurement,
        CONSTS.MEAS.PAUSE: handle_pause_measurement,
        CONSTS.MEAS.CLOSE: handle_close_measurement,
        CONSTS.MEAS.SET_AOI: handle_set_aoi_measurement,
        CONSTS.MEAS.SET_FRAME_NUM: handle_set_frame_num_measurement,
        CONSTS.MEAS.SET_ROLLING_AVG_WINDOW: handle_set_rolling_avg_window_measurement,
        CONSTS.MEAS.SET_ROLLING_AVG_MAX_SWEEPS: handle_set_rolling_avg_max_sweeps_measurement,
        CONSTS.MEAS.SAVE_SWEEP: handle_save_sweep_measurement,
        CONSTS.MEAS.SAVE_SWEEP_W_FIT: handle_save_sweep_w_fit_measurement,
        CONSTS.MEAS.SAVE_FULL_DATA: handle_save_full_data_measurement,
        # CAM
        CONSTS.CAM.SET_CAMERA_PARAMS: handle_set_camera_params,
        CONSTS.CAM.TAKE_SNAPSHOT: handle_take_snapshot,
        CONSTS.CAM.TAKE_AND_SAVE_SNAPSHOT: handle_take_and_save_snapshot,
        CONSTS.CAM.GET_FRAME_SHAPE: handle_get_frame_shape,
        CONSTS.CAM.START_VIDEO: handle_start_video,
        CONSTS.CAM.STOP_VIDEO: handle_stop_video,
        # SEQ GEN
        CONSTS.SEQGEN.LASER_OUTPUT: handle_laser_output,
        CONSTS.SEQGEN.LASER_RF_OUTPUT: handle_laser_rf_output,
        CONSTS.SEQGEN.RF_OUTPUT: handle_rf_output,
    }


# this function is essentially the 'server'
async def request_router(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    request: Request,
):
    logger.debug("*REQUEST* (server<-): {}", request)

    # first get applicable Measurement object if needed
    measurement = None
    if (
        request.command.startswith("CONSTS.MEAS.")
        and request.command != CONSTS.MEAS.ADD
    ):
        try:
            measurement = measurements[request.params["meas_id"]]
        except KeyError:
            logger.error(
                "No measurement matching meas_id: {}, currently operating: {}",
                request.params["meas_id"],
                measurements.keys(),
            )
            pass  # stay as None

    try:
        handler_func = get_router_map()[request.command]
        await handler_func(
            server_connection, req_identity, system, measurements, measurement, request
        )
    except KeyError:
        logger.error("Unknown request: {}", request.command)
        await _send_response(
            server_connection,
            req_identity,
            ErrorResponse(value=f"Unknown request: {request.command}"),
        )


# ============================================================================


def _check_devices_available(
    server_connection: ServerConnection,
    required_roles: tuple[DeviceRole, ...],
    requester: str,
) -> tuple[bool, str]:
    """Check if required device roles are available.

    Parameters
    ----------
    server_connection : ServerConnection
        The server connection
    required_roles : set[DeviceRole]
        Set of required device roles
    requester : str
        Description of what is requesting the devices

    Returns
    -------
    tuple[bool, str]
        (available, error_message)
    """
    locked_roles = []
    for role in required_roles:
        if role in server_connection.device_locks:
            lock = server_connection.device_locks[role]
            if lock.owner != requester:
                locked_roles.append(f"{role} (locked by {lock.description})")

    if locked_roles:
        return False, f"Required device roles are locked: {', '.join(locked_roles)}"
    return True, ""


def _lock_devices(
    server_connection: ServerConnection,
    devices: tuple[DeviceRole, ...],
    owner: str,
    description: str,
):
    """Lock specified devices"""
    now = time.time()
    for device_type in devices:
        server_connection.device_locks[device_type] = DeviceLock(
            owner=owner, description=description, timestamp=now
        )


def _unlock_devices(server_connection: ServerConnection, owner: str):
    """Unlock all devices owned by owner"""
    to_remove = [
        dev_type
        for dev_type, lock in server_connection.device_locks.items()
        if lock.owner == owner
    ]
    for dev_type in to_remove:
        del server_connection.device_locks[dev_type]


# ============================================================================
# ============== Handlers
# ============================================================================


def handler(
    command: str,
    *client_methods: str,
    system_types: tuple[Type[System], ...] = (),
    roles: tuple[DeviceRole, ...] = (),
) -> Callable[
    [Callable[..., Awaitable[None]]],
    Callable[
        [
            ServerConnection,
            bytes,
            System,
            dict[str, Measurement],
            Optional[Measurement],
            Request,
        ],
        Awaitable[None],
    ],
]:
    """Decorator that registers a server handler and its client methods.

    Args:
        command: The command string that identifies this handler
        *client_methods: Names of client methods that use this handler
        system_types: Required system types
        roles: Required device roles

    Returns:
        Decorated handler function

    Example:
        @handler(
            CONSTS.SEQGEN.LASER_RF_OUTPUT,
            "set_laser_rf_output",
            system_types=(SGSystem,),
            roles=(DeviceRoles.SEQUENCE_GEN, DeviceRoles.PRIMARY_RF)
        )
        async def handle_laser_rf_output(...):
            ...
    """

    def decorator(
        func: Callable[..., Awaitable[None]],
    ) -> Callable[
        [
            ServerConnection,
            bytes,
            System,
            dict[str, Measurement],
            Optional[Measurement],
            Request,
        ],
        Awaitable[None],
    ]:
        HANDLER_REGISTRY[command] = HandlerInfo(
            handler_func=func,
            client_methods=list(client_methods),
            command=command,
            system_types=system_types,
            required_roles=roles,
        )

        @wraps(func)
        async def wrapper(
            server_connection: ServerConnection,
            req_identity: bytes,
            system: System,  # Type checking will be done at runtime
            measurements: dict[str, Measurement],
            measurement: Optional[Measurement],
            request: Request,
        ) -> Awaitable[None]:
            # Validate system type
            if system_types and not isinstance(system, system_types):
                await _send_response(
                    server_connection,
                    req_identity,
                    ErrorResponse(
                        value=f"Operation requires one of these system types: "
                        f"{[t.__name__ for t in system_types]}, "
                        f"got {type(system).__name__}"
                    ),
                )
                return

            # Validate required roles
            if roles:
                missing_roles = []
                for role in roles:
                    if not system.has_device_role(role):
                        missing_roles.append(role)

                if missing_roles:
                    await _send_response(
                        server_connection,
                        req_identity,
                        ErrorResponse(
                            value=f"Operation requires device roles: {', '.join(missing_roles)}"
                        ),
                    )
                    return

            await func(
                server_connection,
                req_identity,
                system,
                measurements,
                measurement,
                request,
            )

        return wrapper

    return decorator


# ============================================================================


@handler(
    CONSTS.SEQGEN.LASER_RF_OUTPUT,
    "set_laser_rf_output",
    system_types=(SGSystem,),
    roles=(SEQUENCE_GEN, PRIMARY_RF),
)
async def handle_laser_rf_output(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.set_laser_rf_output
    # FIXME: Ignore hardware lock for now but should check if a measurement is running that
    # is not just the video stream.
    # if await hardware_lock_ok(server_connection, req_identity, request):
    if _check_devices_available(server_connection, (SEQUENCE_GEN,), "laser_rf_ouput"):
        try:
            state = request.params["state"]
            if state:
                _lock_devices(
                    server_connection,
                    (SEQUENCE_GEN,),
                    "laser_rf_ouput",
                    "laser_rf_ouput",
                )
            else:
                _unlock_devices(server_connection, "laser_rf_output")
            system.set_laser_rf_output(
                state, request.params["freq"], request.params["power"]
            )
            await _send_response(
                server_connection,
                req_identity,
                MsgResponse(value="Laser output changed."),
            )
        except Exception:
            await _send_response(
                server_connection,
                req_identity,
                ErrorResponse(value=format_error_response()),
            )


@handler(
    CONSTS.SEQGEN.LASER_OUTPUT,
    "set_laser_output",
    system_types=(SGSystem,),
    roles=(SEQUENCE_GEN, PRIMARY_RF),
)
async def handle_laser_output(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.set_laser_output
    if _check_devices_available(
        server_connection, (SEQUENCE_GEN, PRIMARY_RF), "laser_output"
    ):
        try:
            state = request.params["state"]
            if state:
                _lock_devices(
                    server_connection,
                    (SEQUENCE_GEN, PRIMARY_RF),
                    "laser_output",
                    "laser_output",
                )
            else:
                _unlock_devices(server_connection, "laser_output")
            system.set_laser_output(state)
            await _send_response(
                server_connection,
                req_identity,
                MsgResponse(value="Laser output changed."),
            )
        except Exception:
            await _send_response(
                server_connection,
                req_identity,
                ErrorResponse(value=format_error_response()),
            )


# ============================================================================


# TODO could add a device role param in request (or perhaps duplicate the command?
#  `roles` in @handler would get tricky)
@handler(
    CONSTS.SEQGEN.RF_OUTPUT,
    "set_rf_output",
    system_types=(SGSystem,),
    roles=(SEQUENCE_GEN, PRIMARY_RF),
)
async def handle_rf_output(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.set_rf_output

    if _check_devices_available(
        server_connection,
        (
            SEQUENCE_GEN,
            PRIMARY_RF,
        ),
        "rf_output",
    ):
        try:
            state = request.params["state"]
            if state:
                _lock_devices(
                    server_connection,
                    (
                        SEQUENCE_GEN,
                        PRIMARY_RF,
                    ),
                    "rf_output",
                    "rf_output",
                )
            else:
                _unlock_devices(server_connection, "rf_output")
            system.set_rf_output(state, request.params["freq"], request.params["power"])
            await _send_response(
                server_connection, req_identity, MsgResponse(value="RF output changed.")
            )
        except Exception:
            await _send_response(
                server_connection,
                req_identity,
                ErrorResponse(value=format_error_response()),
            )


# ============================================================================


@handler(CONSTS.COMMS.PING, "ping")  # No system or role requirements for basic comms
async def handle_ping(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    """Handle ping request from client."""
    handles: qscope.server.client.ping
    await _send_response(
        server_connection, req_identity, MsgResponse(value=CONSTS.COMMS.PONG)
    )


# ============================================================================


@handler(
    CONSTS.COMMS.SHUTDOWN, "shutdown_server"
)  # No requirements - should work for all systems
async def handle_shutdown(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.shutdown_server
    logger.info("Shutting down all measurements.")
    for meas_id, meas in measurements.items():
        try:
            if meas.state not in ["paused", "finished"]:
                meas.stop_now()
            while True:
                try:
                    meas.close()
                    break
                except PleaseWait:
                    await asyncio.sleep(0)
        except Exception:
            logger.exception("Error closing {}, continuing.", meas_id)
    logger.info(
        "Shutting down server: you may want to kill server "
        + "subprocess with the `kill_bg_server` method."
    )
    system.packdown()

    # Add shutdown flag to server_connection
    server_connection.shutdown_requested = True

    await _send_response(
        server_connection, req_identity, MsgResponse(value="Shutting down")
    )
    logger.info("Closing connection.")
    # Increase sleep time to give client more time to process the response and close its sockets
    await asyncio.sleep(3)  # Increased from 1 to 3 seconds

    # Close sockets with LINGER=0
    for socket in [
        server_connection.msg_socket,
        server_connection.notif_socket,
        server_connection.stream_socket,
    ]:
        try:
            socket.setsockopt(zmq.LINGER, 0)
            socket.close()
        except Exception as e:
            logger.warning(f"Error closing socket: {e}")

    # Try to terminate the ZMQ context if available
    try:
        if hasattr(server_connection, "context"):
            server_connection.context.term()
    except Exception as e:
        logger.warning(f"Error terminating ZMQ context: {e}")

    logger.info("Closing down server logger.")
    logger.remove()


# ============================================================================


@handler(CONSTS.COMMS.ECHO, "echo")
async def handle_echo(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.echo
    await _send_response(
        server_connection, req_identity, MsgResponse(value=request.params["msg"])
    )


# ============================================================================


@handler(CONSTS.COMMS.GET_SERVER_LOG_PATH, "get_server_log_path")
async def handle_get_server_log_path(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.get_server_log_path
    log_path = qscope.util.get_log_filename()
    logger.info("Server log path: {}", log_path)
    await _send_response(server_connection, req_identity, ValueResponse(value=log_path))


# ============================================================================


@handler(CONSTS.COMMS.STARTUP, "startup")
async def handle_startup(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.startup
    try:
        if not system.hardware_started_up:
            logger.info("Attempting to startup server system.")
            dev_status = system.startup()
            await _send_response(
                server_connection, req_identity, DictResponse(value=dev_status)
            )
        else:
            logger.info("System already started up.")
            await _send_response(
                server_connection,
                req_identity,
                DictResponse(value=system.device_status),
            )
    except Exception:
        await _send_response(
            server_connection,
            req_identity,
            ErrorResponse(value=format_error_response()),
        )


# ============================================================================


@handler(CONSTS.COMMS.PACKDOWN, "packdown")
async def handle_packdown(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.packdown
    try:
        if not system.hardware_started_up:
            logger.info("System not started up, nothing to pack down.")
            await _send_response(
                server_connection,
                req_identity,
                MsgResponse(value="System not started."),
            )
            return

        # NOTE is this what we want? I think it makes sense.
        if len(measurements):
            logger.info("Packdown requested: closing unclosed measurements.")
            for meas_id, meas in measurements.items():
                try:
                    if meas.state not in ["paused", "finished"]:
                        meas.stop_now()
                    while True:
                        try:
                            meas.close()
                            break
                        except PleaseWait:
                            await asyncio.sleep(0)
                except Exception:
                    logger.exception("Error closing {}, continuing.", meas_id)
        logger.info("Attempting to packdown server system.")
        system.packdown()
        await _send_response(
            server_connection, req_identity, MsgResponse(value="System packed down..")
        )
    except Exception:
        await _send_response(
            server_connection,
            req_identity,
            ErrorResponse(value=format_error_response()),
        )


# ============================================================================


@handler(CONSTS.COMMS.GET_ALL_MEAS_INFO, "get_all_meas_info")
async def handle_get_all_meas_info(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.get_all_meas_info
    try:
        meas_info = {k: v.get_info() for k, v in measurements.items()}
        await _send_response(
            server_connection, req_identity, DictResponse(value=meas_info)
        )
    except Exception:
        await _send_response(
            server_connection,
            req_identity,
            ErrorResponse(value=format_error_response()),
        )


# ============================================================================


@handler(CONSTS.COMMS.GET_OTHER_PORTS, "get_other_ports")
async def handle_get_other_ports(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.get_other_ports
    await _send_response(
        server_connection,
        req_identity,
        TupleResponse(
            value=(
                server_connection.notif_port,
                server_connection.stream_port,
            ),
        ),
    )


# ============================================================================


@handler(CONSTS.COMMS.CLIENT_SYNC, "client_sync")
async def handle_client_sync(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.client_sync
    await _send_response(
        server_connection,
        req_identity,
        ClientSyncResponse(
            system_type=system.__class__.__name__,
            system_name=system.system_name,
            is_streaming=system.streaming,
            all_meas_info={k: v.get_info() for k, v in measurements.items()},
            sys_metadata=system.get_metadata(),
            version=qscope.__version__,
            value=None,
            hardware_started_up=system.hardware_started_up
        ),
    )


# ============================================================================


@handler(CONSTS.COMMS.IS_STREAMING, "is_streaming")
async def handle_is_streaming(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.is_streaming
    await _send_response(
        server_connection, req_identity, ValueResponse(value=system.streaming)
    )


# ============================================================================


async def _check_meas_ok(
    server_connection: ServerConnection,
    req_identity: bytes,
    measurement: Optional[qscope.meas.Measurement],
):
    handles: None  # called 'internally' on server-side
    if measurement is None:
        await _send_response(
            server_connection,
            req_identity,
            ErrorResponse(value=f"Measurement not found."),
        )
        return False
    return True


# ============================================================================


@handler(CONSTS.MEAS.STOP, "stop_measurement")
async def handle_stop_measurement(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.stop_measurement
    if await _check_meas_ok(server_connection, req_identity, measurement):
        logger.info("Attempting to stop measurement.")
        try:
            measurement.stop_now()
            _unlock_devices(server_connection, measurement.get_description())
            await _send_response(
                server_connection,
                req_identity,
                MsgResponse(value="Measurement stopped."),
            )
        except Exception:
            await _send_response(
                server_connection,
                req_identity,
                ErrorResponse(value=format_error_response()),
            )


# ============================================================================


@handler(
    CONSTS.MEAS.GET_STATE,
    "measurement_get_state",
    "measurement_is_stopped",
    system_types=(SGSystem, SGCameraSystem),
)
async def handle_get_measurement_state(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.measurement_get_state
    if await _check_meas_ok(server_connection, req_identity, measurement):
        await _send_response(
            server_connection, req_identity, ValueResponse(value=measurement.state)
        )


# ============================================================================


@handler(
    CONSTS.MEAS.GET_INFO,
    "measurement_get_info",
    system_types=(SGSystem, SGCameraSystem),
)
async def handle_get_meas_info(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.measurement_get_info
    if await _check_meas_ok(server_connection, req_identity, measurement):
        meas_info: dict = measurement.get_info()
        await _send_response(
            server_connection, req_identity, DictResponse(value=meas_info)
        )


# ============================================================================


@handler(
    CONSTS.MEAS.GET_FRAME,
    "measurement_get_frame",
    "measurement_get_frame_shape",
    system_types=(SGCameraSystem,),
    roles=(MAIN_CAMERA,),
)
async def handle_get_frame_measurement(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.measurement_get_frame
    if await _check_meas_ok(server_connection, req_identity, measurement):
        frame = measurement.get_frame(
            frame_type=request.params["frame_type"],
            frame_num=request.params["frame_num"],
        )
        await _send_response(
            server_connection, req_identity, ArrayResponse(value=frame)
        )


# ============================================================================


@handler(
    CONSTS.MEAS.GET_SWEEP,
    "measurement_get_sweep",
    system_types=(SGSystem, SGCameraSystem),
)
async def handle_get_sweep_measurement(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.measurement_get_sweep
    if await _check_meas_ok(server_connection, req_identity, measurement):
        sweep = measurement.get_sweep()
        await _send_response(
            server_connection, req_identity, ArrayResponse(value=sweep)
        )


# ============================================================================


@handler(
    CONSTS.MEAS.ADD,
    "add_measurement",
    system_types=(SGSystem, SGCameraSystem),
)
async def handle_add_measurement(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.add_measurement
    try:
        config = qscope.meas.MeasurementConfig.from_dict(request.params)
    except Exception:
        logger.exception("Error unpacking measurement config.")
        await _send_response(
            server_connection,
            req_identity,
            ErrorResponse(value=format_error_response()),
        )
        return
    try:
        meas_map = get_all_subclasses_map(qscope.meas.Measurement)
        meas = meas_map.get(request.params["meas_type"], None)
        if meas is None:
            logger.error("Unknown measurement type: {}", request.params["meas_type"])
            await _send_response(
                server_connection,
                req_identity,
                ErrorResponse(value=format_error_response()),
            )
            return
            # can we avoid raising an error here? Try just continuing.
            # raise CommsError(
            #     f"Unknown measurement type: {request.params['meas_type']}"
            # )
        else:
            meas = meas(system, config, server_connection.notif_queue)
        meas_id = meas.get_meas_id()
        system.add_bg_meas_task(
            meas_id, asyncio.create_task(meas.state_machine())
        )  # run in bg
        await asyncio.sleep(0)  # let above task start
        measurements[meas_id] = meas
        await _send_response(
            server_connection, req_identity, MsgResponse(value=meas_id)
        )
        logger.info("Added measurement {}.", meas_id)
    except Exception:
        logger.exception("Error adding {}.", request.params["meas_type"])
        await _send_response(
            server_connection,
            req_identity,
            ErrorResponse(value=format_error_response()),
        )


# ============================================================================


@handler(
    CONSTS.MEAS.START,
    "start_measurement_nowait",
    "start_measurement_wait",
    system_types=(SGSystem, SGCameraSystem),
)
async def handle_start_measurement(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: qscope.meas.Measurement,
    request: Request,
):
    handles: (
        qscope.server.client.start_measurement_wait
        | qscope.server.client.start_measurement_nowait
    )
    if not await _check_meas_ok(server_connection, req_identity, measurement):
        return

    required_roles = measurement.get_hardware_requirements().device_roles
    available, error_msg = _check_devices_available(
        server_connection, required_roles, measurement.get_description()
    )

    if not available:
        await _send_response(
            server_connection, req_identity, ErrorResponse(value=error_msg)
        )
        return

    try:
        _lock_devices(
            server_connection,
            required_roles,
            measurement.get_description(),
            f"measurement {measurement.__class__.__name__}",
        )
        measurement.start()
        await _send_response(
            server_connection,
            req_identity,
            MsgResponse(value=f"Started measurement {measurement.get_description()}."),
        )
    except PleaseWait:
        await _send_response(
            server_connection,
            req_identity,
            PleaseWaitResponse(value="Wait then try again."),
        )
        return
    except Exception:
        await _send_response(
            server_connection,
            req_identity,
            ErrorResponse(value=format_error_response()),
        )


# ============================================================================


@handler(
    CONSTS.MEAS.PAUSE,
    "pause_endsweep_measurement",
    system_types=(SGSystem, SGCameraSystem),
)
async def handle_pause_measurement(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.pause_endsweep_measurement
    if await _check_meas_ok(server_connection, req_identity, measurement):
        measurement.pause_endsweep()
        await _send_response(
            server_connection,
            req_identity,
            MsgResponse(value=f"Paused measurement {measurement.get_description()}."),
        )
        _unlock_devices(server_connection, measurement.get_description())


# ============================================================================


@handler(
    CONSTS.MEAS.CLOSE,
    "close_measurement_nowait",
    "close_measurement_wait",
    system_types=(SGSystem, SGCameraSystem),
)
async def handle_close_measurement(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: (
        qscope.server.client.close_measurement_wait
        | qscope.server.client.close_measurement_nowait
    )
    if await _check_meas_ok(server_connection, req_identity, measurement):
        # this del tasks stuff is probably a bit OTT, but just to be sure
        meas_id = measurement.get_meas_id()
        try:
            measurement.close()
        except PleaseWait:
            await _send_response(
                server_connection,
                req_identity,
                PleaseWaitResponse(value="Wait then try again."),
            )
            return
        except Exception:
            await _send_response(
                server_connection,
                req_identity,
                ErrorResponse(value=format_error_response()),
            )
            return
        _unlock_devices(server_connection, measurement.get_description())
        system._bg_meas_tasks[meas_id].cancel()
        del system._bg_meas_tasks[meas_id]
        del measurements[meas_id]
        del measurement
        await _send_response(
            server_connection,
            req_identity,
            MsgResponse(value=f"Closed measurement {meas_id}."),
        )
        logger.info("Closed measurement {}.", meas_id)


# ============================================================================


@handler(
    CONSTS.MEAS.SET_AOI,
    "measurement_set_aoi",
    system_types=(SGCameraSystem,),
    roles=(MAIN_CAMERA,),
)
async def handle_set_aoi_measurement(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.measurement_set_aoi
    if await _check_meas_ok(server_connection, req_identity, measurement):
        try:
            measurement.set_aoi(request.params["aoi"])
            await _send_response(
                server_connection,
                req_identity,
                MsgResponse(value=f"AOI set to {request.params['aoi']}."),
            )
        except Exception:
            await _send_response(
                server_connection,
                req_identity,
                ErrorResponse(value="Error setting AOI: " + format_error_response()),
            )


# ============================================================================


@handler(
    CONSTS.MEAS.SET_FRAME_NUM,
    "measurement_set_frame_num",
    system_types=(SGCameraSystem,),
    roles=(MAIN_CAMERA,),
)
async def handle_set_frame_num_measurement(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.measurement_set_frame_num
    if await _check_meas_ok(server_connection, req_identity, measurement):
        if not system.has_camera():
            await _send_response(
                server_connection,
                req_identity,
                ErrorResponse(value="System does not have a camera."),
            )
        else:
            try:
                measurement.set_frame_num(request.params["frame_num"])
                await _send_response(
                    server_connection,
                    req_identity,
                    MsgResponse(
                        value=f"Frame number set to {request.params['frame_num']}."
                    ),
                )
            except Exception as e:
                await _send_response(
                    server_connection,
                    req_identity,
                    ErrorResponse(value="Error setting frame number: " + str(e)),
                )


# ============================================================================


@handler(
    CONSTS.MEAS.SET_ROLLING_AVG_WINDOW,
    "measurement_set_rolling_avg_window",
    system_types=(SGSystem, SGCameraSystem),
)
async def handle_set_rolling_avg_window_measurement(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.measurement_set_rolling_avg_window
    if await _check_meas_ok(server_connection, req_identity, measurement):
        try:
            measurement.set_rolling_avg_window(request.params["window"])
            await _send_response(
                server_connection,
                req_identity,
                MsgResponse(value=f"RA window set to {request.params['window']}."),
            )
        except Exception:
            await _send_response(
                server_connection,
                req_identity,
                ErrorResponse(
                    value="Error setting RW window: " + format_error_response()
                ),
            )


# ============================================================================


@handler(
    CONSTS.MEAS.SET_ROLLING_AVG_MAX_SWEEPS,
    "measurement_set_rolling_avg_max_sweeps",
    system_types=(SGSystem,),
)
async def handle_set_rolling_avg_max_sweeps_measurement(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.measurement_set_rolling_avg_max_sweeps
    if await _check_meas_ok(server_connection, req_identity, measurement):
        try:
            measurement.set_rolling_avg_window(request.params["max_sweeps"])
            await _send_response(
                server_connection,
                req_identity,
                MsgResponse(
                    value=f"RA max sweeps set to {request.params['max_sweeps']}."
                ),
            )
        except Exception:
            await _send_response(
                server_connection,
                req_identity,
                ErrorResponse(
                    value="Error setting RW max sweeps: " + format_error_response()
                ),
            )


# ============================================================================


@handler(
    CONSTS.MEAS.SAVE_SWEEP,
    "measurement_save_sweep",
    system_types=(SGSystem, SGCameraSystem),
)
async def handle_save_sweep_measurement(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.measurement_save_sweep
    if await _check_meas_ok(server_connection, req_identity, measurement):
        path = qscope.util.save_sweep(
            system, measurement, request.params["project_name"], request.params["notes"]
        )
        await _send_response(server_connection, req_identity, MsgResponse(value=path))


# ============================================================================


@handler(
    CONSTS.MEAS.SAVE_SWEEP_W_FIT,
    "measurement_save_sweep_w_fit",
    system_types=(SGSystem, SGCameraSystem),
)
async def handle_save_sweep_w_fit_measurement(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.measurement_save_sweep_w_fit
    if await _check_meas_ok(server_connection, req_identity, measurement):
        path = qscope.util.save_sweep_w_fit(
            system,
            measurement,
            request.params["project_name"],
            request.params["xdata"],
            request.params["ydata"],
            request.params["xfit"],
            request.params["yfit"],
            request.params["fit_results"],
            request.params["comparison_x"],
            request.params["comparison_y"],
            request.params["comparison_label"],
            request.params["color_map"],
            request.params["notes"],
        )
        await _send_response(server_connection, req_identity, MsgResponse(value=path))


# ============================================================================


@handler(
    CONSTS.MEAS.SAVE_FULL_DATA,
    "measurement_save_full_data",
    system_types=(SGSystem, SGCameraSystem),
)
async def handle_save_full_data_measurement(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    """Handle request to save full measurement data.

    Creates an async task to monitor measurement state and save when complete.
    Returns immediately with a message to monitor for SaveFullComplete.
    """
    handles: qscope.server.client.measurement_save_full_data

    if not await _check_meas_ok(server_connection, req_identity, measurement):
        return

    async def save_task():
        # wait for end-of-sweep/paused etc. to save
        while measurement.state == MEAS_STATE.RUNNING:
            await asyncio.sleep(0)

        try:
            # Save the data
            save_path = qscope.util.save_full_data(
                system,
                measurement,
                request.params["project_name"],
                request.params["notes"],
            )

            # Notify client of completion
            server_connection.notif_queue.put_nowait(
                SaveFullComplete(meas_id=measurement.get_meas_id(), save_path=save_path)
            )
        except Exception:
            logger.exception("Error in save task")

    # Start the save task
    asyncio.create_task(save_task())

    # Return immediately
    await _send_response(
        server_connection,
        req_identity,
        MsgResponse(
            value="Save task started, monitor for `SaveFullComplete` `Notification`."
        ),
    )


# ============================================================================


@handler(
    CONSTS.CAM.SET_CAMERA_PARAMS,
    "camera_set_params",
    system_types=(SGCameraSystem,),
    roles=(MAIN_CAMERA,),
)
async def handle_set_camera_params(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.camera_set_params
    available, error_msg = _check_devices_available(
        server_connection, (MAIN_CAMERA,), "camera_params"
    )
    if available:
        if not isinstance(system, (qscope.system.SGCameraSystem,)):
            logger.error("System is not a camera system.")
            await _send_response(
                server_connection,
                req_identity,
                MsgResponse(value="System is not a camera system."),
            )
        try:
            system.set_camera_params(**request.params)
            await _send_response(
                server_connection,
                req_identity,
                MsgResponse(value=f"Set camera parameters: " + str(request.params)),
            )
        except Exception:
            await _send_response(
                server_connection,
                req_identity,
                ErrorResponse(
                    value="Error setting camera parameters: " + format_error_response()
                ),
            )


# ============================================================================


@handler(
    CONSTS.CAM.TAKE_SNAPSHOT,
    "camera_take_snapshot",
    system_types=(SGCameraSystem,),
    roles=(MAIN_CAMERA,),
)
async def handle_take_snapshot(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.camera_take_snapshot
    available, error_msg = _check_devices_available(
        server_connection, (MAIN_CAMERA,), "snapshot"
    )
    if available:
        if not isinstance(system, (qscope.system.SGCameraSystem,)):
            logger.error(
                "{} err: System is not a camera system.", CONSTS.CAM.TAKE_SNAPSHOT
            )
            await _send_response(
                server_connection,
                req_identity,
                MsgResponse(value="System is not a camera system."),
            )
        try:
            frame = system.take_snapshot()
            await _send_response(
                server_connection, req_identity, ArrayResponse(value=frame)
            )
        except Exception:
            await _send_response(
                server_connection,
                req_identity,
                ErrorResponse(value=format_error_response()),
            )


# ============================================================================


@handler(
    CONSTS.CAM.TAKE_AND_SAVE_SNAPSHOT,
    "camera_take_and_save_snapshot",
    system_types=(SGCameraSystem,),
    roles=(MAIN_CAMERA,),
)
async def handle_take_and_save_snapshot(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.camera_take_and_save_snapshot
    available, error_msg = _check_devices_available(
        server_connection, (MAIN_CAMERA,), "save_snapshot"
    )
    if available:
        if not isinstance(system, (qscope.system.SGCameraSystem,)):
            logger.error(
                "{} err: System is not a camera system.",
                CONSTS.CAM.TAKE_AND_SAVE_SNAPSHOT,
            )
            await _send_response(
                server_connection,
                req_identity,
                MsgResponse(value="System is not a camera system."),
            )
        try:
            frame = system.take_snapshot()
            path = qscope.util.save_snapshot(
                system, request.params["project_name"], frame, request.params["notes"]
            )
            await _send_response(
                server_connection, req_identity, MsgResponse(value=path)
            )
        except Exception:
            await _send_response(
                server_connection,
                req_identity,
                ErrorResponse(value=format_error_response()),
            )


# ============================================================================


@handler(
    CONSTS.CAM.GET_FRAME_SHAPE,
    "camera_get_frame_shape",
    system_types=(SGCameraSystem,),
    roles=(MAIN_CAMERA,),
)
async def handle_get_frame_shape(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    handles: qscope.server.client.camera_get_frame_shape
    if not isinstance(system, (qscope.system.SGCameraSystem,)):
        logger.error(
            "{} err: System is not a camera system.", CONSTS.CAM.GET_FRAME_SHAPE
        )
        await _send_response(
            server_connection,
            req_identity,
            ErrorResponse(value="System is not a camera system."),
        )
    await _send_response(
        server_connection, req_identity, Shape2DResponse(value=system.get_frame_shape())
    )


# ============================================================================


@handler(
    CONSTS.CAM.START_VIDEO,
    "camera_start_video",
    system_types=(SGCameraSystem,),
    roles=(MAIN_CAMERA,),
)
async def handle_start_video(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    if not isinstance(system, (qscope.system.SGCameraSystem,)):
        logger.error("{} err: System is not a camera system.", CONSTS.CAM.START_VIDEO)
        await _send_response(
            server_connection,
            req_identity,
            MsgResponse(value="System is not a camera system."),
        )
        return

    available, error_msg = _check_devices_available(
        server_connection, (MAIN_CAMERA,), "video_stream"
    )

    if not available:
        await _send_response(
            server_connection, req_identity, ErrorResponse(value=error_msg)
        )
        return

    try:
        _lock_devices(server_connection, (MAIN_CAMERA,), "video_stream", "video stream")
        await system.start_stream(
            server_connection,
            typ="video",
        )
        await _send_response(
            server_connection,
            req_identity,
            MsgResponse(value="Started video."),
        )
        server_connection.notif_queue.put_nowait(NewStream(stream_type="video"))
    except Exception:
        await _send_response(
            server_connection,
            req_identity,
            ErrorResponse(value=format_error_response()),
        )


# ============================================================================


@handler(
    CONSTS.CAM.STOP_VIDEO,
    "camera_stop_video",
    system_types=(SGCameraSystem,),
    roles=(MAIN_CAMERA,),
)
async def handle_stop_video(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    if not isinstance(system, (qscope.system.SGCameraSystem,)):
        logger.error("{} err: System is not a camera system.", CONSTS.CAM.STOP_VIDEO)
        await _send_response(
            server_connection,
            req_identity,
            MsgResponse(value="System is not a camera system."),
        )
    try:
        system.stop_video()
        _unlock_devices(server_connection, "video_stream")
        await _send_response(
            server_connection,
            req_identity,
            MsgResponse(value="Stopped video."),
        )
    except Exception:
        await _send_response(
            server_connection,
            req_identity,
            ErrorResponse(value=format_error_response()),
        )


# ============================================================================


@handler(CONSTS.COMMS.GET_DEVICE_LOCKS, "get_device_locks")
async def handle_get_device_locks(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    """Handle request for current device locks.

    Returns dictionary mapping device types to their lock information.
    """
    handles: qscope.server.client.get_device_locks
    try:
        dev_locks = server_connection.device_locks
        dev_lock_map = {str(k): astuple(v) for k, v in dev_locks.items()}
        await _send_response(
            server_connection,
            req_identity,
            DictResponse(value=dev_lock_map),
        )
    except Exception:
        await _send_response(
            server_connection,
            req_identity,
            ErrorResponse(value=format_error_response()),
        )


# ============================================================================


@handler(CONSTS.COMMS.SAVE_LATEST_STREAM, "save_latest_stream")
async def handle_save_latest_stream(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    """Handle request to save the latest stream frame.

    Saves the latest frame from the video stream to a file.
    """
    handles: qscope.server.client.save_latest_stream
    if not system.streaming:
        await _send_response(
            server_connection,
            req_identity,
            ErrorResponse(value="System is not streaming."),
        )
        return
    try:
        last_stream = server_connection.get_last_stream_chunk()
        if last_stream is None:
            await _send_response(
                server_connection,
                req_identity,
                ErrorResponse(value="No stream data available."),
            )
            return
        stream_ttrace = server_connection.get_stream_time_trace()
        path = qscope.util.save_latest_stream(
            system,
            request.params["project_name"],
            last_stream,
            stream_ttrace,
            request.params["color_map"],
            request.params["notes"],
        )
        await _send_response(
            server_connection,
            req_identity,
            MsgResponse(value=path),
        )
    except Exception:
        await _send_response(
            server_connection,
            req_identity,
            ErrorResponse(value=format_error_response()),
        )


# ============================================================================


@handler(CONSTS.COMMS.SAVE_NOTES, "save_notes")
async def handle_save_notes(
    server_connection: ServerConnection,
    req_identity: bytes,
    system: qscope.system.System,
    measurements: dict[str, qscope.meas.Measurement],
    measurement: Optional[qscope.meas.Measurement],
    request: Request,
):
    """Handle request to save the notes as markdown on server side."""
    handles: qscope.server.client.save_notes
    try:
        path = qscope.util.save_notes(
            system, request.params["project_name"], request.params["notes"]
        )
        await _send_response(
            server_connection,
            req_identity,
            MsgResponse(value=path),
        )
    except Exception:
        await _send_response(
            server_connection,
            req_identity,
            ErrorResponse(value=format_error_response()),
        )
