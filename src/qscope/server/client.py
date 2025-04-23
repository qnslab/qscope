# -*- coding: utf-8 -*-
"""
Client implementation of the client-server interface.

The client uses a decorator-based framework to maintain correspondence with server handlers:

1. Each client method is decorated with @command to specify which handler it calls
2. The command decorator validates the method maps to a registered handler
3. Client methods use _send_request to communicate with the server
4. The protocol registry maintains the mapping between clients and handlers
5. Tools validate that the correspondence is maintained

This framework ensures that:
- Every client method explicitly declares which handler it calls
- Client methods are validated against the protocol registry
- The mapping between clients and handlers is maintained
- Requests are properly routed to handlers

The protocol correspondence can be validated using:
assert_valid_handler_client_correspondence()

See types.py for the protocol definitions and server.py for the server side.
"""

# ============================================================================

from __future__ import annotations

import asyncio
import os
import pickle
import subprocess
import sys
import time
from functools import wraps
from timeit import default_timer as timer
from tkinter import Tk, filedialog
from typing import TYPE_CHECKING, Any, Callable, Optional, Type, TypeVar, Union, cast

from qscope.meas import MEAS_STATE
from qscope.server import kill_qscope_servers
from qscope.types import PENDING_COMMAND_VALIDATIONS

if TYPE_CHECKING:
    import qscope.server

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import zmq
from loguru import logger
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1 import axes_size, make_axes_locatable

import qscope.util
from qscope.types import (
    CONSTS,
    ArrayResponse,
    ClientConnection,
    ClientSyncResponse,
    CommsError,
    DictResponse,
    ErrorResponse,
    MeasurementConfig,
    MeasurementStoppedError,
    MeasurementUpdate,
    MsgResponse,
    Notification,
    PleaseWaitResponse,
    Request,
    Response,
    Shape2DResponse,
    SweepUpdate,
    TupleResponse,
    ValueResponse,
)

# ============================================================================
from qscope.util import (
    DEFAULT_HOST_ADDR,
    DEFAULT_LOGLEVEL,
    DEFAULT_PORT,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    format_error_response,
    process_qt_events,
)

# ============================================================================

# ==========================c==========================================================
# ----------------------------------
# Connection & Message Classes
# ----------------------------------
# ====================================================================================


def start_bg_server(
    system_name: str,
    host: str = DEFAULT_HOST_ADDR,
    msg_port: int = DEFAULT_PORT,
    notif_port: int = DEFAULT_PORT + 1,
    stream_port: int = DEFAULT_PORT + 2,
    log_path: Optional[str] = None,  # if "" or None defaults to log_default_path_server
    clear_prev_log: bool = True,
    log_to_file: bool = True,
    log_to_stdout: bool = False,
    log_level: str = DEFAULT_LOGLEVEL,
) -> subprocess.Popen:
    """Start a background server process.

    Parameters
    ----------
    system_name : str
        Name of the system to start
    host : str, optional
        Host address to bind to, by default DEFAULT_HOST_ADDR
    msg_port : int, optional
        Port for message socket, by default DEFAULT_PORT
    notif_port : int, optional
        Port for notification socket, by default DEFAULT_PORT + 1
    stream_port : int, optional
        Port for stream socket, by default DEFAULT_PORT + 2
    log_path : str | None, optional
        Path to log file (None/empty for default), by default None
    clear_prev_log : bool, optional
        Whether to clear previous log, by default True
    log_to_file : bool, optional
        Whether to log to file, by default True
    log_to_stdout : bool, optional
        Whether to log to stdout, by default False
    log_level : str, optional
        Logging level, by default DEFAULT_LOGLEVEL

    Returns
    -------
    subprocess.Popen
        The server process handle

    Raises
    ------
    subprocess.SubprocessError
        If there are issues starting the subprocess
    FileNotFoundError
        If python executable or server script is not found
    PermissionError
        If there are permission issues executing the script
    OSError
        For other OS-level errors when starting the process

    Notes
    -----
    This function starts a background server process using the current Python
    interpreter. The server is started with the specified configuration and
    logging options.
    """
    if log_path is None or log_path == "":
        log_path = qscope.util.log_default_path_server()

    killed = (
        kill_qscope_servers()
    )  # kill any running bg servers THERE WILL ONLY BE ONE (per machine)
    logger.info("Killed {} running local servers.", killed)

    current_python_exec_path = sys.executable  # conda env py
    this_dir = os.path.dirname(os.path.realpath(__file__))
    proc = subprocess.Popen(
        [
            current_python_exec_path,
            this_dir + "/server_script.py",
            system_name,
            host,
            str(msg_port),
            str(notif_port),
            str(stream_port),
            log_path,
            str(clear_prev_log),
            str(log_to_file),
            str(log_to_stdout),
            str(log_level),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


# ============================================================================


def kill_bg_server(proc: subprocess.Popen):
    """Kill a background server process.

    Parameters
    ----------
    proc : subprocess.Popen
        The server process handle to kill

    Raises
    ------
    ProcessLookupError
        If the process no longer exists
    PermissionError
        If the current user lacks permission to kill the process
    subprocess.SubprocessError
        If there are issues communicating with the subprocess
    UnicodeDecodeError
        If the process output cannot be decoded as UTF-8

    Notes
    -----
    The process is forcefully terminated
    and its stdout/stderr are captured and logged.
    """
    logger.info("Killing server process.")
    pid = proc.pid
    proc.kill()
    outs, errs = proc.communicate()
    outs = outs.decode("utf-8")
    errs = errs.decode("utf-8")
    if outs:
        logger.info("#======= Server killed, outs: =======#")
        logger.info(outs)
    if errs:
        logger.error("#======= Server killed, errs: =======#")
        logger.error(errs)
        logger.error("PID = {}", pid)


# ====================================================================================


def _get_response(
    client_connection: ClientConnection,
    request: Request,
    request_retries: int = DEFAULT_RETRIES,
) -> Response:
    """Read a single response from the server.

    This function implements a reliable request-reply pattern that handles server failures
    and network issues. (ZMQ Lazy pirate). It will:
    - Poll the REQ socket and receive only when a reply has arrived
    - Resend the request if no reply arrives within timeout
    - Abandon the transaction after several failed retries
    - Handle shutdown commands specially, expecting disconnection

    Parameters
    ----------
    client_connection : ClientConnection
        The connection object containing the ZMQ sockets and connection info
    request : Request
        The request object to send to the server
    request_retries : int, optional
        Number of times to retry sending request before giving up, by default DEFAULT_RETRIES

    Returns
    -------
    Response
        The response object from the server. Could be ErrorResponse if server appears offline.

    Raises
    ------
    zmq.ZMQError
        If there are ZMQ-related errors during socket operations
    CommsError
        If the server appears to be offline after retries
    ValueError
        If the response cannot be deserialized from msgpack format
    """
    retries_left = request_retries + 1 # (+1 to account for the first attempt)
    is_shutdown_request = request.command == CONSTS.COMMS.SHUTDOWN

    logger.debug("*REQUEST* (client->): {}", request)
    client_connection.msg_socket.send(request.to_msgpack())
    logger.trace("*NOTIF* (client<-) Waiting for response...")
    while True:
        try:
            if client_connection.msg_socket.poll(1000 * DEFAULT_TIMEOUT, zmq.POLLIN):
                resp = client_connection.msg_socket.recv()
                resp = Response.from_msgpack(resp)
                logger.debug("*RESPONSE* (client<-): {}", resp)

                # For shutdown requests, we expect the server to close the connection
                # after sending the response, so return immediately
                if is_shutdown_request:
                    return resp

                return resp
        except zmq.ZMQError as e:
            # For shutdown requests, ZMQ errors are expected after response
            if is_shutdown_request:
                logger.info("Expected ZMQ error after shutdown command")
                return MsgResponse(value="Server shutting down")

            logger.warning(f"ZMQ error: {e}")

        retries_left -= 1
        logger.warning("No response from server...")
        # Socket is confused. Close and remove it.
        client_connection.msg_socket.setsockopt(zmq.LINGER, 0)
        client_connection.msg_socket.close()
        if retries_left == 0:
            logger.error("Server seems to be offline, abandoning.")
            return ErrorResponse(value="Server seems to be offline.")
        logger.info("Reconnecting to server...")
        # Create new connection, try request again (only on REQ/msg socket)
        client_connection.msg_socket = client_connection.context.socket(zmq.REQ)
        _reopen_connection(client_connection)
        logger.debug("*REQUEST* (client->): {}", request)
        client_connection.msg_socket.send(request.to_msgpack())


# ====================================================================================


def _confirm_connection(
    client_connection: ClientConnection, request_retries: int = DEFAULT_RETRIES
) -> bool:
    """Check if the server is up and running by sending a ping request.

    Parameters
    ----------
    client_connection : ClientConnection
        The connection object containing the ZMQ sockets
    request_retries : int, optional
        Number of retries for the ping request, by default DEFAULT_RETRIES

    Returns
    -------
    bool
        True if server responds with expected PONG response, False otherwise

    Raises
    ------
    zmq.ZMQError
        If there are ZMQ-related errors during socket operations
    CommsError
        If there are communication errors with the server
    ValueError
        If the response cannot be deserialized from msgpack format

    Notes
    -----
    This function allows extra time for the server process to start up by using
    the retry mechanism. It sends a PING request and expects a PONG response.
    Returns False if the response is incorrect or if an error occurs.
    """
    # similar to _get_response but without recreating connection & fixed request

    request = Request(CONSTS.COMMS.PING)
    _old = client_connection.host, client_connection.msg_port
    retries_left = request_retries
    logger.info("Confirming connection to server...")
    logger.debug("*REQUEST* (client->): {}", request)
    client_connection.msg_socket.send(request.to_msgpack())
    while True:
        process_qt_events()

        if client_connection.msg_socket.poll(1000 * DEFAULT_TIMEOUT, zmq.POLLIN):
            resp = client_connection.msg_socket.recv()
            resp = Response.from_msgpack(resp)
            logger.debug("*RESPONSE* (client<-): {}", resp)
            break
        retries_left -= 1
        logger.warning("No response from server...")
        # Socket is confused. Close and remove it.
        client_connection.msg_socket.setsockopt(zmq.LINGER, 0)
        client_connection.msg_socket.close()
        if retries_left == 0:
            logger.error("Server seems to be offline, abandoning.")
            resp = ErrorResponse(value="Server seems to be offline.")
            break
        client_connection.msg_socket = client_connection.context.socket(zmq.REQ)
        client_connection.msg_socket.connect(f"tcp://{_old[0]}:{_old[1]}")
        logger.info("Trying again...")
        logger.debug("*REQUEST* (client->): {}", request)
        client_connection.msg_socket.send(request.to_msgpack())

    if isinstance(resp, ErrorResponse):
        logger.error("Err during connection confirmation: '{}'", resp.value)
        return False
    if resp.value != CONSTS.COMMS.PONG:
        logger.error("Bad response from server.")
        return False
    return True


# ============================================================================


def _open_connection(
    host: str,
    msg_port: int,
    notif_port: int,
    stream_port: int,
) -> ClientConnection:
    """Open ZMQ sockets for all communication channels with the server.

    Parameters
    ----------
    host : str
        The host address to connect to
    msg_port : int
        Port number for the message socket (REQ/REP)
    notif_port : int
        Port number for the notification socket (PUB/SUB)
    stream_port : int
        Port number for the stream socket (PUB/SUB)

    Returns
    -------
    ClientConnection
        A connection object containing all the ZMQ sockets

    Raises
    ------
    CommsError
        If there are any issues establishing the connections
    zmq.ZMQError
        If there are any ZMQ-specific errors
    """
    logger.info("Attempting full connection to server on {}:{}.", host, msg_port)
    try:
        context = zmq.Context()
        msg_socket = context.socket(zmq.REQ)
        msg_socket.connect(f"tcp://{host}:{msg_port}")
        notif_socket = context.socket(zmq.SUB)
        notif_socket.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all
        notif_socket.connect(f"tcp://{host}:{notif_port}")
        stream_socket = context.socket(zmq.SUB)
        stream_socket.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all
        stream_socket.connect(f"tcp://{host}:{stream_port}")
        client_connection = ClientConnection(
            context,
            msg_socket,
            notif_socket,
            stream_socket,
            host,
            msg_port,
            notif_port,
            stream_port,
        )
    except Exception:
        logger.exception("Error during connection.")
        raise CommsError(f"Error during connection: {format_error_response()}")
    return client_connection

# ===================================== =======================================

def _reopen_connection(
    client_connection: ClientConnection,
) -> None:
    """Reopen ZMQ sockets for all communication channels with the server.

    This function updates the existing `client_connection` object in-place.

    Parameters
    ----------
    client_connection : ClientConnection
        The connection object to update

    Raises
    ------
    CommsError
        If there are any issues establishing the connections
    zmq.ZMQError
        If there are any ZMQ-specific errors
    """
    logger.info("Reopening connection to server on {}:{}.", client_connection.host, client_connection.msg_port)
    try:
        # Close existing sockets
        for socket in [
            client_connection.msg_socket,
            client_connection.notif_socket,
            client_connection.stream_socket,
        ]:
            if socket:
                socket.setsockopt(zmq.LINGER, 0)
                socket.close()

        # Reinitialize sockets
        client_connection.msg_socket = client_connection.context.socket(zmq.REQ)
        client_connection.msg_socket.connect(f"tcp://{client_connection.host}:{client_connection.msg_port}")

        client_connection.notif_socket = client_connection.context.socket(zmq.SUB)
        client_connection.notif_socket.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all
        client_connection.notif_socket.connect(f"tcp://{client_connection.host}:{client_connection.notif_port}")

        client_connection.stream_socket = client_connection.context.socket(zmq.SUB)
        client_connection.stream_socket.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all
        client_connection.stream_socket.connect(f"tcp://{client_connection.host}:{client_connection.stream_port}")

        logger.info("Connection reopened successfully.")
    except Exception:
        logger.exception("Error during connection reopening.")
        raise CommsError(f"Error during connection reopening: {format_error_response()}")

# ===================================== =======================================


def open_connection(
    host=DEFAULT_HOST_ADDR,
    msg_port=DEFAULT_PORT,
    timeout=DEFAULT_TIMEOUT,
    request_retries: int = DEFAULT_RETRIES,
) -> tuple[ClientConnection, ClientSyncResponse]:
    """Establish a connection to the server with retry mechanism.

    This function attempts to connect to the server, confirms the connection
    is working, and performs initial client synchronization.

    Parameters
    ----------
    host : str, optional
        The host address to connect to, by default DEFAULT_HOST_ADDR
    msg_port : int, optional
        Port number for the message socket, by default DEFAULT_PORT
    timeout : float, optional
        Maximum time to wait for connection, by default DEFAULT_TIMEOUT
    request_retries : int, optional
        Number of retry attempts for requests, by default DEFAULT_RETRIES

    Returns
    -------
    tuple[ClientConnection, ClientSyncResponse]
        A tuple containing:
        - The established client connection object
        - The synchronization response from the server

    Raises
    ------
    CommsError
        If connection cannot be established or synchronization fails
    TimeoutError
        If connection attempts exceed the timeout period
    zmq.ZMQError
        If there are any ZMQ-specific errors
    """
    t0 = timer()
    attempts = 0
    context = zmq.Context()
    logger.info("Attempting initial connection to server on {}:{}.", host, msg_port)
    while timer() - t0 < timeout:
        process_qt_events()

        try:
            msg_socket = context.socket(zmq.REQ)
            msg_socket.connect(f"tcp://{host}:{msg_port}")
            logger.info("Initial connection appears successful")
            break  # successful
        except Exception as e:
            attempts += 1
            logger.warning("Attempt {} failed to connect with error {}", attempts, e)
            process_qt_events()
            time.sleep(0.05)  # Always sleep a bit to prevent tight loop
            continue
    else:  # timed out
        logger.error("Connection not established.")
        raise CommsError("Connection not established: timed out.")
    # confirm msg connection works...
    ms, mp = msg_socket, msg_port
    temp_connection = ClientConnection(context, ms, ms, ms, host, mp, mp, mp)
    ok = _confirm_connection(temp_connection, request_retries=request_retries)
    if not ok:
        logger.error("Bad connection - no response from server.")
        try:
            temp_connection.msg_socket.setsockopt(zmq.LINGER, 0)
            temp_connection.msg_socket.close()
        except zmq.ZMQError:
            logger.debug("Socket already closed")
        raise CommsError("Bad connection - no response from server.")
    else:
        logger.info("Initial connection confirmed, now getting other ports.")
    # ok now grab the other ports
    try:
        notif_port, stream_port = get_other_ports(temp_connection, request_retries)
        # this will re-connect to msg_port
        client_connection = _open_connection(host, msg_port, notif_port, stream_port)
    except Exception as e:
        logger.exception("Error getting other ports.")
        raise e
    try:
        sync_response = client_sync(client_connection, request_retries)
        if sync_response.version != qscope.__version__:
            logger.critical(
                "Client-server version mismatch: {} vs {}",
                qscope.__version__,
                sync_response.version,
            )
    except Exception as e:
        logger.exception("Error during client sync.")
        raise e
    logger.info("Connection established on {}", host)
    return client_connection, sync_response


# ============================================================================


def close_connection(client_connection: ClientConnection):
    """Close the connection to the server.

    Arguments
    ---------
    client_connection : ClientConnection
        The connection object to close.
    """
    logger.info("Closing connection.")
    for socket in [
        client_connection.msg_socket,
        client_connection.notif_socket,
        client_connection.stream_socket,
    ]:
        if isinstance(socket, zmq.Socket):
            try:
                socket.setsockopt(zmq.LINGER, 0)
                socket.close()
            except zmq.ZMQError as e:
                logger.debug(f"Error closing socket: {e}")

    # Also terminate the ZMQ context to ensure clean shutdown
    try:
        if hasattr(client_connection, "context") and client_connection.context:
            client_connection.context.term()
    except Exception as e:
        logger.debug(f"Error terminating ZMQ context: {e}")


# ============================================================================


T = TypeVar("T", bound=Response)


def _send_request(
    client_connection: ClientConnection,
    request: Request,
    request_retries: int = DEFAULT_RETRIES,
) -> T:
    """Send a request to the server and get a response.

    Parameters
    ----------
    client_connection : ClientConnection
        The connection object to the server
    request : Request
        The request object to send
    request_retries : int, optional
        Number of retries, by default DEFAULT_RETRIES

    Returns
    -------
    T
        The response object of the expected type

    Raises
    ------
    CommsError
        If the server returns an error
    TypeError
        If response is not of expected type
    """
    resp = _get_response(client_connection, request, request_retries)
    if isinstance(resp, ErrorResponse):
        logger.error("Error during {}: '{}'", request.command, resp.value)
        raise CommsError(f"Error returned from {request.command}: {resp.value}")
    return cast(T, resp)


# ====================================================================================


def command(
    command_str: str, response_type: Type[T] | type[Union[Any, ...]] = "Response"
) -> Callable[[Callable[..., Any]], Callable[..., T]]:
    """Decorator that marks a client method and validates its handler mapping.

    Args:
        command_str: The command string that identifies this client method
        response_type: The expected response type from the server or Union of types

    Returns:
        Decorated client method with proper type information
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., T]:
        # Store for later validation instead of immediate check
        PENDING_COMMAND_VALIDATIONS.append((command_str, func.__name__))

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result = func(*args, **kwargs)
            return cast(T, result)

        wrapper._command = command_str
        wrapper._response_type = response_type
        wrapper._is_client_method = True
        return wrapper

    return decorator


# ====================================================================================
# -----------------
# INTERFACE METHODS
# -----------------
# ====================================================================================

# each of these has a corresponding function in server.py to handle the request.
# -> may want to catch CommsError(s) here?

# -------------------------------------------------------------------------------------
# General server comms
# -------------------------------------------------------------------------------------


@command(CONSTS.COMMS.CLIENT_SYNC, response_type=ClientSyncResponse | ErrorResponse)
def client_sync(
    client_connection: ClientConnection, request_retries: int = DEFAULT_RETRIES
) -> ClientSyncResponse:
    """Synchronize the client with the server.

    Parameters
    ----------
    client_connection : ClientConnection
        The connection to the server
    request_retries : int, optional
        Number of retry attempts, by default DEFAULT_RETRIES

    Returns
    -------
    ClientSyncResponse
        The server's synchronization response
    """
    calls: qscope.server.server.handle_client_sync
    resp = _send_request(
        client_connection, Request(CONSTS.COMMS.CLIENT_SYNC), request_retries
    )
    return resp


# ============================================================================


@command(CONSTS.COMMS.GET_OTHER_PORTS, response_type=TupleResponse | ErrorResponse)
def get_other_ports(
    client_connection: ClientConnection, request_retries: int = DEFAULT_RETRIES
) -> tuple[int, int]:
    """Get the notification and stream ports from the server.

    Parameters
    ----------
    client_connection : ClientConnection
        The connection to the server
    request_retries : int, optional
        Number of retry attempts, by default DEFAULT_RETRIES

    Returns
    -------
    tuple[int, int]
        The notification and stream ports
    """
    calls: qscope.server.server.handle_get_other_ports
    resp = _send_request(
        client_connection, Request(CONSTS.COMMS.GET_OTHER_PORTS), request_retries
    )
    return resp.value


# ============================================================================


@command(CONSTS.COMMS.PING, response_type=MsgResponse | ErrorResponse)
def ping(
    client_connection: ClientConnection, request_retries: int = DEFAULT_RETRIES
) -> str:
    """Send ping request to server.

    Parameters
    ----------
    client_connection : ClientConnection
        The connection to the server
    request_retries : int, optional
        Number of retry attempts, by default DEFAULT_RETRIES

    Returns
    -------
    str
        "pong" if successful
    """
    calls: qscope.server.server.handle_ping
    try:
        resp = _send_request(
            client_connection, Request(CONSTS.COMMS.PING), request_retries
        )
        return "pong"
    except Exception:
        logger.exception("Ping failed.")
        return "-1.0"


# ============================================================================


@command(CONSTS.COMMS.GET_SERVER_LOG_PATH, response_type=ValueResponse | ErrorResponse)
def get_server_log_path(
    client_connection: ClientConnection, request_retries: int = DEFAULT_RETRIES
) -> str:
    """Get the server's log file path.

    Parameters
    ----------
    client_connection : ClientConnection
        The connection to the server
    request_retries : int, optional
        Number of retry attempts, by default DEFAULT_RETRIES

    Returns
    -------
    str
        Path to the server's log file
    """
    calls: qscope.server.server.handle_get_server_log_path
    resp = _send_request(
        client_connection, Request(CONSTS.COMMS.GET_SERVER_LOG_PATH), request_retries
    )
    logger.info("Server log path: {}", resp.value)
    return resp.value


# ============================================================================


@command(CONSTS.COMMS.SHUTDOWN, response_type=MsgResponse | ErrorResponse)
def shutdown_server(
    client_connection: ClientConnection, request_retries: int = DEFAULT_RETRIES
) -> None:
    """Send shutdown request to server and handle expected disconnection.

    This function sends a shutdown command to the server and expects the server
    to close the connection afterward. It handles any connection errors that
    may occur during this process.
    """
    calls: qscope.server.server.handle_shutdown
    try:
        # Send the shutdown request with fewer retries since we expect disconnection
        resp = _send_request(client_connection, Request(CONSTS.COMMS.SHUTDOWN), 1)
        logger.info("Server shutdown initiated successfully")

        # Close all sockets with LINGER=0 to prevent hanging
        for socket in [
            client_connection.msg_socket,
            client_connection.notif_socket,
            client_connection.stream_socket,
        ]:
            try:
                socket.setsockopt(zmq.LINGER, 0)
                socket.close()
            except:
                pass

        # Terminate the ZMQ context to ensure clean shutdown
        try:
            client_connection.context.term()
        except:
            pass

        return resp
    except zmq.ZMQError as e:
        logger.info(f"Expected ZMQ error during shutdown: {e}")
        # Close sockets and terminate context even if there was an error
        for socket in [
            client_connection.msg_socket,
            client_connection.notif_socket,
            client_connection.stream_socket,
        ]:
            try:
                socket.setsockopt(zmq.LINGER, 0)
                socket.close()
            except:
                pass

        try:
            client_connection.context.term()
        except:
            pass

        return None
    except Exception as e:
        logger.warning(f"Unexpected error during shutdown: {e}")
        # Still try to close sockets and terminate context
        for socket in [
            client_connection.msg_socket,
            client_connection.notif_socket,
            client_connection.stream_socket,
        ]:
            try:
                socket.setsockopt(zmq.LINGER, 0)
                socket.close()
            except:
                pass

        try:
            client_connection.context.term()
        except:
            pass

        return None


# ============================================================================


@command(CONSTS.COMMS.ECHO, response_type=MsgResponse | ErrorResponse)
def echo(
    client_connection: ClientConnection,
    msg: str,
    request_retries: int = DEFAULT_RETRIES,
) -> str:
    """Echo a message through the server."""
    calls: qscope.server.server.handle_echo
    resp = _send_request(
        client_connection, Request(CONSTS.COMMS.ECHO, {"msg": msg}), request_retries
    )
    return resp.value


# ============================================================================


@command(CONSTS.COMMS.STARTUP, response_type=DictResponse | ErrorResponse)
def startup(
    client_connection: ClientConnection, request_retries: int = 3*DEFAULT_RETRIES
) -> tuple[bool, dict[str, dict[str, str | bool]]]:
    """Start up the system on the server."""
    calls: qscope.server.server.handle_startup
    # format of dev_status: {"device_name": {"status": bool, "msg": str}, ...}
    logger.info("Starting up system on server.")
    resp = _send_request(
        client_connection, Request(CONSTS.COMMS.STARTUP), request_retries
    )
    if isinstance(resp, ErrorResponse):
        logger.exception("Error during startup.")
        return False, {}
    else:
        dev_status = resp.value
        for devname, devd in dev_status.items():
            if not devd["status"]:
                logger.error("Device status: {}", dev_status)
                return False, dev_status
        logger.info("Device status: {}", dev_status)
        logger.info("System started on server.")
    return True, dev_status


# ============================================================================


@command(CONSTS.COMMS.PACKDOWN, response_type=MsgResponse | ErrorResponse)
def packdown(
    client_connection: ClientConnection, request_retries: int = DEFAULT_RETRIES
) -> None:
    """Send packdown request to server."""
    calls: qscope.server.server.handle_packdown
    resp = _send_request(
        client_connection, Request(CONSTS.COMMS.PACKDOWN), request_retries
    )
    return resp


# ============================================================================


@command(CONSTS.COMMS.GET_ALL_MEAS_INFO, response_type=DictResponse | ErrorResponse)
def get_all_meas_info(
    client_connection: ClientConnection, request_retries: int = DEFAULT_RETRIES
) -> dict:
    """Get info about all measurements."""
    calls: qscope.server.server.handle_get_all_meas_info
    resp = _send_request(
        client_connection, Request(CONSTS.COMMS.GET_ALL_MEAS_INFO), request_retries
    )
    # may want to handle dict -> MeasurementConfig here? Key is "meas_config"
    return resp.value


# ============================================================================


@command(CONSTS.COMMS.SAVE_LATEST_STREAM, response_type=MsgResponse | ErrorResponse)
def save_latest_stream(
    client_connection: ClientConnection,
    project_name: str,
    color_map: str = "seaborn:mako",
    notes: str = "",
    request_retries: int = DEFAULT_RETRIES,
) -> MsgResponse | ErrorResponse:
    """Save the last stream image to a file."""
    calls: qscope.server.server.handle_save_latest_stream
    resp = _send_request(
        client_connection,
        Request(
            CONSTS.COMMS.SAVE_LATEST_STREAM,
            {"project_name": project_name, "color_map": color_map, "notes": notes},
        ),
        request_retries,
    )
    return resp.value


# ============================================================================


@command(CONSTS.COMMS.SAVE_NOTES, response_type=MsgResponse | ErrorResponse)
def save_notes(
    client_connection: ClientConnection,
    project_name: str,
    notes: str = "",
    request_retries: int = DEFAULT_RETRIES,
) -> MsgResponse | ErrorResponse:
    """Save the last stream image to a file."""
    calls: qscope.server.server.handle_save_notes
    resp = _send_request(
        client_connection,
        Request(
            CONSTS.COMMS.SAVE_NOTES,
            {"project_name": project_name, "notes": notes},
        ),
        request_retries,
    )
    return resp.value


# -------------------------------------------------------------------------------------
# Seq gen comms
# -------------------------------------------------------------------------------------


@command(CONSTS.SEQGEN.LASER_OUTPUT, response_type=MsgResponse | ErrorResponse)
def set_laser_output(
    client_connection: ClientConnection,
    state: bool,
    request_retries: int = DEFAULT_RETRIES,
) -> None:
    """Turn laser on or off."""
    calls: qscope.server.server.handle_laser_output
    resp = _send_request(
        client_connection,
        Request(CONSTS.SEQGEN.LASER_OUTPUT, {"state": state}),
        request_retries,
    )
    return resp


# TODO could add a device role param in request
@command(CONSTS.SEQGEN.RF_OUTPUT, response_type=MsgResponse | ErrorResponse)
def set_rf_output(
    client_connection: ClientConnection,
    state: bool,
    freq: float,
    power: float,
    request_retries: int = DEFAULT_RETRIES,
) -> None:
    """Turn RF output on or off."""
    calls: qscope.server.server.handle_rf_output
    resp = _send_request(
        client_connection,
        Request(
            CONSTS.SEQGEN.RF_OUTPUT, {"state": state, "freq": freq, "power": power}
        ),
        request_retries,
    )
    return resp


@command(CONSTS.SEQGEN.LASER_RF_OUTPUT, response_type=MsgResponse | ErrorResponse)
def set_laser_rf_output(
    client_connection: ClientConnection,
    state: bool,
    freq: float,
    power: float,
    request_retries: int = DEFAULT_RETRIES,
) -> None:
    """Turn laser and RF output on or off."""
    calls: qscope.server.server.handle_laser_rf_output
    resp = _send_request(
        client_connection,
        Request(
            CONSTS.SEQGEN.LASER_RF_OUTPUT,
            {"state": state, "freq": freq, "power": power},
        ),
        request_retries,
    )
    return resp


# -------------------------------------------------------------------------------------
# Measurement comms
# -------------------------------------------------------------------------------------


@command(CONSTS.MEAS.STOP, response_type=MsgResponse | ErrorResponse)
def stop_measurement(
    client_connection: ClientConnection,
    meas_id: str,
    request_retries: int = DEFAULT_RETRIES,
) -> bool:
    """Stop a measurement."""
    calls: qscope.server.server.handle_stop_measurement
    _ = _send_request(
        client_connection,
        Request(CONSTS.MEAS.STOP, {"meas_id": meas_id}),
        request_retries,
    )
    logger.info("Measurement stopped.")
    return True


# ============================================================================


@command(CONSTS.MEAS.GET_STATE, response_type=MsgResponse | ErrorResponse)
def measurement_is_stopped(
    client_connection: ClientConnection,
    meas_id: str,
    request_retries: int = DEFAULT_RETRIES,
) -> bool:
    """Check if a measurement is stopped."""
    calls: qscope.server.server.handle_get_measurement_state
    resp = _send_request(
        client_connection,
        Request(CONSTS.MEAS.GET_STATE, {"meas_id": meas_id}),
        request_retries,
    )
    return resp.value not in [MEAS_STATE.RUNNING, MEAS_STATE.PAUSED]


# ============================================================================


# TODO I think meas info should be a dataclass on client end, not a dict?
@command(CONSTS.MEAS.GET_INFO, response_type=DictResponse | ErrorResponse)
def measurement_get_info(
    client_connection: ClientConnection,
    meas_id: str,
    request_retries: int = DEFAULT_RETRIES,
) -> dict:
    calls: qscope.server.server.handle_get_meas_info  # just for code navigation
    resp = _send_request(
        client_connection,
        Request(CONSTS.MEAS.GET_INFO, {"meas_id": meas_id}),
        request_retries,
    )
    return resp.value


# ============================================================================


@command(CONSTS.MEAS.GET_STATE, response_type=MsgResponse | ErrorResponse)
def measurement_get_state(
    client_connection: ClientConnection,
    meas_id: str,
    request_retries: int = DEFAULT_RETRIES,
) -> str:
    calls: qscope.server.server.handle_get_measurement_state  # just for code navigation
    resp = _send_request(
        client_connection,
        Request(CONSTS.MEAS.GET_STATE, {"meas_id": meas_id}),
        request_retries,
    )
    return resp.value


# ============================================================================


@command(CONSTS.MEAS.GET_SWEEP)
def measurement_get_sweep(
    client_connection: ClientConnection,
    meas_id: str,
    request_retries: int = DEFAULT_RETRIES,
) -> np.ndarray:
    calls: qscope.server.server.handle_get_sweep_measurement  # just for code navigation
    resp = _send_request(
        client_connection,
        Request(CONSTS.MEAS.GET_SWEEP, {"meas_id": meas_id}),
        request_retries,
    )
    return resp.value


# ============================================================================


@command(CONSTS.MEAS.GET_FRAME, response_type=ArrayResponse | ErrorResponse)
def measurement_get_frame(
    client_connection: ClientConnection,
    meas_id: str,
    frame_type: str = "sig",
    frame_num: int = 2,
    request_retries: int = DEFAULT_RETRIES,
) -> np.ndarray:
    calls: qscope.server.server.handle_get_frame_measurement  # just for code navigation
    resp = _send_request(
        client_connection,
        Request(
            CONSTS.MEAS.GET_FRAME,
            {"meas_id": meas_id, "frame_type": frame_type, "frame_num": frame_num},
        ),
        request_retries,
    )
    return resp.value


# ============================================================================


@command(CONSTS.MEAS.SAVE_SWEEP, response_type=MsgResponse | ErrorResponse)
def measurement_save_sweep(
    client_connection: ClientConnection,
    meas_id: str,
    project_name: str,
    notes: str = "",
    request_retries: int = DEFAULT_RETRIES,
):
    calls: qscope.server.server.handle_save_sweep_measurement

    resp = _send_request(
        client_connection,
        Request(
            CONSTS.MEAS.SAVE_SWEEP,
            {"meas_id": meas_id, "project_name": project_name, "notes": notes},
        ),
        request_retries,
    )
    return resp.value


# ============================================================================


@command(CONSTS.MEAS.SAVE_SWEEP_W_FIT, response_type=MsgResponse | ErrorResponse)
def measurement_save_sweep_w_fit(
    client_connection: ClientConnection,
    meas_id: str,
    project_name: str,
    xdata: np.ndarray,
    ydata: np.ndarray,
    xfit: np.ndarray,
    yfit: np.ndarray,
    fit_results: str,
    comparison_x: np.ndarray = None,
    comparison_y: np.ndarray = None,
    comparison_label: str = None,
    color_map: str = "gray",
    notes: str = "",
    request_retries: int = DEFAULT_RETRIES,
):
    calls: (
        qscope.server.server.handle_save_sweep_w_fit_measurement
    )  # just for code navigation
    resp = _send_request(
        client_connection,
        Request(
            CONSTS.MEAS.SAVE_SWEEP_W_FIT,
            # FIXME change from dict to dataclass
            {
                "meas_id": meas_id,
                "project_name": project_name,
                "xdata": xdata,
                "ydata": ydata,
                "xfit": xfit,
                "yfit": yfit,
                "fit_results": fit_results,
                "comparison_x": comparison_x,
                "comparison_y": comparison_y,
                "comparison_label": comparison_label,
                "color_map": color_map,
                "notes": notes,
            },
        ),
        request_retries,
    )
    return resp.value


# ============================================================================


@command(CONSTS.MEAS.SAVE_FULL_DATA, response_type=MsgResponse | ErrorResponse)
def measurement_save_full_data(
    client_connection: ClientConnection,
    meas_id: str,
    project_name: str,
    notes: str = "",
    request_retries: int = DEFAULT_RETRIES,
):
    """Request server to save full measurement data.

    Server will start an async task to monitor measurement state and save when ready.
    Listen for SaveFullComplete to know when save is complete.
    """
    calls: (
        qscope.server.server.handle_save_full_data_measurement
    )  # just for code navigation
    resp = _send_request(
        client_connection,
        Request(
            CONSTS.MEAS.SAVE_FULL_DATA,
            {"meas_id": meas_id, "project_name": project_name, "notes": notes},
        ),
        request_retries,
    )
    return resp.value


# ============================================================================


@command(CONSTS.MEAS.ADD, response_type=MsgResponse | ErrorResponse)
def add_measurement(
    client_connection: ClientConnection,
    meas_config: MeasurementConfig,
    request_retries: int = DEFAULT_RETRIES,
):
    calls: qscope.server.server.handle_add_measurement  # just for code navigation
    # logger.warning("{{{ meas_config.to_dict(): " + str(meas_config.to_dict()))
    resp = _send_request(
        client_connection,
        Request(
            CONSTS.MEAS.ADD,
            meas_config.to_dict(),
        ),
        request_retries,
    )
    logger.info(
        "Measurement {} ({}) added.", meas_config.__class__.__name__, resp.value
    )
    return resp.value  # return meas_id


# ============================================================================


@command(
    CONSTS.MEAS.START, response_type=MsgResponse | PleaseWaitResponse | ErrorResponse
)
def start_measurement_nowait(
    client_connection: ClientConnection,
    meas_id: str,
    request_retries: int = DEFAULT_RETRIES,
) -> Union[MsgResponse, PleaseWaitResponse]:
    calls: qscope.server.server.handle_start_measurement  # just for code navigation
    resp = _send_request(
        client_connection,
        Request(CONSTS.MEAS.START, {"meas_id": meas_id}),
        request_retries,
    )
    if not isinstance(resp, PleaseWaitResponse):
        return resp


@command(
    CONSTS.MEAS.START, response_type=MsgResponse | PleaseWaitResponse | ErrorResponse
)
def start_measurement_wait(
    client_connection: ClientConnection,
    meas_id: str,
    request_retries: int = DEFAULT_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
):
    calls: qscope.server.server.handle_start_measurement  # just for code navigation
    start = time.time()
    while time.time() - start < timeout:
        resp = start_measurement_nowait(client_connection, meas_id, request_retries)
        if isinstance(resp, PleaseWaitResponse):
            time.sleep(0.1)
        elif isinstance(resp, MsgResponse):
            return resp
    raise TimeoutError("Start measurement timed out.")


# ============================================================================


@command(CONSTS.MEAS.PAUSE, response_type=MsgResponse | ErrorResponse)
def pause_endsweep_measurement(
    client_connection: ClientConnection,
    meas_id: str,
    request_retries: int = DEFAULT_RETRIES,
):
    calls: qscope.server.server.handle_pause_measurement  # just for code navigation
    resp = _send_request(
        client_connection,
        Request(CONSTS.MEAS.PAUSE, {"meas_id": meas_id}),
        request_retries,
    )
    return resp


# ============================================================================


@command(CONSTS.COMMS.IS_STREAMING, response_type=ValueResponse | ErrorResponse)
def is_streaming(
    client_connection: ClientConnection, request_retries: int = DEFAULT_RETRIES
) -> bool:
    calls: qscope.server.server.handle_is_streaming  # just for code navigation
    resp = _send_request(
        client_connection, Request(CONSTS.COMMS.IS_STREAMING), request_retries
    )
    return resp.value


# ============================================================================


@command(
    CONSTS.MEAS.CLOSE, response_type=MsgResponse | PleaseWaitResponse | ErrorResponse
)
def close_measurement_nowait(
    client_connection: ClientConnection,
    meas_id: str,
    request_retries: int = DEFAULT_RETRIES,
) -> Response:
    calls: qscope.server.server.handle_close_measurement  # just for code navigation
    # this may raise a PleaseWaitResponse, see close_measurement_wait
    resp = _send_request(
        client_connection,
        Request(CONSTS.MEAS.CLOSE, {"meas_id": meas_id}),
        request_retries,
    )
    return resp


@command(
    CONSTS.MEAS.CLOSE, response_type=MsgResponse | PleaseWaitResponse | ErrorResponse
)
def close_measurement_wait(
    client_connection: ClientConnection,
    meas_id: str,
    request_retries: int = DEFAULT_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
):
    calls: qscope.server.server.handle_close_measurement  # just for code navigation
    start = time.time()
    while time.time() - start < timeout:
        resp = close_measurement_nowait(client_connection, meas_id, request_retries)
        if isinstance(resp, PleaseWaitResponse):
            time.sleep(0.1)
        elif isinstance(resp, MsgResponse):
            return resp
    raise TimeoutError("Close measurement timed out.")


# ============================================================================


@command(CONSTS.MEAS.GET_FRAME, response_type=ArrayResponse | ErrorResponse)
def measurement_get_frame_shape(
    client_connection: ClientConnection,
    request_retries: int = DEFAULT_RETRIES,
) -> tuple[int, int]:
    calls: qscope.server.server.handle_get_frame_measurement  # just for code navigation
    request = Request(CONSTS.MEAS.GET_FRAME_SHAPE)
    resp = _send_request(client_connection, request, request_retries)
    return resp.value.shape  # NOTE is this correct?


# ============================================================================


@command(CONSTS.MEAS.SET_FRAME_NUM, response_type=MsgResponse | ErrorResponse)
def measurement_set_frame_num(
    client_connection: ClientConnection,
    meas_id: str,
    frame_num: int = 2,
    request_retries: int = DEFAULT_RETRIES,
):
    calls: (
        qscope.server.server.handle_set_frame_num_measurement
    )  # just for code navigation
    resp = _send_request(
        client_connection,
        Request(
            CONSTS.MEAS.SET_FRAME_NUM,
            {"meas_id": meas_id, "frame_num": frame_num},
        ),
        request_retries,
    )
    return resp.value


# ============================================================================


@command(CONSTS.MEAS.SET_ROLLING_AVG_WINDOW, response_type=MsgResponse | ErrorResponse)
def measurement_set_rolling_avg_window(
    client_connection: ClientConnection,
    meas_id: str,
    window: int,
    request_retries: int = DEFAULT_RETRIES,
):
    calls: (
        qscope.server.server.handle_set_rolling_avg_window_measurement
    )  # just for code navigation
    resp = _send_request(
        client_connection,
        Request(
            CONSTS.MEAS.SET_ROLLING_AVG_WINDOW,
            {"meas_id": meas_id, "window": window},
        ),
        request_retries,
    )
    return resp.value


# ============================================================================


@command(CONSTS.MEAS.SET_AOI, response_type=MsgResponse | ErrorResponse)
def measurement_set_aoi(
    client_connection: ClientConnection,
    meas_id: str,
    aoi: tuple[int, int, int, int] | None,
    request_retries: int = DEFAULT_RETRIES,
):
    calls: qscope.server.server.handle_set_aoi_measurement  # just for code navigation
    resp = _send_request(
        client_connection,
        Request(
            CONSTS.MEAS.SET_AOI,
            {"meas_id": meas_id, "aoi": aoi},
        ),
        request_retries,
    )
    return resp.value


# ============================================================================


@command(
    CONSTS.MEAS.SET_ROLLING_AVG_MAX_SWEEPS, response_type=MsgResponse | ErrorResponse
)
def measurement_set_rolling_avg_max_sweeps(
    client_connection: ClientConnection,
    meas_id: str,
    max_sweeps: int,
    request_retries: int = DEFAULT_RETRIES,
):
    calls: (
        qscope.server.server.handle_set_rolling_avg_max_sweeps_measurement
    )  # just for code navigation
    resp = _send_request(
        client_connection,
        Request(
            CONSTS.MEAS.SET_ROLLING_AVG_MAX_SWEEPS,
            {"meas_id": meas_id, "max_sweeps": max_sweeps},
        ),
        request_retries,
    )
    return resp.value


# -------------------------------------------------------------------------------------
# Camera comms
# -------------------------------------------------------------------------------------


@command(CONSTS.CAM.SET_CAMERA_PARAMS, response_type=MsgResponse | ErrorResponse)
def camera_set_params(
    client_connection: ClientConnection,
    exp_t: float,
    image_size: tuple[int, int],
    binning: tuple[int, int],
    request_retries: int = DEFAULT_RETRIES,
):
    calls: qscope.server.server.handle_set_camera_params  # just for code navigation
    request = Request(
        CONSTS.CAM.SET_CAMERA_PARAMS,
        {
            "exp_t": exp_t,
            "image_size": image_size,
            "binning": binning,
        },
    )
    logger.debug(request)
    return _send_request(client_connection, request, request_retries)


# ============================================================================


@command(CONSTS.CAM.TAKE_SNAPSHOT, response_type=ArrayResponse | ErrorResponse)
def camera_take_snapshot(
    client_connection: ClientConnection, request_retries: int = DEFAULT_RETRIES
) -> np.ndarray:
    calls: qscope.server.server.handle_take_snapshot  # just for code navigation
    request = Request(CONSTS.CAM.TAKE_SNAPSHOT)
    resp = _send_request(client_connection, request, request_retries)
    return resp.value


# ============================================================================


@command(CONSTS.CAM.TAKE_AND_SAVE_SNAPSHOT, response_type=MsgResponse | ErrorResponse)
def camera_take_and_save_snapshot(
    client_connection: ClientConnection,
    project_name: str,
    notes: str = "",
    request_retries: int = DEFAULT_RETRIES,
) -> str:
    calls: (
        qscope.server.server.handle_take_and_save_snapshot
    )  # just for code navigation
    """Returns the save dir path."""
    request = Request(
        CONSTS.CAM.TAKE_AND_SAVE_SNAPSHOT,
        {"project_name": project_name, "notes": notes},
    )
    resp = _send_request(client_connection, request, request_retries)
    return resp.value


# ============================================================================


@command(CONSTS.CAM.GET_FRAME_SHAPE, response_type=Shape2DResponse | ErrorResponse)
def camera_get_frame_shape(
    client_connection: ClientConnection,
    request_retries: int = DEFAULT_RETRIES,
) -> tuple[int, int]:
    calls: qscope.server.server.handle_get_frame_shape  # just for code navigation
    request = Request(CONSTS.CAM.GET_FRAME_SHAPE)
    resp = _send_request(client_connection, request, request_retries)
    return resp.value


# ============================================================================


@command(CONSTS.CAM.START_VIDEO, response_type=MsgResponse | ErrorResponse)
def camera_start_video(
    client_connection: ClientConnection,
    request_retries: int = DEFAULT_RETRIES,
):
    calls: qscope.server.server.handle_start_video  # just for code navigation
    request = Request(CONSTS.CAM.START_VIDEO)
    resp = _send_request(client_connection, request, request_retries)
    return resp


# ============================================================================


@command(CONSTS.CAM.STOP_VIDEO, response_type=MsgResponse | ErrorResponse)
def camera_stop_video(
    client_connection: ClientConnection,
    request_retries: int = DEFAULT_RETRIES,
):
    calls: qscope.server.server.handle_stop_video  # just for code navigation
    request = Request(CONSTS.CAM.STOP_VIDEO)
    resp = _send_request(client_connection, request, request_retries)
    return resp


@command(CONSTS.COMMS.GET_DEVICE_LOCKS, response_type=DictResponse | ErrorResponse)
def get_device_locks(
    client_connection: ClientConnection,
    request_retries: int = DEFAULT_RETRIES,
) -> dict:
    """Get the current device locks from the server.

    Parameters
    ----------
    client_connection : ClientConnection
        The connection to the server
    request_retries : int, optional
        Number of retry attempts, by default DEFAULT_RETRIES

    Returns
    -------
    dict
        Dictionary mapping device types to their lock information
    """
    calls: qscope.server.server.handle_get_device_locks
    resp = _send_request(
        client_connection, Request(CONSTS.COMMS.GET_DEVICE_LOCKS), request_retries
    )
    return resp.value


# ============================================================================
# Simple video fn
# ============================================================================


def video(
    client_connection: ClientConnection,
    exp_t: float,
    image_size: tuple[int, int],
    binning: tuple[int, int],
    poll_interval_ms: float = 100,
    request_retries=DEFAULT_RETRIES,
):
    def _add_colorbar(im, fig, ax, aspect=20, pad_fraction=1, locator=None, **kwargs):
        # TODO move to util
        divider = make_axes_locatable(ax)
        width = axes_size.AxesY(ax, aspect=1.0 / aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        cbar = fig.colorbar(im, cax=cax, **kwargs)
        if locator:
            tick_locator = mpl.ticker.FixedLocator(locator)
        else:
            tick_locator = mpl.ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        if tick_locator()[0] < 0:
            bare0 = lambda y, pos: ("%+g" if y > 0 else "%g") % y
            cbar.formatter = mpl.ticker.FuncFormatter(bare0)
        cbar.update_ticks()
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.linewidth = 0.5

        return cbar

    fig, ax = plt.subplots(figsize=(10, 6))

    camera_set_params(client_connection, exp_t, image_size, binning, request_retries)
    camera_start_video(client_connection, request_retries)

    # get frame shape
    frame_shape = camera_get_frame_shape(client_connection, request_retries)

    # Initialize with a blank frame
    img = ax.imshow(np.full(frame_shape, np.nan), cmap="gray", vmin=0, origin="upper")
    cbar = _add_colorbar(
        img,
        fig,
        ax,
    )
    cbar.ax.set_ylabel("PL (a.u.)", rotation=270)

    def update(*args):
        try:
            header, frame = client_connection.stream_socket.recv_multipart(
                flags=zmq.NOBLOCK
            )
            assert header == "video"
            frame = pickle.loads(frame)
            img.set_array(frame)
            vmin, vmax = np.min(frame), np.max(frame)
            img.set_clim(vmin, vmax)
            logger.debug("Frame received")
        except zmq.Again:
            logger.debug("No new frame received, try again")
            pass  # No new frame received, try again

    initialdir = os.getcwd()

    # Define the save function
    def save(event):
        if not hasattr(save, "dir"):
            save.dir = initialdir
        root = Tk()
        root.withdraw()  # Hide the main window
        file_path = filedialog.asksaveasfilename(
            initialdir=save.dir,
            initialfile="PL_image.png",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
        )
        if file_path:
            frame = img.get_array()
            fig.savefig(file_path)
            np.savetxt(file_path.partition(".")[0] + ".txt", frame)
            save.dir = os.path.dirname(file_path)
        root.destroy()

    _ = animation.FuncAnimation(
        fig, update, interval=poll_interval_ms, cache_frame_data=False
    )

    # Add a button widget
    ax_save = plt.axes((0.81, 0.01, 0.1, 0.075))
    btn_save = Button(ax_save, "Save Frame")
    btn_save.on_clicked(save)

    plt.show()

    camera_stop_video(client_connection, request_retries)


def start_bg_notif_listener(client_connection: ClientConnection):
    qu = asyncio.Queue()

    async def listen(queue):
        logger.info("Starting notification listener")
        while True:
            await asyncio.sleep(0.01)
            try:
                msg = client_connection.notif_socket.recv(flags=zmq.NOBLOCK)
                # msg = client_connection.notif_socket.recv()
                notif = Notification.from_msgpack(msg)
                queue.put_nowait(notif)
                # below is rather loquacious
                logger.trace("*NOTIF* (client<-): {}", notif)
                await asyncio.sleep(0.01)
            except zmq.Again:
                pass
            except Exception as e:
                logger.exception("Error in notif listener.")
                break

    task = asyncio.create_task(listen(qu))
    return task, qu


# move to util
def clean_queue(qu: asyncio.Queue):
    while not qu.empty():
        try:
            qu.get_nowait()
        except asyncio.QueueEmpty:
            break


def queue_to_list(qu: asyncio.Queue, nitems: int = 10) -> list:
    lst = []
    i = 0
    while not qu.empty():
        i += 1
        if i > nitems:
            break
        lst.append(qu.get_nowait())
    return lst


async def wait_for_notif(
    qu: asyncio.Queue, notif_type: Type[Notification], timeout=DEFAULT_TIMEOUT
):
    start = time.time()
    while time.time() - start < timeout:
        try:
            notif = qu.get_nowait()
            if isinstance(notif, notif_type):
                return notif
        except asyncio.QueueEmpty:
            pass
        await asyncio.sleep(0.01)
    raise TimeoutError(f"Timeout waiting for {notif_type} notification.")


async def wait_for_notif_with_meas_check(
    qu: asyncio.Queue, 
    notif_type: Type[Notification], 
    meas_id: str,
    timeout=DEFAULT_TIMEOUT
):
    """Wait for a specific notification type while monitoring measurement state.
    
    This function watches for both the requested notification type and any
    measurement state changes that would indicate the measurement has stopped.
    If the measurement is stopped, it returns the latest matching notification
    seen so far.
    
    Parameters
    ----------
    qu : asyncio.Queue
        Queue of notifications
    notif_type : Type[Notification]
        The notification type to wait for
    meas_id : str
        Measurement ID to check for state changes
    timeout : float, optional
        Maximum time to wait, by default DEFAULT_TIMEOUT
        
    Returns
    -------
    Notification
        The notification of the requested type
        
    Raises
    ------
    MeasurementStoppedError
        If the measurement is stopped while waiting, includes latest notification
    TimeoutError
        If timeout is reached before any relevant notification is received
    """
    start = time.time()
    latest_matching_notif = None
    
    # First, check if there are any notifications already in the queue
    # This helps with testing and cases where notifications arrive before we start waiting
    all_notifs = []
    while not qu.empty():
        try:
            notif = qu.get_nowait()
            all_notifs.append(notif)
            
            # Store the latest matching notification for this measurement
            if (isinstance(notif, notif_type) and 
                hasattr(notif, 'meas_id') and 
                notif.meas_id == meas_id):
                latest_matching_notif = notif
            
            # Check if this is a measurement update indicating the measurement was stopped
            if (isinstance(notif, MeasurementUpdate) and 
                notif.meas_id == meas_id and
                notif.new_state in [MEAS_STATE.FINISHED, MEAS_STATE.CLOSE]):
                
                # If we found a stop notification, raise immediately with the latest matching notif
                raise MeasurementStoppedError(
                    f"Measurement {meas_id} was stopped with state {notif.new_state}",
                    latest_matching_notif
                )
        except asyncio.QueueEmpty:
            break
    
    # Put back any notifications that weren't stop notifications or matching notifications
    for notif in all_notifs:
        if (not isinstance(notif, MeasurementUpdate) or 
            notif.meas_id != meas_id or 
            notif.new_state not in [MEAS_STATE.FINISHED, MEAS_STATE.CLOSE]):
            
            if not (isinstance(notif, notif_type) and 
                   hasattr(notif, 'meas_id') and 
                   notif.meas_id == meas_id):
                qu.put_nowait(notif)
    
    # If we found a matching notification, return it
    if latest_matching_notif is not None:
        return latest_matching_notif
    
    # Otherwise, wait for new notifications
    while time.time() - start < timeout:
        try:
            notif = qu.get_nowait()
            
            # Store the latest matching notification for this measurement
            if (isinstance(notif, notif_type) and 
                hasattr(notif, 'meas_id') and 
                notif.meas_id == meas_id):
                latest_matching_notif = notif
            
            # Check if this is a measurement update indicating the measurement was stopped
            if (isinstance(notif, MeasurementUpdate) and 
                notif.meas_id == meas_id and
                notif.new_state in [MEAS_STATE.FINISHED, MEAS_STATE.CLOSE]):
                
                raise MeasurementStoppedError(
                    f"Measurement {meas_id} was stopped with state {notif.new_state}",
                    latest_matching_notif
                )
                
            # Return immediately if we find a matching notification
            if (isinstance(notif, notif_type) and 
                hasattr(notif, 'meas_id') and 
                notif.meas_id == meas_id):
                return notif
                
        except asyncio.QueueEmpty:
            pass
            
        await asyncio.sleep(0.01)
        
    raise TimeoutError(f"Timeout waiting for {notif_type} notification.")
