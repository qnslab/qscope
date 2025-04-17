"""
Connection manager for client-side server connections.

This class encapsulates all connection handling and server communication,
providing a clean interface for the rest of the application to use.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from functools import wraps
from typing import Any, Callable, Optional, Type, TypeVar

import zmq
from loguru import logger

import qscope.server.client as client

F = TypeVar("F", bound=Callable)


def qt_process_events_if_available(func: F) -> F:
    """Decorator that processes Qt events and shows wait cursor during function execution, if Qt is available.

    This decorator will:
    1. Process any pending Qt events before execution
    2. Set the cursor to WaitCursor during execution
    3. Process any pending Qt events after execution
    4. Restore the original cursor

    Only takes effect if running within a Qt application context.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if we're in a Qt application context
        qt_app = None
        try:
            from PyQt6.QtCore import Qt
            from PyQt6.QtWidgets import QApplication

            qt_app = QApplication.instance()
            if qt_app:
                process_qt_events()
                # Force cursor change for all windows
                while qt_app.overrideCursor() is not None:
                    qt_app.restoreOverrideCursor()
                qt_app.setOverrideCursor(Qt.CursorShape.WaitCursor)
                process_qt_events()  # Ensure cursor change is processed
        except ImportError:
            pass  # Not in Qt environment

        try:
            result = func(*args, **kwargs)
        finally:
            # Always restore cursor and process events, even if exception occurs
            if qt_app:
                # Restore cursor state
                while qt_app.overrideCursor() is not None:
                    qt_app.restoreOverrideCursor()
                process_qt_events()  # Ensure cursor restoration is processed

        return result

    return wrapper


from qscope.types import (
    ClientConnection,
    ClientSyncResponse,
    MeasurementConfig,
    Notification,
)
from qscope.util import (
    DEFAULT_HOST_ADDR,
    DEFAULT_LOGLEVEL,
    DEFAULT_PORT,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
)
from qscope.util.qt import process_qt_events


class ConnectionManager:
    """
    Manages client-side connection to the server.

    This class handles connection lifecycle and state management, while delegating
    protocol operations to the client module functions. It provides a clean OO
    interface by automatically wrapping client functions as methods.

    The separation between connection management and protocol implementation
    allows each layer to evolve independently while maintaining a simple
    interface for users.
    """

    def __init__(self):
        """Initialize the connection manager."""
        self._connection: Optional[ClientConnection] = None
        self._client_sync: Optional[ClientSyncResponse] = None
        self._server_proc: Optional[subprocess.Popen] = None
        self._notif_task: Optional[asyncio.Task] = None
        self._notif_queue: Optional[asyncio.Queue] = None
        self._prev_connection_params: Optional[tuple[str, int, float, int]] = None
        self._prev_connection_params: Optional[tuple[str, int, float, int]] = None

    @property
    def connection(self) -> Optional[ClientConnection]:
        """Get the current connection."""
        return self._connection

    @property
    def client_sync(self) -> Optional[ClientSyncResponse]:
        """Get the last client sync response."""
        return self._client_sync

    def is_connected(self) -> bool:
        """Check if currently connected to server."""
        return self._connection is not None

    @qt_process_events_if_available
    def start_local_server(
        self,
        system_name: str,
        host: str = DEFAULT_HOST_ADDR,
        msg_port: int = DEFAULT_PORT,
        notif_port: int = DEFAULT_PORT + 1,
        stream_port: int = DEFAULT_PORT + 2,
        log_path: Optional[str] = None,
        clear_prev_log: bool = True,
        log_to_file: bool = True,
        log_to_stdout: bool = False,
        log_level: str = DEFAULT_LOGLEVEL,
    ) -> None:
        """Start a server process."""
        if self._server_proc:
            logger.warning("Server already running, stopping first")
            self.stop_server()

        self._server_proc = client.start_bg_server(
            system_name,
            host,
            msg_port,
            notif_port,
            stream_port,
            log_path,
            clear_prev_log,
            log_to_file,
            log_to_stdout,
            log_level,
        )
        logger.info(f"Server process started: {self._server_proc}")
        time.sleep(0.2)

    @qt_process_events_if_available
    def stop_server(self) -> None:
        """Stop the server and handle connection cleanup.

        This method sends a shutdown command to the server if connected,
        and handles the expected disconnection gracefully.
        """
        if self._server_proc:
            self.stop_local_server()
        else:
            if not self._connection and self._prev_connection_params:
                try:
                    self._connection = client.open_connection(
                        *self._prev_connection_params
                    )[0]
                except Exception as e:
                    logger.warning(
                        "Could not reconnect to remote running server: %s", str(e)
                    )
                    # Continue with termination even if reconnect fails

            if self._connection:
                try:
                    # Send shutdown command and expect disconnection
                    client.shutdown_server(self._connection)
                    # Connection is now invalid, so clear it
                    self._connection = None
                    self._client_sync = None

                    if self._notif_task:
                        self._notif_task.cancel()
                        self._notif_task = None

                    logger.info("Server shutdown completed")
                except Exception as e:
                    logger.warning(f"Error during server shutdown: {e}")
                    # Still disconnect
                    self.disconnect()
            else:
                logger.warning("No connection, can't stop remote server.")

    @qt_process_events_if_available
    def stop_local_server(self) -> None:
        """Stop the server process if running."""
        if self._server_proc:
            if not self._connection:
                logger.warning(
                    "No connection, killing server process without shutting down"
                )
            else:
                logger.info("Shutting down local server.")
                client.shutdown_server(self._connection)
            client.kill_bg_server(self._server_proc)
            self._server_proc = None
            self.disconnect()

    @qt_process_events_if_available
    def connect(
        self,
        host: str = DEFAULT_HOST_ADDR,
        msg_port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
        request_retries: int = DEFAULT_RETRIES,
    ) -> None:
        """Connect to a running server."""
        if self._connection:
            logger.warning("Already connected, disconnecting first")
            self.disconnect()

        self._prev_connection_params = (host, msg_port, timeout, request_retries)

        try:
            self._connection, self._client_sync = client.open_connection(
                host, msg_port, timeout, request_retries
            )
        except Exception as e:
            # ensure these aren't set if connection fails
            self._connection = None
            self._client_sync = None
            raise e
        self._prev_connection_params = (host, msg_port, timeout, request_retries)
        # self.start_notification_listener() # don't always want, e.g. for gui
        logger.info("Connected to server: {}", self._client_sync)

    @qt_process_events_if_available
    def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._connection:
            if self._notif_task:
                self._notif_task.cancel()
                self._notif_task = None
            client.close_connection(self._connection)
            self._connection = None
            self._client_sync = None

    def start_notification_listener(self) -> None:
        """Start the notification listener task."""
        if not self._connection:
            raise RuntimeError("Not connected to server")
        self._notif_task, self._notif_queue = client.start_bg_notif_listener(
            self._connection
        )

    T = TypeVar("T", bound=Notification)

    async def wait_for_notification(
        self, notif_type: Type[T], timeout: float = DEFAULT_TIMEOUT
    ) -> T:
        """Wait for a specific type of notification."""
        if not self._notif_queue:
            raise RuntimeError("Notification listener not started")
        return await client.wait_for_notif(self._notif_queue, notif_type, timeout)
        
    async def wait_for_notification_with_meas_check(
        self, notif_type: Type[T], meas_id: str, timeout: float = DEFAULT_TIMEOUT
    ) -> T:
        """Wait for a specific type of notification while checking measurement state.
        
        Parameters
        ----------
        notif_type : Type[T]
            The notification type to wait for
        meas_id : str
            Measurement ID to check for state changes
        timeout : float, optional
            Maximum time to wait, by default DEFAULT_TIMEOUT
        
        Returns
        -------
        T
            The notification of the requested type
            
        Raises
        ------
        RuntimeError
            If notification listener is not started
        MeasurementStoppedError
            If the measurement is stopped while waiting, includes latest notification
        TimeoutError
            If timeout is reached before any relevant notification is received
        """
        if not self._notif_queue:
            raise RuntimeError("Notification listener not started")
        return await client.wait_for_notif_with_meas_check(
            self._notif_queue, notif_type, meas_id, timeout
        )

    def clean_notification_queue(self) -> None:
        """Clear all pending notifications."""
        if self._notif_queue:
            client.clean_queue(self._notif_queue)

    def get_notifications(self, max_items: int = 10) -> list[Notification]:
        """Get pending notifications up to max_items."""
        if not self._notif_queue:
            return []
        return client.queue_to_list(self._notif_queue, max_items)

    def get_notification_socket(self) -> zmq.Socket:
        """Get the notification socket for direct access."""
        if not self._connection:
            raise RuntimeError("Not connected to server")
        return self._connection.notif_socket

    def get_stream_socket(self) -> zmq.Socket:
        """Get the stream socket for direct access."""
        if not self._connection:
            raise RuntimeError("Not connected to server")

        # Configure socket buffer limit if not already done
        if not hasattr(self, "_configured_buffer"):
            self._connection.stream_socket.setsockopt(
                zmq.RCVHWM, 20
            )  # Limit receive buffer to 20 frames
            self._configured_buffer = True
            logger.debug("Configured stream socket with RCVHWM=20")

        return self._connection.stream_socket

    def get_device_locks(self) -> dict[str, tuple[str, str, float]]:
        """Get current device locks.

        Returns:
            Dictionary mapping device class names to tuples of:
            (owner, description, timestamp)
        """
        if not self._connection:
            return {}
        try:
            locks = client.get_device_locks(self._connection)
            return {
                dev.__name__: (lock.owner, lock.description, lock.timestamp)
                for dev, lock in locks.items()
            }
        except Exception:
            return {}

    def is_device_locked(self, device_type: Type) -> bool:
        """Check if a specific device type is locked.

        Args:
            device_type: The device class to check

        Returns:
            True if the device is currently locked
        """
        locks = self.get_device_locks()
        return device_type.__name__ in locks

    def __del__(self):
        """Cleanup on deletion."""
        self.disconnect()
        self.stop_server()

    # ========================================================================
    # Access client.py function 'through' the connection manager w automatic
    # check if connection is open.
    # ========================================================================

    def __getattr__(self, name: str) -> Any:
        """
        Delegate unknown attributes to client protocol functions.

        This provides a clean object-oriented interface to the client protocol functions
        by automatically injecting the connection object.

        Examples:
            # ConnectionManager automatically delegates to client protocol functions:

            manager = ConnectionManager()
            manager.connect()

            # Start/stop server
            manager.start_local_server("MOCK")
            manager.stop_server()

            # Direct protocol access
            manager.ping()  # Calls client.ping(connection)
            manager.echo("test")  # Calls client.echo(connection, "test")

            # Hardware control
            manager.set_laser_output(True)  # Calls client.set_laser_output(connection, True)
            manager.set_rf_params(2870, -20)  # Calls client.set_rf_params(connection, 2870, -20)

            # Camera control
            manager.camera_set_params(0.1, (512, 512), (1, 1))
            manager.camera_start_video()
            manager.camera_stop_video()

            # Measurement control
            meas_id = manager.add_measurement(config)
            manager.start_measurement_wait(meas_id)
            manager.stop_measurement(meas_id)
            manager.close_measurement_wait(meas_id)

        Args:
            name: The attribute name to look up

        Returns:
            A wrapper function that injects the connection object

        Raises:
            RuntimeError: If not connected to server
            AttributeError: If no matching client function exists
        """
        if hasattr(client, name):
            # Get the original function
            func = getattr(client, name)

            # Create a wrapper that injects the connection
            @qt_process_events_if_available
            def wrapper(*args, **kwargs):
                if not self._connection:
                    raise RuntimeError("Not connected to server")
                return func(self._connection, *args, **kwargs)

            return wrapper
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
