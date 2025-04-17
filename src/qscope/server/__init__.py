# -*- coding: utf-8 -*-
"""
Server-client communication module for Qscope.

This module provides the infrastructure for client-server communication,
allowing multiple clients to connect to a server that manages hardware
and measurements. It includes functions for starting and stopping servers,
connecting clients, and sending commands.

The server runs in a separate process from clients, which can be scripts
or GUI applications. Communication uses ZeroMQ sockets for reliable
message passing.

Examples
--------
Starting a server and connecting:
```python
from qscope.server import ConnectionManager
manager = ConnectionManager()
manager.start_local_server("mock")
manager.connect()
manager.startup()  # Initialize hardware
```

Sending commands to the server:
```python
manager.camera_start_video()
manager.add_measurement(config)
manager.start_measurement_wait(meas_id)
```

See Also
--------
qscope.server.client : Client-side communication functions
qscope.server.server : Server implementation
qscope.server.connection_manager : Connection management class
"""

from __future__ import annotations

import subprocess
import time
from typing import TYPE_CHECKING, Optional

from loguru import logger

import qscope.types
from qscope.util import (
    DEFAULT_HOST_ADDR,
    DEFAULT_LOGLEVEL,
    DEFAULT_PORT,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    shutdown_client_log,
    start_client_log,
)

from .bg_killer import kill_qscope_servers
from .client import (
    add_measurement,
    camera_get_frame_shape,
    camera_set_params,
    camera_start_video,
    camera_stop_video,
    camera_take_and_save_snapshot,
    camera_take_snapshot,
    clean_queue,
    client_sync,
    close_connection,
    close_measurement_nowait,
    close_measurement_wait,
    echo,
    get_all_meas_info,
    get_server_log_path,
    kill_bg_server,
    measurement_get_frame,
    measurement_get_frame_shape,
    measurement_get_info,
    measurement_get_state,
    measurement_get_sweep,
    measurement_is_stopped,
    measurement_save_full_data,
    measurement_save_sweep,
    measurement_save_sweep_w_fit,
    measurement_set_aoi,
    measurement_set_frame_num,
    measurement_set_rolling_avg_window,
    open_connection,
    packdown,
    pause_endsweep_measurement,
    ping,
    queue_to_list,
    save_latest_stream,
    set_laser_output,
    set_laser_rf_output,
    set_rf_output,
    shutdown_server,
    start_bg_notif_listener,
    start_bg_server,
    start_measurement_nowait,
    start_measurement_wait,
    startup,
    stop_measurement,
    video,
)
from .connection_manager import ConnectionManager
from .server import start_server

if TYPE_CHECKING:
    from qscope.types import ClientConnection, ClientSyncResponse


# TODO docs, I think this starts & connects in one?
# Not so useful with ConnectionManager now? I think remove, change to CM. FIXME
def start_bg(
    system_name: str,
    host=DEFAULT_HOST_ADDR,
    msg_port=DEFAULT_PORT,
    notif_port=DEFAULT_PORT + 1,
    stream_port=DEFAULT_PORT + 2,
    server_log_path: Optional[str] = None,
    clear_prev_log=True,
    timeout=DEFAULT_TIMEOUT,
    request_retries: int = DEFAULT_RETRIES,
    log_level: str = DEFAULT_LOGLEVEL,
) -> tuple[
    subprocess.Popen,
    ClientConnection,
    ClientSyncResponse,
]:
    proc = start_bg_server(
        system_name,
        log_path=server_log_path,
        host=host,
        msg_port=msg_port,
        notif_port=notif_port,
        stream_port=stream_port,
        clear_prev_log=clear_prev_log,
        log_to_file=True,  # NOTE always want this,
        log_to_stdout=False,  # and never want this, for bg server
        log_level=log_level,
    )
    logger.info("Server started on {}:{} @ pid={}", host, msg_port, proc.pid)
    time.sleep(3)  # give server time to start (~3s required!)
    try:
        client_connection, client_sync = open_connection(
            host=host,
            msg_port=msg_port,
            timeout=timeout,
            request_retries=request_retries,
        )
    except qscope.types.CommsError:
        kill_bg_server(proc)  # adds proc errors to client log
        raise
    return proc, client_connection, client_sync


def close_bg(proc: subprocess.Popen, client_connection: ClientConnection):
    shutdown_server(client_connection)
    kill_bg_server(proc)
    close_connection(client_connection)
    shutdown_client_log()


__all__ = [
    "add_measurement",
    "camera_get_frame_shape",
    "camera_set_params",
    "camera_start_video",
    "camera_stop_video",
    "camera_take_snapshot",
    "clean_queue",
    "close_bg",
    "close_connection",
    "close_measurement_nowait",
    "close_measurement_wait",
    "echo",
    "get_all_meas_info",
    "get_server_log_path",
    "kill_bg_server",
    "set_laser_output",
    "set_laser_rf_output",
    "measurement_get_frame",
    "measurement_get_frame_shape",
    "measurement_get_info",
    "measurement_get_state",
    "measurement_get_sweep",
    "measurement_is_stopped",
    "measurement_save_full_data",
    "measurement_save_sweep",
    "measurement_save_sweep_w_fit",
    "measurement_set_aoi",
    "measurement_set_frame_num",
    "measurement_set_rolling_avg_window",
    "open_connection",
    "packdown",
    "pause_endsweep_measurement",
    "ping",
    "queue_to_list",
    "set_rf_output",
    "shutdown_server",
    "start_bg",
    "start_bg_notif_listener",
    "start_measurement_nowait",
    "start_measurement_wait",
    "start_server",
    "startup",
    "stop_measurement",
    "video",
    "start_client_log",
]
