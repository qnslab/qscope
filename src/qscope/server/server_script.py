# -*- coding: utf-8 -*-
"""
TODO
"""

import asyncio
import os
import sys
from datetime import datetime

from setproctitle import setproctitle

from qscope.server import start_server

if __name__ == "__main__":
    sys.path.append(os.getcwd())

    system_name: str = sys.argv[1]
    host: str = sys.argv[2]
    msg_port: int = int(sys.argv[3])
    notif_port: int = int(sys.argv[4])
    stream_port: int = int(sys.argv[5])
    log_path: str = sys.argv[6]
    clear_prev_log: bool = sys.argv[7] == "True"
    log_to_file: bool = sys.argv[8] == "True"
    log_to_stdout: bool = sys.argv[9] == "True"
    log_level: str = str(sys.argv[10])

    asyncio.run(
        start_server(
            system_name,
            host,
            msg_port,
            notif_port,
            stream_port,
            log_path=log_path,
            log_to_stdout=log_to_stdout,
            clear_prev_log=clear_prev_log,
            log_to_file=log_to_file,
            log_level=log_level,
        )
    )
