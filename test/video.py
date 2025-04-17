from loguru import logger

import qscope.server
import qscope.util
from qscope.util import TEST_LOGLEVEL

qscope.util.start_client_log(log_level=TEST_LOGLEVEL, log_to_file=True)
proc, client_connection, _ = qscope.server.start_bg(
    "mock",
    log_level=TEST_LOGLEVEL,
)
qscope.server.startup(client_connection)

qscope.server.video(client_connection, 10e-3, (128, 128), (1, 1), 5)

qscope.server.packdown(client_connection)
qscope.server.close_bg(proc, client_connection)
qscope.util.shutdown_client_log()
