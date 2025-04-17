# -*- coding: utf-8 -*-

import tempfile

DEFAULT_HOST_ADDR = "127.0.0.1"
DEFAULT_PORT = 8850
DEFAULT_RETRIES = 3  # Number of times to retry a failed req operation
DEFAULT_TIMEOUT = 5  # seconds
DEFAULT_LOGLEVEL = "INFO"
TEST_LOGLEVEL = "TRACE"
TEMP_DIR = tempfile.gettempdir()
SINGLE_LINE_ERR_LOG = False  # reformat tracebacks into a single line for err comms

MEAS_SWEEP_TIMEOUT = 120  # seconds (change this to a measurement config param?)
