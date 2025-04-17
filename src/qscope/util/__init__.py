# -*- coding: utf-8 -*-
"""
Utility functions and constants for the Qscope framework.

This module provides various utility functions used throughout the Qscope
framework, including:

- Logging configuration and management
- Data saving and loading
- Parameter sweep list generation
- Hardware port detection
- Data normalization and processing
- Qt event processing helpers

These utilities are designed to be used by other Qscope modules and by
user scripts.

Examples
--------
Generating a parameter sweep:
```python
from qscope.util import gen_linear_sweep_list
freqs = gen_linear_sweep_list(2.7e9, 3.0e9, 101)
```

Saving measurement data:
```python
from qscope.util import save_sweep
save_sweep(x_data, y_data, "esr_measurement")
```

See Also
--------
qscope.util.logging : Logging configuration
qscope.util.save : Data saving functions
qscope.util.list_gen : Parameter sweep generation
"""
# everything here will be exported at top level of qdm

from .check_hw import get_hw_ports
from .decimate import decimate_uneven_data
from .defaults import (
    DEFAULT_HOST_ADDR,
    DEFAULT_LOGLEVEL,
    DEFAULT_PORT,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    SINGLE_LINE_ERR_LOG,
    TEST_LOGLEVEL,
)
from .list_gen import (
    gen_centred_sweep_list,
    gen_exp_centered_list,
    gen_exp_tau_list,
    gen_gauss_sweep_list,
    gen_linear_sweep_list,
    gen_multicentre_sweep_list,
    gen_multigauss_sweep_list,
)
from .logging import (
    clear_log,
    format_error_response,
    get_log_filename,
    log_default_dir,
    log_default_path_client,
    log_default_path_server,
    shutdown_client_log,
    start_client_log,
    start_server_log,
)
from .normalisation import norm
from .qt import (
    process_qt_events,
)

# start_client_log_loguru,
from .save import (
    save_full_data,
    save_latest_stream,
    save_notes,
    save_snapshot,
    save_sweep,
    save_sweep_w_fit,
)

__all__ = [
    "DEFAULT_HOST_ADDR",
    "DEFAULT_LOGLEVEL",
    "DEFAULT_PORT",
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT",
    "SINGLE_LINE_ERR_LOG",
    "TEST_LOGLEVEL",
    "clear_log",
    "format_error_response",
    "get_hw_ports",
    "get_log_filename",
    "log_default_dir",
    "log_default_path_client",
    "log_default_path_server",
    "save_full_data",
    "save_snapshot",
    "save_sweep",
    "save_sweep_w_fit",
    "save_latest_stream",
    "shutdown_client_log",
    "start_client_log",
    "start_server_log",
    "gen_linear_sweep_list",
    "gen_centred_sweep_list",
    "gen_gauss_sweep_list",
    "gen_multigauss_sweep_list",
    "gen_multicentre_sweep_list",
    "gen_exp_tau_list",
    "gen_exp_centered_list",
    "norm",
    "decimate_uneven_data",
]
