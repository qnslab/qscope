# -*- coding: utf-8 -*-
"""
TODO
"""

import os
import pathlib
import sys
import traceback

from loguru import logger

from . import DEFAULT_LOGLEVEL
from .defaults import SINGLE_LINE_ERR_LOG, TEMP_DIR


def format_error_response():
    err_str = traceback.format_exc()
    if SINGLE_LINE_ERR_LOG:
        return "\t".join(line.strip() for line in err_str.splitlines())
    else:
        return err_str


def start_client_log(
    log_to_file=True,
    log_to_stdout=False,
    log_path=None,
    clear_prev=True,
    log_level=DEFAULT_LOGLEVEL,
):
    if log_path is None:
        log_path = log_default_path_client()
    else:
        log_path = os.path.abspath(log_path)

    if clear_prev:
        clear_log(log_path)

    # first remove (default) stderr output
    logger.remove()

    # start a client logger
    if log_to_file:
        logger.add(log_path, level=log_level, enqueue=True, colorize=False)
    if log_to_stdout:
        logger.add(sys.stderr, level=log_level, enqueue=True, colorize=True)
        logger.our_naughty_log_path_attr = log_path
    if log_to_file:
        logger.info("Client log started at {}", log_path)
    else:
        logger.info("Client log started.")


def start_server_log(
    log_to_file=True,
    log_to_stdout=False,
    log_path=None,
    clear_prev=True,
    log_level=DEFAULT_LOGLEVEL,
):
    if log_path is None or log_path == "":
        log_path = log_default_path_server()
    else:
        log_path = os.path.abspath(log_path)

    if clear_prev:
        clear_log(log_path)

    # first remove (default) stderr output
    logger.remove()

    # start a server logger
    if log_to_file:
        # logger.add(log_path, level=log_level)
        logger.add(log_path, level=log_level, enqueue=True, colorize=False)
        logger.our_naughty_log_path_attr = log_path
    if log_to_stdout:
        logger.add(sys.stderr, level=log_level, enqueue=True, colorize=True)
    if log_to_file:
        logger.info("Server log started at {}", log_path)
    else:
        logger.info("Server log started.")


def log_default_path_client() -> str:
    return str(pathlib.Path.home().joinpath(".qscope/client.log"))


def log_default_path_server() -> str:
    return str(pathlib.Path.home().joinpath(".qscope/server.log"))


def log_default_dir():
    return TEMP_DIR


def clear_log(log_path: str):
    """
    Clear the logger file at the given path. Raises FileNotFoundError if the file is
    not found.

    Arguments
    ---------
    log_path : str
        The path to the logger file. Can get logger default dir with
        log_default_dir() and logger default path with log_default_path_client()
        or log_default_path_server().
    """

    # delete file
    if os.path.exists(log_path):
        try:
            os.remove(log_path)
        except PermissionError:
            logger.error(
                f"Could not clear log file {log_path}. Permission denied. Continuing."
            )


def shutdown_client_log():
    try:
        logger.info("Closing down client log.")
        logger.remove()
    except Exception as e:
        logger.exception("Error shutting down client log - skipping.")


def get_log_filename() -> str:
    """Finds the logger filename."""
    if hasattr(logger, "our_naughty_log_path_attr"):
        return logger.our_naughty_log_path_attr
    else:
        return ""
