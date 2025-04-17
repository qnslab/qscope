# -*- coding: utf-8 -*-
# pydoit tast file
# see https://pydoit.org/
# run from this dir with `doit`, or `doit list`, `doit help` etc. (pip install doit 1st)

from doit.action import CmdAction


def _build_pytest_command(
    test_dir,
    keyword="",
    speed="",
    retry=False,
    print_logs=False,
    full_trace=False,
    show_time=False,
):
    """Helper function to build pytest commands for test tasks."""
    cmd = ["pytest"]

    # Add options
    if print_logs:
        cmd.append("--capture=no")
    if full_trace:
        cmd.append("--full-trace")
    if show_time:
        cmd.append("--durations=0")

    # Add common flags
    cmd.extend(["--color=yes", "-vv", "-x"])

    # Add filters
    if retry:
        cmd.append("--lf")
    if keyword:
        cmd.extend(["-k", keyword])
    if speed:
        if speed == "slow":
            cmd.extend(["-m", "slow"])
        elif speed in ["not slow", "fast"]:
            cmd.extend(["-m", '"not slow"'])
        elif speed == "all":
            pass
        elif speed:
            raise ValueError(
                f"Invalid speed filter: {speed}. Use 'slow', 'not slow', 'fast', or 'all'"
            )

    # Add test directory
    cmd.append(test_dir)

    return " ".join(cmd)


def task_make_env():
    """Create a conda environment"""
    return {
        "actions": ["conda create --prefix ./conda_env python=3.11"],
        "targets": ["./conda_env"],
        "uptodate": [True],  # Only run if target doesn't exist
        "verbosity": 2,
    }


# def task_dump_env():
#     """Dump the conda environment to env.txt"""
#     return {"actions": ["conda list --explicit > env.txt"], "verbosity": 2}


def task_install():
    """Install qscope in editable mode"""
    return {
        "actions": ["pip install -e ."],
        "task_dep": ["make_env"],
        "verbosity": 2,
    }


# I think this needs the __init__.py 's to work.
def task_prospector():
    """Run prospector static analysis"""
    return {
        "file_dep": ["qscope.prospector.yaml"],
        "actions": [
            "prospector --profile %(dependencies)s -o grouped:%(targets)s ./src/qscope",
        ],
        "targets": ["qscope.prospector.log"],
        "verbosity": 2,
    }


def task_test_logic():
    """Run the logic test suite (test in test/logic/)."""

    def router(keyword, speed, retry, print_logs, full_trace, show_time, help=False):
        if help:
            return """echo '
Test Logic Runner Help
====================

Filter Options:
  -k, --keyword TEXT    Only run test matching the keyword expression
                        Example: -k "camera and not slow"
  -s, --speed TEXT      Filter test by speed:
                        - "slow": Run only slow test
                        - "not slow" or "fast": Skip slow test
                        - "all": Run all test regardless of speed
  -r, --retry           Only run previously failed test

Output Options:
  -p, --print-logs      Print test logs to console instead of capturing
  -f, --full-trace      Show full traceback on errors
  -t, --show-time       Display duration of all test

Examples:
  doit test_logic                     # Run all test
  doit test_logic -k camera           # Run test containing "camera"
  doit test_logic -s fast -p          # Run fast test with logs
  doit test_logic --retry --show-time # Rerun failed test with timing
  '"""
        try:
            return _build_pytest_command(
                "test/logic/",
                keyword=keyword,
                speed=speed,
                retry=retry,
                print_logs=print_logs,
                full_trace=full_trace,
                show_time=show_time,
            )
        except ValueError as e:
            return f"echo 'Error: {str(e)}' && exit 1"

    return {
        # "actions": [router],
        "actions": [CmdAction(router)],
        "params": [
            {
                "name": "help",
                "long": "help",
                "default": False,
                "type": bool,
            },
            {
                "name": "keyword",
                "short": "k",
                "default": "",
            },
            {
                "name": "speed",
                "short": "s",
                "default": "",
            },
            {
                "name": "retry",
                "short": "r",
                "default": False,
                "type": bool,
            },
            {
                "name": "print_logs",
                "short": "p",
                "default": False,
                "type": bool,
            },
            {
                "name": "full_trace",
                "short": "f",
                "default": False,
                "type": bool,
            },
            {
                "name": "show_time",
                "short": "t",
                "default": False,
                "type": bool,
            },
        ],
        "verbosity": 2,
    }


def task_test_hardware():
    """Run the hardware test suite (test in test/hardware/)."""

    def router(keyword, speed, retry, print_logs, full_trace, show_time, help=False):
        if help:
            return """echo '
Test Hardware Runner Help
=======================

Filter Options:
  -k, --keyword TEXT    Only run test matching the keyword expression
                        Example: -k "camera and not slow"
  -s, --speed TEXT      Filter test by speed:
                        - "slow": Run only slow test
                        - "not slow" or "fast": Skip slow test
                        - "all": Run all test regardless of speed
  -r, --retry           Only run previously failed test

Output Options:
  -p, --print-logs      Print test logs to console instead of capturing
  -f, --full-trace      Show full traceback on errors
  -t, --show-time       Display duration of all test

Examples:
  doit test_hardware                     # Run all test
  doit test_hardware -k camera           # Run test containing "camera"
  doit test_hardware -s fast -p          # Run fast test with logs
  doit test_hardware --retry --show-time # Rerun failed test with timing
  '"""
        try:
            return _build_pytest_command(
                "test/hardware/",
                keyword=keyword,
                speed=speed,
                retry=retry,
                print_logs=print_logs,
                full_trace=full_trace,
                show_time=show_time,
            )
        except ValueError as e:
            return f"echo 'Error: {str(e)}' && exit 1"

    return {
        "actions": [CmdAction(router)],
        # "task_dep": ["install"],
        "params": [
            {
                "name": "help",
                "long": "help",
                "default": False,
                "type": bool,
            },
            {
                "name": "keyword",
                "short": "k",
                "default": "",
            },
            {
                "name": "speed",
                "short": "s",
                "default": "",
            },
            {
                "name": "retry",
                "short": "r",
                "default": False,
                "type": bool,
            },
            {
                "name": "print_logs",
                "short": "p",
                "default": False,
                "type": bool,
            },
            {
                "name": "full_trace",
                "short": "f",
                "default": False,
                "type": bool,
            },
            {
                "name": "show_time",
                "short": "t",
                "default": False,
                "type": bool,
            },
        ],
        "verbosity": 2,
    }


def task_andor_get_cameras():
    """Query andor sdk for available cameras, print their info."""
    return {
        "actions": [
            'python -c "from qscope.util.andor import get_available_andor_cameras as go; go()"'
        ],
        "verbosity": 2,
    }


def task_andor_restart_lib():
    """Attempt to restart the andor lib."""
    return {
        "actions": [
            'python -c "from qscope.util.andor import restart_lib as go; go()"'
        ],
        "verbosity": 2,
    }


def task_kill_servers():
    """Kill orphaned QScope server processes."""

    def router(help=False):
        if help:
            return """echo '
Server Process Killer Help
========================

This task finds and terminates any orphaned QScope server processes
that may be running on the system. Useful for:
- Cleaning up after crashed servers
- Resolving port conflicts
- Ensuring a clean slate before starting new servers

No options required - simply run:
  doit kill_servers'"""
        return "python src/qscope/server/kill_qscope_servers.py"

    return {
        "actions": [CmdAction(router)],
        "params": [
            {
                "name": "help",
                "long": "help",
                "default": False,
                "type": bool,
            },
        ],
        "verbosity": 2,
    }


def task_format():
    """Format code using ruff."""

    def router(help=False):
        if help:
            return """echo '
Code Formatter Help
=================

This task runs the ruff formatter to ensure consistent code style:
- Sorts imports (ruff check --select I --fix)
- Formats code (ruff format)

Formats these locations:
- src/qscope/
- test/
- dodo.py

No options required - simply run:
  doit format
  '"""
        return [
            "ruff check --select I --fix src/qscope",
            "ruff format src/qscope",
            "ruff check --select I --fix test/",
            "ruff format test/",
            "ruff check --select I --fix dodo.py",
            "ruff format dodo.py",
        ]

    return {
        "actions": [CmdAction(router)],
        "params": [
            {
                "name": "help",
                "long": "help",
                "default": False,
                "type": bool,
            },
        ],
        "verbosity": 2,
    }


def task_docs():
    """Generate documentation using pdoc3."""

    def router(help=False):
        if help:
            return """echo '
Documentation Generator Help
==========================

This task runs pdoc3 to generate HTML documentation for the qscope package:
- Outputs to docs/ directory
- Uses templates from docs/ directory (if available)
- Forces regeneration of all files
- Skips errors during generation

No options required - simply run:
  doit docs
  '"""
        return "pdoc3 --output-dir docs/ --html --template-dir docs/ --force --skip-errors ./src/qscope/"

    return {
        "actions": [CmdAction(router)],
        "params": [
            {
                "name": "help",
                "long": "help",
                "default": False,
                "type": bool,
            },
        ],
        "verbosity": 2,
    }
