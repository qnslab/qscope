from datetime import datetime
from typing import Optional

import click

from qscope.gui.main_gui import main_gui
from qscope.server.bg_killer import kill_qscope_servers, list_running_servers
from qscope.server.server import start_server
from qscope.util import (
    DEFAULT_HOST_ADDR,
    DEFAULT_LOGLEVEL,
    DEFAULT_PORT,
    format_error_response,
)
from qscope.util.check_hw import get_hw_ports, list_visa_devices


def print_tree(cmd, prefix="", parent_ctx=None):
    """Print command tree starting from given command."""
    ctx = click.Context(cmd, info_name=cmd.name, parent=parent_ctx)

    # Only print root name if no parent
    if not parent_ctx:
        click.echo(cmd.name)

    for sub in sorted(cmd.list_commands(ctx)):
        sub_cmd = cmd.get_command(ctx, sub)
        if isinstance(sub_cmd, click.Group):
            click.echo(f"{prefix}└── {sub}")
            print_tree(sub_cmd, prefix + "    ", ctx)
        else:
            click.echo(f"{prefix}└── {sub}")


def tree_option(f):
    """Add --tree option to command."""

    def callback(ctx, param, value):
        if not value or ctx.resilient_parsing:
            return
        print_tree(ctx.command)
        ctx.exit()

    return click.option(
        "--tree",
        is_flag=True,
        help="Show command tree from this point",
        expose_value=False,
        is_eager=True,
        callback=callback,
    )(f)


@click.group()
@tree_option
def cli():
    """QScope - Quantum Diamond Microscope (QDM) control software.

    A comprehensive control system for quantum diamond microscopes, providing:

    - Server-based hardware control and coordination

    - GUI interface for microscope operation and data visualization

    - Command-line tools for system management, hardware charac etc.
    """
    pass


@cli.command()
@click.option(
    "--system-name",
    "-n",
    help='Name of the system configuration to use (e.g. "mock", "hqdm")',
)
@click.option(
    "--host-address",
    "-ha",
    default=DEFAULT_HOST_ADDR,
    help="Network address to bind server to (default: localhost)",
)
@click.option(
    "--msg-port",
    "-mp",
    default=DEFAULT_PORT,
    type=int,
    help="Port for command/response messages (default: 5555)",
)
@click.option(
    "--notif-port",
    "-np",
    default=lambda: DEFAULT_PORT + 1,
    type=int,
    help="Port for server notifications (default: 5556)",
)
@click.option(
    "--stream-port",
    "-sp",
    default=lambda: DEFAULT_PORT + 2,
    type=int,
    help="Port for video streaming (default: 5557)",
)
@click.option(
    "--log-to-file/--no-log-to-file",
    "-ltf/",
    default=True,
    help="Enable/disable logging to file (default: enabled)",
)
@click.option(
    "--log-to-stdout/--no-log-to-stdout",
    "-lts/",
    default=True,
    help="Enable/disable console logging (default: enabled)",
)
@click.option(
    "--log-path",
    "-lp",
    default="",
    help="Custom path for log file (default: auto-generated)",
)
@click.option(
    "--clear-prev-log/--no-clear-prev-log",
    "-c/",
    default=True,
    help="Clear previous log file on startup (default: enabled)",
)
@click.option(
    "--log-level",
    "-ll",
    default=DEFAULT_LOGLEVEL,
    help="Logging level (DEBUG, INFO, WARNING, ERROR) (default: INFO)",
)
def server(**kwargs):
    """Start the QScope server.

    Launches a server instance that manages hardware devices and provides
    network-accessible control interfaces. The server handles:

    - Device initialization and control

    - Command processing and response

    - Real-time notifications

    - Video streaming
    """
    import asyncio

    # Convert host_address to host for start_server
    kwargs["host"] = kwargs.pop("host_address")
    asyncio.run(start_server(**kwargs))


@cli.command()
@click.option(
    "--system-name",
    "-n",
    help='System configuration to use or connect to (e.g. "mock", "hqdm")',
)
@click.option(
    "--host-address",
    "-ha",
    default="",
    help="Server address to connect to (required with --msg-port)",
)
@click.option(
    "--msg-port",
    "-mp",
    default="",
    help="Server message port to connect to (required with --host-address)",
)
@click.option(
    "--log-to-file/--no-log-to-file",
    "-ltf/",
    default=True,
    help="Enable/disable logging to file (default: enabled)",
)
@click.option(
    "--log-to-stdout/--no-log-to-stdout",
    "-lts/",
    default=True,
    help="Enable/disable console logging (default: enabled)",
)
@click.option(
    "--log-path", "-lp", help="Custom path for log file (default: auto-generated)"
)
@click.option(
    "--clear-prev-log/--no-clear-prev-log",
    "-c/",
    default=True,
    help="Clear previous log file on startup (default: enabled)",
)
@click.option(
    "--log-level",
    "-ll",
    default=DEFAULT_LOGLEVEL,
    help="Logging level (DEBUG, INFO, WARNING, ERROR) (default: INFO)",
)
@click.option(
    "--auto-connect/--no-auto-connect",
    "-a/",
    default=False,
    help="Auto-connect to first available local system (default: disabled)",
)
def gui(**kwargs):
    """Start the QScope GUI.

    Launches the graphical user interface for microscope control. Features include:

    - Live camera view and recording

    - Hardware device control panels

    - Measurement configuration and execution

    - Data visualization tools

    Can connect to a local or remote server instance.
    """
    if bool(kwargs["host_address"]) != bool(kwargs["msg_port"]):
        raise click.UsageError(
            "Must define both --host-address and --msg-port if defining one."
        )

    # Convert host_address to host for main_gui
    kwargs["host"] = kwargs.pop("host_address")
    main_gui(**kwargs)


@cli.command()
def list():
    """List all running QScope servers.

    Displays information about each running server instance:

    - Process ID (PID)

    - Running status

    - Start time

    - Network configuration (host and ports)

    - System configuration
    """
    servers = list_running_servers()

    click.echo("\nRunning qscope servers:")
    click.echo("------------------------")

    if not servers:
        click.echo("No servers found")
        click.echo("")
        return

    for server in servers:
        status = "(RUNNING)" if server.get("running", False) else "(NOT RUNNING)"
        click.echo(f"\nPID: {server['pid']} {status}")
        click.echo(f"Started: {server['timestamp']}")
        click.echo(f"Host: {server['host']}")
        click.echo(
            f"Ports: msg={server['ports']['msg']}, "
            f"notif={server['ports']['notif']}, "
            f"stream={server['ports']['stream']}"
        )
    click.echo("")


@cli.command()
def ports():
    """List all available COM ports.

    Displays information about serial/COM ports:
    - Port name (e.g. COM1, /dev/ttyUSB0)
    - Device description
    - Hardware information
    - Manufacturer details
    """
    ports = get_hw_ports()

    click.echo("\nAvailable COM ports:")
    click.echo("-------------------")

    if not ports:
        click.echo("No COM ports found")
        click.echo("")
        return

    for port, info in ports.items():
        click.echo(f"\nPort: {port}")
        if len(info) >= 2:
            description, hwid = info
            click.echo(f"Description: {description}")
            click.echo(f"Hardware ID: {hwid}")

    click.echo("")


@cli.command()
def kill():
    """Kill all running QScope servers.

    Forcefully terminates all running QScope server processes.
    Useful for cleaning up orphaned processes or resolving port conflicts.
    Will attempt to gracefully shut down each server before force-killing.
    """
    killed = kill_qscope_servers()
    if killed:
        click.echo(f"Killed {killed} qscope server(s)")
    else:
        click.echo("No running qscope servers found")
    click.echo("")


@cli.group()
@tree_option
def system():
    """Manage system configurations."""
    pass


@system.command(name="list")
def list_systems():
    """List available system configurations."""
    from qscope.system.sysconfig import list_available_systems

    systems = list_available_systems()

    click.echo("\nAvailable system configurations:")
    click.echo("-----------------------------")

    if not systems:
        click.echo("No system configurations found")
        click.echo("")
        return

    # Group by source
    package_systems = [name for name, src in systems.items() if src == "package"]
    user_systems = [name for name, src in systems.items() if src == "user"]

    if package_systems:
        click.echo("\nPackage defaults:")
        for system in sorted(package_systems):
            click.echo(f"  - {system}")

    if user_systems:
        click.echo("\nUser configurations:")
        for system in sorted(user_systems):
            click.echo(f"  - {system}")
    click.echo("")


@system.command()
@click.argument("source")
@click.argument("destination")
def copy(source: str, destination: str):
    """Copy a system configuration.

    SOURCE: Name of system configuration to copy from
    DESTINATION: Name for new system configuration
    """
    from qscope.system.sysconfig import copy_system_config

    try:
        copy_system_config(source, destination)
        click.echo(f"Copied system configuration '{source}' to '{destination}'")
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {format_error_response()}", err=True)


@system.command()
@click.argument("name")
def install(name: str):
    """Install a package system config to user directory.

    NAME: Name of system configuration to install
    """
    from qscope.system.sysconfig import install_system_config

    try:
        install_system_config(name)
        click.echo(f"Installed system configuration '{name}' to user directory")
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {format_error_response()}", err=True)


@system.command()
def install_all():
    """Install all package system sysconfig to user directory."""
    from qscope.system.sysconfig import install_all_system_configs

    try:
        installed = install_all_system_configs()
        if installed:
            click.echo(
                f"Installed {len(installed)} system configurations to user directory:"
            )
            for name in installed:
                click.echo(f"  - {name}")
        else:
            click.echo("No system configurations to install")
    except Exception as e:
        click.echo(f"Error: {format_error_response()}", err=True)


@system.command(name="check_hardware")
@click.option(
    "-s",
    "--system",
    help="Check specific system configuration",
    default="",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show detailed diagnostic logs",
    default=False,
)
def check_hardware(system: str, verbose: bool):
    """Check hardware availability for system configurations.

    Tests each system configuration by attempting to initialize and connect
    to required hardware devices. Displays a summary of:

    - Fully available systems (all devices working)
    - Partially available systems (some devices working)
    - Unavailable systems (no devices working)

    For each device, shows connection status and any error messages.
    """
    from loguru import logger

    from qscope.util.system_check import check_systems, print_summary

    if not verbose:
        logger.disable("qscope")

    if system:
        results = check_systems([system.lower()])
    else:
        results = check_systems()

    print_summary(results)


@cli.command()
@click.option(
    "--reset/--no-reset", "-r/", default=False, help="Reset devices during discovery"
)
@click.option(
    "--filter", "-f", help='Filter devices by resource string (e.g. "USB" or "GPIB")'
)
@click.option("--model", "-m", help="Filter devices by model string")
def visa(reset: bool, filter: Optional[str], model: Optional[str]):
    """List all available VISA devices.

    Displays information about connected VISA instruments:
    - Resource address (e.g. USB, GPIB)
    - Device identification string
    - Connection status and errors
    - Reset status (if --reset specified)
    """
    with click.progressbar(length=100, label="Scanning devices") as bar:

        def progress_callback(current, total, msg):
            bar.update(int(100 * current / total))
            if msg:
                click.echo(f"\n{msg}")

        devices = list_visa_devices(
            filter_string=filter,
            model_filter=model,
            reset_devices=reset,
            progress_callback=progress_callback,
        )

    click.echo("\nAvailable VISA devices:")
    click.echo("----------------------")

    if not devices:
        click.echo("No VISA devices found")
        click.echo("")
        return

    for addr, info in devices.items():
        click.echo(f"\nAddress: {addr}")
        click.echo(f"Status: {info['status']}")

        if info["status"] == "connected":
            click.echo(f"Device: {info['idn']}")
        elif info["error"]:
            click.echo(f"Error: {info['error']}")

    click.echo("")
