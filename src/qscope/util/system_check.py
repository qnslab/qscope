from typing import Dict, Optional, Tuple

import serial
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import qscope.util
from qscope.system.sysconfig import list_available_systems, load_system_config


def categorize_error(error: Exception) -> tuple[str, str]:
    """Categorize error type for more user-friendly output.

    Returns
    -------
    tuple[str, str]
        (error_category, detailed_message)
    """
    error_str = str(error)

    # Configuration errors
    if isinstance(error, TypeError) and "unexpected keyword" in error_str:
        param = error_str.split("'")[-2] if "'" in error_str else "unknown"
        return (
            "Configuration Error",
            f"Invalid config parameter '{param}' - {error_str}",
        )

    # Hardware access errors
    if isinstance(error, (FileNotFoundError, serial.SerialException)):
        return "Hardware Access Error", error_str

    # PulseBlaster specific errors
    if isinstance(error, RuntimeError) and error_str.strip() in ["b''", "b", "b''"]:
        return "Hardware Access Error", "No PulseBlaster board found"

    # Serial port errors
    if "could not open port" in error_str.lower() or "serial port" in error_str.lower():
        return "Hardware Access Error", error_str

    # Windows/Linux compatibility errors
    if "windll" in error_str or "Windows driver not available" in error_str:
        return (
            "Platform Compatibility Error",
            error_str,
        )  # "Windows driver not available on Linux"

    # Role configuration errors
    if isinstance(error, ValueError) and "Invalid roles" in error_str:
        roles = (
            error_str.split("[")[-1].strip("]") if "[" in error_str else "unknown roles"
        )
        return "Configuration Error", f"Invalid {roles} for camera system"

    # Implementation errors
    if isinstance(error, (NotImplementedError, AttributeError)):
        if "windll" not in error_str:  # Skip Windows compatibility errors
            return "Implementation Error", error_str

    # Default case - but try to be smart about common patterns
    if "not found" in error_str.lower() or "no such" in error_str.lower():
        return "Hardware Access Error", error_str
    if "permission" in error_str.lower():
        return "Access Error", error_str
    if "configuration" in error_str.lower():
        return "Configuration Error", error_str

    # Additional checks for specific error messages
    if "No PulseBlaster board found" in error_str:
        return "Hardware Access Error", "No PulseBlaster board found"
    if "Serial port" in error_str and "not found" in error_str:
        return "Hardware Access Error", error_str
    if "Windows driver not available" in error_str:
        return (
            "Platform Compatibility Error",
            error_str,
        )  # "Windows driver not available on Linux"

    return "Unknown Error", error_str or "Unknown error occurred"


def check_systems(
    systems: Optional[list[str]] = None,
) -> Dict[str, Tuple[bool, Dict[str, Dict[str, str | bool]], Optional[str]]]:
    """Check availability of system configurations.

    Parameters
    ----------
    systems : Optional[list[str]]
        List of system names to check. If None, checks all systems.

    Returns
    -------
    Dict[str, Tuple[bool, Dict[str, Dict[str, str|bool]], Optional[str]]]
        Dictionary mapping system names to (success, device_status, packdown_error) tuples
    """
    results = {}
    if systems is not None:
        systems = [sys.lower() for sys in systems]

    # Get available systems
    available_systems = list_available_systems()

    # Filter systems to check
    systems_to_check = [
        name for name in available_systems if systems is None or name.lower() in systems
    ]

    for sys_name in systems_to_check:
        logger.info(
            f"\n=================================\nTesting system: {sys_name}\n================================="
        )
        system = None
        try:
            # Create system instance from config
            config = load_system_config(sys_name)
            system = config.system_type(config)

            # Attempt startup
            dev_status = system.startup()
            logger.debug(dev_status)

            # Check if all devices connected successfully
            success = all(status["status"] for status in dev_status.values())

            # Store results
            results[sys_name] = (success, dev_status, None)

        except Exception as e:
            # Get error type and message
            error_type, error_msg = categorize_error(e)

            # Create a device status dictionary that matches the structure from successful cases
            dev_status = {}

            # Try to get device name from the traceback
            if hasattr(e, "device_name"):
                dev_name = e.device_name
            else:
                # Try to extract device name from the traceback
                import traceback

                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    if "open" in frame.name:  # Look for the device's open() method
                        dev_name = frame.filename.split("/")[-1].split(".")[0].title()
                        break
                else:
                    dev_name = "error"  # fallback name if no specific device found

            # Clean up the error message
            clean_msg = clean_error_message(error_msg)

            dev_status[dev_name] = {
                "status": False,
                "message": clean_msg,
                "error_type": error_type,
            }
            results[sys_name] = (False, dev_status, None)

            # Log for debugging
            logger.debug(f"Error type: {error_type}")
            logger.debug(f"Error message: {error_msg}")

        finally:
            if system is not None:
                try:
                    system.packdown()
                except Exception as e:
                    error_type, error_msg = categorize_error(e)
                    # Store packdown error with the results tuple
                    results[sys_name] = (
                        results[sys_name][0],
                        results[sys_name][1],
                        f"{error_type}: {error_msg}",
                    )

    return results


def clean_error_message(error_msg: str) -> str:
    """Clean up error message by removing traceback and extracting key info."""
    # Handle empty or None
    if not error_msg:
        return "Unknown error"

    # Clean up the message first
    error_msg = str(error_msg).strip()

    # If message contains a traceback, extract just the final error
    if "Traceback" in error_msg:
        lines = [line.strip() for line in error_msg.split("\n")]
        error_lines = []
        for line in lines:
            if (
                not line.startswith('File "')
                and not line.startswith("Traceback")
                and not line.startswith("During handling")
                and not line.lstrip().startswith("^")
                and line
            ):  # Skip empty lines
                error_lines.append(line)

        if len(error_lines) >= 2:
            error_msg = ": ".join(error_lines[-2:])  # Take the last two error messages
        elif error_lines:
            error_msg = error_lines[
                -1
            ]  # Take the last error message if only one exists

    # Remove any remaining file paths
    parts = []
    for part in error_msg.split():
        if not (
            part.startswith("/") or part.startswith("line") or part.startswith('"/')
        ):
            parts.append(part)
    error_msg = " ".join(parts)

    # If it's an exception with a message, just keep the message
    if ": " in error_msg:
        error_msg = error_msg.split(": ", 1)[1]

    # Handle special cases
    if "RuntimeError: b''" in error_msg:  # PulseBlaster errors
        return "No PulseBlaster board found"
    if "windll" in error_msg:
        return error_msg  # "Windows driver not available on Linux"
    if isinstance(error_msg, str) and "SerialException" in error_msg:
        # Extract COM port from the error message if present
        import re

        port_match = re.search(r"COM\d+", error_msg)
        port = port_match.group(0) if port_match else "specified port"
        return f"Serial port {port} not found"

    # Final cleanup
    error_msg = error_msg.strip("\"'")

    # Handle empty result after cleanup
    if not error_msg:
        return "Unknown error occurred"

    # Truncate if too long
    MAX_COL = 150
    if len(error_msg) > MAX_COL:
        error_msg = error_msg[: MAX_COL - 3] + "..."

    return error_msg


def print_summary(
    results: Dict[str, Tuple[bool, Dict[str, Dict[str, str | bool]], Optional[str]]],
):
    """Print a formatted summary of system test results with rich formatting."""
    console = Console(color_system="standard")

    ya = "[green]+[/green]"
    na = "[red]-[/red]"
    wa = "[yellow]![/yellow]"

    # Create main table
    table = Table(show_header=False, box=None)
    table.add_column("Status")

    # Process results
    available_systems = []
    partially_available = []
    unavailable_systems = []

    for sys_name, (success, dev_status, packdown_error) in results.items():
        if success:
            available_systems.append((sys_name, dev_status, packdown_error))
        elif any(status["status"] for status in dev_status.values()):
            partially_available.append((sys_name, dev_status, packdown_error))
        else:
            unavailable_systems.append((sys_name, dev_status, packdown_error))

    # Add sections to table
    if available_systems:
        table.add_row("\n[bold]Fully Available Systems:[/bold]")
        for sys_name, dev_status, packdown_error in available_systems:
            table.add_row(f"{ya} {sys_name}")
            # Show device status
            for dev_name, status in dev_status.items():
                table.add_row(f"  {ya} {dev_name}")
            if packdown_error:
                table.add_row(
                    f"  {wa} Packdown error: {clean_error_message(packdown_error)}"
                )

    if partially_available:
        table.add_row("\n[bold]Partially Available Systems:[/bold]")
        for sys_name, dev_status, packdown_error in partially_available:
            table.add_row(f"{wa} {sys_name}")
            for dev_name, status in dev_status.items():
                if status["status"]:
                    table.add_row(f"  {ya} {dev_name}")
                else:
                    table.add_row(f"  {na} {dev_name}")
                    table.add_row(
                        f"     {clean_error_message(status.get('message', 'Unknown error'))}"
                    )
            if packdown_error:
                table.add_row(
                    f"  {wa} Packdown error: {clean_error_message(packdown_error)}"
                )

    if unavailable_systems:
        table.add_row("\n[bold]Unavailable Systems:[/bold]")
        for sys_name, dev_status, packdown_error in unavailable_systems:
            table.add_row(f"{na} {sys_name}")
            # Show status for each device
            for dev_name, status in dev_status.items():
                if status["status"]:
                    table.add_row(f"  {ya} {dev_name}")
                else:
                    table.add_row(f"  {na} {dev_name}")
                    table.add_row(
                        f"     {clean_error_message(status.get('message', 'Unknown error'))}"
                    )
            if packdown_error:
                table.add_row(
                    f"  {wa} Packdown error: {clean_error_message(packdown_error)}"
                )

    # Print everything in a nice panel
    console.print(
        Panel(
            table,
            title="Hardware Availability Summary",
            border_style="blue",
            padding=(1, 2),
        )
    )


def main():
    """Main entry point for system availability checking."""

    qscope.util.start_client_log(log_level=0, log_to_stdout=True, log_to_file=False)

    results = check_systems()
    print_summary(results)

    qscope.util.shutdown_client_log()

    # Return 0 if at least one system is available, 1 otherwise
    return 0 if any(success for success, _ in results.values()) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
