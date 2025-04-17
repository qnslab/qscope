import time
from typing import Dict, List, Optional

import pyvisa
import serial.tools.list_ports
from loguru import logger


def get_hw_ports():
    port_dict = dict()
    for p in list(serial.tools.list_ports.comports()):
        # Skip virtual serial ports on Unix systems
        # (unsure if we want to do this)
        # if p.device.startswith('/dev/ttyS'):
        #     continue
        # Only include if there's actual hardware info
        if p.hwid != "n/a":
            port_dict[p.device] = tuple(p)[1:]
    return port_dict


def list_visa_devices(
    filter_string: Optional[str] = None,
    model_filter: Optional[str] = None,
    detailed: bool = True,
    reset_devices: bool = False,
    resource_manager: Optional[pyvisa.ResourceManager] = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Dict[str, str]] | Dict[str, str]:
    """List available VISA devices and query their information.

    Args:
        filter_string: Optional string to filter resources (e.g., "USB" or "GPIB")
        model_filter: Optional string to filter devices by model name
        detailed: If True, return detailed status info. If False, just return IDN strings
        reset_devices: If True, attempt to reset each device found
        resource_manager: Optional ResourceManager to use. If None, creates one
        progress_callback: Optional callback function(current, total, message) for progress updates

    Returns:
        If detailed=True:
            Dictionary mapping VISA addresses to device info dictionaries containing:
            - idn: Full identification string
            - status: Connection/query status ('connected', 'error', 'reset_failed')
            - error: Error message if query failed
            - reset_status: Reset result if reset_devices=True
        If detailed=False:
            Dictionary mapping VISA addresses to IDN strings
    """
    owns_rm = False
    if resource_manager is None:
        resource_manager = pyvisa.ResourceManager()
        owns_rm = True

    try:
        devices = {}
        resources = resource_manager.list_resources()
        total_resources = len(resources)

        # Filter out virtual ports once at the start
        resources = [
            r
            for r in resources
            if not any(x in r for x in ["/dev/ttyS", "COM", "ASRL/dev/ttyS"])
        ]

        for idx, resource in enumerate(resources):
            if progress_callback:
                progress_callback(idx, total_resources, f"Scanning {resource}")

            # Apply resource filter if provided
            if filter_string and filter_string not in resource:
                continue

            device_info = (
                {
                    "idn": "",
                    "status": "unknown",
                    "error": "",
                    "reset_status": "not_attempted",
                }
                if detailed
                else None
            )

            inst = None
            try:
                # Try to open and query the device
                inst = resource_manager.open_resource(resource)
                inst.timeout = 2000  # 2 second timeout

                # Set proper termination characters
                inst.read_termination = "\n"
                inst.write_termination = "\n"

                if reset_devices:
                    try:
                        inst.write("*RST")
                        time.sleep(0.1)
                        inst.write("*CLS")
                        time.sleep(0.1)
                        inst.write("*WAI")
                        time.sleep(0.1)
                        if detailed:
                            device_info["reset_status"] = "success"
                        logger.info(f"Reset successful for {resource}")
                    except Exception as e:
                        if detailed:
                            device_info["reset_status"] = "failed"
                        logger.error(f"Reset failed for {resource}: {e}")

                # Try SCPI identification first - it gives the most information
                try:
                    idn = inst.query("*IDN?").strip()
                    if idn.count(",") >= 3:  # Looks like a valid SCPI response
                        logger.debug(f"Got SCPI response from {resource}")
                    else:
                        logger.debug(
                            f"Got non-standard response from {resource}: {idn}"
                        )
                except Exception as e:
                    # Device might not support SCPI, try simple read
                    logger.debug(f"SCPI query failed for {resource}, trying basic read")
                    try:
                        idn = inst.read().strip()
                    except Exception:
                        idn = "Unknown device"

                # Apply model filter if provided
                if model_filter and model_filter not in idn:
                    continue

                if detailed:
                    device_info["idn"] = idn
                    device_info["status"] = "connected"
                    devices[resource] = device_info
                else:
                    devices[resource] = idn

                logger.debug(f"Found device at {resource}: {idn}")

            except Exception as e:
                if detailed:
                    device_info["status"] = "error"
                    device_info["error"] = str(e)
                    devices[resource] = device_info
                logger.debug(f"Error with resource {resource}: {str(e)}")
            finally:
                if inst is not None:
                    try:
                        inst.close()
                    except Exception:
                        pass

        return devices

    finally:
        if owns_rm:
            resource_manager.close()


def check_smu_available(smu_address: Optional[str] = None) -> bool:
    """Check if an SMU is available before attempting measurement.

    Args:
        smu_address: Optional VISA address to check for specific device.
            If None, checks for any available SMU2450.

    Returns:
        bool: True if specified/any SMU is available, False otherwise.
    """
    try:
        devices = list_visa_devices(model_filter="MODEL 2450", detailed=False)
        if smu_address:
            return smu_address in devices
        return len(devices) > 0
    except Exception as e:
        logger.error(f"Error checking for SMU: {e}")
        return False
