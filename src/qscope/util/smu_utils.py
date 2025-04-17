"""Utilities for SMU device operations.

This module provides common utility functions for working with Source Measure Unit
(SMU) devices across different scripts and modules.
"""

from typing import Optional

from loguru import logger

from qscope.util.check_hw import list_visa_devices


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
