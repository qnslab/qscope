"""Utilities for caching device connection information."""

import json
import os
from pathlib import Path
from typing import Optional

from loguru import logger

CACHE_DIR = Path.home() / ".qscope" / "device_cache"


def get_cached_address(device_type: str) -> Optional[str]:
    """Get cached VISA address for device type.

    Args:
        device_type: Type of device (e.g. 'smu2450')

    Returns:
        Cached VISA address if found, None otherwise
    """
    try:
        cache_file = CACHE_DIR / f"{device_type}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
                return data.get("address")
    except Exception as e:
        logger.debug(f"Error reading cache for {device_type}: {e}")
    return None


def update_cached_address(device_type: str, address: str) -> None:
    """Update cached VISA address for device type.

    Args:
        device_type: Type of device (e.g. 'smu2450')
        address: VISA address to cache
    """
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{device_type}.json"
        with open(cache_file, "w") as f:
            json.dump({"address": address}, f)
    except Exception as e:
        logger.debug(f"Error updating cache for {device_type}: {e}")
