"""Shared utilities for MPL measurements."""

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np
from loguru import logger

from qscope.device import SMU2450, Picoscope5000a
from qscope.util import DEFAULT_LOGLEVEL

# SMU limits
SMU_MAX_VOLTAGE = 20.0  # Maximum voltage output in volts
SMU_MAX_CURRENT = 1.05  # Maximum current output in amps (1.05A for DC)


def setup_logging(
    log_path: str = "",
    clear_prev_log: bool = True,
    log_to_file: bool = True,
    log_to_stdout: bool = True,
    log_level: str = DEFAULT_LOGLEVEL,
) -> None:
    """Configure logging based on parameters.

    Parameters
    ----------
    log_path : str, optional
        Path to log file. If empty, uses default path.
    clear_prev_log : bool, optional
        Whether to clear previous log file, by default True
    log_to_file : bool, optional
        Enable logging to file, by default True
    log_to_stdout : bool, optional
        Enable console logging, by default True
    log_level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR), by default DEFAULT_LOGLEVEL
    """
    from qscope.util.logging import clear_log, log_default_path_client, start_client_log

    log_path = log_path or log_default_path_client()
    if clear_prev_log:
        clear_log(log_path)

    start_client_log(
        log_to_file=log_to_file,
        log_to_stdout=log_to_stdout,
        log_path=log_path,
        log_level=log_level,
    )


def configure_scope_acquisition(
    scope: Picoscope5000a,
    pl_range: float,
    pl_coupling: str,
    use_scope_current: bool,
    current: float = 0.0,
    coil_resistance: float = 27.0,
    downsample_ratio: int = 1,
    downsample_mode: str = "AVERAGE",
) -> None:
    """Configure scope channels and acquisition parameters.

    Parameters
    ----------
    scope : Picoscope5000a
        Configured picoscope instance
    pl_range : float
        Voltage range for PL channel
    pl_coupling : str
        Coupling mode for PL channel ("AC" or "DC")
    use_scope_current : bool
        Whether to enable current measurement channel
    current : float, optional
        Peak current for range calculation, by default 0.0
    coil_resistance : float, optional
        Coil resistance for voltage calculation, by default 27.0
    downsample_ratio : int, optional
        Hardware downsampling ratio, by default 1
    downsample_mode : str, optional
        Downsampling mode, by default "AVERAGE"
    """
    channels = [0]  # PL channel always enabled
    ranges = [pl_range]
    couplings = [pl_coupling]

    if use_scope_current:
        voltage_range = current * coil_resistance * 1.2
        channels.append(1)
        ranges.append(voltage_range)
        couplings.append("DC")

    # Validate all ranges at once
    valid_ranges = {
        ch: min((r for r in scope.VOLTAGE_RANGES if r > rng), default=None)
        for ch, rng in zip(channels, ranges)
    }

    if not all(valid_ranges.values()):
        invalid = [ch for ch, r in valid_ranges.items() if not r]
        raise ValueError(f"Invalid voltage range for channels {invalid}")

    scope.configure_channels(
        channels=channels,
        ranges=[valid_ranges[ch] for ch in channels],
        coupling=couplings,
    )

    # Configure hardware downsampling if enabled
    if downsample_ratio > 1:
        scope.set_downsampling(mode=downsample_mode, downsample_ratio=downsample_ratio)
        logger.info(f"Using hardware downsampling with ratio {downsample_ratio}")


@contextmanager
def managed_devices(
    smu_address: str,
    current: float,
    coil_resistance: float,
    additional_resistance: float = 0.0,
    voltage_limit: Optional[float] = None,
    source_mode: str = "voltage",
    force_sourcing: bool = False,
    nplc: float = 0.1,
    auto_zero: bool = True,
    source_delay: float = 0.0,
    scope_resolution: int = 12,
) -> Generator[tuple[SMU2450, Picoscope5000a], Any, None]:
    """Context manager for device setup and cleanup.

    Parameters
    ----------
    smu_address : str
        VISA address for the SMU
    current : float
        Peak current in amps
    coil_resistance : float
        Electromagnet coil resistance in ohms
    additional_resistance : float, optional
        Additional resistance in series with coil in ohms, by default 0.0
    voltage_limit : float, optional
        Maximum voltage to apply, by default None (calculated from current and resistance)
    source_mode : str, optional
        SMU source mode ("voltage" or "current"), by default "voltage"
    force_sourcing : bool, optional
        Force operation even when approaching SMU limits, by default False
    nplc : float, optional
        Number of power line cycles for SMU, by default 0.1
    auto_zero : bool, optional
        Enable SMU auto-zero, by default True
    source_delay : float, optional
        SMU source delay in seconds, by default 0.0
    scope_resolution : int, optional
        Scope resolution in bits, by default 12
    """
    smu, scope = setup_devices(
        smu_address=smu_address,
        current=current,
        coil_resistance=coil_resistance,
        additional_resistance=additional_resistance,
        voltage_limit=voltage_limit,
        source_mode=source_mode,
        force_sourcing=force_sourcing,
        nplc=nplc,
        auto_zero=auto_zero,
        source_delay=source_delay,
        scope_resolution=scope_resolution,
    )
    try:
        yield smu, scope
    finally:
        if smu is not None:
            try:
                smu.zero_output()
                smu.close()
            except Exception:
                logger.exception("Error closing SMU")

        if scope is not None:
            try:
                scope.close()
            except Exception:
                logger.exception("Error closing scope")


def setup_devices(
    smu_address: str,
    current: float,
    coil_resistance: float,
    additional_resistance: float = 0.0,
    voltage_limit: Optional[float] = None,
    source_mode: str = "voltage",
    force_sourcing: bool = False,
    nplc: float = 0.1,
    auto_zero: bool = True,
    source_delay: float = 0.0,
    scope_resolution: int = 12,
) -> Tuple[SMU2450, Picoscope5000a]:
    """Initialize and configure measurement devices.

    Parameters
    ----------
    smu_address : str
        VISA address for the SMU
    current : float
        Peak current in amps
    coil_resistance : float
        Electromagnet coil resistance in ohms
    additional_resistance : float, optional
        Additional resistance in series with coil in ohms, by default 0.0
    voltage_limit : float, optional
        Maximum voltage to apply, by default None (calculated from current and resistance)
    source_mode : str, optional
        SMU source mode ("voltage" or "current"), by default "voltage"
    force_sourcing : bool, optional
        Force operation even when approaching SMU limits, by default False
    nplc : float, optional
        Number of power line cycles for SMU, by default 0.1
    auto_zero : bool, optional
        Enable SMU auto-zero, by default True
    source_delay : float, optional
        SMU source delay in seconds, by default 0.0
    scope_resolution : int, optional
        Scope resolution in bits, by default 12
    """
    # Initialize devices
    smu = SMU2450(smu_address)
    scope = Picoscope5000a()

    # Calculate total resistance and required voltage
    total_resistance = coil_resistance + additional_resistance
    required_voltage = abs(current) * total_resistance

    # Safety checks before configuring SMU
    if current > SMU_MAX_CURRENT:
        if not force_sourcing:
            raise ValueError(
                f"Requested current ({current}A) exceeds SMU maximum ({SMU_MAX_CURRENT}A)"
            )
        logger.warning(f"Current limited to {SMU_MAX_CURRENT}A")
        current = SMU_MAX_CURRENT
        # Recalculate required voltage with limited current
        required_voltage = abs(current) * total_resistance

    # Check if we'll exceed voltage limits in voltage mode
    if source_mode == "voltage" and required_voltage > SMU_MAX_VOLTAGE:
        if not force_sourcing:
            raise ValueError(
                f"Required voltage ({required_voltage:.2f}V) exceeds SMU maximum ({SMU_MAX_VOLTAGE}V). "
                f"Consider reducing current, adding resistance, or using current sourcing mode."
            )
        # If forcing, limit to max voltage and warn about reduced current
        limited_current = SMU_MAX_VOLTAGE / total_resistance
        logger.warning(
            f"Voltage limited to {SMU_MAX_VOLTAGE}V. "
            f"Maximum achievable current will be {limited_current:.3f}A instead of {current:.3f}A"
        )
        # Adjust the current for later calculations
        current = limited_current
        required_voltage = SMU_MAX_VOLTAGE

    # Use provided voltage limit or calculate with headroom
    if voltage_limit is None:
        voltage_headroom = 1.5
        voltage_limit = min(required_voltage * voltage_headroom, SMU_MAX_VOLTAGE)
    elif voltage_limit > SMU_MAX_VOLTAGE:
        logger.warning(
            f"Voltage limit ({voltage_limit}V) exceeds SMU maximum ({SMU_MAX_VOLTAGE}V)"
        )
        voltage_limit = SMU_MAX_VOLTAGE

    # Configure SMU
    smu.open()
    smu.configure(nplc=nplc, auto_zero=auto_zero, source_delay=source_delay)

    # Provide informative message about expected operation
    logger.info(
        f"Electromagnet configuration: {coil_resistance}Ω coil + {additional_resistance}Ω additional"
    )
    logger.info(
        f"Target current: {current}A, Expected voltage: {required_voltage:.2f}V"
    )

    if source_mode == "current":
        # Current sourcing mode
        smu.set_mode("current")
        smu._communicate(f":SOUR:CURR:RANG {current * 1.2}")
        smu._communicate(f":SENS:VOLT:RANG {voltage_limit}")
        smu.set_compliance(voltage_limit)
        logger.info(
            f"Configured SMU in current mode: {current}A, compliance: {voltage_limit}V"
        )
    else:
        # Voltage sourcing mode
        smu.set_mode("voltage")
        smu._communicate(f":SOUR:VOLT:RANG {required_voltage * 1.2}")

        # For high currents near the maximum, use less headroom
        if current > 0.8:  # If we're near the max current
            current_limit = min(1.05, current * 1.05)  # Use only 5% headroom
            logger.info(
                f"High current detected ({current:.3f}A), using reduced headroom"
            )
        else:
            current_limit = current * 1.2  # Normal 20% headroom

        # First disable auto-ranging for current measurement
        smu._communicate(":SENS:CURR:RANG:AUTO OFF")

        # For high currents, set to max range directly
        if current > 0.8:
            smu._communicate(":SENS:CURR:RANG 1.05")
            logger.info("Setting current measurement range to maximum (1.05A)")
        else:
            smu._communicate(f":SENS:CURR:RANG {current_limit}")

        # Then set current limit
        smu.set_compliance(current_limit)
        logger.info(
            f"Configured SMU in voltage mode: {required_voltage}V, compliance: {current_limit}A"
        )

    smu.set_output_state(True)

    # Configure scope
    scope.set_resolution(scope_resolution)
    scope.open()

    return smu, scope


def analyze_mpl_response(pl: np.ndarray, current: np.ndarray) -> Dict[str, float]:
    """Analyze magnetophotoluminescence (MPL) response characteristics."""
    mask = current > 0.5 * np.max(current)
    pl_high, pl_low = np.mean(pl[mask]), np.mean(pl[~mask])
    contrast = (pl_low - pl_high) / pl_low if pl_low else 0

    return {
        "contrast": 0 if np.isnan(contrast) else contrast,
        "pl_high": 0 if np.isnan(pl_high) else pl_high,
        "pl_low": 0 if np.isnan(pl_low) else pl_low,
    }
