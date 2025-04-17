"""Class for controlling the Keithley SMU2450 SourceMeter.

Uses VISA communication and SCPI commands to control the Keithley 2450.
Implements voltage/current source and measurement capabilities with
safety features like ramping and compliance limits.
"""

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import pyvisa
from loguru import logger

from qscope.device import Device
from qscope.util.check_hw import list_visa_devices
from qscope.util.device_cache import get_cached_address, update_cached_address

# Error handling constants
ERROR_MESSAGES = {
    "not_connected": "Device not connected",
    "invalid_voltage": "Voltage {voltage}V outside instrument range (-20V to +20V)",
    "invalid_current": "Current {current}A outside instrument range (-1.05A to +1.05A)",
    "invalid_mode": 'Mode must be either "voltage" or "current"',
    "invalid_nplc": "NPLC must be between 0.01 and 10",
    "invalid_source_delay": "Source delay cannot be negative",
    "compliance_reached": "Compliance limit reached during {operation}",
    "verification_failed": "Failed to verify {param} setting",
    "unexpected_device": "Unexpected device ID: {idn}",
    "connection_failed": "Failed to connect to instrument: {error}",
}

# Default settings
DEFAULT_NPLC = 1.0  # Integration time in power line cycles
DEFAULT_AUTO_ZERO = True  # Enable/disable auto-zero
DEFAULT_SOURCE_DELAY = 0.0  # Delay after setting source value


class TriggerEdge(Enum):
    """Trigger edge direction."""

    RISING = auto()
    FALLING = auto()
    EITHER = auto()


@dataclass
class TriggerConfig:
    """Configuration for SMU triggering and digital I/O.

    Attributes
    ----------
    source : Literal["external", "timer", "manual"]
        Trigger source selection
    edge : TriggerEdge
        Trigger edge direction for external trigger
    pin : int
        Digital I/O pin number (1-6) for external trigger
    delay : float
        Delay after trigger before action (seconds)
    timeout : float
        Time to wait for trigger before timeout (seconds)
    high_value : float
        Value to output when triggered (current/voltage depending on mode)
    low_value : float
        Value to output when not triggered
    count : int
        Number of trigger cycles to execute (0 for infinite)
    resistance : float
        Resistance value for voltage-to-current calculations (ohms)
    """

    source: Literal["external", "timer", "manual"] = "external"
    edge: TriggerEdge = TriggerEdge.RISING
    pin: int = 3
    delay: float = 0.0
    timeout: float = 1.0
    high_value: float = 0.0
    low_value: float = 0.0
    count: int = 1
    resistance: float = 20.0  # Default resistance for calculations

    def __post_init__(self):
        """Validate configuration immediately after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate trigger configuration."""
        validators = {
            "pin": (1 <= self.pin <= 6, "Digital I/O pin must be between 1 and 6"),
            "delay": (self.delay >= 0, "Trigger delay cannot be negative"),
            "timeout": (self.timeout > 0, "Trigger timeout must be positive"),
            "count": (self.count >= 0, "Trigger count cannot be negative"),
        }

        # Check all validation rules
        for param, (valid, message) in validators.items():
            if not valid:
                raise ValueError(f"{message} (got {getattr(self, param)})")


class SMU2450(Device):
    """Keithley 2450 SourceMeter implementation.

    Provides high-precision voltage/current sourcing and measurement capabilities.
    Supports voltage/current sourcing with optional ramping, compliance limits,
    and various measurement configurations.

    Parameters
    ----------
    visa_address : str, optional
        VISA resource address of the instrument.
        If None, will attempt to find first available 2450.
    resource_manager : pyvisa.ResourceManager, optional
        PyVISA ResourceManager instance.
        If None, will create one with Python backend.

    Attributes
    ----------
    visa_address : str
        VISA address of the connected device
    rm : pyvisa.ResourceManager
        PyVISA resource manager instance
    inst : pyvisa.Resource
        PyVISA resource instance for device communication
    _nplc : float
        Integration time in power line cycles (0.01 to 10)
    _auto_zero : bool
        Auto-zero setting state
    _source_delay : float
        Delay after setting source value in seconds
    _mode : str
        Current source mode ("voltage" or "current")
    _output_enabled : bool
        Current output state
    """

    def __init__(
        self,
        visa_address: Optional[str] = None,
        resource_manager: Optional[pyvisa.ResourceManager] = None,
    ):
        """Initialize SMU2450 device.

        Args:
            visa_address: Optional VISA resource address of the instrument.
                         If None, will attempt to find first available 2450.
            resource_manager: Optional PyVISA ResourceManager instance.
                            If None, will create one with Python backend.
        """
        super().__init__()
        # Store the resource manager reference
        self.rm = resource_manager
        self.inst = None
        self._owns_rm = False  # Track if we created our own resource manager

        # Create resource manager if none provided
        if self.rm is None:
            self.rm = pyvisa.ResourceManager()
            self._owns_rm = True  # We created it, so we should close it

        if visa_address is None:
            # Try cached address first
            cached_addr = get_cached_address("smu2450")
            if cached_addr:
                try:
                    # Quick test if device at cached address is available
                    inst = self.rm.open_resource(cached_addr)
                    idn = inst.query("*IDN?")
                    if "MODEL 2450" in idn:
                        self.visa_address = cached_addr
                        inst.close()
                        logger.debug(f"Using cached SMU address: {cached_addr}")
                        return
                except Exception:
                    logger.trace("Cached address not valid, falling back to discovery")

            # Fall back to discovery if cache miss or invalid
            devices = list_visa_devices(
                model_filter="MODEL 2450", detailed=False, resource_manager=self.rm
            )
            if not devices:
                raise RuntimeError("No Keithley 2450 devices found")
            self.visa_address = next(iter(devices.keys()))
            # Cache the discovered address
            update_cached_address("smu2450", self.visa_address)
            logger.info(
                f"Auto-discovered Keithley 2450 at {self.visa_address}: {devices[self.visa_address]}"
            )
        else:
            self.visa_address = visa_address
            # Cache explicitly provided address
            update_cached_address("smu2450", self.visa_address)

        # Configuration settings
        self._nplc = DEFAULT_NPLC
        self._auto_zero = DEFAULT_AUTO_ZERO
        self._source_delay = DEFAULT_SOURCE_DELAY

        # Current state
        self._mode: Optional[str] = None
        self._output_enabled = False

    def _check_error(self) -> None:
        """Check for device errors and raise if found."""
        error = self._communicate("SYST:ERR?", query=True, check_errors=False)
        if not error.startswith("0"):
            raise RuntimeError(f"Device error: {error}")

    def _handle_error(self, action: str, e: Exception) -> None:
        """Centralized error handling with logging."""
        logger.error(f"Error during {action}: {e}")
        self.abort()  # Safe abort on errors
        raise RuntimeError(f"{action} failed: {str(e)}")

    def _recover_from_pipe_error(self) -> bool:
        """Attempt to recover from a pipe error by resetting communication.

        Returns
        -------
        bool
            True if recovery was successful, False otherwise
        """
        logger.trace("Attempting to recover from pipe error")

        # Define recovery steps
        recovery_steps = [
            lambda: self._close_connection(),
            lambda: time.sleep(0.5),  # Shorter wait time
            lambda: self._recreate_resource_manager(),
            lambda: self._open_connection(),
            lambda: self._reset_device(),
            lambda: self._verify_connection(),
        ]

        # Try recovery sequence up to 2 times
        for attempt in range(2):
            try:
                # Execute each recovery step
                for step in recovery_steps:
                    step()

                # If we get here, recovery was successful
                logger.trace("Successfully recovered from pipe error")
                return True

            except Exception as e:
                logger.trace(f"Recovery attempt {attempt + 1} failed: {e}")
                time.sleep(1.0)  # Wait between attempts

        logger.error("All recovery attempts failed")
        return False

    def _close_connection(self) -> None:
        """Close the current connection safely."""
        if self.inst:
            try:
                self.inst.close()
            except Exception:
                pass
            self.inst = None

    def _recreate_resource_manager(self) -> None:
        """Recreate the resource manager if we own it."""
        if self._owns_rm:
            self.rm = pyvisa.ResourceManager()

    def _open_connection(self) -> None:
        """Open a new connection to the device."""
        self.inst = self.rm.open_resource(self.visa_address)
        self.inst.timeout = 5000  # Longer timeout during recovery

    def _reset_device(self) -> None:
        """Reset the device to a known state."""
        self.inst.write("*RST")
        time.sleep(0.2)
        self.inst.write("*CLS")
        time.sleep(0.2)

    def _verify_connection(self) -> None:
        """Verify the connection is working properly."""
        response = self.inst.query("*IDN?")
        if "MODEL 2450" not in response:
            raise RuntimeError("Device identification failed")
        self.inst.timeout = 2000  # Reset to normal timeout

    def _communicate(
        self,
        command: str,
        query: bool = False,
        check_errors: bool = False,
        description: str = None,
    ) -> Optional[str]:
        """Unified communication method with error handling and recovery.

        Parameters
        ----------
        command : str
            Command to send to the device
        query : bool
            Whether this is a query command
        check_errors : bool
            Whether to check for device errors after command execution
        description : str
            Human-readable description of what this command does

        Returns
        -------
        Optional[str]
            Response string if query=True, None otherwise

        Raises
        ------
        RuntimeError
            If the device is not connected or command execution fails
            If check_errors=True and the command results in a device error
        """
        if not self.inst:
            raise RuntimeError(ERROR_MESSAGES["not_connected"])

        max_retries = 2
        for attempt in range(max_retries):
            try:
                logger.trace(f"{'Querying' if query else 'Writing'}: {command}")
                if query:
                    time.sleep(0.02)  # Reduced delay before queries
                    result = self.inst.query(command).strip()
                    logger.trace(f"Query result: {result}")

                    # Check for errors if requested
                    if check_errors:
                        error = self.inst.query("SYST:ERR?").strip()
                        if not error.startswith("0"):
                            cmd_desc = f" ({description})" if description else ""
                            raise RuntimeError(
                                f"Error executing query{cmd_desc}: {command}\nDevice error: {error}"
                            )

                    return result
                else:
                    self.inst.write(command)
                    time.sleep(0.01)  # Reduced delay after writes
                    logger.trace(f"Command write successful")

                    # Check for errors if requested
                    if check_errors:
                        error = self.inst.query("SYST:ERR?").strip()
                        if not error.startswith("0"):
                            cmd_desc = f" ({description})" if description else ""
                            raise RuntimeError(
                                f"Error executing command{cmd_desc}: {command}\nDevice error: {error}"
                            )

                    return None

            except Exception as e:
                logger.trace(f"Communication error with command {command}: {str(e)}")

                if "Pipe" in str(e) and attempt < max_retries - 1:
                    logger.trace("Pipe error detected, attempting recovery")
                    if self._recover_from_pipe_error():
                        logger.trace("Recovery successful, retrying command")
                        continue

                self._handle_error(f"{'query' if query else 'command'} {command}", e)

    def _communicate_with_check(
        self, command: str, query: bool = False, description: str = None
    ) -> Optional[str]:
        """Execute a command and immediately check for errors.

        This is a convenience wrapper around _communicate with check_errors=True.

        Parameters
        ----------
        command : str
            Command to send to the device
        query : bool
            Whether this is a query command
        description : str
            Human-readable description of what this command does

        Returns
        -------
        Optional[str]
            Response string if query=True, None otherwise
        """
        return self._communicate(
            command, query=query, check_errors=True, description=description
        )

    def check_compliance(self) -> bool:
        """Check if device is in compliance (limit reached)."""
        try:
            if not self.inst:
                return False

            # First check if output is even enabled
            if not self.get_output_state():
                return False

            # Get current mode and appropriate command
            mode = self.get_mode()
            if mode == "voltage":
                cmd = ":SOUR:VOLT:ILIM:TRIP?"
            else:
                cmd = ":SOUR:CURR:VLIM:TRIP?"

            # Query compliance state without error checking (to avoid false errors)
            try:
                result = self._communicate(cmd, query=True, check_errors=False)
                return bool(int(result))
            except Exception as e:
                logger.trace(f"Compliance check failed with command {cmd}: {e}")
                # Try alternative compliance check method
                status = self._communicate(
                    ":STAT:OPER:COND?", query=True, check_errors=False
                )
                return bool(int(status) & 0x0800)  # Check compliance bit

        except Exception as e:
            logger.trace(f"Compliance check failed: {e}")
            return False  # Return False on any error to avoid false positives

    def _ramp_value(
        self, start: float, target: float, ramp_rate: float, set_func
    ) -> None:
        """Ramp a value smoothly with adaptive timing.

        Args:
            start: Starting value
            target: Target value
            ramp_rate: Rate of change per second
            set_func: Function to set the value

        Raises:
            ValueError: If ramp_rate is not positive
            RuntimeError: If compliance limit reached or other error occurs
        """
        if ramp_rate <= 0:
            raise ValueError("Ramp rate must be positive")

        # Calculate optimal step parameters based on ramp rate
        delta = abs(target - start)
        min_step_time = 0.02  # Reduced minimum step time
        max_step_time = 0.1  # Maximum step time

        # Calculate number of steps based on ramp rate and timing constraints
        ideal_time = delta / ramp_rate
        n_steps = max(
            min(
                int(ideal_time / min_step_time),  # Steps based on min time
                int(delta * 20),  # Max 20 steps per unit change
            ),
            5,  # Minimum 5 steps for smooth ramping
        )

        # Generate step values with smaller steps near endpoints
        t = np.linspace(0, 1, n_steps)
        # Use smooth sigmoid-like function for step sizes
        smoothing = 0.5 * (1 - np.cos(np.pi * t))
        values = start + (target - start) * smoothing

        # Calculate timing
        step_times = np.diff(smoothing) * delta / ramp_rate

        start_time = time.time()
        next_check_time = start_time
        check_interval = max(
            min_step_time * 2, ideal_time / 10
        )  # Dynamic compliance check interval

        try:
            for i, (val, step_time) in enumerate(zip(values, np.append(step_times, 0))):
                current_time = time.time()

                # Set value
                set_func(val)

                # Check compliance on a time basis rather than step count
                if current_time >= next_check_time:
                    if self.check_compliance():
                        raise RuntimeError(f"Compliance limit reached at {val}")
                    next_check_time = current_time + check_interval

                # Adaptive timing adjustment
                elapsed = time.time() - current_time
                if step_time > elapsed:
                    sleep_time = min(max_step_time, step_time - elapsed)
                    if sleep_time > 0.001:  # Only sleep for meaningful intervals
                        time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Error during ramp: {e}")
            self.abort()
            raise RuntimeError(f"Ramp failed: {str(e)}")

    def configure(
        self,
        nplc: Optional[float] = None,
        auto_zero: Optional[bool] = None,
        source_delay: Optional[float] = None,
        voltage_range: Optional[float] = None,
        current_range: Optional[float] = None,
    ) -> None:
        """Configure measurement settings.

        Args:
            nplc: Integration time in power line cycles (0.01 to 10)
            auto_zero: Enable/disable auto-zero functionality
            source_delay: Delay after setting source value, in seconds
            voltage_range: Voltage measurement range (0 for auto-range)
            current_range: Current measurement range (0 for auto-range)
        """
        if not self.inst:
            raise RuntimeError("Device not connected")

        original_timeout = self.inst.timeout
        try:
            # Longer timeout for configuration
            self.inst.timeout = 10000  # 10 seconds

            if nplc is not None:
                if not 0.01 <= nplc <= 10:
                    raise ValueError("NPLC must be between 0.01 and 10")
                self._communicate(f":SENS:CURR:NPLC {nplc}")
                self._communicate(f":SENS:VOLT:NPLC {nplc}")
                error = self._communicate("SYST:ERR?", query=True)
                if not error.startswith("0"):
                    raise RuntimeError(f"NPLC setting failed: {error}")
                self._nplc = nplc

            if auto_zero is not None:
                self._communicate(f":SYST:AZER {'ON' if auto_zero else 'OFF'}")
                error = self._communicate("SYST:ERR?", query=True)
                if not error.startswith("0"):
                    raise RuntimeError(f"Auto-zero setting failed: {error}")
                self._auto_zero = auto_zero

            if source_delay is not None:
                if source_delay < 0:
                    raise ValueError("Source delay cannot be negative")
                self._communicate(f":SOUR:DEL {source_delay}")
                error = self._communicate("SYST:ERR?", query=True)
                if not error.startswith("0"):
                    raise RuntimeError(f"Source delay setting failed: {error}")
                self._source_delay = source_delay

            if voltage_range is not None:
                if voltage_range == 0:
                    self._communicate(":SENS:VOLT:RANG:AUTO ON")
                    error = self._communicate("SYST:ERR?", query=True)
                    if not error.startswith("0"):
                        raise RuntimeError("Voltage auto-range setting failed: {error}")
                else:
                    self._communicate(":SENS:VOLT:RANG:AUTO OFF")
                    self._communicate(f":SENS:VOLT:RANG {abs(voltage_range)}")
                    error = self._communicate("SYST:ERR?", query=True)
                    if not error.startswith("0"):
                        raise RuntimeError(f"Voltage range setting failed: {error}")

            if current_range is not None:
                if current_range == 0:
                    self._communicate(":SENS:CURR:RANG:AUTO ON")
                else:
                    self._communicate(":SENS:CURR:RANG:AUTO OFF")
                    self._communicate(f":SENS:CURR:RANG {abs(current_range)}")
                error = self._communicate("SYST:ERR?", query=True)
                if not error.startswith("0"):
                    raise RuntimeError(f"Current range setting failed: {error}")
        finally:
            self.inst.timeout = original_timeout

    def open(self) -> None:
        """Open connection to the instrument and configure defaults."""
        try:
            self.inst = self.rm.open_resource(self.visa_address)
            self.inst.timeout = 2000  # 2 second timeout

            # Set proper termination characters
            self.inst.read_termination = "\n"
            self.inst.write_termination = "\n"

            # Basic initialization sequence
            idn = self._communicate("*IDN?", query=True, check_errors=True)
            if "MODEL 2450" not in idn:
                raise RuntimeError(f"Unexpected device ID: {idn}")

            # Basic configuration
            self._communicate("*RST", check_errors=True)  # Reset
            time.sleep(0.1)
            self._communicate("*CLS", check_errors=True)  # Clear status
            time.sleep(0.1)

            # Basic configuration - standard safe state
            self._communicate(
                ":SOUR:FUNC VOLT", check_errors=True
            )  # Start in voltage source mode
            self._communicate(
                ":SOUR:VOLT:RANG 20", check_errors=True
            )  # Set to 20V range for safety
            self._communicate(
                ":SOUR:VOLT:ILIM 0.5", check_errors=True
            )  # Set 500mA compliance
            self._communicate(':SENS:FUNC "CURR"', check_errors=True)  # Measure current
            self._communicate(
                ":SENS:CURR:RANG 1.05", check_errors=True
            )  # Set to max current range
            self._communicate(
                ":SENS:CURR:NPLC 1", check_errors=True
            )  # Set integration time
            self._communicate(":OUTP OFF", check_errors=True)  # Ensure output is off

            self._mode = "voltage"  # Set initial mode

        except Exception as e:
            if self.inst:
                try:
                    self.inst.close()
                except:
                    pass
                self.inst = None
            raise RuntimeError(f"Failed to connect to instrument: {str(e)}")

    def close(self) -> None:
        """Safely close connection to the instrument."""
        if self.inst:
            try:
                self.zero_output()  # Safely return to zero
                self._communicate(":OUTP OFF")
                self.inst.close()
            except Exception as e:
                logger.error(f"Error closing instrument: {e}")
            finally:
                self.inst = None

        # Only close the resource manager if we created it
        if self._owns_rm and self.rm:
            try:
                self.rm.close()
            except Exception as e:
                logger.error(f"Error closing resource manager: {e}")
            finally:
                self.rm = None
                self._owns_rm = False

    def _measure_point(
        self, mode: str, value: float, delay: float
    ) -> Tuple[float, float]:
        """Measure a single point in the sweep."""
        if mode == "voltage":
            self._communicate(f":SOUR:VOLT {value}")
        else:
            self._communicate(f":SOUR:CURR {value}")
        time.sleep(delay)

        # Measure source value and the complementary measurement
        if mode == "voltage":
            source_val = float(self._communicate(":MEAS:VOLT?", query=True))
            meas_val = float(self._communicate(":MEAS:CURR?", query=True))
        else:
            source_val = float(self._communicate(":MEAS:CURR?", query=True))
            meas_val = float(self._communicate(":MEAS:VOLT?", query=True))

        return source_val, meas_val

    def measure_iv_curve(
        self,
        start: float,
        stop: float,
        points: int,
        delay: float = 0.1,
        sweep_type: Literal["linear", "log"] = "linear",
        bidirectional: bool = False,
    ) -> Tuple[list[float], list[float]]:
        """Perform an I-V curve measurement."""
        if not self.inst:
            raise RuntimeError("Device not connected")

        mode = self.get_mode()
        source_vals = []
        meas_vals = []

        try:
            # Enable output before starting sweep
            self.set_output_state(True)

            # Generate source values
            if sweep_type == "linear":
                forward = np.linspace(start, stop, points)
                if bidirectional:
                    backward = np.linspace(stop, start, points)
                    values = np.concatenate([forward, backward])
                else:
                    values = forward
            else:
                forward = np.logspace(
                    np.log10(abs(start)), np.log10(abs(stop)), points
                ) * np.sign(start)
                if bidirectional:
                    backward = np.logspace(
                        np.log10(abs(stop)), np.log10(abs(start)), points
                    ) * np.sign(stop)
                    values = np.concatenate([forward, backward])
                else:
                    values = forward

            # Measure point by point
            for val in values:
                source, meas = self._measure_point(mode, val, delay)
                source_vals.append(source)
                meas_vals.append(meas)

                # Check compliance periodically
                if self.check_compliance():
                    raise RuntimeError("Compliance limit reached during measurement")

            return source_vals, meas_vals

        except Exception as e:
            self.abort()  # Safely abort on any error
            raise RuntimeError(f"IV curve measurement failed: {str(e)}")

    def set_mode(self, mode: Literal["voltage", "current"]) -> None:
        """Set the source mode.

        Args:
            mode: Either "voltage" for voltage source mode or "current" for current source mode

        Raises:
            ValueError: If mode is not "voltage" or "current"
            RuntimeError: If device is not connected
        """
        if mode not in ["voltage", "current"]:
            raise ValueError('Mode must be either "voltage" or "current"')

        func = "VOLT" if mode == "voltage" else "CURR"
        self._communicate(f":SOUR:FUNC {func}")
        time.sleep(0.1)
        self._check_error()
        self._mode = mode

    def set_voltage(self, voltage: float, ramp_rate: Optional[float] = None) -> None:
        """Set output voltage, optionally ramping to target.

        Args:
            voltage: Target voltage in volts (-20V to +20V)
            ramp_rate: Optional ramp rate in V/s. If None, sets immediately.

        Raises:
            RuntimeError: If device not connected
            ValueError: If voltage outside valid range
        """
        if not -20 <= voltage <= 20:
            raise ValueError(
                f"Voltage {voltage}V outside instrument range (-20V to +20V)"
            )

        try:
            if ramp_rate is not None:
                current = float(self._communicate(":SOUR:VOLT?", query=True))
                self._ramp_value(
                    current,
                    voltage,
                    ramp_rate,
                    lambda v: self._communicate(f":SOUR:VOLT {v}"),
                )
            else:
                self._communicate(f":SOUR:VOLT {voltage}")

            # Add settling time and verification
            time.sleep(0.1)

            # Verify voltage was set correctly
            attempts = 3
            for _ in range(attempts):
                try:
                    actual = float(self._communicate(":SOUR:VOLT?", query=True))
                    if abs(actual - voltage) < 0.01:  # Within 10mV
                        return
                    time.sleep(0.1)
                except Exception as e:
                    logger.trace(f"Voltage verification attempt failed: {e}")
                    time.sleep(0.1)

            raise RuntimeError("Failed to verify voltage setting")

        except Exception as e:
            self._handle_error(f"setting voltage to {voltage}V", e)

    def set_current(
        self,
        current: float,
        ramp_rate: Optional[float] = None,
        fast: Optional[bool] = False,
    ) -> None:
        """Set output current, optionally ramping to target.

        Args:
            current: Target current in amps (-1.05A to +1.05A)
            ramp_rate: Optional ramp rate in A/s. If None, sets immediately.

        Raises:
            RuntimeError: If device not connected
            ValueError: If current outside valid range
        """
        if not -1.05 <= current <= 1.05:
            raise ValueError(
                f"Current {current}A outside instrument range (-1.05A to +1.05A)"
            )

        try:
            if ramp_rate is not None:
                current_val = float(self._communicate(":SOUR:CURR?", query=True))
                self._ramp_value(
                    current_val,
                    current,
                    ramp_rate,
                    lambda i: self._communicate(f":SOUR:CURR {i}"),
                )
            else:
                self._communicate(f":SOUR:CURR {current}")

            if not fast:
                # Add settling time and verification
                time.sleep(0.05)
                actual = float(self._communicate(":SOUR:CURR?", query=True))
                if abs(actual - current) > 0.001:  # Within 1mA
                    raise RuntimeError(
                        f"Failed to verify current setting: got {actual}A, expected {current}A"
                    )

        except Exception as e:
            self._handle_error(f"setting current to {current}A", e)

    def set_compliance(self, value: float) -> None:
        """Set compliance limit based on current mode.

        In voltage source mode, sets current compliance.
        In current source mode, sets voltage compliance.

        Args:
            value: Compliance limit value

        Raises:
            RuntimeError: If device not connected
        """
        mode = self.get_mode()
        if mode == "voltage":
            self._communicate(f":SOUR:VOLT:ILIM {value}")
        else:
            self._communicate(f":SOUR:CURR:VLIM {value}")
        error = self._communicate("SYST:ERR?", query=True)
        if not error.startswith("0"):
            raise RuntimeError(f"Error setting compliance limit: {error}")

    def set_output_state(self, state: bool) -> None:
        """Enable or disable the output.

        Args:
            state: True to enable output, False to disable

        Raises:
            RuntimeError: If device not connected
        """
        self._communicate(f":OUTP {1 if state else 0}")
        time.sleep(0.1)
        self._check_error()
        self._output_enabled = state

    def get_voltage(self) -> float:
        """Get the present output voltage.

        Returns:
            float: Present voltage in volts

        Raises:
            RuntimeError: If device not connected
        """
        return float(self._communicate(":MEAS:VOLT?", query=True))

    def get_current(self) -> float:
        """Get the present output current.

        Returns:
            float: Present current in amps

        Raises:
            RuntimeError: If device not connected
        """
        return float(self._communicate(":MEAS:CURR?", query=True))

    def get_output_state(self) -> bool:
        """Get the present output state.

        Returns:
            bool: True if output enabled, False if disabled

        Raises:
            RuntimeError: If device not connected
        """
        result = self._communicate(":OUTP?", query=True)
        return bool(int(result))

    def get_mode(self) -> str:
        """Get the present source mode.

        Returns
        -------
        str
            Either "voltage" or "current"

        Raises
        ------
        RuntimeError
            If device not connected
        """
        func = self._communicate(":SOUR:FUNC?", query=True).strip('"')
        return "voltage" if func == "VOLT" else "current"

    def abort(self) -> None:
        """Abort any ongoing operations safely."""
        if not self.inst:
            return

        logger.trace("Attempting abort sequence")

        # Define critical commands to execute in sequence
        commands = [
            ("ABOR", "abort command"),
            ("TRIG:LOAD:EMPTY", "reset trigger model"),
            (":OUTP OFF", "disable output"),
        ]

        # Execute each command with basic error handling
        for cmd, description in commands:
            try:
                self.inst.write(cmd)
                time.sleep(0.05)  # Shorter delay between commands
                logger.trace(f"Executed {description}")
            except Exception as e:
                logger.warning(f"Error during {description}: {e}")
                # Continue with next command even if this one failed

        # Final emergency output disable if needed
        try:
            self.inst.write(":OUTP OFF")
        except Exception:
            pass

        logger.trace("Abort sequence completed")

    def zero_output(self) -> None:
        """Safely return output to zero.

        Ramps the output to zero and disables it.
        """
        logger.trace("Attempting zero_output")
        try:
            # Simpler zero sequence
            logger.trace("Setting voltage to 0")
            self.inst.write(":SOUR:VOLT 0")
            time.sleep(0.1)
            logger.trace("Disabling output")
            self.inst.write(":OUTP 0")
            time.sleep(0.1)
            logger.trace("zero_output completed successfully")
        except Exception as e:
            logger.error(f"Error during zero_output: {e}")
            try:
                logger.trace("Attempting emergency output disable")
                self.inst.write(":OUTP 0")  # One last try to disable output
                logger.trace("Emergency disable successful")
            except Exception as e2:
                logger.error(f"Emergency disable failed: {e2}")

    def get_compliance_limit(self) -> float:
        """Get the current compliance limit.

        Returns:
            float: Current compliance limit (current in voltage mode, voltage in current mode)
        """
        mode = self.get_mode()
        if mode == "voltage":
            result = self._communicate(":SOUR:VOLT:ILIM?", query=True)
            error = self._communicate("SYST:ERR?", query=True)
            if not error.startswith("0"):
                raise RuntimeError(f"Error getting voltage compliance: {error}")
            return float(result)
        else:
            result = self._communicate(":SOUR:CURR:VLIM?", query=True)
            error = self._communicate("SYST:ERR?", query=True)
            if not error.startswith("0"):
                raise RuntimeError(f"Error getting current compliance: {error}")
            return float(result)

    def is_remote(self) -> bool:
        """Check if device is in remote control mode.

        Returns:
            bool: True if in remote mode, False if in local mode
        """
        result = self._communicate(":SYST:REM?", query=True)
        error = self._communicate("SYST:ERR?", query=True)
        if not error.startswith("0"):
            raise RuntimeError(f"Error checking remote mode: {error}")
        return bool(int(result))

    def set_local(self) -> None:
        """Return control to front panel."""
        self._communicate(":SYST:LOC")
        error = self._communicate("SYST:ERR?", query=True)
        if not error.startswith("0"):
            raise RuntimeError(f"Error setting local mode: {error}")

    def get_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get available ranges for voltage and current.

        Returns:
            Dict with 'voltage' and 'current' keys mapping to (min, max) tuples
        """
        # 2450 ranges from datasheet
        return {
            "voltage": (-20.0, 20.0),
            "current": (-1.05, 1.05),  # ±1.05A DC, ±1.0A pulse
        }

    def get_mode_state(self) -> str:
        """Get current source mode ("voltage" or "current")."""
        return self._mode

    def get_output_enabled(self) -> bool:
        """Get current output state."""
        return self._output_enabled

    def get_nplc(self) -> float:
        """Get integration time in power line cycles."""
        return self._nplc

    def get_auto_zero(self) -> bool:
        """Get auto-zero setting state."""
        return self._auto_zero

    def get_source_delay(self) -> float:
        """Get delay after setting source value in seconds."""
        return self._source_delay

    def configure_trigger(self, config: TriggerConfig) -> None:
        """Configure triggering behavior.

        Parameters
        ----------
        config : TriggerConfig
            TriggerConfig instance with trigger settings

        Raises
        ------
        ValueError
            If trigger configuration is invalid
        RuntimeError
            If device is not connected or trigger configuration fails
        """
        if not self.inst:
            raise RuntimeError(ERROR_MESSAGES["not_connected"])

        # Configure the trigger model
        self._configure_trigger_simple(config)
        self._check_error()

    def wait_for_trigger(self, timeout: float = 1.0) -> bool:
        """Wait for trigger event.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if triggered, False if timed out

        Raises:
            RuntimeError: If device not connected or error occurs
        """
        if not self.inst:
            raise RuntimeError("Device not connected")

        start = time.time()
        while time.time() - start < timeout:
            status = self._communicate(":STAT:OPER:COND?", query=True)
            if int(status) & 0x0002:  # Check trigger bit
                return True
            time.sleep(0.01)
        return False

    def configure_digital_io(
        self, pin: int, mode: Literal["trigger", "input", "output"] = "trigger"
    ) -> None:
        """Configure a digital I/O pin.

        Args:
            pin: Digital I/O pin number (1-6)
            mode: Pin mode - "trigger" for trigger input, "input" for digital input,
                  "output" for digital output
        """
        if not 1 <= pin <= 6:
            raise ValueError("Pin must be between 1 and 6")

        if mode == "trigger":
            self._communicate(f"DIG:LINE{pin}:MODE TRIG, IN")
        elif mode == "input":
            self._communicate(f"DIG:LINE{pin}:MODE DIG, IN")
        elif mode == "output":
            self._communicate(f"DIG:LINE{pin}:MODE DIG, OUT")

    def start_trigger_model(self) -> None:
        """Start the configured trigger model."""
        self._communicate(":INIT")
        self._check_error()

    def stop_trigger_model(self, max_wait: float = 2.0) -> str:
        """Stop the trigger model execution and wait for it to fully stop.

        Parameters
        ----------
        max_wait : float
            Maximum time to wait in seconds for the trigger model to stop

        Returns
        -------
        str
            Final trigger model state after stopping
        """
        logger.debug("Stopping trigger model")

        # Define stop sequence commands
        stop_commands = [
            ("ABOR", "abort trigger model"),
            # ("TRIG:LOAD:EMPTY", "reset trigger model")
        ]

        # Execute stop sequence
        for cmd, description in stop_commands:
            try:
                self._communicate(cmd, check_errors=False)
                logger.trace(f"Executed {description}")
            except Exception as e:
                logger.warning(f"Error during {description}: {e}")

        # Wait for trigger model to fully stop with timeout
        acceptable_states = ["idle", "empty", "aborted"]
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                state = self.get_trigger_state()
                if state in acceptable_states:
                    logger.debug(f"Trigger model is now in {state} state")
                    return state
                time.sleep(0.1)  # Short sleep between checks
            except Exception:
                break
        raise RuntimeError(f"Could not abort, state: {state}")

        # # If we get here, try a more aggressive reset
        # try:
        #     self._communicate("*RST", check_errors=False)
        #     self._communicate("*CLS", check_errors=False)
        #     return "reset"
        # except Exception:
        #     return "unknown"

    def get_trigger_state(self) -> str:
        """Get the current trigger model state.

        Returns:
            str: Current state ("idle", "running", "waiting", "empty",
                               "building", "failed", "aborting", "aborted")

        Note:
            The device returns both the state and the last executed block number
            in format like "IDLE;IDLE;9". This method extracts just the state.
        """
        try:
            response = self._communicate(":TRIG:STATE?", query=True)

            # Handle empty response
            if not response:
                return "unknown"

            # Extract just the state part from the response
            # Response format is typically "STATE;STATE;BLOCK_NUMBER"
            if ";" in response:
                state = response.split(";")[0]
            else:
                state = response

            # Clean up the state - remove any block information
            if ")" in state:
                state = "running"  # If we see block info, the model is running

            # Normalize state to lowercase
            return state.lower()
        except Exception as e:
            logger.warning(f"Error getting trigger state: {e}")
            return "unknown"

    def _configure_trigger_simple(self, config: TriggerConfig) -> None:
        """Configure a simple trigger model for toggling output based on digital input.

        Parameters
        ----------
        config : TriggerConfig
            Trigger configuration
        """
        logger.debug("Configuring simple trigger model")

        try:
            # Step 1: Reset and prepare the device
            self._setup_device_for_triggering(config)

            # Step 2: Create configuration list with high and low states
            list_name = "sourceList"
            self._create_source_config_list(list_name, config)

            # Step 3: Build the trigger model blocks
            self._build_trigger_model_blocks(config, list_name)

            # Step 4: Enable output
            self._communicate_with_check(":OUTP ON", description="Enable output")

            self._check_error()

            logger.debug("Trigger model configured successfully")

        except Exception as e:
            self._handle_error("configuring simple trigger", e)

    def _setup_device_for_triggering(self, config: TriggerConfig) -> None:
        """Prepare the device for triggering by setting up basic configuration.

        Parameters
        ----------
        config : TriggerConfig
            Trigger configuration
        """
        # Reset trigger model and device
        self._communicate_with_check("*RST", description="Reset device")
        self._communicate_with_check("*CLS", description="Clear status")

        # Explicitly set the source mode
        if self._mode == "current":
            self._communicate_with_check(
                ":SOUR:FUNC CURR", description="Set to current source mode"
            )
            # Set voltage limit for current source mode
            self._communicate_with_check(
                ":SOUR:CURR:VLIM 20", description="Set voltage limit to 20V"
            )
        else:
            self._communicate_with_check(
                ":SOUR:FUNC VOLT", description="Set to voltage source mode"
            )
            # Set smart current limit for voltage source mode based on expected current
            # Use 1.05A as default max, but calculate based on config if possible
            if hasattr(config, "high_value") and config.high_value > 0:
                # Calculate current from voltage using Ohm's law with the provided resistance
                resistance = getattr(
                    config, "resistance", 20.0
                )  # Use provided resistance or default to 20 ohms
                expected_current = config.high_value / resistance

                # For high currents near the maximum, use less headroom
                if expected_current > 0.8:  # If we're near the max current
                    estimated_current = min(
                        1.05, expected_current * 1.05
                    )  # Use only 5% headroom
                    logger.info(
                        f"High current detected ({expected_current:.3f}A), using reduced headroom"
                    )
                else:
                    estimated_current = min(
                        1.05, expected_current * 1.2
                    )  # Normal 20% headroom

                # First disable auto-ranging for current measurement
                self._communicate_with_check(
                    ":SENS:CURR:RANG:AUTO OFF", description="Disable current auto-range"
                )

                # For high currents, set to max range directly
                if expected_current > 0.8:
                    self._communicate_with_check(
                        ":SENS:CURR:RANG 1.05",
                        description="Set current measurement range to maximum (1.05A)",
                    )
                else:
                    # Set current measurement range before setting compliance limit
                    self._communicate_with_check(
                        f":SENS:CURR:RANG {estimated_current}",
                        description=f"Set current measurement range to {estimated_current}A",
                    )

                # Then set current limit
                self._communicate_with_check(
                    f":SOUR:VOLT:ILIM {estimated_current}",
                    description=f"Set current limit to {estimated_current}A (using {resistance}Ω)",
                )
            else:
                # Default to max current if we can't estimate
                self._communicate_with_check(
                    ":SOUR:VOLT:ILIM 1.05", description="Set current limit to 1.05A"
                )
                self._communicate_with_check(
                    ":SENS:CURR:RANG:AUTO OFF", description="Disable current auto-range"
                )
                self._communicate_with_check(
                    ":SENS:CURR:RANG 1.05",
                    description="Set current measurement range to 1.05A",
                )

        # Verify the mode is set correctly
        actual_mode = self._communicate_with_check(
            ":SOUR:FUNC?", query=True, description="Verify source mode"
        )
        logger.debug(f"Source mode set to: {actual_mode}")

        # Clear any existing trigger model
        self._communicate_with_check(
            "TRIG:LOAD:EMPTY", description="Empty trigger model"
        )

        # Configure digital line for external triggering
        self._communicate_with_check(
            f"DIG:LINE{config.pin}:MODE TRIG, IN",
            description=f"Set digital line {config.pin} as trigger input",
        )

        # Set edge detection
        edge_str = (
            "EITH"
            if config.edge == TriggerEdge.EITHER
            else ("RIS" if config.edge == TriggerEdge.RISING else "FALL")
        )
        self._communicate_with_check(
            f"TRIG:DIG{config.pin}:IN:EDGE {edge_str}",
            description=f"Set trigger edge to {edge_str}",
        )

        # Optimize for fast transitions
        self._communicate_with_check(
            ":SOUR:DEL:AUTO OFF", description="Disable auto delay"
        )
        self._communicate_with_check(":SOUR:DEL 0", description="Set source delay to 0")
        self._communicate_with_check(
            ":SENS:CURR:NPLC 0.01", description="Set NPLC to minimum"
        )

    def _create_source_config_list(self, list_name: str, config: TriggerConfig) -> None:
        """Create a configuration list with high and low states.

        Parameters
        ----------
        list_name : str
            Name of the configuration list to create
        config : TriggerConfig
            Trigger configuration with high and low values
        """
        # Create a new configuration list
        self._communicate_with_check(
            f'SOUR:CONF:LIST:CRE "{list_name}"',
            description="Create source configuration list",
        )

        if self._mode == "current":
            # Set appropriate current range
            self._communicate_with_check(
                ":SOUR:CURR:RANG:AUTO OFF", description="Disable auto range"
            )

            # Calculate appropriate range based on high value
            high_current = 1.2 * config.high_value
            current_range = 1.0  # Default to max range
            if high_current <= 0.01:
                current_range = 0.01
            elif high_current <= 0.1:
                current_range = 0.1
            elif high_current <= 1.0:
                current_range = 1.0

            self._communicate_with_check(
                f":SOUR:CURR:RANG {current_range}",
                description=f"Set current range to {current_range}A",
            )

            # Configure and store low state
            self._communicate_with_check(
                f":SOUR:CURR {config.low_value}", description="Set low current"
            )
            self._communicate_with_check(
                f'SOUR:CONF:LIST:STOR "{list_name}"', description="Store low state"
            )

            # Configure and store high state
            self._communicate_with_check(
                f":SOUR:CURR {config.high_value}", description="Set high current"
            )
            self._communicate_with_check(
                f'SOUR:CONF:LIST:STOR "{list_name}"', description="Store high state"
            )
        else:
            # Set appropriate voltage range for voltage mode
            self._communicate_with_check(
                ":SOUR:VOLT:RANG:AUTO OFF", description="Disable auto range"
            )

            # Calculate appropriate range based on high value
            high_voltage = 1.2 * config.high_value
            voltage_range = 20.0  # Default to max range
            if high_voltage <= 2.0:
                voltage_range = 2.0
            elif high_voltage <= 7.0:
                voltage_range = 7.0
            elif high_voltage <= 20.0:
                voltage_range = 20.0

            self._communicate_with_check(
                f":SOUR:VOLT:RANG {voltage_range}",
                description=f"Set voltage range to {voltage_range}V",
            )

            # Calculate appropriate current limit based on expected current
            resistance = getattr(
                config, "resistance", 20.0
            )  # Use provided resistance or default to 20 ohms
            expected_current = config.high_value / resistance

            # For high currents near the maximum, use less headroom
            if expected_current > 0.8:  # If we're near the max current
                current_limit = min(
                    1.05, expected_current * 1.05
                )  # Use only 5% headroom
                logger.info(
                    f"High current detected ({expected_current:.3f}A), using reduced headroom"
                )
            else:
                current_limit = min(1.05, expected_current * 1.2)  # Normal 20% headroom

            # First disable auto-ranging for current measurement
            self._communicate_with_check(
                ":SENS:CURR:RANG:AUTO OFF", description="Disable current auto-range"
            )

            # For high currents, set to max range directly
            if expected_current > 0.8:
                self._communicate_with_check(
                    ":SENS:CURR:RANG 1.05",
                    description="Set current measurement range to maximum (1.05A)",
                )
            else:
                # Set current measurement range before setting compliance limit
                self._communicate_with_check(
                    f":SENS:CURR:RANG {current_limit}",
                    description=f"Set current measurement range to {current_limit}A",
                )

            # Then set current limit
            self._communicate_with_check(
                f":SOUR:VOLT:ILIM {current_limit}",
                description=f"Set current limit to {current_limit}A (using {resistance}Ω)",
            )

            # Configure and store low state for voltage mode
            self._communicate_with_check(
                f":SOUR:VOLT {config.low_value}", description="Set low voltage"
            )
            self._communicate_with_check(
                f'SOUR:CONF:LIST:STOR "{list_name}"', description="Store low state"
            )

            # Configure and store high state
            self._communicate_with_check(
                f":SOUR:VOLT {config.high_value}", description="Set high voltage"
            )

            # For high currents, maintain the same approach as above
            if expected_current > 0.8:
                self._communicate_with_check(
                    ":SENS:CURR:RANG 1.05",
                    description="Set current measurement range to maximum (1.05A)",
                )
            else:
                self._communicate_with_check(
                    f":SENS:CURR:RANG {current_limit}",
                    description="Set current measurement range for high state",
                )

            # Set current limit again before storing high state to ensure it's included
            self._communicate_with_check(
                f":SOUR:VOLT:ILIM {current_limit}",
                description="Set current limit for high state",
            )
            self._communicate_with_check(
                f'SOUR:CONF:LIST:STOR "{list_name}"', description="Store high state"
            )

        # Set back to low state initially
        self._communicate_with_check(
            f'SOUR:CONF:LIST:REC "{list_name}", 1', description="Recall low state"
        )

    def _build_trigger_model_blocks(
        self, config: TriggerConfig, list_name: str
    ) -> None:
        """Build the trigger model blocks for toggling between states.

        Parameters
        ----------
        config : TriggerConfig
            Trigger configuration
        list_name : str
            Name of the configuration list to use
        """
        # Build a trigger model that alternates between high and low states on rising edges
        commands = [
            # Block 1: Wait for rising edge trigger
            f"TRIG:BLOC:WAIT 1, DIG{config.pin}",
            # Block 2: Recall high state configuration
            f'TRIG:BLOC:CONF:REC 2, "{list_name}", 2',
            # Block 3: Wait for rising edge trigger again
            f"TRIG:BLOC:WAIT 3, DIG{config.pin}",
            # Block 4: Recall low state configuration
            f'TRIG:BLOC:CONF:REC 4, "{list_name}", 1',
        ]

        # Add the appropriate branching block based on count
        if config.count > 0:
            commands.append(f"TRIG:BLOC:BRAN:COUN 5, {config.count}, 1")
        else:
            commands.append("TRIG:BLOC:BRAN:ALW 5, 1")

        # Execute all commands
        for cmd in commands:
            self._communicate_with_check(cmd, description="Build trigger model block")

    def is_responding(self) -> bool:
        """Check if device is connected and responding.

        Returns
        -------
        bool
            True if device is connected and responding
        """
        if not self.inst:
            return False

        try:
            # Try to query device ID as a simple check
            response = self._communicate("*IDN?", query=True, check_errors=False)
            return "MODEL 2450" in response
        except Exception as e:
            logger.trace(f"Device not responding: {e}")
            return False

    def get_all_attrs(self) -> Dict[str, Any]:
        """Get all device attributes.

        Returns:
            Dict containing current device state and settings
        """
        return {
            "voltage": self.get_voltage(),
            "current": self.get_current(),
            "output_state": self.get_output_state(),
            "mode": self.get_mode(),
            "nplc": self._nplc,
            "auto_zero": self._auto_zero,
            "source_delay": self._source_delay,
        }
