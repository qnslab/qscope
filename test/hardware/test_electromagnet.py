import time

import matplotlib.pyplot as plt
import numpy as np
import pytest
from loguru import logger

import qscope.util
from qscope.device import SMU2450
from qscope.util import TEST_LOGLEVEL


def measure_magnet_response(
    smu: SMU2450,
    target_current: float,
    sample_rate: float = 100,
    duration: float = 1.0,
    voltage_compliance: float = 20.0,
) -> tuple[list[float], list[float], list[float]]:
    """Measure electromagnet response to current step.

    Args:
        smu: SMU2450 instance
        target_current: Target current in amps
        sample_rate: Measurements per second
        duration: Measurement duration in seconds
        voltage_compliance: Maximum voltage allowed

    Returns:
        times: List of measurement times (seconds)
        currents: List of measured currents
        voltages: List of measured voltages
    """
    # Configure SMU
    smu.configure(nplc=0.01)  # Fast measurements
    smu.set_mode("current")
    smu.set_compliance(voltage_compliance)

    # Calculate timing
    delay = 1.0 / sample_rate
    points = int(duration * sample_rate)

    # Initialize data arrays
    times = []
    currents = []
    voltages = []

    try:
        # Start with output off at zero current
        smu.set_current(0)
        smu.set_output_state(True)
        time.sleep(0.1)  # Let settle

        # Apply current step
        start_time = time.time()
        smu.set_current(target_current)

        # Measure response
        for i in range(points):
            t = time.time() - start_time
            times.append(t)
            currents.append(smu.get_current())
            voltages.append(smu.get_voltage())

            # Check compliance
            if smu.check_compliance():
                logger.warning(f"Compliance reached at t={t:.3f}s")
                break

            # Delay until next sample
            elapsed = time.time() - start_time
            next_sample = (i + 1) / sample_rate
            if next_sample > elapsed:
                time.sleep(next_sample - elapsed)

    finally:
        # Safely return to zero
        smu.zero_output()

    return times, currents, voltages


def analyze_magnet_response(
    times: list[float], currents: list[float], voltages: list[float]
) -> dict:
    """Analyze electromagnet step response data.

    Args:
        times: Measurement times in seconds
        currents: Measured currents in amps
        voltages: Measured voltages in volts

    Returns:
        Dict containing:
        - rise_time: Time to reach 90% of final current
        - inductance: Calculated inductance in henries
        - resistance: Calculated DC resistance in ohms
        - time_constant: L/R time constant in seconds
    """
    times = np.array(times)
    currents = np.array(currents)
    voltages = np.array(voltages)

    # Find final (steady-state) values
    final_current = currents[-10:].mean()
    final_voltage = voltages[-10:].mean()

    # Calculate rise time (time to reach 90% of final)
    target = 0.9 * final_current
    rise_time = times[currents >= target][0]

    # Calculate DC resistance from steady state
    resistance = final_voltage / final_current

    # Calculate inductance from initial di/dt
    dt = times[1] - times[0]
    di_dt = np.gradient(currents, dt)
    initial_di_dt = di_dt[0]
    initial_voltage = voltages[0]
    inductance = initial_voltage / initial_di_dt

    # Calculate time constant
    time_constant = inductance / resistance

    return {
        "rise_time": rise_time,
        "inductance": inductance,
        "resistance": resistance,
        "time_constant": time_constant,
    }


def plot_magnet_response(
    times: list[float], currents: list[float], voltages: list[float], analysis: dict
) -> None:
    """Plot electromagnet step response data.

    Args:
        times: Measurement times in seconds
        currents: Measured currents in amps
        voltages: Measured voltages in volts
        analysis: Analysis results from analyze_magnet_response()
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Current vs time
    ax1.plot(times, currents, "b-", label="Current")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Current (A)")
    ax1.grid(True)

    # Add rise time annotation
    ax1.axvline(
        analysis["rise_time"],
        color="r",
        linestyle="--",
        label=f"Rise time: {analysis['rise_time']:.3f}s",
    )
    ax1.legend()

    # Voltage vs time
    ax2.plot(times, voltages, "g-", label="Voltage")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Voltage (V)")
    ax2.grid(True)

    # Add analysis results
    text = (
        f"Inductance: {analysis['inductance'] * 1000:.1f} mH\n"
        f"Resistance: {analysis['resistance']:.2f} Ω\n"
        f"Time constant: {analysis['time_constant'] * 1000:.1f} ms"
    )
    ax2.text(
        0.02,
        0.98,
        text,
        transform=ax2.transAxes,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    plt.show()


class TestElectromagnet:
    @pytest.fixture()
    def client_log(self):
        qscope.util.start_client_log(log_to_file=True, log_level=TEST_LOGLEVEL)
        yield
        qscope.util.shutdown_client_log()

    @pytest.fixture
    def smu(self):
        """Create and open SMU connection for each test."""
        smu = SMU2450()  # Auto-discover device
        try:
            smu.open()
            yield smu
        finally:
            smu.close()

    @pytest.mark.usefixtures("client_log")
    def test_magnet_step_response(self, smu: SMU2450) -> None:
        """Test electromagnet step response measurement."""
        logger.info("Starting electromagnet step response test")

        # Test parameters
        target_current = 0.1  # 100mA test current
        sample_rate = 100  # 100 Hz sampling
        duration = 0.5  # 500ms measurement
        voltage_compliance = 20.0  # 20V compliance

        try:
            # Measure response
            times, currents, voltages = measure_magnet_response(
                smu=smu,
                target_current=target_current,
                sample_rate=sample_rate,
                duration=duration,
                voltage_compliance=voltage_compliance,
            )

            # Analyze results
            analysis = analyze_magnet_response(times, currents, voltages)

            # Log results
            logger.info("Electromagnet analysis results:")
            logger.info(f"  Rise time: {analysis['rise_time'] * 1000:.1f} ms")
            logger.info(f"  Inductance: {analysis['inductance'] * 1000:.1f} mH")
            logger.info(f"  Resistance: {analysis['resistance']:.2f} Ω")
            logger.info(f"  Time constant: {analysis['time_constant'] * 1000:.1f} ms")

            # Plot results
            plot_magnet_response(times, currents, voltages, analysis)

        finally:
            smu.zero_output()


if __name__ == "__main__":
    # Setup logging
    qscope.util.start_client_log(log_to_file=True, log_level=TEST_LOGLEVEL)

    # Create test instance
    test = TestElectromagnet()

    # Create SMU instance
    smu = SMU2450()
    smu.open()

    try:
        # Run test
        test.test_magnet_step_response(smu)
    finally:
        smu.close()
