import time

import matplotlib.pyplot as plt
import pyvisa
from loguru import logger


def plot_iv_curve(
    bipolar: bool = False, voltage: float = 1.0, num: int = 101, delay: float = 0.1
):
    """Demonstrate IV curve measurement and plotting with SMU.

    Args:
        bipolar: If True, performs measurement in both directions
        voltage: Maximum voltage for IV sweep in volts (default: 1.0V)
        num: Number of points for IV curve (default: 101)
        delay: Delay between points in seconds (default: 0.1s)
    """
    import numpy as np

    from qscope.device import SMU2450

    smu = SMU2450()

    try:
        # Open with default voltage source mode
        smu.open()
        smu.set_mode("voltage")
        smu.set_compliance(0.5)  # 500mA compliance

        # Measure IV curve with optional bipolar sweep
        voltages, currents = smu.measure_iv_curve(
            start=-voltage if bipolar else 0,  # Use provided voltage limit
            stop=voltage,
            points=num,  # More points for smoother curve
            delay=delay,
            bidirectional=bipolar,
        )

        # Calculate dynamic resistance, excluding points near zero
        voltages = np.array(voltages)
        currents = np.array(currents)

        # Define threshold relative to compliance current (0.5A)
        zero_threshold = 0.5 * 0.001  # 0.1% of compliance current

        # Create mask for valid points (away from zero)
        valid_mask = np.abs(currents) > zero_threshold

        # Calculate resistance only for valid points
        resistances = np.zeros_like(voltages)
        resistances[valid_mask] = np.abs(voltages[valid_mask] / currents[valid_mask])

        # Calculate average resistance excluding zero region
        avg_resistance = np.mean(resistances[valid_mask])

        # Create the plot
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # IV curve plot
        ax1.plot(voltages, currents, "b.-", label="IV Curve")
        if bipolar:
            mid = len(voltages) // 2
            ax1.plot(voltages[:mid], currents[:mid], "g.-", label="Forward")
            ax1.plot(voltages[mid:], currents[mid:], "r.-", label="Reverse")
        ax1.grid(True)
        ax1.set_xlabel("Voltage (V)")
        ax1.set_ylabel("Current (A)")
        ax1.set_title("IV Characteristic Curve")
        ax1.legend()

        # Resistance plot
        ax2.semilogy(
            voltages[valid_mask],
            resistances[valid_mask],
            "k.-",
            label=f"R_avg = {avg_resistance:.2e} Ω",
        )
        ax2.grid(True)
        ax2.set_xlabel("Voltage (V)")
        ax2.set_ylabel("Resistance (Ω)")
        ax2.set_title("Dynamic Resistance")
        ax2.legend()

        # Add some annotations
        plt.axhline(y=0, color="k", linestyle=":")
        plt.axvline(x=0, color="k", linestyle=":")

        # Format axis for better readability
        ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.error(f"Error during IV curve measurement: {e}")

    finally:
        if "smu" in locals():
            smu.zero_output()
            smu.close()


def measure_time_response():
    """Measure and analyze time response to current step."""
    import numpy as np
    from scipy import optimize

    from qscope.device import SMU2450

    smu = SMU2450()

    try:
        # Open and configure for current source mode
        smu.open()
        smu.configure(nplc=0.01)  # Fast measurements
        smu.set_mode("current")
        smu.set_compliance(20)  # 20V compliance for inductive kick

        # Setup measurement parameters
        target_current = 0.4  # 100mA step
        sample_rate = 1000  # Hz
        duration = 0.1  # seconds
        points = int(duration * sample_rate)

        # Arrays to store results
        times = []
        currents = []
        voltages = []
        start_time = time.time()

        # Enable output at zero
        smu.set_current(0)
        smu.set_output_state(True)
        time.sleep(0.1)

        # Apply current step and measure response
        smu.set_current(target_current)

        for i in range(points):
            t = time.time() - start_time
            v = smu.get_voltage()
            i = smu.get_current()
            times.append(t)
            voltages.append(v)
            currents.append(i)

            # Delay for sample rate
            time.sleep(1 / sample_rate)

        # Convert to numpy arrays
        times = np.array(times)
        voltages = np.array(voltages)
        currents = np.array(currents)

        # Fit exponential decay to estimate L/R time constant
        def exp_decay(t, tau, a, b):
            return a * (1 - np.exp(-t / tau)) + b

        try:
            popt, _ = optimize.curve_fit(exp_decay, times, voltages)
            tau, a, b = popt
            fit_v = exp_decay(times, tau, a, b)

            # Calculate inductance
            resistance = a / target_current  # V/I at steady state
            inductance = tau * resistance

            # Plot results
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Voltage response
            ax1.plot(times * 1000, voltages, "b.", label="Measured")
            ax1.plot(times * 1000, fit_v, "r-", label=f"Fit (τ = {tau * 1000:.1f} ms)")
            ax1.grid(True)
            ax1.set_xlabel("Time (ms)")
            ax1.set_ylabel("Voltage (V)")
            ax1.set_title("Step Response")
            ax1.legend()

            # Current profile
            ax2.plot(times * 1000, currents * 1000, "g.-")
            ax2.grid(True)
            ax2.set_xlabel("Time (ms)")
            ax2.set_ylabel("Current (mA)")
            ax2.set_title("Applied Current")

            plt.tight_layout()
            plt.show()

            logger.info(f"Estimated parameters:")
            logger.info(f"Time constant (τ): {tau * 1000:.1f} ms")
            logger.info(f"Resistance: {resistance:.2f} Ω")
            logger.info(f"Inductance: {inductance * 1000:.2f} mH")

        except RuntimeError as e:
            logger.error(f"Curve fitting failed: {e}")

    except Exception as e:
        logger.error(f"Error during measurement: {e}")

    finally:
        if "smu" in locals():
            smu.zero_output()
            smu.close()


def use_pyvisa():
    """Demonstrate basic PyVISA usage with SMU."""
    rm = pyvisa.ResourceManager()
    logger.info(f"Available resources: {rm.list_resources()}")

    try:
        # Open connection to SMU
        smu = rm.open_resource("USB0::1510::9296::04300231::0::INSTR")
        smu.timeout = 2000

        # Basic device info and reset
        logger.info(f"Device ID: {smu.query('*IDN?')}")
        smu.write("*RST")
        time.sleep(0.1)

        # Configure and test output
        smu.write("SOUR:FUNC VOLT")
        smu.write("SOUR:VOLT:RANG 20")
        smu.write("SOUR:VOLT:ILIM 0.1")
        smu.write("SOUR:VOLT 0")
        smu.write("OUTP ON")

        # Read back values
        voltage = float(smu.query("MEAS:VOLT?"))
        current = float(smu.query("MEAS:CURR?"))
        logger.info(f"Measured: {voltage:.3f}V, {current:.3e}A")

    finally:
        if "smu" in locals():
            smu.write("OUTP OFF")
            smu.close()
        rm.close()


def use_qscope(
    voltage: float = 1.0, ramp_rate: float = 0.5, visa_address: str | None = None
):
    """Demonstrate QScope SMU interface with advanced features.

    Args:
        voltage: Test voltage to ramp to (V)
        ramp_rate: Voltage ramp rate (V/s)
        visa_address: Optional VISA address of SMU
    """
    from qscope.device import SMU2450

    # Auto-discover or specify address
    smu = SMU2450(visa_address)

    try:
        # Open and configure
        smu.open()
        smu.configure(nplc=1.0, auto_zero=True)

        # Basic voltage sourcing
        smu.set_mode("voltage")
        smu.set_compliance(0.45)  # 450mA limit

        # Demonstrate ramped voltage changes
        logger.info("Testing voltage ramping...")
        smu.set_output_state(True)
        smu.set_voltage(0)  # Start at 0V
        time.sleep(0.5)

        # Ramp to target voltage at specified rate
        smu.set_voltage(voltage, ramp_rate=ramp_rate)
        logger.info(f"After ramp: {smu.get_voltage():.3f}V, {smu.get_current():.3e}A")

    except Exception as e:
        logger.error(f"Error during test: {e}")

    finally:
        if "smu" in locals():
            smu.zero_output()
            smu.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SMU example operations")
    parser.add_argument("--address", type=str, help="Optional VISA address of SMU")
    parser.add_argument(
        "--test",
        choices=["pyvisa", "qscope", "iv", "time", "all"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument(
        "--voltage",
        type=float,
        default=1.0,
        help="Test voltage for ramping example, and min/max for IV curve",
    )
    parser.add_argument(
        "--ramp-rate", type=float, default=0.5, help="Voltage ramp rate in V/s"
    )
    parser.add_argument(
        "--bipolar", action="store_true", help="Perform bipolar IV sweep"
    )
    parser.add_argument(
        "--num", type=int, default=101, help="Number of points for IV curve"
    )
    parser.add_argument(
        "--delay", type=float, default=0.1, help="Delay between IV curve points"
    )

    args = parser.parse_args()

    if args.test in ["pyvisa", "all"]:
        logger.info("\n=== Testing PyVISA Interface ===")
        use_pyvisa()

    if args.test in ["qscope", "all"]:
        logger.info("\n=== Testing QScope Interface ===")
        use_qscope(
            voltage=args.voltage, ramp_rate=args.ramp_rate, visa_address=args.address
        )

    if args.test in ["iv", "all"]:
        logger.info("\n=== Testing IV Curve Measurement and Plotting ===")
        plot_iv_curve(
            bipolar=args.bipolar, voltage=args.voltage, num=args.num, delay=args.delay
        )

    if args.test in ["time", "all"]:
        logger.info("\n=== Testing Time Response Measurement ===")
        measure_time_response()
