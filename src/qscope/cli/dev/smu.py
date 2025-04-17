import time

import click
from loguru import logger

from qscope.cli.base import tree_option
from qscope.device import SMU2450
from qscope.util.check_hw import check_smu_available

# TODO in future allow other types of SMUs?


@click.group()
@tree_option
def smu():
    """SourceMeasureUnit control commands."""
    pass


@smu.command()
@click.option("--voltage", "-v", type=float, help="Set output voltage")
@click.option("--current", "-c", type=float, help="Set output current")
@click.option("--ramp-rate", "-r", type=float, help="Optional ramp rate (V/s or A/s)")
@click.option(
    "--compliance",
    type=float,
    help="Set compliance limit (A for voltage mode, V for current mode)",
)
@click.option("--address", "-a", type=str, default=None, help="Optional VISA address")
def set(voltage, current, ramp_rate, compliance, address):
    """Set SMU output voltage or current with optional compliance."""
    if voltage is not None and current is not None:
        raise click.UsageError("Cannot set both voltage and current")
    if voltage is None and current is None:
        raise click.UsageError("Must specify either voltage or current")

    if not check_smu_available(address):
        raise click.UsageError("No Keithley 2450 SMU found")

    smu = SMU2450(address)
    try:
        smu.open()
        if voltage is not None:
            smu.set_mode("voltage")
            if compliance:
                smu.set_compliance(compliance)
            smu.set_output_state(True)
            smu.set_voltage(voltage, ramp_rate=ramp_rate)
            click.echo(f"Set voltage to {voltage}V")
            if compliance:
                click.echo(f"Current compliance: {compliance}A")
        else:
            smu.set_mode("current")
            if compliance:
                smu.set_compliance(compliance)
            smu.set_output_state(True)
            smu.set_current(current, ramp_rate=ramp_rate)
            click.echo(f"Set current to {current}A")
            if compliance:
                click.echo(f"Voltage compliance: {compliance}V")
    finally:
        # if 'smu' in locals():
        # smu.close()
        pass


@smu.command()
def zero():
    """Safely zero the SMU output and disable."""
    if not check_smu_available():
        raise click.UsageError("No Keithley 2450 SMU found")

    smu = SMU2450()
    try:
        smu.open()
        smu.zero_output()
        click.echo("SMU output zeroed and disabled")
    finally:
        if "smu" in locals():
            smu.close()


@smu.command()
@click.option("--address", "-a", type=str, default=None, help="Optional VISA address")
@click.option(
    "--watch/--no-watch", "-w/", default=False, help="Continuously monitor values"
)
@click.option(
    "--interval",
    "-i",
    type=float,
    default=1.0,
    help="Update interval for watch mode (seconds)",
)
def read(address, watch, interval):
    """Read current SMU voltage and current with optional continuous monitoring."""
    if not check_smu_available(address):
        raise click.UsageError("No Keithley 2450 SMU found")

    smu = SMU2450(address)
    try:
        smu.open()

        def display_values():
            v = smu.get_voltage()
            i = smu.get_current()
            mode = smu.get_mode()
            enabled = smu.get_output_state()
            compliance = smu.check_compliance()
            click.echo(f"\nSMU Status:")
            click.echo(f"Mode: {mode}")
            click.echo(f"Output: {'enabled' if enabled else 'disabled'}")
            click.echo(f"Voltage: {v:.6f} V")
            click.echo(f"Current: {i * 1000:.6f} mA")
            if compliance:
                click.echo("WARNING: Compliance limit reached!")
            click.echo("")

        if watch:
            click.echo("Press Ctrl+C to stop monitoring")
            try:
                while True:
                    display_values()
                    time.sleep(interval)
            except KeyboardInterrupt:
                click.echo("\nMonitoring stopped")
        else:
            display_values()
    finally:
        pass
        # if 'smu' in locals():
        # smu.close()
