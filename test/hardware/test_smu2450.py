import time

import numpy as np
import pytest
import pyvisa
from loguru import logger

import qscope.util
from qscope.device import SMU2450
from qscope.util import TEST_LOGLEVEL
from qscope.util.check_hw import list_visa_devices


class TestSMU2450Local:
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
    def test_auto_discovery(self) -> None:
        """Test automatic device discovery."""
        logger.info("Testing auto-discovery")

        # Should work if device available
        try:
            smu = SMU2450()
            smu.open()
            logger.info(f"Successfully connected to {smu.visa_address}")
            smu.close()
        except RuntimeError as e:
            logger.warning(f"No device found: {e}")

        # Should still work with explicit address if valid
        rm = pyvisa.ResourceManager()
        try:
            devices = list_visa_devices(
                model_filter="MODEL 2450", detailed=False, resource_manager=rm
            )
            if devices:
                smu = SMU2450(next(iter(devices.keys())))
                smu.open()
                smu.close()
        finally:
            rm.close()

    @pytest.mark.usefixtures("client_log")
    def test_basic_voltage_source(self, smu: SMU2450) -> None:
        """Test basic voltage sourcing and measurement."""
        logger.info("Starting basic voltage source test")

        try:
            # Configure as voltage source
            smu.set_mode("voltage")
            smu.set_compliance(0.1)  # 100mA compliance

            # Test voltage setpoints
            test_voltages = [0, 1, -1, 0]
            measured_currents = []

            for voltage in test_voltages:
                smu.set_voltage(voltage)
                smu.set_output_state(True)
                time.sleep(0.1)  # Allow settling

                # Verify voltage setting
                measured_v = smu.get_voltage()
                assert abs(measured_v - voltage) < 0.01, f"Voltage error at {voltage}V"

                # Measure current
                measured_currents.append(smu.get_current())

            logger.info("Voltage source test complete")

        finally:
            smu.set_output_state(False)

    @pytest.mark.usefixtures("client_log")
    def test_voltage_ramp(self, smu: SMU2450) -> None:
        """Test voltage ramping functionality."""
        logger.info("Starting voltage ramp test")

        try:
            smu.set_mode("voltage")
            smu.set_compliance(0.1)

            # Test ramping
            start_v = 0
            target_v = 5
            ramp_rate = 1.0  # V/s

            smu.set_output_state(True)
            start_time = time.time()
            smu.set_voltage(target_v, ramp_rate=ramp_rate)
            end_time = time.time()

            # Verify ramp timing with allowed overhead
            elapsed = end_time - start_time
            expected_time = abs(target_v - start_v) / ramp_rate

            # Allow for communication overhead and safety delays
            overhead_factor = 1.5  # 50% overhead allowance
            max_allowed_time = (
                expected_time * overhead_factor + 1.0
            )  # Add 1s fixed overhead

            assert elapsed <= max_allowed_time, (
                f"Ramp too slow: took {elapsed:.1f}s, "
                f"expected <= {max_allowed_time:.1f}s"
            )

            # Verify we're not too fast (which would be unsafe)
            assert elapsed >= expected_time * 0.9, (
                f"Ramp too fast: took {elapsed:.1f}s, "
                f"expected >= {expected_time * 0.9:.1f}s"
            )

            # Verify final voltage
            measured_v = smu.get_voltage()
            assert abs(measured_v - target_v) < 0.01, "Final voltage error"

        finally:
            smu.zero_output()

    @pytest.mark.usefixtures("client_log")
    def test_iv_curve(self, smu: SMU2450) -> None:
        """Test IV curve measurement."""
        logger.info("Starting IV curve measurement test")

        try:
            smu.set_mode("voltage")
            smu.set_compliance(0.5)

            # Enable output and set initial voltage
            smu.set_voltage(0)  # Start at 0V
            smu.set_output_state(True)
            time.sleep(0.1)  # Allow settling

            # Perform IV sweep
            start_v = -1
            stop_v = 1
            points = 21

            voltages, currents = smu.measure_iv_curve(
                start=start_v,
                stop=stop_v,
                points=points,
                delay=0.1,
            )

            # Verify data properties
            assert len(voltages) == points, "Wrong number of voltage points"
            assert len(currents) == points, "Wrong number of current points"

            # Check voltages with appropriate tolerance for instrument precision
            rtol = 1e-3  # 0.1% relative tolerance
            atol = 1e-3  # 1mV absolute tolerance
            assert np.allclose(voltages[0], start_v, rtol=rtol, atol=atol), (
                f"Wrong start voltage: got {voltages[0]}, expected {start_v}"
            )
            assert np.allclose(voltages[-1], stop_v, rtol=rtol, atol=atol), (
                f"Wrong stop voltage: got {voltages[-1]}, expected {stop_v}"
            )

            # Verify voltage spacing with appropriate tolerance
            voltage_steps = np.diff(voltages)
            expected_step = (stop_v - start_v) / (points - 1)
            assert np.allclose(voltage_steps, expected_step, rtol=rtol, atol=atol), (
                f"Non-uniform voltage steps: expected {expected_step:.3f}V steps, "
                f"got steps ranging from {min(voltage_steps):.3f}V to {max(voltage_steps):.3f}V"
            )

        finally:
            smu.zero_output()

    @pytest.mark.usefixtures("client_log")
    def test_error_handling(self, smu: SMU2450) -> None:
        """Test error handling and safety features."""
        # Test compliance
        smu.set_mode("voltage")
        smu.set_compliance(0.001)  # 1mA compliance
        smu.set_voltage(1)
        smu.set_output_state(True)
        time.sleep(0.1)

        # Check compliance state
        if smu.check_compliance():
            logger.warning("Device in compliance as expected")

        # Test abort functionality
        smu.abort()
        assert not smu.get_output_state(), "Output not disabled after abort"

    @pytest.mark.usefixtures("client_log")
    def test_device_discovery(self) -> None:
        """Test device discovery functionality."""
        logger.info("Checking for available SMU2450 devices")

        # List available devices
        devices = list_visa_devices(model_filter="MODEL 2450", detailed=False)

        # Log found devices
        if devices:
            logger.info(f"Found {len(devices)} SMU2450 device(s):")
            for addr, idn in devices.items():
                logger.info(f"  {addr}: {idn}")
        else:
            logger.warning("No SMU2450 devices found")

        # Verify return type
        assert isinstance(devices, dict), "Expected dict return type"

        # Verify device info format if any found
        for addr, idn in devices.items():
            assert isinstance(addr, str), "Expected string VISA address"
            assert isinstance(idn, str), "Expected string IDN"

    @pytest.mark.usefixtures("client_log")
    def test_configuration(self, smu: SMU2450) -> None:
        """Test device configuration settings."""
        # Test NPLC setting
        smu.configure(nplc=1.0)
        assert smu._nplc == 1.0, "NPLC not set correctly"

        # Test invalid NPLC
        with pytest.raises(ValueError):
            smu.configure(nplc=11.0)  # Above 10 NPLC limit

        # Test auto-zero setting
        smu.configure(auto_zero=True)
        assert smu._auto_zero is True, "Auto-zero not set correctly"

        # Test source delay
        smu.configure(source_delay=0.1)
        assert smu._source_delay == 0.1, "Source delay not set correctly"


if __name__ == "__main__":
    # Setup logging when run directly
    qscope.util.start_client_log(log_to_file=True, log_level=TEST_LOGLEVEL)
    test = TestSMU2450Local()

    # Create SMU instance
    # smu = SMU2450("USB0::0x05E6::0x2450::04424197::INSTR")
    smu = SMU2450()

    # Run test
    test.test_basic_voltage_source(smu)
    test.test_voltage_ramp(smu)
    test.test_iv_curve(smu)
    test.test_configuration(smu)
