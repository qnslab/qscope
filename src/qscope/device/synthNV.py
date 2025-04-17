# Class for controlling Windfreak SynthNV
# author: David Broadway
import time
from typing import Sequence

import serial  # pyserial package
from loguru import logger

from qscope.device import Device
from qscope.util import format_error_response, get_hw_ports


class SynthNV(Device):
    connected = False
    port: str  # "COM3" etc., com port for the windfreak
    required_config = {"port": str}

    def __init__(self, port: str):
        super().__init__(port=port)
        self.port = port

        self.cmd_wait = (
            10e-6  # time (s) to wait after sending a command to the windfreak
        )

        # set the default values
        self._freq = 2870
        self._power = -30

    def open(self) -> tuple[bool, str]:
        """
        Opens the serial port connection to the RF source.

        If the port is already opened, it will be closed and opened again.
        If the connection is successful, the `rf` attribute will be set to the opened serial port.

        Returns:
            None: If an unknown error occurs while opening the serial port.
        """
        try:
            self.rf = serial.Serial(self.port)
            logger.info("Connected to RF source: windfreak on port {}", self.port)
        except serial.serialutil.SerialException:
            logger.exception("Error opening winfreak serial port.")
            self.connected = False
            return (
                False,
                f"Error opening winfreak serial port: {format_error_response()}",
            )
        except IOError:
            try:
                self.rf = serial.Serial(self.port)
                # self.rf.open()
                logger.info("Connected to RF source: windfreak on port: {}", self.port)
            except:
                logger.exception("Error opening winfreak serial port.")
                self.connected = False
                return (
                    False,
                    f"Error opening winfreak serial port: {format_error_response()}",
                )
        # save this port info so later we check if connected is still made
        ports = get_hw_ports()
        self.port_info = [
            portinfo for port, portinfo in ports.items() if self.port in port
        ][0]
        self.connected = True
        return True, "Connected to RF source: windfreak on port " + self.port

    def close(self):
        """
        Closes the connection to the RF device.
        """
        if self.is_connected():
            self.rf.close()

    def reconnect(self):
        if self.is_connected():
            self.rf.close()
            self.rf = serial.Serial(self.port)

    def is_connected(self) -> bool:
        hw_port_info = get_hw_ports()
        return (
            self.port in hw_port_info and self.port_info == hw_port_info[self.port]
            if hasattr(self, "port_info")
            else False and self.connected
        )

    def _get_version(self):  # untested
        self.rf.write(b"+")
        time.sleep(self.cmd_wait)
        return int(self.rf.read_all())

    def start_fm_mod(self, freq):
        """
        Start frequency modulation.

        Args:
            freq (float): Frequency in Hz.
        """
        self.write_command("<100>" + str(freq) + ",100;0/1y9")

    def stop_fm_mod(self):
        """
        Stops the frequency modulation (FM) modulation.

        This method sends the command '/0y3' to stop the FM modulation.

        Parameters:
            None

        Returns:
            None
        """
        self.write_command("/0y3")

    ###################################################################
    # set/get
    ###################################################################
    # NOTE we never really need the 'gets' here. Superfluous

    def get_freq(self):
        self.rf.read_all()
        self.rf.write(b"f?")
        time.sleep(self.cmd_wait)
        self._freq = float(self.rf.read_all())
        return self._freq

    def set_freq(self, freq):
        # freq in MHz
        logger.debug("Setting frequency to {} type {}", freq, type(freq))
        self.write_command("f" + str(freq))
        time.sleep(self.cmd_wait)
        self._freq = freq

    def get_power(self):
        self.rf.read_all()
        self.rf.write(b"&0w")
        time.sleep(3 * self.cmd_wait)
        self._power = self.rf.read_all()
        # convert from bytes to string
        # self._power = self._power.decode("utf-8")
        return self._power

    def set_power(self, power):
        logger.debug("Setting power to {} type {}", power, type(power))
        # power in dBm
        self.write_command("W" + str(power))
        time.sleep(self.cmd_wait)
        self._power = power

    def get_state(self):
        self.rf.read_all()
        self.rf.write(b"E?")
        time.sleep(self.cmd_wait)
        self._output = int(self.rf.read_all())
        return self._output

    def set_state(self, state):
        if state == 1 or state == True:
            self.rf.write(b"E1h1")
            logger.debug("Setting output to ON")
        else:
            self.rf.write(b"E0h0")
            logger.debug("Setting output to OFF")
        time.sleep(self.cmd_wait)
        self._output = state

    # change name?
    # def query(self, command):
    #     self.rf.read_all()
    #     self.rf.write(bytes(command, "utf-8"))
    #     time.sleep(500 * self.cmd_wait)
    #     return self.rf.read_all()

    ###################################################################
    # queries
    ###################################################################

    def poll_device_attrs(self):
        """
        Retrieves all attributes from the device.

        This method reads all attributes from the device by sending a command and
        waiting for the device to respond. It then parses the response and returns
        a dictionary of the attributes.

        Returns:
            dict: A dictionary containing the attributes of the device.
        """
        self.rf.read_all()
        self.rf.write(b"e?")
        # long wait to make sure the device has time to respond
        time.sleep(20 * self.cmd_wait)
        logger.info("Asking winfreak synthNV for all attributes")
        current_status = self.rf.read_until(expected=b"EOM.")
        # break the string into lines
        lines = (current_status).decode("ascii").split("\n")
        # Make into a dictionary of the attributes
        attr_dict = dict()
        for line in lines:
            if line[1] == ")":
                attr_dict[line[0]] = line[2::]
        logger.info("SynthNV attrs: {}", attr_dict)
        return attr_dict

    ###################################################################
    # Sweep functions
    ###################################################################

    def set_freq_list(self, rf_freqs: Sequence, step_time=0.1):
        self.set_f_table(rf_freqs, self._power, step_time)
        return rf_freqs

    def set_f_table(self, freq_list, power, step_time=0.1):
        # command is '@fap' @ frequency a power
        # set the windfreak delay to default
        command = ""
        # Add the sweep mode to tabular command
        command += "X1"
        # Add the frequency list
        for i in range(len(freq_list)):
            if i < 10:
                command += (
                    "L0"
                    + str(i)
                    + "f"
                    + str(freq_list[i])
                    + "L0"
                    + str(i)
                    + "a"
                    + str(power)
                )
            else:
                command += (
                    "L"
                    + str(i)
                    + "f"
                    + str(freq_list[i])
                    + "L"
                    + str(i)
                    + "a"
                    + str(power)
                )
        # Add the last frequency to the end of the list
        command += (
            "L"
            + str(len(freq_list))
            + "f"
            + str(0)
            + "L"
            + str(len(freq_list))
            + "a"
            + str(0)
        )

        # Add command to make the sweep from the start of the table
        command += "^1"
        # FIXME: Set the stepping time to equal the camera trigger time.
        # define the step timming
        command += "t" + str(step_time * 1e3)  # convert to ms

        # Don't sweep automatically
        command += "g0"
        # set sweep mode to continuous
        command += "c1"
        # set trigger to step
        command += "y2"
        # Turn the output on
        command += "E1h1"

        # return command
        self.write_command(command)

    def set_trigger(self, mode=2):
        """
        Sets the trigger mode for the Windfreak SynthNV device.

        Parameters:
            trigger (int): The trigger mode to set. Valid values are:
                - 0: Software trigger
                - 1: Sweep trigger
                - 2: Step trigger
                - 3: Hold all trigger

        Returns:
            None
        """
        self.write_command("y" + str(mode))

    def start_sweep(self):
        """
        Starts the sweep run.

        This method sends a command to start the sweep run on the Windfreak SynthNV device.

        Parameters:
            None

        Returns:
            None
        """
        # g) Sweep run (on=1 / off=0) 0
        self.write_command("g0y2")

    def reset_sweep(self):
        """
        Resets the sweep settings of the Windfreak SynthNV device.
        """
        self.write_command("g0y2")

    def stop_sweep(self):
        """
        Stops the sweep run.

        This method sends a command to stop the sweep run of the Windfreak SynthNV device.

        Parameters:
            None

        Returns:
            None
        """
        # g) Sweep run (on=1 / off=0) 0
        self.write_command("g0E0h0")

    ###################################################################
    # Utility functions
    ###################################################################

    def write_command(self, command):
        """
        Writes a string command to the serial port after converting it into a byte command.

        Args:
            command (str): The command to be written to the serial port.

        Returns:
            None
        """
        self.rf.write(bytes(command, "utf-8"))
        time.sleep(self.cmd_wait)
