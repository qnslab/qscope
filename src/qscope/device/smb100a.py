# Class for controlling smb100a
# author: David Broadway
import time
from typing import Sequence

import pyvisa
from loguru import logger

from qscope.device import Device


class SMB100a(Device):
    visa_addr: str
    required_config = {"visa_addr": str}

    def __init__(self, visa_addr="US?*::0x0AAD::0x0054::108563::INSTR"):
        super().__init__(visa_addr=visa_addr)

        # set default device parameters
        self.cmd_wait = 0.1  # wait time after sending instruction

    def open(self) -> bool:
        self.rm = pyvisa.ResourceManager()
        if (self.rm.list_resources(self.visa_addr)[0]) == None:
            raise Exception("Rohde&Schwarz SMB100A instrument not connected")
        self.SMB100A_addr = self.rm.list_resources(self.visa_addr)[0]
        print(self.SMB100A_addr)
        a = self.rm.open_resource(self.SMB100A_addr)
        a.close()
        self.timeout = 3000
        self.commands = []

        # need to handle errors here
        self.check_connection()
        return True

    def reconnect(self):
        self.rm.close()
        self.open()

    def check_connection(
        self,
    ):
        try:
            with self.rm.open_resource(self.SMB100A_addr) as sg:
                sg.timeout = self.timeout
                msg = sg.query("*IDN?")
            if "SMB100A" not in msg:
                raise Exception(
                    "Device not responding correctly to identification \n\tIDN:%s" % msg
                )
        except pyvisa.VisaIOError:
            logger.exception("SMB100a Instrument disconnected, check USB connection.")
            time.sleep(0.5)

    def close(self):
        self.rm.close()

    ###################################################################
    # set/get
    ###################################################################
    # NOTE we never really need the 'gets' here. Superfluous

    def get_freq(self):
        self.commands.append("SOUR:FREQ?")
        self._freq = float(self.query()) * 1e-6  # return in MHz
        return self._freq

    def set_freq(self, freq):
        # freq in MHz
        self.commands.append("FREQ %.3f MHz" % freq)
        self.send_commands()
        self._freq = freq

    def get_pow(self):
        self.commands.append("SOUR:POW?")
        self._power = float(self.query())
        return self._power

    def set_power(self, power):
        # power in dBm
        self.commands.append("SOUR:POW:LEV:IMM:AMPL %.1f dBM" % power)
        self.send_commands()
        self._power = power

    def get_state(self):
        self.commands.append("OUTP?")
        result = int(self.query())
        self._output = result
        return result

    def set_state(self, state):
        if state == 1:
            self.commands.append("OUTP ON")
        else:
            self.commands.append("OUTP OFF")
        self.send_commands()
        self._output = state

    def get_freq_list(self):
        return self._freq_list

    def set_freq_list(self, rf_freqs: Sequence, step_time=0):
        self.set_f_list(rf_freqs)
        return self._freq_list

    ###################################################################
    # Sweep functions
    ###################################################################

    def set_f_list(self, f_list, p_list=None, power=None):
        """Set the signal generator in list mode.
        writes a list of frequencies and powers and set the system
        to going through the list by an external trigger

         fqlist: list of frequency in MHz
         pwlist: list of powers in dBm

        """
        if p_list is None:
            if power is None:
                power = self._power
            p_list = [power for idx in range(len(f_list))]

        if len(f_list) != len(p_list):
            print("frequency and power lists have different lengths")
            raise Exception("frequency and power lists have different lengths")
        else:
            fq_list, pw_list = "", ""
            for i, j in zip(f_list, p_list):
                fq_list = fq_list + " %.3f MHz," % i
                pw_list = pw_list + " %.1f dBm," % j

            self.commands.append("SOUR:LIST:FREQ %s" % fq_list[:-1])
            self.commands.append("SOUR:LIST:POW %s" % pw_list[:-1])
            self.commands.append("SOUR:LIST:DWEL 1ms")
            self.commands.append("SOUR:LIST:TRIG:SOUR EXT")
            self.commands.append("SOUR:LIST:MODE STEP")
            self.commands.append("FREQ:MODE LIST")
            self.send_commands()
            self._freq_list = f_list
            print("Set rf frequency list")
        return self._freq_list

    def set_trigger(self, mode=2):
        # This is taken care of in the RF_list function
        return

    def reset_sweep(self, *args, **kwargs):
        # The SMB100A will reset with the start command so this is not needed
        return

    ###################################################################
    # Utility functions
    ###################################################################

    def send_commands(self):
        """function to send the list of command to the SMB100A"""
        rm = pyvisa.ResourceManager()
        try:
            with rm.open_resource(self.SMB100A_addr) as v:
                v.timeout = self.timeout
                for c in self.commands:
                    v.write(str(c))
                self.commands = []
        except pyvisa.VisaIOError:
            print("Instrument not responding, check USB connection")
        time.sleep(self.cmd_wait)

    def query(self):
        rm = pyvisa.ResourceManager()
        with rm.open_resource(self.SMB100A_addr) as sg:
            sg.timeout = self.timeout
            for c in self.commands:
                msg = sg.query(str(c))
        return msg
