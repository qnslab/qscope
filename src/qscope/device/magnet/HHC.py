import time

import numpy as np
import serial
from loguru import logger
from serial.serialutil import SerialBase

from qscope.device.magnet.magnet import Magnet
from qscope.util import format_error_response

#### WRITTEN ASSUMING ONLY POSITIVE CURRENT AND VOLTAGES!!!!!


# From APS100 key changes
# xlim changed to ixlim, etc for y and z
# add ivxlim, ivylim, ivzlim, for overpotection limits
# HHC has 3 supplies
# PSU can only provide positive current and voltages
# Currently want to run in CC mode, so set Voltage to V overprotection limit, user can't change
# Need to make functions with serial as input internal i.e. start with "_"
# B is in Gauss and angles are all in degrees

# Sam: We need to allow for the contingency that one of the PSU is dead
# Sam: Do we want set and gets to be separate naming or just use the property decorator?
# i.e iz_set iz_get or iz = and iz
# Sam currently reading measured volts and current, not set values
# Do we need to add the option to read desired values? reading iz if output is off is 0 A.
# Sam once we set a current value, if it is different to the desired value
# do we want to add a fine tuning loop where we add or substract 0.0001 A from the
# desired value, measure the actual value and repeat until we are within 0.001 A of the desired value?
# Sam: PSU adjust sequentially when setting b_sph, do we want to adjust all at once using asyncio?
# Sam: I haven't done logging

##### Adjust system values:
# Current limits --> poor naming choice use ixlim, etc
# magnet_info["ixlim"] # 5 A
# magnet_info["iylim"] # 5 A
# magnet_info["izlim"] # 5 A

# magnet_info["vxlim"] # 18 V
# magnet_info["vylim"] # 10 V
# magnet_info["vzlim"] # 5 V


# # Define the conversion rates for the system. This should be in the system info
# # Rather than in the device class.
# system.magnet_info["ix2bx"]  # A -> mT 30/4.57
# system.magnet_info["iy2by"]  # A -> mT 30/4.6
# system.magnet_info["iz2bz"]  # A -> mT 30/4.6

# system.magnet_info["ixrate"] = 0.2 # A/s # Revise
# system.magnet_info["iyrate"] = 0.2 # A/s
# system.magnet_info["izrate"] = 0.2 # A/s

# system.magnet_info["istep"] = 0.2 # A

# system.ports["magnet_x"] = COM11
# system.ports["magnet_y"] = COM12
# system.ports["magnet_z"] = COM13


class HHC(Magnet):
    x_port: str
    y_port: str
    z_port: str
    ixlim: float | int
    iylim: float | int
    izlim: float | int
    vxlim: float | int
    vylim: float | int
    vzlim: float | int
    ix2bx: float | int  # A -> mT
    iy2by: float | int
    iz2bz: float | int
    ixrate: float | int  # A/s
    iyrate: float | int
    izrate: float | int
    istep: float | int

    # required_config = {
    #     "x_port": str,
    #     "y_port": str,
    #     "z_port": str,
    #     "ixlim": float | int,
    #     "iylim": float | int,
    #     "izlim": float | int,
    #     "vxlim": float | int,
    #     "vylim": float | int,
    #     "vzlim": float | int,
    #     "ix2bx": float | int,
    #     "iy2by": float | int,
    #     "iz2bz": float | int,
    #     "ixrate": float | int,
    #     "iyrate": float | int,
    #     "izrate": float | int,
    #     "istep": float | int,
    # }

    def __init__(
        self,
        x_port: str,
        y_port: str,
        z_port: str,
        ixlim: float | int,
        iylim: float | int,
        izlim: float | int,
        vxlim: float | int,
        vylim: float | int,
        vzlim: float | int,
        ix2bx: float | int,
        iy2by: float | int,
        iz2bz: float | int,
        ixrate: float | int,
        iyrate: float | int,
        izrate: float | int,
        istep: float | int,
    ):
        super().__init__()

        self.x_port = x_port
        self.y_port = y_port
        self.z_port = z_port
        self._ixlim = ixlim
        self._iylim = iylim
        self._izlim = izlim
        self._vxlim = vxlim
        self._vylim = vylim
        self._vzlim = vzlim
        self._ix2bx = ix2bx
        self._iy2by = iy2by
        self._iz2bz = iz2bz
        self._ixrate = ixrate
        self._iyrate = iyrate
        self._izrate = izrate
        self._istep = istep

        self.cmd_wait = 1e-3  # seconds

        self._ix_state = 0
        self._iy_state = 0
        self._iz_state = 0

    def open(self):
        # connect to the 3 current sources.
        try:
            self._mag_x = serial.Serial(
                self.x_port,
                baudrate=115200,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1,
            )
            resp = self.query_command(self._mag_x, "*IDN?\n")
            if not resp:
                raise RuntimeError("No response from power supply.")
            logger.info(
                "Connected to power supply for x-axis coil: on port {}", self.x_port
            )
        except serial.serialutil.SerialException:
            logger.exception("Error opening power supply for x-axis coil serial port.")
            return (
                False,
                f"Error opening power supply for x-axis coil serial port: {format_error_response()}",
            )

        try:
            self._mag_y = serial.Serial(
                self.y_port,
                baudrate=115200,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1,
            )
            resp = self.query_command(self._mag_y, "*IDN?\n")
            if not resp:
                raise RuntimeError("No response from power supply.")
            logger.info(
                "Connected to power supply for y-axis coil: on port {}", self.y_port
            )
        except serial.serialutil.SerialException:
            logger.exception("Error opening power supply for y-axis coil serial port.")
            return (
                False,
                f"Error opening power supply for y-axis coil serial port: {format_error_response()}",
            )

        try:
            self._mag_z = serial.Serial(
                self.z_port,
                baudrate=115200,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1,
            )
            resp = self.query_command(self._mag_z, "*IDN?\n")
            if not resp:
                raise RuntimeError("No response from power supply.")
            logger.info(
                "Connected to power supply for z-axis coil: on port {}", self.z_port
            )
        except serial.serialutil.SerialException:
            logger.exception("Error opening power supply for z-axis coil serial port.")
            return (
                False,
                f"Error opening power supply for z-axis coil serial port: {format_error_response()}",
            )

        self.initial_psu(self._mag_x, self._ixlim, self._vxlim)
        self.initial_psu(self._mag_y, self._iylim, self._vylim)
        self.initial_psu(self._mag_z, self._izlim, self._vzlim)

    def is_connected(self) -> bool:
        # check if the serial ports are open
        # return self._mag_x.is_open and self._mag_y.is_open and self._mag_z.is_open
        return self._mag_z.is_open

    def initial_psu(self, ser: SerialBase, ilim: int | float, vlim: int | float):
        self.write_command(ser, "OUTP 0\n")
        self.i_prot_set(ser, ilim)
        self.v_prot_set(ser, vlim)
        self.write_command(ser, f"CURR 0.00\n")  # set current to 0 A
        self.write_command(ser, f"VOLT {vlim:.3f}\n")  # force cc mode

    def stop(self):
        # Step power to 0 W via current then turn off, each PSU
        self.output_off(self._mag_x, self._ixlim, self._ixrate)
        self.output_off(self._mag_y, self._iylim, self._iyrate)
        self.output_off(self._mag_z, self._izlim, self._izrate)

    def close(self):
        self.stop()
        self._mag_x.close()
        self._mag_y.close()
        self._mag_z.close()

    def output_on(self, ser: SerialBase, ilim: int | float, irate: int | float):
        # set current to 0 A, then to previous value
        if int(float(self.query_command(ser, "OUTP?\n"))):
            logger.debug("HHC Magnet: Output already on")
        else:
            # i_prev = self.i_read(ser)
            i_prev = float(self.query_command(ser, f"SOUR:CURR?\n"))
            self.write_command(ser, f"CURR 0.0\n")
            self.write_command(ser, "OUTP 1\n")
            self.i_sweep(ser, ilim, i_prev, irate)

    def output_off(self, ser: SerialBase, ilim: int | float, irate: int | float):
        if int(float(self.query_command(ser, "OUTP?\n"))):
            self.i_sweep(ser, ilim, 0, irate)
            self.write_command(ser, "OUTP 0\n")
        else:
            logger.debug("HHC Magnet: Output already off")

    def i_sweep(
        self, ser: SerialBase, ilim: int | float, iend: int | float, irate: int | float
    ):
        if abs(iend) > ilim:
            logger.debug("HHC Magnet: CURRENT VALUE IS TOO HIGH. DID NOT CHANGE VALUE!")
            return

        # If output is on, sweep current, else set current to iend
        if int(float(self.query_command(ser, "OUTP?\n"))):
            istart = float(self.query_command(ser, "SOUR:CURR?\n"))
            num = int(np.ceil(abs(iend - istart) / self._istep))
            vals = np.linspace(istart, iend, num + 1)[1:]
            # added if to prevent repeated attempts to set to same value
            # difference between set and actual value is often > 0.001, so this fails
            if abs(iend - istart) > 0.001:
                for val in vals:
                    self.write_command(ser, f"CURR {val:.3f}\n")
                    time.sleep(max(self._istep / irate - self.cmd_wait, 0.001))

                time.sleep(0.1)
                # If the current is not set to the desired value, set it to the desired value
                count = 0
                inew = iend
                while abs(iend - self.i_read(ser) > 0.001) & (count < 10):
                    count += 1
                    if iend - self.i_read(ser) > 0.001:
                        inew = inew + 0.001
                    elif self.i_read(ser) - iend > 0.001:
                        inew = inew - 0.001

                    self.write_command(ser, f"CURR {inew:.3f}\n")
                    time.sleep(0.1)

                if count == 10:
                    logger.error("Failed to set current to desired value")
        else:
            self.write_command(ser, f"CURR {iend:.3f}\n")

    ###################################################################
    # Get functions
    ###################################################################
    def i_read(self, ser: SerialBase):
        return float(self.query_command(ser, "MEAS:CURR?\n"))

    def v_read(self, ser: SerialBase):
        return float(self.query_command(ser, "MEAS:VOLT?\n"))

    @property
    def ix(self):
        # Query the current value of the x magnet
        self._ix = self.i_read(self._mag_x)
        return self._ix

    @property
    def iy(self):
        # Query the current value of the y magnet
        self._iy = self.i_read(self._mag_y)
        return self._iy

    @property
    def iz(self):
        # Query the current value of the z magnet
        self._iz = self.i_read(self._mag_z)
        return self._iz

    @property
    def vx(self):
        # Query the voltage value of the x magnet
        self._vx = self.v_read(self._mag_x)
        return self._vx

    @property
    def vy(self):
        # Query the voltage value of the y magnet
        self._vy = self.v_read(self._mag_y)
        return self._vy

    @property
    def vz(self):
        # Query the voltage value of the z magnet
        self._vz = self.v_read(self._mag_z)
        return self._vz

    # B field read last set values, can update by running update_params
    @property
    def bx(self):
        self._bx = self._ix / self._ix2bx
        return self._bx

    @property
    def by(self):
        self._by = self._iy / self._iy2by
        return self._by

    @property
    def bz(self):
        self._bz = self._iz / self._iz2bz
        return self._bz

    @property
    def ix_state(self):
        self._ix_state = self.query_command(self._mag_x, "OUTP?\n")
        return self.ix_state

    @property
    def iy_state(self):
        self._iy_state = self.query_command(self._mag_y, "OUTP?\n")
        return self._iy_state

    @property
    def iz_state(self):
        self._iz_state = self.query_command(self._mag_z, "OUTP?\n")
        return self._iz_state

    ###################################################################
    # Set functions
    ###################################################################
    def i_prot_set(self, ser: SerialBase, ilim: float | int):
        self.write_command(ser, f"CURR:PROT {ilim:.3f}\n")
        logger.debug("Setting current overprotection to {} A", ilim)

    def v_prot_set(self, ser: SerialBase, vlim: float | int):
        self.write_command(ser, f"VOLT:PROT {vlim:.3f}\n")
        logger.debug("Setting voltage overprotection to {} V", vlim)

    @ix.setter  #### update logging
    def ix(self, val: float | int):
        # check if the requested current vaules are within the limits
        if abs(val) > self._ixlim:
            print("Ix CURRENT VALUE IS TOO HIGH. DID NOT CHANGE VALUE")
            return

        self.i_sweep(self._mag_x, self._ixlim, abs(val), self._ixrate)
        self._ix = float(self.query_command(self._mag_x, "CURR?\n"))

    @iy.setter  #### update logging
    def iy(self, val: float | int):
        # check if the requested current vaules are within the limits
        if abs(val) > self._iylim:
            print("Iy CURRENT VALUE IS TOO HIGH. DID NOT CHANGE VALUE")
            return

        self.i_sweep(self._mag_y, self._iylim, abs(val), self._iyrate)
        self._iy = float(self.query_command(self._mag_y, "CURR?\n"))

    @iz.setter  #### update logging
    def iz(self, val: float | int):
        # check if the requested current vaules are within the limits
        if abs(val) > self._izlim:
            print("Iz CURRENT VALUE IS TOO HIGH. DID NOT CHANGE VALUE")
            return

        self.i_sweep(self._mag_z, self._izlim, abs(val), self._izrate)
        self._iz = float(self.query_command(self._mag_z, "CURR?\n"))

    @ix_state.setter
    def ix_state(self, val: int):
        if val:
            self.output_on(self._mag_x, self._ixlim, self._ixrate)
        else:
            self.output_off(self._mag_x, self._ixlim, self._ixrate)

    @iy_state.setter
    def iy_state(self, val: int):
        if val:
            self.output_on(self._mag_y, self._iylim, self._iyrate)
        else:
            self.output_off(self._mag_y, self._iylim, self._iyrate)

    @iz_state.setter
    def iz_state(self, val: int):
        if val:
            self.output_on(self._mag_z, self._izlim, self._izrate)
        else:
            self.output_off(self._mag_z, self._izlim, self._izrate)

    # Set current to achieve desired field, update _b
    @bx.setter
    def bx(self, val: float | int):
        self.ix = abs(val) / self._ix2bx
        self._bx = abs(val)

    @by.setter
    def by(self, val: float | int):
        self.iy = abs(val) / self._iy2by
        self._by = abs(val)

    @bz.setter
    def bz(self, val: float | int):
        self.iz = abs(val) / self._iz2bz
        self._bz = abs(val)

    ##########################################
    #      b spherical properties
    ##########################################
    @property
    def b_sph(self):
        # use previously determined bx, by, bz values,
        # run magnet.update_params to update based on actual current values
        self._bnorm, self._theta, self._phi = self.cart2sph(
            self._bx, self._by, self._bz
        )
        return self._bnorm, self._theta, self._phi

    @b_sph.setter
    def b_sph(self, value):
        try:
            bnorm, theta, phi = value
        except ValueError:
            raise ValueError("Pass an iterable with B norm (G), theta (°), and phi (°)")
        # b is in Gauss and assume degrees
        # Set bx, by, bz
        self.bx = bnorm * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
        self.by = bnorm * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
        self.bz = bnorm * np.cos(np.deg2rad(theta))

    ###################################################################
    # Utility functions
    ###################################################################
    def write_command(self, ser: SerialBase, command: str):
        # this function is just to convert a string command into a byte command and
        # write it to the serial port
        ser.write(bytes(command, "utf-8"))
        time.sleep(self.cmd_wait)

    def query_command(self, ser: SerialBase, command: str) -> str:
        self.write_command(ser, command)
        ##### ser.write(bytes(command, "utf-8")) already done in write_command
        time.sleep(1e-6)
        return ser.readline().decode("utf-8")


# Useful commands for PSU
# OUTP #1/0
# OUTP?

# CURR # A
# CURR? # A # Set value
# CURR:PROT # A
# CURR:PROT? # A

# VOLT # V
# VOLT? # V
# VOLT:PROT # V mV uV
# VOLT:PROT? # V mV uV

# MEAS:CURR? # A # measure actual value
# MEAS:VOLT? # V
# MEAS:OUTP:COND? # CV/CC
