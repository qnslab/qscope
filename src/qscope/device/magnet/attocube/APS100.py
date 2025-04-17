# Class for controlling the current source APS100
# which is used to control a superconducting vector magnetic.

import time

import numpy as np
import serial

from qscope.device.magnet.magnet import Magnet


class APS100(Magnet):
    # also need to define 'channels'?
    xy_port: str
    z_port: str
    xlim: float
    ylim: float
    zlim: float
    ix2bx: float  # A -> mT
    iy2by: float
    iz2bz: float
    xrate: float  # A/s
    yrate: float
    zrate: float

    required_config = {
        "xy_port": str,
        "z_port": str,
        "xlim": float,
        "ylim": float,
        "zlim": float,
        "ix2bx": float,
        "iy2by": float,
        "iz2bz": float,
        "xrate": float,
        "yrate": float,
        "zrate": float,
    }

    def __init__(
        self,
        system,
        xy_port: str,
        z_port: str,
        xlim: float,
        ylim: float,
        zlim: float,
        ix2bx: float,
        iy2by: float,
        iz2bz: float,
        xrate: float,
        yrate: float,
        zrate: float,
        **other_config,
    ):
        super().__init__(
            system,
            xy_port=xy_port,
            z_port=z_port,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            ix2bx=ix2bx,
            iy2by=iy2by,
            iz2bz=iz2bz,
            xrate=xrate,
            yrate=yrate,
            zrate=zrate,
            **other_config,
        )
        self.cmd_wait = (
            0.1  # time (s) to wait after sending a command to the windfreak >> FIXME?
        )

        # FIXME what do we want to do here?
        self._xrate = self.xrate * 1e-3  # A/s
        self._yrate = self.yrate * 1e-3  # A/s
        self._zrate = self.zrate * 1e-3  # A/s

        self.sweeping_tolerance = 0.5  # percentage

    def open(self):
        # connect to the x current source.
        self.magnet_xy = serial.Serial(self.xy_port, 9600, timeout=1)
        self.magnet_z = serial.Serial(self.z_port, 9600, timeout=1)
        self.init_channels()

    def init_channels(self):
        # initalise Bx channel
        self.write_command(self.magnet_xy, "REMOTE\n")
        self.write_command(self.magnet_xy, "CHAN 1\n")
        self.write_command(self.magnet_xy, "LLIM -" + str(self.xlim) + "\n")

        self.write_command(self.magnet_xy, "REMOTE\n")
        self.write_command(self.magnet_xy, "CHAN 2\n")
        self.write_command(self.magnet_xy, "LLIM -" + str(self.ylim) + "\n")

        self.write_command(self.magnet_z, "REMOTE\n")
        self.write_command(self.magnet_z, "LLIM -" + str(self.zlim) + "\n")

    def stop(self):
        self.write_command(self.magnet_xy, "CHAN 1\n")
        self.write_command(self.magnet_xy, "SWEEP ZERO\n")

        self.write_command(self.magnet_xy, "CHAN 2\n")
        self.write_command(self.magnet_xy, "SWEEP ZERO\n")
        self.write_command(self.magnet_z, "SWEEP ZERO\n")

    def close(self):
        self.stop()
        self.magnet_xy.close()
        self.magnet_z.close()

    ###################################################################
    # Get functions
    ###################################################################

    @property
    def ix(self):
        # clear the read buffer
        self.magnet_xy.read(1000)

        self.write_command(self.magnet_xy, "CHAN 1\n")
        self.magnet_xy.readline()
        self.write_command(self.magnet_xy, "IMAG?\n")
        self.magnet_xy.readline()
        ix = self.magnet_xy.readline().decode()
        self._ix = float(ix[:-3])
        return self._ix

    @property
    def iy(self):
        # clear the read buffer
        self.magnet_xy.read(1000)

        self.write_command(self.magnet_xy, "CHAN 2\n")
        self.magnet_xy.readline()
        self.write_command(self.magnet_xy, "IMAG?\n")
        self.magnet_xy.readline()
        iy = self.magnet_xy.readline().decode()
        self._iy = float(iy[:-3])
        return self._iy

    @property
    def iz(self):
        # clear the read buffer
        self.magnet_z.read(1000)
        self.write_command(self.magnet_z, "IMAG?\n")
        self.magnet_z.readline()
        iz = self.magnet_z.readline().decode()
        self._iz = float(iz[:-3])
        return self._iz

    @property
    def bx(self):
        self._bx = self.ix * self.ix2bx
        return self._bx

    @property
    def by(self):
        self._by = self.iy * self.iy2by
        return self._by

    @property
    def bz(self):
        self._bz = self.iz * self.iz2bz
        return self._bz

    @property
    def bnorm(self, read=True):
        if read:
            [_, _, bspher] = self.get_current(print_b_field=False)
            self._bnorm = bspher[0]
        return self._bnorm

    @property
    def theta(self, read=True):
        if read:
            [_, _, bspher] = self.get_current(print_b_field=False)
            self._theta = bspher[1]
        return self._theta

    @property
    def phi(self, read=True):
        if read:
            [_, _, bspher] = self.get_current(print_b_field=False)
            self._phi = bspher[1]
        return self._phi

    def get_current(self, print_b_field=True):
        # convert the current values into a bxyz
        self._bx = self._ix * self.ix2bx
        self._by = self._iy * self.iy2by
        self._bz = self._iz * self.iz2bz
        # convert the bxyz into bspherical
        self._bnorm, self._theta, self._phi = self.convert_cartesian_to_spherical(
            self._bx, self._by, self._bz
        )

        if print_b_field:
            print("Current field values")
            print(f"B = {self._bnorm:.3f} (mT)")
            print(f"theta = {self._theta:.3f} (deg)")
            print(f"phi = {self._phi:.3f} (deg)")
            aligned_freq_1 = np.abs(2870 - 28 * self._bnorm)
            aligned_freq_2 = 2870 + 28 * self._bnorm
            print(f"Upper frequency = {aligned_freq_2:.5} (MHz))")
            print(f"Lower frequency = {aligned_freq_1:.5} (MHz))")

        curr = [self._ix, self._iy, self._iz]
        b = [self._bx, self._by, self._bz]
        bspher = [self._bnorm, self._theta, self._phi]
        return curr, b, bspher

    ###################################################################
    # Set functions
    ###################################################################

    def set_b(self, bnorm, theta, phi):
        bx, by, bz = self.convert_spherical_to_cartesian(bnorm, theta, phi)
        print(
            f"Setting B to {bnorm:.3f} (mT) at theta = {theta:.3f} (deg) and phi = {phi:.3f} (deg)"
        )

        set_ix = bx / self.ix2bx
        set_iy = by / self.iy2by
        set_iz = bz / self.iz2bz

        self.ix = set_ix
        self.iy = set_iy
        self.iz = set_iz

        self.check_if_field_reached(set_ix, set_iy, set_iz)

    @ix.setter
    def ix(self, val):
        # check if the requested current vaules are within the limits
        if abs(val) > self.xlim:
            print("Ix CURRENT VALUE IS TOO HIGH. DID NOT CHANGE VALUE")
            return

        # Set all of the current values at once
        self.write_command(self.magnet_xy, "CHAN 1\n")
        self.write_command(self.magnet_xy, "ULIM " + str(val) + "\n")
        self.write_command(self.magnet_xy, "SWEEP UP\n")
        self._ix = val

    @iy.setter
    def iy(self, val):
        # check if the requested current vaules are within the limits
        if abs(val) > self.ylim:
            print("Iy CURRENT VALUE IS TOO HIGH. DID NOT CHANGE VALUE")
            return

        # Set all of the current values at once
        self.write_command(self.magnet_xy, "CHAN 2\n")
        self.write_command(self.magnet_xy, "ULIM " + str(val) + "\n")
        self.write_command(self.magnet_xy, "SWEEP UP\n")
        self._iy = val

    @iz.setter
    def iz(self, val):
        # check if the requested current vaules are within the limits
        if abs(val) > self.zlim:
            print("Iz CURRENT VALUE IS TOO HIGH. DID NOT CHANGE VALUE")
            return

        # Set all of the current values at once
        self.write_command(self.magnet_z, "ULIM " + str(val) + "\n")
        self.write_command(self.magnet_z, "SWEEP UP\n")
        self._iz = val

    def set_curr(self, ix, iy, iz):
        # check if the requested current vaules are within the limits
        if abs(ix) > self.xlim or abs(iy) > self.ylim or abs(iz) > self.zlim:
            print("CURRENT VALUE IS TOO HIGH. DID NOT CHANGE VALUE")
            return

        self.ix = ix
        self.iy = iy
        self.iz = iz

        self.check_if_field_reached(ix, iy, iz)

    @bx.setter
    def bx(self, bx):
        ix = bx / self.ix2bx
        self.ix = ix

    @by.setter
    def by(self, by):
        iy = by / self.iy2by
        self.iy = iy

    @bz.setter
    def bz(self, bz):
        iz = bz / self.iz2bz
        self.iz = iz

    @bnorm.setter
    def bnorm(self, val):
        theta = self.theta(read=False)
        phi = self.phi(read=False)
        self.set_b(val, theta, phi)
        self._bnorm = val

    @theta.setter
    def theta(self, val):
        bnorm = self.bnorm(read=False)
        phi = self.phi(read=False)
        self.set_b(bnorm, val, phi)
        self._theta = val

    @phi.setter
    def phi(self, val):
        bnorm = self.bnorm(read=False)
        theta = self.theta(read=False)
        self.set_b(bnorm, theta, val)
        self._phi = val

    def check_if_field_reached(self, ix, iy, iz):
        b_sweeping = True

        # Get the difference between the current and the requested
        # current as a percentage
        x_diff = np.abs(ix - self.ix)
        y_diff = np.abs(iy - self.iy)
        z_diff = np.abs(iz - self.iz)

        # Waiting for sweeping to finish
        sweeping_time = np.max(
            [x_diff / self._xrate, y_diff / self._yrate, z_diff / self._zrate]
        )
        print(
            f"Sweeping to new field value. Estimated sweeping time: {sweeping_time:0.2f} (s)"
        )
        time.sleep(sweeping_time)

        while b_sweeping:
            curr, _, _ = self.get_current(print_b_field=False)

            # Get the difference between the current and the requested
            # current as a percentage
            x_diff = np.abs(ix - self.ix) * 100
            y_diff = np.abs(iy - self.iy) * 100
            z_diff = np.abs(iz - self.iz) * 100

            # if all of the differences are less than the tolerance then stop sweeping
            if (
                x_diff < self.sweeping_tolerance
                and y_diff < self.sweeping_tolerance
                and z_diff < self.sweeping_tolerance
            ):
                print("Field has been reached")
                b_sweeping = False
            else:
                time.sleep(0.5)
                print("Field has not been reached yet")

        aligned_freq_1 = np.abs(2870 - 28 * self._bnorm)
        aligned_freq_2 = 2870 + 28 * self._bnorm
        print(f"B = {self._bnorm:.3f} (mT)")
        print(f"NV freq approx [{aligned_freq_1:.5}, {aligned_freq_2:.5}] (MHz)")

    ###################################################################
    # Utility functions
    ###################################################################

    def convert_cartesian_to_spherical(self, x, y, z):
        # convert cartesian coordinates to spherical coordinates
        r = np.round(np.sqrt(x**2 + y**2 + z**2), decimals=3)
        theta = np.round(np.rad2deg(np.arccos(z / r)), decimals=3)
        phi = np.round(np.rad2deg(np.arctan2(y, x)), decimals=3)
        return r, theta, phi

    def convert_spherical_to_cartesian(self, r, theta, phi):
        # convert spherical coordinates to cartesian coordinates
        x = r * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
        y = r * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
        z = r * np.cos(np.deg2rad(theta))
        return x, y, z

    def write_command(self, serial, command):
        # this function is just to convert a string command into a byte command and
        # write it to the serial port
        serial.write(bytes(command, "utf-8"))
        time.sleep(self.cmd_wait)
