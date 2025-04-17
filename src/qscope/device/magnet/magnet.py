# This is a abstract class for defining a RF source
#

import numpy as np

from qscope.device.device import Device


# FIXME remove set/get noise
# almost all of this stuff is either useless or internal.
class Magnet(Device):
    def __init__(self, **config_kwargs):
        super().__init__(**config_kwargs)
        self._ix = 0  # current for each coil
        self._iy = 0
        self._iz = 0
        self._vx = 0  # voltage for each coil
        self._vy = 0
        self._vz = 0

        self._bx = 0  # magnetic field for each coil
        self._by = 0
        self._bz = 0
        self._theta = 0  # spherical coordinates
        self._phi = 0
        self._bnorm = 0

        self._ixlim = 0  # current limits
        self._iylim = 0
        self._izlim = 0

        self._vxlim = 0  # voltage limits
        self._vylim = 0
        self._vzlim = 0

        self._ix2bx = 0  # current to magnetic field conversion
        self._iy2by = 0
        self._iz2bz = 0

        self._ixrate = 0  # current sweep speed
        self._iyrate = 0
        self._izrate = 0

        self._istep = 0  # current step

    def open(self):
        pass

    def close(self):
        pass

    ##########################################
    #      b spherical properties
    ##########################################
    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = theta

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        self._phi = phi

    @property
    def bnorm(self):
        return self._bnorm

    @bnorm.setter
    def bnorm(self, bnorm):
        self._bnorm = bnorm

    ##########################################
    #      current xyz properties
    ##########################################

    @property
    def ix(self):
        return self._ix

    @ix.setter
    def ix(self, ix):
        self._ix = ix

    @property
    def iy(self):
        return self._iy

    @iy.setter
    def iy(self, iy):
        self._iy = iy

    @property
    def iz(self):
        return self._iz

    @iz.setter
    def iz(self, iz):
        self._iz = iz

    ##########################################
    #      voltage xyz properties
    ##########################################

    @property
    def vx(self):
        return self._vx

    @vx.setter
    def vx(self, vx):
        self._vx = vx

    @property
    def vy(self):
        return self._vy

    @vy.setter
    def vy(self, vy):
        self._vy = vy

    @property
    def vz(self):
        return self._vz

    @vz.setter
    def vz(self, vz):
        self._vz = vz

    ##########################################
    #      Bxyz properties
    ##########################################

    @property
    def bx(self):
        return self._bx

    @bx.setter
    def bx(self, bx):
        self._bx = bx

    @property
    def by(self):
        return self._by

    @by.setter
    def by(self, by):
        self._by = by

    @property
    def bz(self):
        return self._bz

    @bz.setter
    def bz(self, bz):
        self._bz = bz

    ##########################################
    #      limits properties
    ##########################################
    ##### Current limits
    @property
    def ixlim(self):
        return self._ixlim

    @ixlim.setter
    def ixlim(self, ixlim):
        self._ixlim = ixlim

    @property
    def iylim(self):
        return self._iylim

    @iylim.setter
    def iylim(self, iylim):
        self._iylim = iylim

    @property
    def izlim(self):
        return self._izlim

    @izlim.setter
    def izlim(self, izlim):
        self._izlim = izlim

    ##### Voltage limits
    @property
    def vxlim(self):
        return self._vxlim

    @vxlim.setter
    def vxlim(self, val):
        self._vxlim = val

    @property
    def vylim(self):
        return self._vylim

    @vylim.setter
    def vylim(self, val):
        self._vylim = val

    @property
    def vzlim(self):
        return self._vzlim

    @vzlim.setter
    def vzlim(self, val):
        self._vzlim = val

    ##########################################
    #      sweep speed properties
    ##########################################

    @property
    def ixrate(self):
        return self._ixrate

    @property
    def iyrate(self):
        return self._iyrate

    @property
    def izrate(self):
        return self._izrate

    ##########################################
    #  current to b conversion properties
    ##########################################

    @property
    def ix2bx(self):
        return self._ix2bx

    @property
    def iy2by(self):
        return self._iy2by

    @property
    def iz2bz(self):
        return self._iz2bz

    @property
    def b_sph(self):
        # use previously determined bx, by, bz values
        self._bnorm, self._theta, self._phi = self.cart2sph(
            self._bx, self._by, self._bz
        )
        return self._bnorm, self._theta, self._phi

    # Update the parameters
    def update_params(self):
        # Call the get functions to read off the current values
        self._ix = self.ix
        self._iy = self.iy
        self._iz = self.iz
        # Run the function to convert the current into a bxyz and bsph values
        self.i2b()

    ##########################################
    #       functions for coversions
    ##########################################
    def i2b(self):
        self._bx = self._ix * self._ix2bx
        self._by = self._iy * self._iy2by
        self._bz = self._iz * self._iz2bz

        self._bnorm, self._theta, self._phi = self.cart2sph(
            self._bx, self._by, self._bz
        )

    def cart2sph(self, x, y, z):
        # convert cartesian coordinates to spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, np.rad2deg(theta), np.rad2deg(phi)

    def sph2cart(self, r, theta, phi):
        # convert spherical coordinates to cartesian coordinates
        x = r * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
        y = r * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
        z = r * np.cos(np.deg2rad(theta))
        return x, y, z
