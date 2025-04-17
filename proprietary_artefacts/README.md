NOTES
-----

Extraneous notes on the libs here, & how to get them etc.

# SpinAPI / Pulseblaster

Docs: https://www.spincore.com/support/spinapi/reference/production/2013-09-25/index.html
OR: https://web.archive.org/web/20221013050136/https://www.spincore.com/support/spinapi/reference/production/2013-09-25/spinapi_8h.html

libspinapi64.so was built by Sam on his laptop. Unsure if it will work for 32bit systems.
Built from code @: https://www.spincore.com/support/spinapi/

_libspinapi.so & _libspinapi.so from Chris Billington (https://github.com/chrisjbillington/spinapi/)
which did not work for Sam

spinapi64.dll was taken from one of the lab PC's, I assume the universal installer at the above
spincore link was used.

# Andor

The Andor dlls were bought a long time ago. Currently only have windows versions, and we don't check the os platform.
-> in future, have diff directories for each platform of andor dll, and check before call to `pylablib.par`

# Photometrics

See:
https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads/?c=1073743373__CatalogContent&p=19688__CatalogContent&page=1
Probably just run installer:
PVCam_3.10.1.1-SDK60_Setup.zip in proprietary_artefacts.

(Unsure what's required otherwise)
Manuals etc. here:
https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads/photometrics/teledyne-photometrics-drivers-software-instructions/


# Picoscope

See e.g. https://www.picotech.com/downloads/linux
On Sam's laptop the picoscope shared objects are in /opt/picoscope/lib/
& was installed along with some other libraries.