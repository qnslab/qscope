"""# Supported Hardware

This page documents what devices/hardware are currently supported by `qscope`.

The table below shows hardware compatibility across operating systems:

| Device              | &nbsp;&nbsp;Windows&nbsp;&nbsp; | &nbsp;&nbsp;Linux&nbsp;&nbsp; | Connection&nbsp;&nbsp;  | Notes                                  |
| :------------------ | :-----------------------------: | :---------------------------: | :---------------------  | :------------------------------------- |
| Andor camera        | &nbsp;&nbsp;✅&nbsp;&nbsp;      | &nbsp;&nbsp;❓&nbsp;&nbsp;     | USB/PCIe               | Install Solis (win only) or SDK        |
| Basler camera       | &nbsp;&nbsp;✅&nbsp;&nbsp;      | &nbsp;&nbsp;❓&nbsp;&nbsp;     | USB/GigE               | Install pylon                          |
| Thorlabs camera     | &nbsp;&nbsp;✅&nbsp;&nbsp;      | &nbsp;&nbsp;❓&nbsp;&nbsp;     | USB                    | Install Thorcam                        |
| Photometrics camera | &nbsp;&nbsp;✅&nbsp;&nbsp;      | &nbsp;&nbsp;❓&nbsp;&nbsp;     | USB/PCIe               | Install PVCAM                          |
| Atto magnet         | &nbsp;&nbsp;✅&nbsp;&nbsp;      | &nbsp;&nbsp;❓&nbsp;&nbsp;     | Serial                 | Serial communication                   |
| Winfreak            | &nbsp;&nbsp;✅&nbsp;&nbsp;      | &nbsp;&nbsp;❓&nbsp;&nbsp;     | Serial                 | Serial communication                   |
| R&S smb100          | &nbsp;&nbsp;✅&nbsp;&nbsp;      | &nbsp;&nbsp;❓&nbsp;&nbsp;     | VISA                   | VISA communication                     |
| Agilent SigGen      | &nbsp;&nbsp;✅&nbsp;&nbsp;      | &nbsp;&nbsp;❓&nbsp;&nbsp;     | VISA                   | VISA communication                     |
| DAQ                 | &nbsp;&nbsp;❓&nbsp;&nbsp;      | &nbsp;&nbsp;❓&nbsp;&nbsp;     | USB/PCIe               | See pylablib                           |
| Picoscope           | &nbsp;&nbsp;✅&nbsp;&nbsp;      | &nbsp;&nbsp;✅&nbsp;&nbsp;     | USB                    | Picoscope SDK. Win/Ubuntu/OpenSUSE/Mac |
| Pulseblaster        | &nbsp;&nbsp;✅&nbsp;&nbsp;      | &nbsp;&nbsp;✅&nbsp;&nbsp;     | PCIe                   | spinapi DLL from spincore              |
| Keithley SMU        | &nbsp;&nbsp;✅&nbsp;&nbsp;      | &nbsp;&nbsp;✅&nbsp;&nbsp;     | VISA                   | VISA communication                     |
| Helmholtz coils     | &nbsp;&nbsp;❓&nbsp;&nbsp;      | &nbsp;&nbsp;❓&nbsp;&nbsp;     | Serial                 | Serial communication                   |
| Temp controller     | &nbsp;&nbsp;❓&nbsp;&nbsp;      | &nbsp;&nbsp;❓&nbsp;&nbsp;     | VISA/USB               | See pylablib                           |
| AWG                 | &nbsp;&nbsp;✅&nbsp;&nbsp;      | &nbsp;&nbsp;✅&nbsp;&nbsp;     | PCIe/USB               | Generic, see pylablib, need NI runtime |
| Spectrometer        | &nbsp;&nbsp;❓&nbsp;&nbsp;      | &nbsp;&nbsp;❓&nbsp;&nbsp;     | USB                    | Swap to pyseabreeze                    |

Legend:

- ✅ = Supported
- ❓ = Needs testing, or partially tested
- ❌ = requires external libs

"""