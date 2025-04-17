# Developer Documentation

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
- [Development Workflow](#development-workflow)
- [Build System](#build-system)
- [Hardware Support](#hardware-support)
- [Roadmap](#roadmap)
- [Coding Standards](#coding-standards)
- [Testing](#testing)

## Architecture Overview

Qscope uses a distributed architecture with these key components:

```
GUI/Script Client <--> Server <--> System <--> Devices
                         |
                         v
                    Measurements
```

### Component Hierarchy

- **Client**: The Python process the user controls (e.g., script, notebook)
- **GUI**: Qt-based interface that uses client functions
- **Server**: Runs the system in a separate process
- **System**: Represents the entire experimental setup
- **Device**: Individual hardware components
- **Measurement**: Experiment logic for parameter sweeps

## Core Components

### Client

- Provides API for controlling the server
- Handles communication with the server
- Manages connections and data transfer

### Server

- Runs in a separate process from clients
- Manages the system and devices
- Executes measurements
- Handles device locking and resource management

### System

- Represents a specific experimental setup
- Manages multiple hardware devices
- Provides system-specific implementations

### Device

- Represents a single piece of hardware
- Provides a unified interface for hardware control
- Implements device-specific functionality

### Measurement

- Manages experiment logic
- Handles parameter sweeps
- Collects and processes data
- Server-side execution

## Role System Architecture

### Overview

The role system is a key architectural feature of Qscope that provides hardware abstraction. It allows the framework to work with different hardware devices through a consistent interface, making it possible to swap hardware implementations without changing measurement code.

### Components

The role system consists of four main components:

1. **Protocols** (`protocols.py`): Define the methods a device must implement to fulfill a role
2. **Roles** (`roles.py`): Connect devices to interfaces and validate compatibility
3. **Interfaces** (`interfaces.py`): Provide a clean, type-safe API for accessing device functionality
4. **Devices** (`device/`): Concrete implementations of hardware drivers

```
GUI/Script Client <--> Server <--> System <--> Devices
                         |            |
                         v            v
                    Measurements   Role System
```

### How It Works

1. **Protocol Definition**: Protocols define the methods a device must implement to fulfill a role
   ```python
   @runtime_checkable
   class RFSourceProtocol(Protocol):
       def set_freq(self, freq: float) -> None: ...
       def set_power(self, power: float) -> None: ...
       def set_state(self, state: bool) -> None: ...
   ```

2. **Interface Implementation**: Interfaces wrap devices and provide a clean API
   ```python
   class RFSourceInterface(RoleInterface):
       def __init__(self, device: RFSourceProtocol):
           super().__init__(device)
           
       def set_freq(self, freq: float) -> None:
           return self._device.set_freq(freq)
           
       def set_power(self, power: float) -> None:
           return self._device.set_power(power)
   ```

3. **Role Definition**: Roles connect protocols to interfaces
   ```python
   class PrimaryRFSource(DeviceRole[RFSourceProtocol]):
       interface_class = RFSourceInterface
   ```

4. **Device Implementation**: Devices implement the methods required by protocols
   ```python
   class SynthNV(Device):
       def set_freq(self, freq: float) -> None:
           # Hardware-specific implementation
           self._device.write(f"FREQ {freq}")
           
       def set_power(self, power: float) -> None:
           # Hardware-specific implementation
           self._device.write(f"POW {power}")
   ```

5. **System Integration**: The system validates and connects devices to roles
   ```python
   # Validates that SynthNV implements RFSourceProtocol
   system.add_device_with_role(SynthNV(...), PRIMARY_RF)
   ```

6. **Measurement Usage**: Measurements use role interfaces
   ```python
   # Returns RFSourceInterface wrapping SynthNV
   rf = system.get_device_by_role(PRIMARY_RF)
   
   # Calls interface method, which calls device method
   rf.set_freq(2.87e9)
   ```

### Type Safety

The role system provides both static and runtime type safety:

1. **Static Type Checking**: Mypy can verify that devices implement required protocols
2. **Runtime Validation**: The system validates that devices implement required methods
3. **Interface Type Safety**: Interfaces ensure that only role-specific methods are available

### Creating New Components

#### New Protocol

```python
# In protocols.py
@runtime_checkable
class TemperatureControllerProtocol(Protocol):
    def set_temperature(self, temp: float) -> None: ...
    def get_temperature(self) -> float: ...
```

#### New Interface

```python
# In interfaces.py
class TemperatureControllerInterface(RoleInterface):
    def __init__(self, device: TemperatureControllerProtocol):
        super().__init__(device)
    
    def set_temperature(self, temp: float) -> None:
        return self._device.set_temperature(temp)
    
    def get_temperature(self) -> float:
        return self._device.get_temperature()
```

#### New Role

```python
# In roles.py
class TemperatureController(DeviceRole[TemperatureControllerProtocol]):
    interface_class = TemperatureControllerInterface

# Singleton instance
TEMP_CONTROLLER = TemperatureController()
```

#### New Device

```python
# In device/temperature.py
class LakeshoreController(Device):
    required_config = {
        "visa_addr": str,
    }
    
    def __init__(self, **config_kwargs):
        super().__init__(**config_kwargs)
        self._connected = False
        
    def open(self) -> tuple[bool, str]:
        # Connect to hardware
        self._connected = True
        return True, "Connected to Lakeshore"
        
    def close(self):
        # Disconnect from hardware
        self._connected = False
        
    def is_connected(self) -> bool:
        return self._connected
        
    # Implement TemperatureControllerProtocol methods
    def set_temperature(self, temp: float) -> None:
        # Implementation
        pass
        
    def get_temperature(self) -> float:
        # Implementation
        return 25.0
```

### Using the New Components

```python
# In system configuration
system.add_device_with_role(LakeshoreController(visa_addr="TCPIP0::192.168.1.1::INSTR"), TEMP_CONTROLLER)

# In measurement
temp_controller = system.get_device_by_role(TEMP_CONTROLLER)
temp_controller.set_temperature(25.0)
current_temp = temp_controller.get_temperature()
```

## Development Workflow

### Code Organization

- `src/qscope/` - Main package
  - `cli/` - Command-line interfaces
  - `device/` - Hardware device implementations
  - `fitting/` - Data analysis and fitting models
  - `gui/` - Qt GUI components
  - `meas/` - Measurement implementations
  - `server/` - Server and client communication
  - `system/` - System implementations
  - `types/` - Type definitions and interfaces
  - `util/` - Utility functions

## Hardware Support

The table below shows hardware compatibility across operating systems:

| Device              | Windows | Linux   | Vendored? | Connection | Notes                                  |
|---------------------|---------|---------|-----------|------------|----------------------------------------|
| Andor camera        | ✅      | ✅ ?    | ❌        | USB/PCIe   | Install Solis (win only) or SDK        |
| Basler camera       | ✅      | ✅ ?    | ❌        | USB/GigE   | Install pylon                          |
| Thorlabs camera     | ✅      | ✅ ?    | ❌        | USB        | Install Thorcam                        |
| Photometrics camera | ✅      | ✅ ?    | ❌        | USB/PCIe   | Install PVCAM                          |
| Atto magnet         | ✅      | ✅ ?    | ✅        | Serial     | Serial communication                   |
| Winfreak            | ✅      | ✅ ?    | ✅        | Serial     | Serial communication                   |
| R&S smb100          | ✅      | ✅ ?    | ✅        | VISA       | VISA communication                     |
| Agilent SG          | ✅      | ✅ ?    | ✅        | VISA       | VISA communication                     |
| DAQ                 | ❓      | ❓      | ❓        | USB/PCIe   | See pylablib                           |
| Picoscope           | ✅      | ✅      | ❌        | USB        | Picoscope SDK. Win/Ubuntu/OpenSUSE/Mac |
| Pulseblaster        | ✅      | ✅      | ❓        | PCIe       | spinapi DLL from spincore              |
| Keithley SMU        | ✅      | ✅      | N/A       | VISA       | VISA communication                     |
| Helmholtz coils     | ❓      | ❓      | ❓        | Serial     | Serial communication                   |
| Temp controller     | ❓      | ❓      | ❓        | VISA/USB   | See pylablib                           |
| AWG                 | ✅      | ✅      | ❌        | PCIe/USB   | Generic, see pylablib, need NI runtime |
| Spectrometer        | ✅ ?    | ✅ ?    | ❌        | USB        | Swap to pyseabreeze                    |

Legend:
- ✅ = Supported
- ❓ = Needs testing
- ? = Partially tested

### Driver Notes

- **pylablib**: Not fully tested for Linux, but should work
- **Andor drivers**: Windows distribution via Solis, Linux requires SDK
- **Repository limitations**: Cannot include proprietary DLLs in public repository

## Roadmap

- [ ] Refactor into core library + project templates
- [ ] Improve test coverage
- [ ] Add support for more hardware devices
- [ ] Enhance GUI capabilities
- [ ] Support for rastered microscopy systems
- [ ] Distributed measurement capabilities
- [ ] Machine learning integration for experiment optimization
