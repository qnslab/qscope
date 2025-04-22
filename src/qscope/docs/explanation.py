"""# Explanations

This section provides conceptual explanations of QScope's architecture and design.

## Architecture Overview

QScope uses a distributed architecture with these key components:

```
GUI/Script Client <--> Server <--> System <--> Devices
                         |
                         v
                    Measurements
```

The architecture separates hardware control (server) from user interfaces (clients), allowing for flexible experiment control.

## Role System

The role system provides hardware abstraction through:

1. **Protocols**: Define required device methods
2. **Interfaces**: Provide clean APIs for accessing devices
3. **Roles**: Connect devices to interfaces
4. **Devices**: Implement hardware-specific functionality

This allows measurements to work with abstract roles rather than specific hardware implementations. See the types module for more discussion.

## Measurement Framework

The measurement framework is designed to standardize how experiments are defined and executed in QScope.

### Measurement Lifecycle

A measurement in QScope follows this lifecycle:

1. **Configuration**: Define parameters through a configuration class
2. **Initialization**: Create a measurement instance with the configuration
3. **Setup**: Prepare devices and initialize data structures
4. **Execution**: Run one or more sweeps to collect data
5. **Cleanup**: Release resources and reset devices

### Measurement State Machine

Measurements use a state machine to track their progress (SEE NOTE BELOW):

```
CREATED → SETUP → RUNNING → PAUSED → FINISHED
                    ↑  ↓     ↑  ↓
                    ↓  ↑     ↓  ↑
                   --- STOPPING ---
```

NB: An AI made the above. It isn't correct. It would have been a good idea to create a flowchart when I understood the flow.

### Data Flow

Data flows through the measurement system as follows:

1. **Collection**: Raw data from devices
2. **Processing**: Apply calibrations and transformations
3. **Analysis**: Fit models and extract parameters
4. **Visualization**: Display results in the GUI
5. **Storage**: Save data to disk

## Client-Server Communication

QScope uses a message-passing architecture for client-server communication:

1. **Messages**: Structured data packets for commands and responses
2. **Requests**: Client-initiated commands to the server
3. **Responses**: Server replies to client requests
4. **Notifications**: Server-initiated messages about state changes

This approach enables:
- Multiple simultaneous client connections
- Asynchronous operation
- Robust error handling
"""
