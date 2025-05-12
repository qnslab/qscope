import asyncio
import matplotlib.pyplot as plt
import qscope.server
import qscope.system
import qscope.types
import qscope.util
from qscope.scripting import meas_close_after_nsweeps

# Set the total number of sweeps to be taken
NUM_SWEEPS = 5

# Start the client server
qscope.server.start_client_log(log_to_stdout=True, log_level="INFO")

# Start the connection manager
manager = qscope.server.ConnectionManager()
# Start connection manager and define the system
manager.start_local_server(
    # "mock",
    "hqdm",
    # "gmx",
) # logs go to ~./qscope/server.log

# Connect to the server
manager.connect()
manager.startup()

# Define the frequency sweep that will be taken
# f_list = qscope.util.gen_linear_sweep_list(2770, 2970, 100)

f_list = qscope.util.gen_multicentre_sweep_list([2840, 2910], 40, 41)

# Configure the measurement
config = qscope.types.SGAndorCWESRConfig(
    sweep_x=f_list,
    ref_mode="no_rf",
    exposure_time=30e-3,
    frame_shape=(512, 512),
    hardware_binning=(1, 1),
    avg_per_point=1,
    fmod_freq=0,
    rf_pow=-40.0,
    laser_delay=0.0,
)
# Add the measurement to the manager
meas_id = manager.add_measurement(config)
# Start the measurement
manager.start_measurement_wait(meas_id)
# Close the measurement after the number of sweeps and get the data
sweep_data = meas_close_after_nsweeps(manager, meas_id, NUM_SWEEPS, timeout=60)

# close the connection
manager.disconnect()
# Stop the local server
# manager.stop_local_server()

# Plot the data
fig, ax = plt.subplots()
x, y_sig, y_ref = sweep_data
ax.plot(x, y_sig, "-o", label="Signal")
ax.plot(x, y_ref, "-o", label="Reference")
ax.legend()
plt.show()
