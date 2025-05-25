import asyncio
import matplotlib.pyplot as plt
import qscope.server
import qscope.system
import qscope.types
import qscope.util
from qscope.scripting import meas_stop_after_nsweeps


PROJECT_NAME = "testing_long_exposures_2"

# Set the total number of sweeps to be taken
NUM_SWEEPS = 100
AVG_PER_POINT = 5

# Start the client server
qscope.server.start_client_log(log_to_stdout=True, log_level="INFO")

# Start the connection manager
manager = qscope.server.ConnectionManager()

# Start connection manager and define the system
try:
    manager.start_local_server(
        # "zyla",
        # "hqdm",
        "gmx",
    ) # logs go to ~./qscope/server.log
    # Connect to the server
    manager.connect()
    manager.startup()
except qscope.server.ServerAlreadyRunning:
    manager.connect()
    pass


# Define the frequency sweep that will be taken
# f_list = qscope.util.gen_linear_sweep_list(2770, 2970, 100)
# f_list = qscope.util.gen_multicentre_sweep_list([2840, 2910], 40, 41)
f_list = qscope.util.gen_centred_sweep_list(1330, 500, 101)

# Configure the measurement
config = qscope.types.SGAndorCWESRLongExpConfig(
    sweep_x=f_list,
    ref_mode="fmod",
    exposure_time=30e-3,
    frame_shape=(256, 256),
    hardware_binning=(1, 1),
    avg_per_point=AVG_PER_POINT,
    fmod_freq=25,
    rf_pow=-40.0,
    laser_delay=0.0,
    long_exp=True,
)
# Add the measurement to the manager
meas_id = manager.add_measurement(config)
# Start the measurement
manager.start_measurement_wait(meas_id)
# Stop the measurement after the number of sweeps and get the data
sweep_data = meas_stop_after_nsweeps(manager, meas_id, NUM_SWEEPS, timeout=1e6)

# save the measurement
manager.measurement_save_sweep(
    meas_id,
    PROJECT_NAME,
)

# close the measurement
manager.close_measurement_wait(meas_id)

# close the connection
manager.disconnect()
# Stop the local server
# manager.stop_local_server()

x, y_sig, y_ref = sweep_data
# remove the first point of the sweep becasuse it is commonly a bad point
x = x[1:]
y_sig = y_sig[1:]
y_ref = y_ref[1:]

# Plot the data
fig = plt.figure()
plt.subplot(121)

plt.plot(x, y_sig, "-o", label="Signal")
plt.plot(x, y_ref, "-o", label="Reference")
plt.legend()

plt.subplot(122)
plt.plot(x, y_sig/y_ref, "-o", label="Signal/ Reference")
plt.show()

