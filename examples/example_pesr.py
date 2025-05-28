import asyncio
import matplotlib.pyplot as plt
import qscope.server
import qscope.system
import qscope.types
import qscope.util
from qscope.scripting import meas_stop_after_nsweeps


PROJECT_NAME = "pulsed_esr_GLU_test"

# Set the total number of sweeps to be taken
NUM_SWEEPS = 10

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
f_list = qscope.util.gen_centred_sweep_list(1328, 300, 101)

# Configure the measurement
config = qscope.types.SGAndorPESRConfig(
    sweep_x=f_list,
    ref_mode="no_rf",
    exposure_time=25e-3,
    frame_shape=(256, 256),
    hardware_binning=(1, 1),
    avg_per_point=1,
    fmod_freq=20,
    rf_pow=-30.0,
    laser_delay=340e-9,
    laser_dur=2e-6,
    rf_dur= 1e-6,
    rf_delay=0.0,
    laser_to_rf_delay=0,
    rf_to_laser_delay=0,
)
# Add the measurement to the manager
meas_id = manager.add_measurement(config)
# Start the measurement
manager.start_measurement_wait(meas_id)
# Stop the measurement after the number of sweeps and get the data
sweep_data = meas_stop_after_nsweeps(manager, meas_id, NUM_SWEEPS, timeout=1e9)

x, y_sig, y_ref = sweep_data
# remove the first point of the sweep becasuse it is commonly a bad point
x = x[1:]
y_sig = y_sig[1:]
y_ref = y_ref[1:]

# save the measurement
manager.measurement_save_sweep_w_fit(
    meas_id,
    PROJECT_NAME,
    xdata = x, 
    ydata = 100*(y_sig/y_ref -1), 
    xfit = None, 
    yfit = None,
    fit_results = ''
)

# close the measurement
# manager.close_measurement_wait(meas_id)

# close the connection
# manager.disconnect()
# Stop the local server
# manager.stop_local_server()


# Plot the data
fig = plt.figure()
plt.subplot(121)

plt.plot(x, y_sig, "-o", label="Signal")
plt.plot(x, y_ref, "-o", label="Reference")
plt.legend()

plt.subplot(122)
plt.plot(x, y_sig/y_ref, "-o", label="Signal/ Reference")
plt.show()

