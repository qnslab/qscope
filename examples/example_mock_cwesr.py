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
    "mock",
    # "hqdm",
    # "gmx",
) # logs go to ~./qscope/server.log

# Connect to the server
manager.connect()
manager.startup()

# Define the frequency sweep that will be taken
f_list = qscope.util.gen_linear_sweep_list(2770, 2970, 100)

# f_list = qscope.util.gen_multicentre_sweep_list([2840, 2910], 40, 41)


config_dict = {
        # General settings
        "ref_mode": 'no_rf',
        "avg_per_point": 1,
        # Camera settings
        "exposure_time":15e-3,
        "frame_shape": (
            256,
            256,
        ),
        "hardware_binning": (1, 1),  # FIXME
        # "camera_trig_time": 10e-3, # Not needed should be set by the system
        # Sweep
        "sweep_x":f_list,
        # Laser settings
        "laser_delay":30 * 1e-9,
        "laser_dur": 30 * 1e-9,
        "laser_to_rf_delay": 0,  # FIXME
        # RF settings
        "rf_pow": -40.0,
        "rf_dur": 1 * 1e-9,
        "rf_delay": 1 * 1e-9,
        # test data params
        "peak_contrasts": (-0.4, -0.4),
        "peak_widths": (18, 22),
        "bg_zeeman": 0.0,
        "ft_zeeman": 50.0,
        "ft_width_dif": 5.0,
        "ft_height_dif": -0.004,
        "ft_centre": (None, None),
        "ft_rad": None,
        "ft_linewidth": 4.0,
        "noise_sigma": 500.0,
    }

# Configure the measurement
config = qscope.types.MockSGAndorESRConfig(
    **config_dict
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
