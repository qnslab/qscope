'''
This script is for running a series of single ESR measurement to determine the drift on the ESR position over time. 

This can be used to obtain a colormap of the drift itself over time but it can also be used to calculate the Allan deviation or other metrics. 

'''
import asyncio
from loguru import logger
import os
import matplotlib.pyplot as plt
import time
import qscope.server
import qscope.system
import qscope.types
import qscope.util
from qscope.scripting import (
    meas_close_after_nsweeps, 
    meas_stop_after_nsweeps,
    meas_wait_for_nsweeps,
)

import numpy as np

# TODO fix the manager so it is opened and closed in the script itself.

# Save Name
SAVE_DIR = 'C:\\ExperimentalData\\2025\\2025-04\\2025-04-11_drift_testing\\'
SAVE_NAME = 'esr_drift'
# set the number of measurements to take
NUM_MEAS = 1
# Set the total number of sweeps to be taken in each esr
NUM_SWEEPS = 1

# Start the client server
qscope.server.start_client_log(log_to_stdout=True, log_level="INFO")

# Start the connection manager
manager = qscope.server.ConnectionManager()
# Start connection manager and define the system
try:
    manager.start_local_server(
        # "zyla",
        "hqdm",
        # "gmx",
    ) # logs go to ~./qscope/server.log
    # Connect to the server
    manager.connect()
    manager.startup()
except qscope.server.ServerAlreadyRunning:
    manager.connect()
    pass


# Define the frequency sweep that will be taken
# f_list = qscope.util.gen_linear_sweep_list(2770, 2970, 100)
f_list = qscope.util.gen_centred_sweep_list(2988.4, 10, 101)

# f_list = qscope.util.gen_multicentre_sweep_list([2840, 2910], 40, 41)

# Configure the measurement
config = qscope.types.SGAndorCWESRConfig(
    sweep_x=f_list,
    ref_mode="no_rf",
    exposure_time=15e-3,
    frame_shape=(256, 256),
    hardware_binning=(1, 1),
    avg_per_point=4,
    fmod_freq=0,
    rf_pow=-40.0,
    laser_delay=0.0,
)

# preallocate the full dataset
full_data = np.zeros((NUM_MEAS, 3, len(f_list)))


# Make loop for all of the measurements 
i = 0
while True:
    try:
        print(f"Measurement {i+1}/{NUM_MEAS}")
        # Add the measurement to the manager
        meas_id = manager.add_measurement(config)
        time.sleep(0.1)
        # Start the measurement
        manager.start_measurement_wait(meas_id)
        time.sleep(0.1)
        # Close the measurement after the number of sweeps and get the data
        sweep_data = meas_stop_after_nsweeps(manager, meas_id, NUM_SWEEPS, timeout=60)
        # sweep_data = meas_wait_for_nsweeps(manager, NUM_SWEEPS, timeout=60)
        manager.close_measurement_wait(meas_id)
        # Add the data to the full dataset
        full_data[i, ::] = sweep_data
        # wait for a bit to let the system settle
        time.sleep(0.1)
        if i == NUM_MEAS:
            break
        i += 1
    except Exception as e:
        logger.error(f"Error during script: {e}")
        logger.error("Atempting to continue with the next measurement")
    

# Close the connection
# manager.shutdown()
manager.disconnect()

# Save the data with autoincrementing name
i = 0
while True:
    # Check if the file already exists at SAVE_NAME_{i}.npy
    # If it does, break the loop and increment i

    if os.path.exists(SAVE_DIR + f'{SAVE_NAME}_{i}.npy'):
        # If it exists, increment i
        i += 1
        continue
    else:
        # If it does not exist, break the loop
        break

# If it does not exist, save the data
# Save the data 
print(f"Saving data to {SAVE_NAME}_{i}.npy")
# Save the data
np.save(SAVE_DIR + f'{SAVE_NAME}_{i}.npy', full_data)

# Save the frequency list
np.save(SAVE_DIR + f'{SAVE_NAME}_{i}_f_list.npy', f_list)

# get the mean frequency so the scale is in difference from the mean
mean_freq = np.mean(f_list)


# Plot the data
plt.figure()
plt.imshow(full_data[:,1,:] / full_data[:,2,:], aspect='auto', extent=[f_list[0,] - mean_freq, f_list[-1] - mean_freq, 0, NUM_MEAS], cmap='viridis')
plt.colorbar()

plt.xlabel('Frequency (MHz)')
plt.ylabel('Measurement number')
plt.title('ESR drift over time')
# Save the figure
plt.savefig(SAVE_DIR + f'{SAVE_NAME}_{i}_plot.png')
plt.show()




