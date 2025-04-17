import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

def decimate_uneven_data(time, data, target_factor):
    """
    Decimate unevenly sampled time series data to a target resampling factor

    Parameters:
    - time: array-like, the time stamps of the original data
    - data: array-like, the values of the original data
    - target_rate: float, the target sampling rate (Hz)

    Returns:
    - new_time: array, the time stamps of the decimated data
    - new_data: array, the values of the decimated data
    """

    # Create a uniform time grid based on the resampling factor
    start_time = time[0]
    end_time = time[-1]

    num_step = len(time) / target_factor
    # total_time = end_time - start_time
    # time_step = total_time / num_step
    # uniform_time = np.arange(start_time, end_time, time_step)
    uniform_time = np.linspace(start_time, end_time, int(num_step+1))

    # Interpolate the data to the uniform time grid
    interpolator = interp1d(time, data, kind='linear', fill_value='extrapolate')
    uniform_data = interpolator(uniform_time)

    # Downsample the uniformly sampled data
    decimation_factor = int(len(time) / len(uniform_time))
    new_time = uniform_time[::decimation_factor]
    new_data = uniform_data[::decimation_factor]

    return new_time, new_data

# Example usage
time = np.array([3, 6, 9, 12, 13, 20, 25, 27, 30, 35, 36])  # Uneven time stamps
data = np.array([10, 20, 15, 25, 30, 12, 22, 19, 18, 28, 20])     # Corresponding data values
print(f"len(time) = {len(time)}")
target_factor = 2

new_time, new_data = decimate_uneven_data(time, data, target_factor)
print(f"len(new_time) = {len(new_time)}")

fig, ax = plt.subplots()
ax.plot(time, data, "o-", label="Original Data")
ax.plot(new_time, new_data, "x-", label="Decimated Data")
ax.legend()
plt.show()