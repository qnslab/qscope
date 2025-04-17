import numpy as np
from scipy.interpolate import interp1d


def decimate_uneven_data(
    time: list[int], data: list[np.ndarray[float]], target_factor: float
) -> (tuple)[list[int], list[np.ndarray[float]]]:
    """
    Decimate unevenly sampled time series data to a target resampling factor

    Parameters:
    - time: array-like, the time stamps of the original data
    - data: array-like, the values of the original data (shape = [n_time, n_data])
    - target_rate: float, the target sampling rate (Hz)

    Returns:
    - new_time: array, the time stamps of the decimated data
    - new_data: array, the values of the decimated data
    """
    data = np.asarray(data)  # convert to 2D numpy array

    # Create a uniform time grid based on the resampling factor
    start_time = time[0]
    end_time = time[-1]

    num_step = len(time) / target_factor
    # total_time = end_time - start_time
    # time_step = total_time / num_step
    # uniform_time = np.arange(start_time, end_time, time_step)
    uniform_time = np.linspace(start_time, end_time, int(num_step + 1))

    # Interpolate the data to the uniform time grid
    # logger.debug(f"time shape {np.shape(time)}, data shape {np.shape(data)}")
    interpolator = interp1d(
        time, data, kind="linear", fill_value="extrapolate", assume_sorted=True, axis=0
    )
    uniform_data = interpolator(uniform_time)

    # Downsample the uniformly sampled data
    decimation_factor = int(len(time) / len(uniform_time))
    new_time = uniform_time[::decimation_factor]
    new_data = uniform_data[::decimation_factor, :]

    return list(new_time), list(new_data)
