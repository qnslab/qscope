'''
This script demonstrates how to calculate the Allan deviation of a time series of ESR measurements.

'''
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the lorentzian function
def lorentzian(x, x0, gamma, A):
    return A * gamma**2 / ((x - x0)**2 + gamma**2)


FILE_NAME = 'esr_drift.npy'
use_mock_data = True

if use_mock_data:
    NUM_MEAS = 200
    # Generate some mock data
    f_list = np.linspace(975, 1025, 100)
    data = np.zeros((100, NUM_MEAS))
    # make some random data with some positive drift
    for i in range(0, NUM_MEAS):
        noise = np.random.rand()
        drift = 0.001 * i
        data[:, i] = lorentzian(f_list, 1000 + noise + drift, 1, 0.5)

else:
    # load the data
    data = np.load(FILE_NAME)
    f_list = np.load('{FILE_NAME}_f_list.npy')



# Plot the data
fig, ax = plt.subplots()
ax.imshow(data, aspect='auto', extent=[ 0, NUM_MEAS, f_list[0], f_list[-1]])
ax.set_ylabel('Frequency (MHz)')
ax.set_xlabel('Measurement number')
plt.show()


# Preallocate the variation
variation = np.zeros(NUM_MEAS)

# Loop over the data
for i in range(NUM_MEAS):
    # Fit the data
    p0 = [f_list[np.argmax(data[:, i])], 10, 0.1]
    popt, pcov = curve_fit(lorentzian, f_list, data[:, i], p0=p0)
    # Calculate the allan deviation
    variation[i] = popt[0]

# Plot the allan deviation
fig, ax = plt.subplots()
ax.plot(variation, '-o')
ax.set_xlabel('Measurements')
ax.set_ylabel('Variation in frequency')
plt.show()


# define the function to average over the tau values and fit the newly averaged data
def fit_and_average(data, f_list, tau_values, tau, p0):
    avg_data = np.zeros((len(f_list), len(tau_values) - tau))
    for i in range(len(tau_values) - tau):
        avg_data[:, i] = np.mean(data[:, i:i+tau], axis=1)
    # Fit the averaged data
    popt = np.zeros((3, len(tau_values) - tau))
    for i in range(len(tau_values) - tau):
        popt[:, i], pcov = curve_fit(lorentzian, f_list, avg_data[:, i], p0)
    return popt


def calculate_mdev_per_timepoint(data, f_list, tau_values, tau, popt):
    # Calculate the modified Allan variance from the fit parameters
    mdev = 0
    for i in range(popt.shape[1] - tau):
        mdev += (popt[0, i] - popt[0, i+tau])**2
    return np.sqrt(mdev)


# Define the time intervals (tau) over which to average the data 
# (this is the x-axis of the modified Allan variance plot)
tau_values = np.arange(1, NUM_MEAS)
tau_list =  np.arange(1, NUM_MEAS//3)

# Calculate and plot the modified Allan variance for each tau
mdev = np.zeros(len(tau_list))
i=0
for tau in tau_list:
    popt = fit_and_average(data, f_list, tau_values, tau, p0)
    mdev[i] = calculate_mdev_per_timepoint(data, f_list, tau_values, tau, popt)
    i+=1


fig, ax = plt.subplots()
# Plot the modified Allan variance as a y semi-log plot
ax.plot(tau_list, mdev, '-o')
ax.set_xlabel('number of measurements')
ax.set_ylabel('Modified Allan Variance')
ax.legend()
plt.show()

