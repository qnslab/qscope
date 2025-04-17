import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import curve_fit

from .fit_model import FitModel


def get_time_unit(multiplier):
    if multiplier == 1:
        return "s"
    elif multiplier == 1e3:
        return "ms"
    elif multiplier == 1e6:
        return "µs"
    elif multiplier == 1e9:
        return "ns"
    elif multiplier == 1e12:
        return "ps"
    elif multiplier == 1e15:
        return "fs"


class OscillationModel(FitModel):
    def __init__(self):
        super().__init__()
        self.param_names = []
        self.param_units = []
        self.x_multiplier = 1

    def set_data(self, xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata
        max_time = np.max(xdata)
        if max_time > 1e-3:
            self.x_multiplier = 1e3
        elif max_time > 1e-6:
            self.x_multiplier = 1e6
        elif max_time > 1e-9:
            self.x_multiplier = 1e9
        elif max_time > 1e-12:
            self.x_multiplier = 1e12
        elif max_time > 1e-15:
            self.x_multiplier = 1e15
        else:
            self.x_multiplier = 1
        self.xdata = xdata * self.x_multiplier

    def set_x_multiplier(self, multiplier):
        self.x_multiplier = multiplier

    def get_fit_results_txt(self):
        results = "Fit results:\n"
        for name, val, error, unit in zip(
            self.param_names, self.fit_results, self.fit_error, self.param_units
        ):
            if unit == "s":
                unit = get_time_unit(self.x_multiplier)
            results += f"{name}: {val:0.2e} ± {error:0.2e} {unit}\n"
        return results


class Sine(OscillationModel):
    def __init__(self):
        self.param_names = ["frequency", "amp", "phase", "background"]
        self.param_units = ["MHz", "counts", "degrees", "counts"]

    def guess_parameters(self, **kwargs):
        # Guess the parameters for the fit

        # Determine if the peak is positive or negative
        # We are assuming that there is more data below the FWHM than above
        avg_val = np.mean(self.ydata)
        max_val = np.max(self.ydata)
        min_val = np.min(self.ydata)

        a_guess = (max_val - min_val) / 2

        # Take the FFT to estimate the frequency
        fft = np.fft.fft(self.ydata)
        freqs = np.fft.fftfreq(len(self.xdata), self.xdata[1] - self.xdata[0])
        freq_guess = np.abs(freqs[np.argmax(np.abs(fft))])

        freq_guess = 10 / np.max(self.xdata)

        # Assume that the phase is 0
        p_guess = 0

        pl_guess = np.mean(self.ydata)

        self.p0 = [freq_guess, a_guess, p_guess, pl_guess]
        return

    def function(self, x, f, a, p, c):
        return a * np.sin(2 * np.pi * f * x + np.deg2rad(p)) + c

    def generate_test_data(self):
        self.p0 = [1, 1, 0, 0]
        return super().generate_test_data()

    def plot_fft(self):
        plt.figure()
        fft = np.fft.fft(self.ydata)
        freqs = np.fft.fftfreq(len(self.xdata), self.xdata[1] - self.xdata[0])
        # Remove the DC component
        fft[0] = 0
        # Remove the negative frequencies
        fft = fft[: len(fft) // 2]
        freqs = freqs[: len(freqs) // 2]

        plt.plot(freqs, np.abs(fft))
        plt.ylabel("Amplitude")
        plt.xlabel("Frequency (Hz)")
        plt.show()


class DampedSine(OscillationModel):
    def __init__(self):
        self.param_names = ["frequency", "amp", "phase", "background", "decay"]
        self.param_units = ["MHz", "counts", "degrees", "counts", "us"]

    def guess_parameters(self, **kwargs):
        # Guess the parameters for the fit

        # Determine if the peak is positive or negative
        # We are assuming that there is more data below the FWHM than above
        avg_val = np.mean(self.ydata)
        max_val = np.max(self.ydata)
        min_val = np.min(self.ydata)

        a_guess = (max_val - min_val) / 2

        # Take the FFT to estimate the frequency
        fft = np.fft.fft(self.ydata)
        freqs = np.fft.fftfreq(len(self.xdata), self.xdata[1] - self.xdata[0])
        freq_guess = np.abs(freqs[np.argmax(np.abs(fft))])

        freq_guess = 10 / np.max(self.xdata)

        # Assume that the phase is 0
        p_guess = 0

        pl_guess = np.mean(self.ydata)

        # Assume that the decay is 0.5 of the total time range
        decay_guess = 0.5 * (self.xdata[-1] - self.xdata[0])

        self.p0 = [freq_guess, a_guess, p_guess, pl_guess, decay_guess]
        return

    def function(self, x, f, a, p, c, t):
        return a * np.sin(2 * np.pi * f * x + np.deg2rad(p)) * np.exp(-x / t) + c

    def generate_test_data(self):
        self.p0 = [1, 1, 0, 0, 4]
        return super().generate_test_data()

    def get_fit_results_txt(self, *args, **kwargs):
        # use the parent class method then add a pi time to it
        results = super().get_fit_results_txt()

        results += f"Pi time: {0.5 / self.fit_results[0]:0.2f} us\n"
        return results

    def plot_fft(self):
        plt.figure()
        fft = np.fft.fft(self.ydata)
        freqs = np.fft.fftfreq(len(self.xdata), self.xdata[1] - self.xdata[0])
        # Remove the DC component
        fft[0] = 0
        # Remove the negative frequencies
        fft = fft[: len(fft) // 2]
        freqs = freqs[: len(freqs) // 2]

        plt.plot(freqs, np.abs(fft))
        plt.ylabel("Amplitude")
        plt.xlabel("Frequency (Hz)")
        plt.show()
