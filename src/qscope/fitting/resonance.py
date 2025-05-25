import matplotlib.pyplot as plt
import numpy as np
import scipy
from loguru import logger
from scipy.optimize import curve_fit

from .fit_model import FitModel


class Lorentzian(FitModel):
    def __init__(self):
        self.param_names = ["position", "amp", "width", "pl"]
        self.param_units = ["MHz", "%", "MHz", "counts"]

    def guess_parameters(self, **kwargs):
        # Guess the parameters for the fit

        # Determine if the peak is positive or negative
        # We are assuming that there is more data below the FWHM than above
        avg_val = np.mean(self.ydata)
        max_val = np.max(self.ydata)
        min_val = np.min(self.ydata)

        if np.abs(avg_val - max_val) > np.abs(avg_val - min_val):
            a_guess = max_val - avg_val
        else:
            a_guess = min_val - avg_val

        # Find the index of the position
        if a_guess < 0:
            peak_val = np.min(self.ydata)
        else:
            peak_val = np.max(self.ydata)

        # Find the index of the maximum value
        peak_index = np.where(self.ydata == peak_val)[0][0]
        # Find the frequency at the maximum value
        pos_guess = self.xdata[peak_index]

        # estimate the FWHM
        # Find the index of the half max value
        half_max = (peak_val + avg_val) / 2
        # Find the index that is closest to this value
        half_max_index = (np.abs(self.ydata - half_max)).argmin()
        # Find the frequency at the half max value
        w_guess = np.abs(self.xdata[half_max_index] - pos_guess)

        pl_guess = np.mean(self.ydata)

        self.p0 = [pos_guess, a_guess, w_guess, pl_guess]
        return

    def function(self, x, x0, a, w, c):
        return a * w**2 / (w**2 + (x - x0) ** 2) + c

    def sensitivity(self, total_counts,  ith_loop=1, *args, **kwargs):
        # To get the sensitivity we need to know the absolute counts.
        # Thus we need to pass this value.

        amp = self.fit_results[1]
        width = self.fit_results[2]

        if self.normalisation == "sig - ref" or self.normalisation == "ref - sig":
            # The amplitude is the difference between the peak value and the mean value
            amp = total_counts/(amp* ith_loop)
        elif self.normalisation == "norm sig / ref" or self.normalisation =="norm ref / sig":
            # The amplitude is the ratio between the peak value and the mean value
            amp = amp /100
        elif self.normalisation == "intensity" or self.normalisation == "sig / ref" or self.normalisation == "ref / sig":
            amp = amp
        else:
            # To catch the None case
            amp = total_counts/(amp* ith_loop)

        planks_const = 6.62607015e-34
        muB = 9.274009994e-24
        ge = 2.0023

        eta = (
            (4 / (3 * np.sqrt(3)))
            * (planks_const / (ge * muB))
            * np.abs(width)
            * 1e6
            / (np.abs(amp) * np.sqrt(total_counts))
        )
        return eta

    def generate_test_data(self):
        self.p0 = [5, 1, 1, 1]
        return super().generate_test_data()

    def get_fit_results_txt(self, total_counts, ith_loop=1):
        results = super().get_fit_results_txt()
        results += f"sens.: {self.sensitivity(total_counts, ith_loop=ith_loop):0.3e} (T/sqrt(Hz))\n"
        return results


class Gaussian(FitModel):
    def __init__(self):
        self.param_names = ["position", "amp", "width", "pl"]
        self.param_units = ["MHz", "%", "MHz", "counts"]

    def guess_parameters(self, **kwargs):
        # Guess the parameters for the fit

        # Determine if the peak is positive or negative
        # We are assuming that there is more data below the FWHM than above
        avg_val = np.mean(self.ydata)
        max_val = np.max(self.ydata)
        min_val = np.min(self.ydata)

        if np.abs(avg_val - max_val) > np.abs(avg_val - min_val):
            a_guess = max_val - avg_val
        else:
            a_guess = min_val - avg_val

        # Find the index of the position
        if a_guess < 0:
            peak_val = np.min(self.ydata)
        else:
            peak_val = np.max(self.ydata)

        # Find the index of the maximum value
        peak_index = np.where(self.ydata == peak_val)[0][0]
        # Find the frequency at the maximum value
        pos_guess = self.xdata[peak_index]

        # estimate the FWHM
        # Find the index of the half max value
        half_max = (peak_val + avg_val) / 2
        # Find the index that is closest to this value
        half_max_index = (np.abs(self.ydata - half_max)).argmin()
        # Find the frequency at the half max value
        w_guess = np.abs(self.xdata[half_max_index] - pos_guess)

        pl_guess = np.mean(self.ydata)

        self.p0 = [pos_guess, a_guess, w_guess, pl_guess]

        return self.p0

    def get_fit_results_txt(self, total_counts, ith_loop=1):
        results = super().get_fit_results_txt()
        results += f"sens.: {self.sensitivity(total_counts, ith_loop=ith_loop):0.3e} (T/sqrt(Hz))\n"

        return results

    def generate_test_data(self):
        self.p0 = [5, 1, 1, 0]
        return super().generate_test_data()

    def function(self, x, x0, a, w, c):
        # Gaussian function
        return a * np.exp(-((x - x0) ** 2) / w**2) + c

    def sensitivity(self, total_counts, ith_loop=1):
        # To get the sensitivity we need to know the absolute counts.
        # Thus we need to pass this value.
        amp = self.fit_results[1]
        width = self.fit_results[2]

        planks_const = 6.62607015e-34
        muB = 9.274009994e-24
        ge = 2.0023

        if self.normalisation == "sig - ref" or self.normalisation == "ref - sig":
            # The amplitude is the difference between the peak value and the mean value
            amp = total_counts/(amp* ith_loop)
        elif self.normalisation == "norm sig / ref" or self.normalisation =="norm ref / sig":
            # The amplitude is the ratio between the peak value and the mean value
            amp = amp /100
        elif self.normalisation == "intensity" or self.normalisation == "sig / ref" or self.normalisation == "ref / sig":
            amp = amp
        else:
            # To catch the None case
            amp = total_counts/(amp* ith_loop)

        eta = (
            0.7
            * (planks_const / (ge * muB))
            * np.abs(width)
            * 1e6
            / (np.abs(amp) * np.sqrt(total_counts))
        )
        return eta


class DifferentialLorentzian(FitModel):
    def __init__(self):
        self.param_names = ["pos.", "amp.", "wid.", "pl"]
        self.param_units = ["MHz", "%", "MHz", "counts"]

    def guess_parameters(self, fmod=0):
        # Guess the parameters for the fit
        if not hasattr(self, "fmod"):
            self.fmod = fmod

        # Determine if the peak is positive or negative
        # We are assuming that there is more data below the FWHM than above
        avg_val = np.mean(self.ydata)
        max_val = np.max(self.ydata)
        min_val = np.min(self.ydata)

        a_guess = (max_val - min_val) / 2

        # cemter is inbetween the max and min
        pos_guess = (
            self.xdata[np.where(self.ydata == max_val)[0][0]]
            + (
                self.xdata[np.where(self.ydata == min_val)[0][0]]
                - self.xdata[np.where(self.ydata == max_val)[0][0]]
            )
            / 2
        )

        # assume that the fmod has been set to the width
        w_guess = self.fmod

        pl_guess = np.mean(self.ydata)

        self.p0 = [pos_guess, a_guess, w_guess, pl_guess]

        return self.p0

    def function(self, x, x0, a, w, c):
        return (
            a * w**2 / (w**2 + (x - 0.5 * self.fmod - x0) ** 2)
            - a * w**2 / (w**2 + (x + 0.5 * self.fmod - x0) ** 2)
            + c
        )

    def get_fit_results_txt(self, total_counts, ith_loop=1):
        results = super().get_fit_results_txt()
        results += (
            f"sens.: {self.sensitivity(total_counts, ith_loop=ith_loop):0.3e} (T/sqrt(Hz)) \n"
        )
        return results

    def generate_test_data(self):
        self.p0 = [5, 1, 1, 0]
        self.fmod = 1
        return super().generate_test_data()

    def sensitivity(self, total_counts, ith_loop=1):
        # To get the sensitivity we need to know the absolute counts.
        # Thus we need to pass this value.

        amp = self.fit_results[1]
        width = self.fit_results[2]

        planks_const = 6.62607015e-34
        muB = 9.274009994e-24
        ge = 2.0023


        if self.normalisation == "sig - ref" or self.normalisation == "ref - sig":
            # The amplitude is the difference between the peak value and the mean value
            amp = total_counts/(amp* ith_loop)
        elif self.normalisation == "norm sig / ref" or self.normalisation =="norm ref / sig":
            # The amplitude is the ratio between the peak value and the mean value
            amp = amp /100
        elif self.normalisation == "intensity" or self.normalisation == "sig / ref" or self.normalisation == "ref / sig":
            amp = amp
        else:
            # To catch the None case
            amp = total_counts/(amp* ith_loop)


        eta = (
            (4 / (3 * np.sqrt(3)))
            * (planks_const / (ge * muB))
            * np.abs(width)
            * 1e6
            / (np.abs(2 * amp) * np.sqrt(2 * total_counts))
        )
        return eta


class DifferentialGaussian(FitModel):
    def __init__(self):
        self.param_names = ["position", "amp", "width", "pl"]
        self.param_units = ["MHz", "%", "MHz", "counts"]

    def guess_parameters(self, fmod=0):
        # Guess the parameters for the fit
        if not hasattr(self, "fmod"):
            self.fmod = fmod

        # Determine if the peak is positive or negative
        # We are assuming that there is more data below the FWHM than above
        avg_val = np.mean(self.ydata)
        max_val = np.max(self.ydata)
        min_val = np.min(self.ydata)

        a_guess = (max_val - min_val) / 2

        # cemter is inbetween the max and min
        pos_guess = (
            self.xdata[np.where(self.ydata == max_val)[0][0]]
            + (
                self.xdata[np.where(self.ydata == min_val)[0][0]]
                - self.xdata[np.where(self.ydata == max_val)[0][0]]
            )
            / 2
        )

        # assume that the fmod has been set to the width
        w_guess = self.fmod
        w_guess = 1

        pl_guess = np.mean(self.ydata)

        self.p0 = [pos_guess, a_guess, w_guess, pl_guess]

        return self.p0

    def function(self, x, x0, a, w, c):
        return (
            a * np.exp(-((x - 0.5 * self.fmod - x0) ** 2) / w**2)
            - a * np.exp(-((x + 0.5 * self.fmod - x0) ** 2) / w**2)
            + c
        )

    def get_fit_results_txt(self, total_counts, ith_loop=1):
        results = super().get_fit_results_txt()
        results += (
            f"sens.: {self.sensitivity(total_counts, ith_loop=ith_loop):0.3e} (T/sqrt(Hz)) \n"
        )
        return results

    def generate_test_data(self):
        self.p0 = [5, 1, 1, 1]
        self.fmod = 1
        return super().generate_test_data()

    def sensitivity(self, total_counts, ith_loop=1):
        # To get the sensitivity we need to know the absolute counts.
        # Thus we need to pass this value.

        amp = self.fit_results[1]
        width = self.fit_results[2]

        planks_const = 6.62607015e-34
        muB = 9.274009994e-24
        ge = 2.0023

        if self.normalisation == "sig - ref" or self.normalisation == "ref - sig":
            # The amplitude is the difference between the peak value and the mean value
            amp = total_counts/(amp* ith_loop)
        elif self.normalisation == "norm sig / ref" or self.normalisation =="norm ref / sig":
            # The amplitude is the ratio between the peak value and the mean value
            amp = amp /100
        elif self.normalisation == "intensity" or self.normalisation == "sig / ref" or self.normalisation == "ref / sig":
            amp = amp
        else:
            # To catch the None case
            amp = total_counts/(amp* ith_loop)


        eta = (
            0.7
            * (planks_const / (ge * muB))
            * np.abs(width)
            * 1e6
            / (np.abs(2 * amp) * np.sqrt(2 * total_counts))
        )
        return eta


class Linear(FitModel):
    def __init__(self):
        self.param_names = ["slope", "intercept"]
        self.param_units = ["cps/MHz", "counts"]

    def guess_parameters(self, **kwargs):
        # Guess the parameters for the fit
        slope_guess = (self.ydata[-1] - self.ydata[0]) / (
            self.xdata[-1] - self.xdata[0]
        )
        intercept_guess = np.mean(self.ydata) - slope_guess * np.mean(self.xdata)

        self.p0 = [slope_guess, intercept_guess]
        return self.p0

    def function(self, x, m, c):
        return m * x + c

    def generate_test_data(self):
        self.p0 = [1, 1]
        return super().generate_test_data()
