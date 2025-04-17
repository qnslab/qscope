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


class DecayModel(FitModel):
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

    def get_fit_results_txt(self, *args, **kwargs):
        results = "Fit results:\n"
        for name, val, error, unit in zip(
            self.param_names, self.fit_results, self.fit_error, self.param_units
        ):
            if unit == "s":
                unit = get_time_unit(self.x_multiplier)
            results += f"{name}: {val:0.2e} ± {error:0.2e} {unit}\n"
        return results


class ExponentialDecay(DecayModel):
    def __init__(self):
        self.param_names = [
            "amp",
            "decay",
            "background",
        ]
        self.param_units = [
            "(a.u.)",
            "s",
            "cps",
        ]

    def guess_parameters(self, **kwargs):
        # Guess the parameters for the fit

        # Determine if the peak is positive or negative
        # We are assuming that there is more data below the FWHM than above
        avg_val = np.mean(self.ydata)
        max_val = np.max(self.ydata)
        min_val = np.min(self.ydata)

        a_guess = max_val - min_val

        pl_guess = np.mean(self.ydata)

        # Assume that the decay is 0.5 of the total time range
        decay_guess = 0.5 * (self.xdata[-1] - self.xdata[0])

        self.p0 = [a_guess, decay_guess, pl_guess]

        return

    def function(self, x, a, t, c):
        return a * np.exp(-x / t) + c

    def generate_test_data(self):
        self.p0 = [1, 1, 0]
        return super().generate_test_data()


class GuassianDecay(DecayModel):
    def __init__(self):
        self.param_names = [
            "amp",
            "decay",
            "background",
        ]
        self.param_units = [
            "(a.u.)",
            "s",
            "cps",
        ]

    def guess_parameters(self, **kwargs):
        # Guess the parameters for the fit

        # Determine if the peak is positive or negative
        # We are assuming that there is more data below the FWHM than above
        avg_val = np.mean(self.ydata)
        max_val = np.max(self.ydata)
        min_val = np.min(self.ydata)

        a_guess = max_val - min_val

        pl_guess = np.mean(self.ydata)

        # Assume that the decay is 0.5 of the total time range
        decay_guess = 0.5 * (self.xdata[-1] - self.xdata[0])

        self.p0 = [a_guess, decay_guess, pl_guess]

        return

    def function(self, x, a, t, c):
        return a * np.exp(-((x / t) ** 2)) + c

    def generate_test_data(self):
        self.p0 = [1, 3, 0]
        return super().generate_test_data()


class StretchedExponentialDecay(DecayModel):
    def __init__(self):
        self.param_names = ["amp", "decay", "expon", "background"]
        self.param_units = ["(a.u.)", "s", "a.u.", "cps"]

    def guess_parameters(self, **kwargs):
        # Guess the parameters for the fit

        # Determine if the peak is positive or negative
        # We are assuming that there is more data below the FWHM than above
        avg_val = np.mean(self.ydata)
        max_val = np.max(self.ydata)
        min_val = np.min(self.ydata)

        a_guess = max_val - min_val

        pl_guess = np.mean(self.ydata)

        p_guess = 1.5

        # Assume that the decay is 0.5 of the total time range
        decay_guess = 0.3 * (self.xdata[-1] - self.xdata[0])

        self.p0 = [a_guess, decay_guess, p_guess, pl_guess]

        return

    def function(self, x, a, t, p, c):
        return a * np.exp(-((x / t) ** p)) + c

    def generate_test_data(self):
        self.p0 = [1, 1, 1.5, 0]
        return super().generate_test_data()
