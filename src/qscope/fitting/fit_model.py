import matplotlib.pyplot as plt
import numpy as np
import scipy
from loguru import logger
from scipy.optimize import curve_fit


class FitModel:
    def __init__(self):
        pass

    def set_data(self, xdata, ydata_sig, ydata_ref):
        self.xdata = xdata
        self.ydata_signal = ydata_sig
        self.ydata_ref = ydata_ref

    def set_normalization(self, normalisation):
        self.normalisation = normalisation
        if normalisation == "None":
            self.ydata = self.ydata_signal
        elif normalisation == "sig - ref":
            self.ydata = self.ydata_signal - self.ydata_ref
        elif normalisation == "sig / ref":
            self.ydata = self.ydata_signal / self.ydata_ref
        elif normalisation == "ref - sig":
            self.ydata = self.ydata_ref - self.ydata_signal
        elif normalisation == "ref / sig":
            self.ydata = self.ydata_ref / self.ydata_signal
        elif normalisation == "norm sig / ref":
            self.ydata = 100 * ((self.ydata_signal / self.ydata_ref) - 1)
        elif normalisation == "norm ref / sig":
            self.ydata = 100 * ((self.ydata_ref / self.ydata_signal) - 1)
        elif normalisation == "intensity":
            self.ydata = (self.ydata_signal - self.ydata_ref)/ (self.ydata_signal + self.ydata_ref)
        else:
            self.ydata = self.ydata_signal
            self.normalisation = "None"

    def fit(self):
        if not hasattr(self, "p0"):
            self.guess_parameters()

        self.fit_results, pcov = curve_fit(
            self.function, self.xdata, self.ydata, self.p0
        )
        # get the fit error
        self.fit_error = np.sqrt(np.diag(pcov))

        logger.debug(f"Fit initial guess: {self.p0}")

        return

    def best_fit(self):
        self.x_fit = np.linspace(
            np.min(self.xdata), np.max(self.xdata), 3 * len(self.xdata)
        )
        self.y_fit = self.function(self.x_fit, *self.fit_results)
        return self.x_fit, self.y_fit

    def generate_test_data(self):
        self.xdata = np.linspace(0, 10, 200)
        self.ydata = self.function(self.xdata, *self.p0) + np.random.normal(
            0, 0.1, len(self.xdata)
        )

    def guess_parameters(self, **kwargs):
        raise NotImplementedError

    def function(self):
        raise NotImplementedError

    def get_results_txt(self):
        raise NotImplementedError

    def get_initial_guess_txt(self):
        results = "Initial guess:\n"
        for name, val, unit in zip(self.param_names, self.p0, self.param_units):
            results += f"{name}: {val:0.2f} {unit}\n"
        return results

    def print_initial_guess(self):
        print(self.get_initial_guess_txt())

    def get_fit_results_txt(self, *args, **kwargs):
        results = "Fit results:\n"
        for name, val, error, unit in zip(
            self.param_names, self.fit_results, self.fit_error, self.param_units
        ):
            if name == "position":
                results += f"{name}: {val:0.3f} ± {error:0.3f} {unit}\n"
            else:
                results += f"{name}: {val:0.5e} ± {error:0.5e} {unit}\n"
        return results

    def print_results(self):
        print(self.get_fit_results_txt())

    def plot(self):
        plt.figure()
        x_fit, y_fit = self.best_fit()
        guess = self.function(x_fit, *self.p0)

        plt.plot(self.xdata, self.ydata, "o", label="data", markersize=1)

        plt.plot(x_fit, guess, "--", color=[0.6, 0.6, 0.6], label="guess")
        plt.plot(x_fit, y_fit, "-", label="fit")
        plt.legend()
        plt.xlabel("x data (a.u.)")
        plt.ylabel("PL (a.u.)")
