# -*- coding: utf-8 -*-

"""
FROM QDMPY/dukit

Faster numba-compiled version of fitmodel in model.py

It's a little messy in here as I want to completely preserve the API defined in fitmodel
numba's @jitclass would be much neater.
"""

# ============================================================================

from collections import OrderedDict

import numpy as np
from numba import njit

# ============================================================================


# ======================================================================================
# ======================================================================================
#
# FastFitModel Class
#
# ======================================================================================
# ======================================================================================


class FastFitModel:
    """
    FitModel used to fit to data.
    """

    # =================================

    def __call__(self, param_ar, sweep_vec):
        """
        Evaluates fitmodel for given parameter values and sweep (affine) param values

        Arguments
        ---------
        param_ar : np array, 1D
            Array of parameters fed into each fitfunc (these are what are fit by sc)
        sweep_vec : np array, 1D or number
            Affine parameter where the fit model is evaluated

        Returns
        -------
        Fit model evaluates at sweep_vec (output is same format as sweep_vec input)
        """
        return self._eval(sweep_vec, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _eval(x, fit_params):
        raise NotImplementedError()

    # =================================

    def residuals_scipyfit(self, param_ar, sweep_vec, pl_vals):
        """Evaluates residual: fit model (affine params/sweep_vec) - pl values"""
        # NB: pl_vals unused, but left for compat with FitModel & Hamiltonian
        return self._resid(sweep_vec, pl_vals, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _resid(x, pl_vals, fit_params):
        raise NotImplementedError()

    # =================================

    def jacobian_scipyfit(self, param_ar, sweep_vec, pl_vals):
        """Evaluates (analytic) jacobian of fitmodel in format expected by
        scipy least_squares"""
        return self._jac(sweep_vec, pl_vals, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _jac(x, pl_vals, fit_params):
        raise NotImplementedError()

    # =================================

    def jacobian_defined(self):
        """Check if analytic jacobian is defined for this fit model."""
        return True  # always true for fastmodel

    # =================================

    def get_param_defn(self):
        """
        Returns list of parameters in fit_model, note there will be duplicates, and
        they do not have numbers e.g. 'pos_0'.
        Use `qdmpy.pl.model.FitModel.get_param_odict` for that purpose.

        Returns
        -------
        param_defn_ar : list
            List of parameter names (param_defn) in fit model.
        """
        raise NotImplementedError()

    # =================================

    def get_param_odict(self):
        """
        get ordered dict of key: param_key (param_name), val: param_unit for all
        parameters in fit_model

        Returns
        -------
        param_dict : dict
            Dictionary containing key: params, values: units.
        """
        raise NotImplementedError()

    # =================================

    def get_param_unit(self, param_name, param_number):
        """Get unit for a given param_key (given by param_name + "_" + param_number)

        Arguments
        ---------
        param_name : str
            Name of parameter, e.g. 'pos'
        param_number : float or int
            Which parameter to use, e.g. 0 for 'pos_0'

        Returns
        -------
        unit : str
            Unit for that parameter, e.g. "constant" -> "Amplitude (a.u.)""
        """
        if param_name == "residual":
            return "Error: sum( || residual(sweep_params) || ) over affine param (a.u.)"
        return self.get_param_odict()[param_name + "_" + str(param_number)]


# ====================================================================================


class ConstStretchedExp(FastFitModel):
    fit_functions = {"constant": 1, "stretched_exponential": 1}

    @staticmethod
    @njit(fastmath=True)
    def _eval(x, fit_params):
        c, charac_exp_t, amp_exp, power_exp = fit_params
        return amp_exp * np.exp(-((x / charac_exp_t) ** power_exp)) + c

    @staticmethod
    @njit(fastmath=True)
    def _resid(x, pl_vals, fit_params):
        c, charac_exp_t, amp_exp, power_exp = fit_params
        return amp_exp * np.exp(-((x / charac_exp_t) ** power_exp)) + c - pl_vals

    @staticmethod
    @njit(fastmath=True)
    def _jac(x, pl_vals, fit_params):
        c, charac_exp_t, amp_exp, power_exp = fit_params
        j = np.empty((x.shape[0], 4))
        j[:, 0] = 1
        j[:, 1] = (1 / charac_exp_t) * (
            amp_exp
            * power_exp
            * np.exp(-((x / charac_exp_t) ** power_exp))
            * (x / charac_exp_t) ** power_exp
        )
        # just lose the 'a'
        j[:, 2] = np.exp(-((x / charac_exp_t) ** power_exp))
        # a e^(-(x/t)^p) (x/t)^p log(x/t)
        j[:, 3] = (
            -amp_exp
            * np.exp(-((x / charac_exp_t) ** power_exp))
            * (x / charac_exp_t) ** power_exp
            * np.log(x / charac_exp_t)
        )
        return j

    def get_param_defn(self):
        return ["constant", "charac_exp_t", "amp_exp", "power_exp"]

    def get_param_odict(self):
        return OrderedDict(
            [
                ("constant_0", "Amplitude (a.u.)"),
                ("charac_exp_t_0", "Time (s)"),
                ("amp_exp_0", "Amplitude (a.u.)"),
                ("power_exp_0", "Unitless"),
            ]
        )


# ====================================================================================


class ConstDampedRabi(FastFitModel):
    fit_functions = {"constant": 1, "damped_rabi": 1}

    @staticmethod
    @njit(fastmath=True)
    def _eval(x, fit_params):
        c, omega, pos, amp, tau = fit_params
        return amp * np.exp(-(x / tau)) * np.cos(omega * (x - pos)) + c

    @staticmethod
    @njit(fastmath=True)
    def _resid(x, pl_vals, fit_params):
        c, omega, pos, amp, tau = fit_params
        return amp * np.exp(-(x / tau)) * np.cos(omega * (x - pos)) + c - pl_vals

    @staticmethod
    @njit(fastmath=True)
    def _jac(x, pl_vals, fit_params):
        c, omega, pos, amp, tau = fit_params
        j = np.empty((x.shape[0], 5))
        j[:, 0] = 1
        j[:, 1] = (
            amp * (pos - x) * np.sin(omega * (x - pos)) * np.exp(-x / tau)
        )  # wrt omega
        j[:, 2] = (amp * omega * np.sin(omega * (x - pos))) * np.exp(
            -x / tau
        )  # wrt pos
        j[:, 3] = np.exp(-x / tau) * np.cos(omega * (x - pos))  # wrt amp
        j[:, 4] = (amp * x * np.cos(omega * (x - pos))) / (
            np.exp(x / tau) * tau**2
        )  # wrt tau
        return j

    def get_param_defn(self):
        return ["constant", "rabi_freq", "rabi_t_offset", "rabi_amp", "rabi_decay_time"]

    def get_param_odict(self):
        return OrderedDict(
            [
                ("constant_0", "Amplitude (a.u.)"),
                ("rabi_freq_0", "Omega (rad/s)"),
                ("rabi_t_offset_0", "Tau_0 (s)"),
                ("rabi_amp_0", "Amp (a.u.)"),
                ("rabi_decay_time_0", "Tau_d (s)"),
            ]
        )


# ====================================================================================


class LinearLorentzians(FastFitModel):
    def __init__(self, n_lorentzians):
        self.n_lorentzians = n_lorentzians
        self.fit_functions = {"linear": 1, "lorentzian": n_lorentzians}

    def __call__(self, param_ar, sweep_vec):
        return self._eval(self.n_lorentzians, sweep_vec, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _eval(n, x, fit_params):
        c = fit_params[0]
        m = fit_params[1]
        val = m * x + c
        for i in range(n):
            fwhm = fit_params[i * 3 + 2]
            pos = fit_params[i * 3 + 3]
            amp = fit_params[i * 3 + 4]
            hwhmsqr = (fwhm**2) / 4
            val += amp * hwhmsqr / ((x - pos) ** 2 + hwhmsqr)
        return val

    def residuals_scipyfit(self, param_ar, sweep_vec, pl_vals):
        """Evaluates residual: fit model (affine params/sweep_vec) - pl values"""
        return self._resid(self.n_lorentzians, sweep_vec, pl_vals, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _resid(n, x, pl_vals, fit_params):
        c = fit_params[0]
        m = fit_params[1]
        val = m * x + c
        for i in range(n):
            fwhm = fit_params[i * 3 + 2]
            pos = fit_params[i * 3 + 3]
            amp = fit_params[i * 3 + 4]
            hwhmsqr = (fwhm**2) / 4
            val += amp * hwhmsqr / ((x - pos) ** 2 + hwhmsqr)
        return val - pl_vals

    def jacobian_scipyfit(self, param_ar, sweep_vec, pl_vals):
        """Evaluates (analytic) jacobian of fitmodel in format expected
        by scipy least_squares"""
        return self._jac(self.n_lorentzians, sweep_vec, pl_vals, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _jac(n, x, pl_vals, fit_params):
        j = np.empty((x.shape[0], 2 + 3 * n), dtype=np.float64)
        j[:, 0] = 1  # wrt constant
        j[:, 1] = x  # wrt m
        for i in range(n):
            fwhm = fit_params[i * 3 + 2]
            c = fit_params[i * 3 + 3]
            a = fit_params[i * 3 + 4]
            g = fwhm / 2

            j[:, 2 + i * 3] = (a * g * (x - c) ** 2) / ((x - c) ** 2 + g**2) ** 2
            j[:, 3 + i * 3] = (2 * a * g**2 * (x - c)) / (g**2 + (x - c) ** 2) ** 2
            j[:, 4 + i * 3] = g**2 / ((x - c) ** 2 + g**2)
        return j

    def get_param_defn(self):
        defn = ["c", "m"]
        for i in range(self.n_lorentzians):
            defn += ["fwhm", "pos", "amp"]
        return defn

    def get_param_odict(self):
        defn = [("c_0", "Amplitude (a.u.)"), ("m_0", "Amplitude per Freq (a.u.)")]
        for i in range(self.n_lorentzians):
            defn += [(f"fwhm_{i}", "Freq (MHz)")]
            defn += [(f"pos_{i}", "Freq (MHz)")]
            defn += [(f"amp_{i}", "Amp (a.u.)")]
        return OrderedDict(defn)


# ====================================================================================


class ConstLorentzians(FastFitModel):
    def __init__(self, n_lorentzians):
        self.n_lorentzians = n_lorentzians
        self.fit_functions = {"constant": 1, "lorentzian": n_lorentzians}

    def __call__(self, param_ar, sweep_vec):
        return self._eval(self.n_lorentzians, sweep_vec, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _eval(n, x, fit_params):
        c = fit_params[0]
        val = c * np.ones(np.shape(x))
        for i in range(n):
            fwhm = fit_params[i * 3 + 1]
            pos = fit_params[i * 3 + 2]
            amp = fit_params[i * 3 + 3]
            hwhmsqr = (fwhm**2) / 4
            val += amp * hwhmsqr / ((x - pos) ** 2 + hwhmsqr)
        return val

    def residuals_scipyfit(self, param_ar, sweep_vec, pl_vals):
        """Evaluates residual: fit model (affine params/sweep_vec) - pl values"""
        return self._resid(self.n_lorentzians, sweep_vec, pl_vals, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _resid(n, x, pl_vals, fit_params):
        c = fit_params[0]
        val = c * np.ones(np.shape(x))
        for i in range(n):
            fwhm = fit_params[i * 3 + 1]
            pos = fit_params[i * 3 + 2]
            amp = fit_params[i * 3 + 3]
            hwhmsqr = (fwhm**2) / 4
            val += amp * hwhmsqr / ((x - pos) ** 2 + hwhmsqr)
        return val - pl_vals

    def jacobian_scipyfit(self, param_ar, sweep_vec, pl_vals):
        """Evaluates (analytic) jacobian of fitmodel in format expected
        by scipy least_squares"""
        return self._jac(self.n_lorentzians, sweep_vec, pl_vals, param_ar)

    @staticmethod
    @njit(fastmath=True)
    def _jac(n, x, pl_vals, fit_params):
        j = np.empty((x.shape[0], 1 + 3 * n), dtype=np.float64)
        j[:, 0] = 1  # wrt constant
        for i in range(n):
            fwhm = fit_params[i * 3 + 1]
            c = fit_params[i * 3 + 2]
            a = fit_params[i * 3 + 3]
            g = fwhm / 2

            j[:, 2 + i * 3] = (a * g * (x - c) ** 2) / ((x - c) ** 2 + g**2) ** 2
            j[:, 3 + i * 3] = (2 * a * g**2 * (x - c)) / (g**2 + (x - c) ** 2) ** 2
            j[:, 4 + i * 3] = g**2 / ((x - c) ** 2 + g**2)
        return j

    def get_param_defn(self):
        defn = ["constant"]
        for i in range(self.n_lorentzians):
            defn += ["fwhm", "pos", "amp"]
        return defn

    def get_param_odict(self):
        defn = [("constant_0", "Amplitude (a.u.)")]
        for i in range(self.n_lorentzians):
            defn += [(f"fwhm_{i}", "Freq (MHz)")]
            defn += [(f"pos_{i}", "Freq (MHz)")]
            defn += [(f"amp_{i}", "Amp (a.u.)")]
        return OrderedDict(defn)
