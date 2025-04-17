# -*- coding: utf-8 -*-

import concurrent.futures
import copy
import os
from itertools import repeat

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import axes_size, make_axes_locatable
from rebin import rebin
from scipy.linalg import svd
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from tqdm.autonotebook import tqdm  # auto detects jupyter

# ============================================================================
from qscope.fitting.image_fitting import fastmodel

# ==========================================================================


def define_fit_model():
    """Define (and return) fit_model object,"""
    return fastmodel.LinearLorentzians(1)


def prepare_data(image, smoothing, binning):
    image = _smooth_image(image, smoothing)
    image_rebinned, sig, ref, sig_norm = _rebin_image(image, binning)
    return image_rebinned, sig, ref, sig_norm


def _rebin_image(image, binning):
    """
    Reshapes raw data into more useful shape, according to image size in metadata.

    Arguments
    ---------
    image : np array, 3D
        Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and has
        not been rebinned.
    binning : int

    Returns
    -------
    image_rebinned : np array, 3D
        Format: [sweep values, y, x]. Same as image, but now rebinned (x size and y size
        have changed). Not cut down to ROI.
    sig : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps not removed yet.
        Format: [sweep_vals, y, x]. Not cut down to ROI.
    ref : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps not removed yet.
        Not cut down to ROI. Format: [sweep_vals, y, x].
    sig_norm : np array, 3D
        Signal normalised by reference (via subtraction or normalisation, chosen in options),
        reshaped and rebinned.  Unwanted sweeps not removed yet.
        Not cut down to ROI. Format: [sweep_vals, y, x].
    """

    image_rebinned = rebin(image, factor=(1, binning, binning), func=np.mean)

    sig = image_rebinned[::2, :, :]
    ref = image_rebinned[1::2, :, :]
    sig_norm = sig / ref
    return image_rebinned, sig, ref, sig_norm


def _smooth_image(image, smoothing):
    """Smooth each frame of image (3d dataset) with direct (non-ft) gaussian filter."""
    return gaussian_filter(image, sigma=[0, smoothing, smoothing])


def to_squares_wrapper(fun, p0, sweep_vec, shaped_data, fit_optns):
    """
    Simple wrapper of scipy.optimize.least_squares to allow us to keep track of which
    solution is which (or where).

    Arguments
    ---------
    fun : function
        Function object acting as residual (fit model minus pl value)
    p0 : np array
        Initial guess: array of parameters
    sweep_vec : np array
        Array (or I guess single value, anything iterable) of affine parameter (tau/freq)
    shaped_data : list (3 elements)
        array returned by `qdmpy.pl.common.pixel_generator`: [y, x, sig_norm[:, y, x]]
    fit_optns : dict
        Other options (dict) passed to least_squares

    Returns
    -------
    wrapped_squares : tuple
        (y, x), least_squares(...).x, leas_squares(...).jac
        I.e. the position of the fit result, the fit result parameters array, jacobian at solution
    """
    # shaped_data: [y, x, pl]
    # output: (y, x), result_params, jac
    try:
        fitres = least_squares(fun, p0, args=(sweep_vec, shaped_data[2]), **fit_optns)
        fitp = fitres.x
        fitj = fitres.jac
    except ValueError:
        fitp = np.empty(np.shape(p0))
        fitj = None
    return ((shaped_data[0], shaped_data[1]), fitp, fitj)


# ==========================================================================


def fit_all_pixels_pl_scipyfit(
    sig_norm, sweep_list, fit_model, init_guess_params, fit_optns
):
    """
    Fits each pixel and returns dictionary of param_name -> param_image.

    Arguments
    ---------
    sig_norm : np array, 3D
        Normalised meas array, shape: [sweep_list, y, x].
    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq)
    fit_model : `qdmpy.pl.model.FitModel`
        The model we're fit to.


    Returns
    -------
    fit_image_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual' as a key.
    """
    sweep_ar = np.array(sweep_list)
    threads = 4
    num_pixels = np.shape(sig_norm)[1] * np.shape(sig_norm)[2]

    # divide pixels by numbers of threads (workers) to use
    chunksize = int(num_pixels / threads)

    pixel_data = sig_norm

    if not chunksize:
        print("chunksize was 0, setting to 1")
        chunksize = 1

    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        fit_results = list(
            tqdm(
                executor.map(
                    to_squares_wrapper,
                    repeat(fit_model.residuals_scipyfit),
                    repeat(init_guess_params),
                    repeat(sweep_ar),
                    pixel_generator(pixel_data),
                    repeat(fit_optns),
                    chunksize=chunksize,
                ),
                desc="pl-scipyfit",
                ascii=True,
                mininterval=1,
                total=num_pixels,
                unit=" PX",
                disable=True,
            )
        )

    res, sigmas = get_pixel_fitting_results(
        fit_model, fit_results, pixel_data, sweep_ar
    )

    return res, sigmas


# ==========================================================================


def get_pixel_fitting_results(fit_model, fit_results, pixel_data, sweep_list):
    """
    Take the fit result data from scipyfit/gpufit and back it down to a dictionary of arrays.

    Each array is 2D, representing the values for each parameter (specified by the dict key).

    Arguments
    ---------
    fit_model : `qdmpy.pl.model.FitModel`
        Model we're fit to.
    fit_results : list of [(y, x), result, jac] objects
        (see `qdmpy.pl.scipyfit.to_squares_wrapper`, or corresponding gpufit method)
        A list of each pixel's parameter array, as well as position in image denoted by (y, x).
    pixel_data : np array, 3D
        Normalised meas array, shape: [sweep_list, y, x]. i.e. sig_norm.
        May or may not already be shuffled (i.e. matches fit_results).
    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq).

    Returns
    -------
    fit_image_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual' as a key.
    sigmas : dict
        Dictionary, key: param_keys, val: image (2D) of param uncertainties across FOV.
    """

    roi_shape = np.shape(pixel_data)[1:]

    # initialise dictionary with key: val = param_name: param_units
    fit_image_results = fit_model.get_param_odict()
    sigmas = copy.copy(fit_image_results)

    # override with correct size empty arrays using np.zeros
    for key in fit_image_results.keys():
        fit_image_results[key] = np.zeros((roi_shape[0], roi_shape[1])) * np.nan
        sigmas[key] = np.zeros((roi_shape[0], roi_shape[1])) * np.nan

    fit_image_results["residual_0"] = np.zeros((roi_shape[0], roi_shape[1])) * np.nan

    # Fill the arrays element-wise from the results function, which returns a
    # 1D array of flattened best-fit parameters.
    for (y, x), result, jac in fit_results:
        filled_params = {}  # keep track of index, i.e. pos_0, for this pixel

        if jac is None:  # can't fit this pixel
            fit_image_results["residual_0"][y, x] = np.nan
            perr = np.empty(np.shape(result))
            perr[:] = np.nan
        else:
            # NOTE we decide not to call each backend separately here
            resid = fit_model.residuals_scipyfit(
                result, sweep_list, pixel_data[:, y, x]
            )
            fit_image_results["residual_0"][y, x] = np.sum(
                np.abs(resid, dtype=np.float64), dtype=np.float64
            )
            # uncertainty (covariance matrix), copied from scipy.optimize.curve_fit (not abs. sigma)
            _, s, vt = svd(jac, full_matrices=False)
            threshold = np.finfo(float).eps * max(jac.shape) * s[0]
            s = s[s > threshold]
            vt = vt[: s.size]
            pcov = np.dot(vt.T / s**2, vt)
            # NOTE using/assuming linear cost fn,
            cost = 2 * 0.5 * np.sum(fit_model(result, sweep_list) ** 2)
            s_sq = cost / (len(resid) - len(result))
            pcov *= s_sq
            perr = np.sqrt(np.diag(pcov))  # array of standard deviations

        for param_num, param_name in enumerate(fit_model.get_param_defn()):
            # keep track of what index we're up to, i.e. pos_1
            if param_name not in filled_params.keys():
                key = param_name + "_0"
                filled_params[param_name] = 1
            else:
                key = param_name + "_" + str(filled_params[param_name])
                filled_params[param_name] += 1

            fit_image_results[key][y, x] = result[param_num]
            sigmas[key][y, x] = perr[param_num]

    return fit_image_results, sigmas


# ==========================================================================


def pixel_generator(our_array):
    """
    Simple generator to shape data as expected by to_squares_wrapper in scipy concurrent method.

    Also allows us to track *where* (i.e. which pixel location) each result corresponds to.
    See also: `qdmpy.pl.scipyfit.to_squares_wrapper`, and corresponding gpufit method.

    Arguments
    ---------
    our_array : np array, 3D
        Shape: [sweep_list, y, x]

    Returns
    -------
    generator : list
        [y, x, our_array[:, y, x]] generator (yielded)
    """
    _, len_y, len_x = np.shape(our_array)
    for y in range(len_y):
        for x in range(len_x):
            yield [y, x, our_array[:, y, x]]


# ==========================================================================
#                       PLOTTING FUNCTIONS
# ==========================================================================


def pl_param_image(
    image,
    param_name,
    c_label,
    errorplot=False,
    c_map="viridis",
    c_range=None,
    save_plot=False,
    dir_path=None,
    pixel_size=1.0,
    clear_plot=False,
):
    """
    Plots an image corresponding to a single parameter in pixel_fit_params.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    fit_model : `qdmpy.pl.model.FitModel`
        Model we're fit to.
    pixel_fit_params : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
    param_name : str
        Name of parameter you want to plot, e.g. 'fwhm'. Can also be 'residual'.

    Optional arguments
    ------------------
    param_number : int
        Which version of the parameter you want. I.e. there might be 8 independent parameters
        in the fit model called 'pos', each labeled 'pos_0', 'pos_1' etc. Default: 0.
    errorplot : bool
        Default: false. Denotes that errors dict has been passed in (e.g. sigmas), so
        ylabel & save names are changed accordingly. Can't be True if param_name='residual'.

    Returns
    -------
    fig : matplotlib Figure object
    """

    fig, ax = plt.subplots()

    if c_range is None:
        c_range = [np.min(image), np.max(image)]

    im = ax.imshow(image, vmin=c_range[0], vmax=c_range[1], cmap=c_map)

    ax.set_title(param_name)

    cbar = _add_colorbar(im, fig, ax)
    cbar.ax.set_ylabel(c_label, rotation=270)

    # scalebar = ScaleBar(pixel_size)
    # ax.add_artist(scalebar)
    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])

    if dir_path[-1] != "/":
        dir_path = dir_path + "/"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if save_plot:
        if errorplot:
            path = dir_path + "/" + (param_name + "_sigma." + ".png")
        else:
            path = dir_path + "/" + (param_name + ".png")
        fig.savefig(path)

        # save data
        data_path = dir_path + "data/"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        np.savetxt(data_path + param_name + ".txt", image)
    if clear_plot:
        plt.close(fig)
        return None
    else:
        return fig


def _add_colorbar(im, fig, ax, aspect=20, pad_fraction=1, **kwargs):
    """
    Adds a colorbar to matplotlib axis

    Arguments
    ---------
    im : image as returned by ax.imshow
    fig : matplotlib Figure object
    ax : matplotlib Axis object

    Returns
    -------
    cbar : matplotlib colorbar object

    Optional Arguments
    ------------------
    aspect : int
        Reciprocal of aspect ratio passed to new colorbar axis width. Default: 20.
    pad_fraction : int
        Fraction of new colorbar axis width to pad from image. Default: 1.

    **kwargs : other keyword arguments
        Passed to fig.colorbar.

    """
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1.0 / aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    cbar = fig.colorbar(im, cax=cax, **kwargs)
    tick_locator = mpl.ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.linewidth = 0.5
    return cbar
