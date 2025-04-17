from matplotlib import ticker
from mpl_toolkits.axes_grid1 import axes_size, make_axes_locatable


def add_colorbar(
    im,
    fig,
    ax,
    aspect=20,
    pad_fraction=1,
    locator=None,
    orientation="vertical",
    labelpad=15,
    **kwargs,
):
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
    locator : array-like
        List of locations for ticks. Default: None (uses MaxNLocator nbins=7)

    **kwargs : other keyword arguments
        Passed to fig.colorbar.

    """
    if orientation == "vertical":
        divider = make_axes_locatable(ax)
        if aspect:
            width = axes_size.AxesY(ax, aspect=1.0 / aspect)
        else:
            width = axes_size.AxesY(ax)
        if pad_fraction:
            pad = axes_size.Fraction(pad_fraction, width)
            cax = divider.append_axes("right", size=width, pad=pad)
        else:
            cax = divider.append_axes("right", size=width)
        cbar = fig.colorbar(im, cax=cax, **kwargs)
        if locator:
            tick_locator = ticker.FixedLocator(locator)
        else:
            tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.ax.get_yaxis().labelpad = labelpad
        cbar.ax.linewidth = 0.5
    else:
        divider = make_axes_locatable(ax)
        width = axes_size.AxesX(ax, aspect=1.0 / aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("top", size=width, pad=pad)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal", **kwargs)
        if locator:
            tick_locator = ticker.FixedLocator(locator)
        else:
            tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.ax.get_xaxis().labelpad = labelpad
        cbar.ax.linewidth = 0.5

    return cbar
