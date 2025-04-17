def set_matplotlib_style(style: str = "dark"):
    import matplotlib.pyplot as plt

    # TODO define:
    #  font family
    #  common linewidths etc.?

    # Set the color scheme
    if style == "dark":
        dark_grey = "#282828"
        light_grey = "#505050"
        plt.rcParams.update(
            {
                "lines.color": "white",
                "patch.edgecolor": "white",
                "text.color": dark_grey,
                "axes.facecolor": dark_grey,
                "figure.facecolor": dark_grey,
                "axes.edgecolor": "white",
                "axes.labelcolor": "white",
                "xtick.color": "white",
                "ytick.color": "white",
                "grid.color": light_grey,
                # make the background the axis transparent
                "axes.labelsize": 12,
                # Set the legend font color
                "legend.labelcolor": "white",
                "legend.facecolor": dark_grey,
                "legend.framealpha": 0,
                "legend.edgecolor": light_grey,
                "figure.facecolor": dark_grey,
                "figure.edgecolor": dark_grey,
                "savefig.facecolor": dark_grey,
                "savefig.edgecolor": dark_grey,
            }
        )
    elif style == "light":
        light_grey = "#F0F0F0"
        dark_grey = "#D0D0D0"
        plt.rcParams.update(
            {
                "lines.color": "black",
                "patch.edgecolor": "black",
                "text.color": "black",
                "axes.facecolor": "white",
                "axes.edgecolor": "black",
                "axes.labelcolor": "black",
                "xtick.color": "black",
                "ytick.color": "black",
                "grid.color": dark_grey,
                # Set the legend font color
                "legend.labelcolor": "black",
                "legend.facecolor": "white",
                "legend.framealpha": 0,
                "figure.facecolor": "white",
                "figure.edgecolor": "white",
                "savefig.facecolor": "white",
                "savefig.edgecolor": "white",
            }
        )


def get_contrasting_text_color(background_color):
    """
    Determine if text should be black or white based on background color.
    Uses luminance calculation to determine contrast.

    Parameters
    ----------
    background_color : str or tuple
     The background color as a matplotlib color string or RGB(A) tuple

    Returns
    -------
    str
     'black' or 'white' depending on which provides better contrast
    """
    from matplotlib import colors as mcolors

    # Convert color to RGB if it's a string
    if isinstance(background_color, str):
        rgb = mcolors.to_rgb(background_color)
    else:
        rgb = background_color[:3]  # Take first 3 values if RGBA

    # Calculate luminance using standard weights
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

    # Use white text for dark backgrounds, black text for light backgrounds
    return "white" if luminance < 0.5 else "black"
