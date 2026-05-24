import colorsys
import os
import re
import warnings
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objects as go
from matplotlib.axes import Axes
from plotly.graph_objs import Figure
from scipy.stats import t

T = TypeVar('T')

COLORS_SNS_10 = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)
]


def cummax(x: np.ndarray, return_ind=False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """ Return array containing at index `i` the value max(X)[:i] """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    cmaxind: list[int] = [0]
    cmax: list[float] = [x[0].item()]
    for i, xx in enumerate(x[1:]):
        assert isinstance(xx, np.number), xx
        i += 1
        if xx > cmax[-1]:
            cmax.append(xx.item())
            cmaxind.append(i)
        else:
            cmax.append(cmax[-1])
            cmaxind.append(cmaxind[-1])
    cmax_np = np.array(cmax)
    assert np.all(x[cmaxind] == cmax_np), (x, x[cmaxind], cmax_np)
    if return_ind:
        return cmax_np, np.array(cmaxind)
    return cmax_np


def get_cummax(scores: list[np.ndarray] | np.ndarray) -> list[np.ndarray]:
    """ Compute cumulative max for each array in a list

    Args:
        scores: list of the arrays on which `cummax` will be applied

    Returns:
        cmaxs:
    """
    if not isinstance(scores, list) and isinstance(scores, np.ndarray):
        scores = np.atleast_2d(scores)
    else:
        raise TypeError(f'Expected list[np.ndarray] or np.ndarray, got {type(scores)}')

    cmaxs: list[np.ndarray] = []
    for score in scores:
        cmaxs.append(cummax(score))
    return cmaxs


def get_cummin(scores: list[np.ndarray] | np.ndarray) -> list[np.ndarray]:
    """ Compute cumulative min for each array in a list

    Args:
        scores: list of the arrays on which `cummin` will be applied

    Returns:
        cmins:
    """
    if not isinstance(scores, list) and isinstance(scores, np.ndarray):
        scores = np.atleast_2d(scores)
    else:
        raise TypeError(f'Expected list[np.ndarray] or np.ndarray, got {type(scores)}')
    cmins: list[np.ndarray] = []
    for score in scores:
        cmins.append(-cummax(-score))
    return cmins


def get_common_chunk_sizes(ys: list[np.ndarray]) -> list[tuple[np.ndarray, list[list[int]]]]:
    """ From a list of arrays of various sizes, get a list of `list of arrays having the same size`

     Example:
         >>> y_s = [[1, 3 ,4, 5],
                   [0, 7, 8 , 2, 9],
                   [-1]]
         >>> get_common_chunk_sizes(y_s)
         ---> [
         --->   ([0], [[1], [0], [-1]]),               # gather all elements of index in [0]
         --->   ([1, 2, 3], [[3, 4, 5], [7, 8, 2]]),   # gather all elements of index in [1, 2, 3]
         --->   ([4], [[9]])                           # gather all elements of index in [4]
         ---> ]
     """
    ys = [y for y in ys if len(y) > 0]
    lens = [0] + sorted(set([len(y) for y in ys]))

    output = []
    for i in range(1, len(lens)):
        x_s = np.arange(lens[i - 1], lens[i])
        y = [y[lens[i - 1]:lens[i]] for y in ys if len(y) >= lens[i]]
        output.append((x_s, y))
    return output


def plot_mean_std(
        *args, n_std: float | None = 1,
        ax: Axes | None = None, alpha: float = .3, errbar: bool = False,
        lb: float | np.ndarray | None = None,
        ub: float | np.ndarray | None = None,
        linewidth: int = 3,
        show_std_error: bool = False,
        ci_level: float | None = None,
        **plot_mean_kwargs
        ) -> Axes:
    """ Plot mean and std (with fill between) of sequential data Y of shape (n_trials, lenght_of_a_trial)

    Args:
        args: x-values (if None, we will take `range(0, len(Y))`), y-values
        n_std: number of std to plot around the mean (if `0` only the mean is plotted)
        ax: axis on which to plot the curves
        alpha: parameter for `fill_between`
        errbar: use error bars instead of shaded area
        ci_level: show confidence interval over the mean at specified level (e.g. 0.95), otherwise uncertainty shows
          n_std std around the mean
        linewidth: width of line
        lb: lower bound (to clamp uncertainty region)
        ub: upper bound (to clamp uncertainty region)
        show_std_error: show standard error (std / sqrt(n_samples)) as shadow area around mean curve

    Returns:
        The axis.
    """
    warnings.warn("Use plotly version `plotly_mean_std`", DeprecationWarning)
    if len(args) == 1:
        y = args[0]
        x = None
    elif len(args) == 2:
        x, y = args
    else:
        raise RuntimeError('Wrong number of arguments (should be [X], Y,...)')

    assert len(y) > 0, 'Y should be a non-empty array, nothing to plot'
    y = np.atleast_2d(y)
    if x is None:
        x = np.arange(y.shape[1])
    assert x.ndim == 1, f'X should be of rank 1, got {x.ndim}'
    mean = y.mean(0)
    std = y.std(0)
    if ax is None:
        ax = plt.subplot()

    if len(x) == 0:
        return ax

    if ci_level is not None and len(y) > 1:
        # student
        t_crit = np.abs(t.ppf((1 - ci_level) / 2, len(y) - 1))
        n_std = t_crit / np.sqrt(len(y))
    elif show_std_error:
        n_std = 1 / np.sqrt(len(y))

    if errbar:
        n_errbars = min(10, len(std))
        errbar_inds = len(std) // n_errbars
        ax.errorbar(x, mean, yerr=n_std * std, errorevery=errbar_inds, linewidth=linewidth, **plot_mean_kwargs)
    else:
        line_plot = ax.plot(x, mean, linewidth=linewidth, **plot_mean_kwargs)

        if n_std > 0 and y.shape[0] > 1:
            uncertainty_lb = mean - n_std * std
            uncertainty_ub = mean + n_std * std
            if lb is not None:
                uncertainty_lb = np.maximum(uncertainty_lb, lb)
            if ub is not None:
                uncertainty_ub = np.minimum(uncertainty_ub, ub)

            ax.fill_between(x, uncertainty_lb, uncertainty_ub, alpha=alpha, color=line_plot[0].get_c())

    return ax


def plotly_mean_std(
        *args, color: tuple[float, float, float], n_std: float | None = 1, fig: Figure | None = None,
        label: str | None = None,
        alpha: float = .3, errbar: bool = False,
        lb: float | np.ndarray | None = None,
        ub: float | np.ndarray | None = None,
        linewidth: int = 3,
        show_std_error: bool | None = False,
        ci_level: float | None = None,
        **plot_mean_kwargs
        ) -> Figure:
    """ Plot mean and std (with fill between) of sequential data Y of shape (n_trials, length_of_a_trial)

    Args:
        args: x-values (if None, we will take `range(0, len(Y))`), y-values
        n_std: number of std to plot around the mean (if `0` only the mean is plotted)
        alpha: parameter for `fill_between`
        fig: plotly figure
        label: label of plot (used when interacting with the plots, but not in the legend as showlegend is False)
        color: specified in rgb
        errbar: use error bars instead of shaded area
        ci_level: show confidence interval over the mean at specified level (e.g. 0.95), otherwise uncertainty shows
          n_std std around the mean
        linewidth: width of line
        lb: lower bound (to clamp uncertainty region)
        ub: upper bound (to clamp uncertainty region)
        show_std_error: show standard error (std / sqrt(n_samples)) as shadow area around mean curve
    """
    if len(args) == 1:
        y = args[0]
        x = None
    elif len(args) == 2:
        x, y = args
    else:
        raise RuntimeError('Wrong number of arguments (should be [X], Y,...)')

    assert len(y) > 0, 'Y should be a non-empty array, nothing to plot'
    y = np.atleast_2d(y)
    if x is None:
        x = np.arange(y.shape[1])
    assert x.ndim == 1, f'X should be of rank 1, got {x.ndim}'
    mean = y.mean(0)
    std = y.std(0)

    if fig is None:
        fig = go.Figure()

    if len(x) == 0:
        return fig

    # Plot the mean line
    fig.add_trace(
        go.Scatter(
            x=x, y=mean,
            mode='lines',
            line=dict(width=linewidth, color='rgb{}'.format(color), **plot_mean_kwargs),
            name=label,
            showlegend=False,
        )
    )

    if ci_level is not None and len(y) > 1:
        # student
        t_crit = np.abs(t.ppf((1 - ci_level) / 2, len(y) - 1))
        n_std = t_crit / np.sqrt(len(y))
    elif show_std_error:
        n_std = 1 / np.sqrt(len(y))

    if errbar:
        n_errbars = min(10, len(std))
        errbar_inds = np.arange(0, len(x), n_errbars)
        fig.add_trace(
            go.Scatter(
                x=x[errbar_inds],
                y=mean[errbar_inds],
                mode='markers',
                marker=dict(color=f"rgb{color}", size=6),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=n_std * std[errbar_inds],  # upper deviation
                    arrayminus=n_std * std[errbar_inds],  # lower deviation
                    thickness=1.5,
                    width=3
                ),
                name=label,
                showlegend=False
            )
        )
    else:
        if n_std > 0 and y.shape[0] > 1:
            uncertainty_lb = mean - n_std * std
            uncertainty_ub = mean + n_std * std
            if lb is not None:
                uncertainty_lb = np.maximum(uncertainty_lb, lb)
            if ub is not None:
                uncertainty_ub = np.minimum(uncertainty_ub, ub)

            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([uncertainty_ub, uncertainty_lb[::-1]]),
                    fill='toself',
                    fillcolor='rgba({}, {}, {}, {})'.format(color[0], color[1], color[2], alpha),
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    showlegend=False,
                    name=label
                )
            )
    return fig


def fill_y_prime(x: np.ndarray, y: np.ndarray, full_x: np.ndarray, init_value: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function to fill a new y that was defined on x based on full_x

    Args:
        x: original x values (sorted)
        y: y values for each entry of x
        full_x: new extended x space (should contain the values of x, and be sorted)
        init_value: value to assign to the extended y array for values smaller than x[0] in x_full

    Returns:
        new_x: extended x space (contains all values of full_x smaller or equal to x[-1]
        new_y: values of y on the extended space new_x -- constant on interval [x[i], x[i][
    """
    y_prime = []
    current_x_ind = -1
    current_full_x_ind = 0
    while current_full_x_ind < len(full_x):
        if full_x[current_full_x_ind] == x[current_x_ind + 1]:
            y_prime.append(y[current_x_ind + 1])
            current_x_ind += 1
            if current_x_ind == len(x) - 1:
                current_full_x_ind += 1
                break
        elif current_x_ind < 0:
            y_prime.append(init_value)
        else:
            y_prime.append(y_prime[-1])
        current_full_x_ind += 1
    assert len(full_x[:current_full_x_ind]) == len(y_prime), (len(full_x[:current_full_x_ind]), len(y_prime))
    return full_x[:current_full_x_ind], np.array(y_prime)


def create_interpolated_series(xs: list[np.ndarray], ys: list[np.ndarray], init_value):
    # Step 1: Combine x1 and x2 into a unique sorted array X
    for x in xs:
        assert len(set(x)) == len(x)

    full_x = np.sort(np.unique(np.concatenate(xs)))

    new_data = [fill_y_prime(x=x, y=y, full_x=full_x, init_value=init_value) for x, y in zip(xs, ys)]
    new_xs = [new[0] for new in new_data]
    new_ys = [new[1] for new in new_data]

    return new_xs, new_ys


def darken_color(hex_color: str, factor=0.8) -> str:
    """
    Darkens a hex color by a given factor (0 to 1).
    Ensures that the color remains within the valid range [0, 255].
    """
    # Remove the '#' symbol if it exists
    hex_color = hex_color.lstrip("#")

    # Convert hex to RGB (0-255)
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    # Convert to HLS (0-1 scale)
    h, light, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)

    # Reduce lightness (l)
    light = max(0., light * factor)

    # Convert back to RGB (0-1 scale)
    r, g, b = colorsys.hls_to_rgb(h, light, s)

    # Convert RGB values back to 0-255 and ensure they are within the valid range
    r = max(0, min(255, int(r * 255)))
    g = max(0, min(255, int(g * 255)))
    b = max(0, min(255, int(b * 255)))

    # Return the darkened color in hex format
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def hex_to_rgba(hex_color: str, alpha=1.0) -> str:
    """ Convert hex color to RGBA format

    Args:
        hex_color: color in hex format
        alpha: target transparency

    Returns:
          rgba: color in RGBA format
    """
    hex_color = hex_color.lstrip("#")
    rgba = f"rgba({int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}, {alpha})"
    return rgba


def get_luminance(rgb: str) -> float:
    """
    Convert an RGB color (hex '#RRGGBB' or 'rgb(r, g, b)') to relative luminance (0–1).
    This is useful for determining text color (black/white) based on background.

    Args:
        rgb (str): A string like '#112233' or 'rgb(17, 34, 51)'

    Returns:
        float: Luminance value between 0 (dark) and 1 (light)
    """
    rgb = rgb.strip()

    if rgb.startswith('#') and len(rgb) == 7:
        r = int(rgb[1:3], 16)
        g = int(rgb[3:5], 16)
        b = int(rgb[5:7], 16)
    elif rgb.startswith('rgb'):
        match = re.match(r'rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)', rgb)
        if not match:
            raise ValueError("Invalid rgb() format. Expected 'rgb(r, g, b)'.")
        r, g, b = map(int, match.groups())
        if not all(0 <= val <= 255 for val in (r, g, b)):
            raise ValueError("RGB values must be in the range 0–255.")
    else:
        raise ValueError("Unsupported color format. Use '#RRGGBB' or 'rgb(r, g, b)'.")

    # Normalize to [0, 1]
    r /= 255.0
    g /= 255.0
    b /= 255.0

    # Apply sRGB luminance formula
    def channel_lum(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    y = 0.2126 * channel_lum(r) + 0.7152 * channel_lum(g) + 0.0722 * channel_lum(b)

    # Convert luminance Y to perceptual lightness L*
    def y_to_ell_star(y_: float) -> float:
        epsilon = 216 / 24389  # ≈ 0.008856
        kappa = 24389 / 27     # ≈ 903.296
        return y_ * kappa if y_ <= epsilon else (y_ ** (1/3)) * 116 - 16

    return y_to_ell_star(y_=y)


def get_text_color(rgb: str) -> str:
    """ Determine text color based on background color """
    luminance = get_luminance(rgb=rgb)
    return 'black' if luminance < 50 else 'white'


def plot_matrix(
        matrix: np.ndarray, x_ticks: list[str], y_ticks: list[str], mask_is: list[int] | None = None,
        mask_js: list[int] | None = None, colorscale: list[str] | None = None, zmin: float | None = None,
        zmax: float | None = None
) -> go.Figure:
    """
    Plot matrix.

    Args:
        matrix: content of the matrix to show
        x_ticks: names of the x-ticks
        y_ticks: names of the y-ticks
        mask_is: if set, will mask entry [m_i, m_j] from the matrix where m_i in mask_is, m_j in mask_js
        mask_js: if set, will mask entry [m_i, m_j] from the matrix where m_i in mask_is, m_j in mask_js
        colorscale: color scale used.
        zmin: minimum value to show
        zmax: max value to show

    Returns:

    """
    if colorscale is None:
        colorscale = plotly.colors.sequential.Blues

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            text=matrix,
            hoverinfo='text',
            showscale=False,
        )
    )

    if mask_is is None:
        mask_is = []
    if mask_js is None:
        mask_js = []

    mask_shapes = []

    for i in mask_is:
        for j in mask_js:
            x0, x1 = j - 0.5, j + 0.5
            y0, y1 = i - 0.5, i + 0.5
            # Add a black rectangle over matrix[i, j]
            mask_shapes.append(
                dict(
                    type="rect",
                    x0=x0, x1=x1,  # X-coordinates
                    y0=y0, y1=y1,  # Y-coordinates
                    fillcolor="black",
                    line=dict(width=0),  # No border
                    layer="above"
                )
            )
    fig.update_layout(shapes=mask_shapes)

    # Extract RGB values for each cell from the color scale
    # Normalize the values to be between 0 and 1
    if zmin is not None:
        scale = zmax - zmin
    else:
        scale = np.max(matrix) - np.min(matrix)
    if scale == 0:
        normalized_values = np.ones_like(matrix) / 2
    else:
        normalized_values = (matrix - np.min(matrix)) / scale
    formatted_text = [[f"{int(value)}" for value in row] for row in matrix]

    # Get RGB for each value in the matrix based on the color scale
    text_colors = [
        [get_text_color(colorscale[int(value * (len(colorscale) - 1))]) for value in row] for row in normalized_values
    ]

    # Add annotations with text color for each cell
    annotations = []
    for i, row in enumerate(formatted_text):
        for j, val in enumerate(row):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=val,
                    showarrow=False,
                    font=dict(size=12, color=text_colors[i][j], weight=1000),
                    align='center'
                )
            )

    fig.update_layout(annotations=annotations)

    #     Set X and Y axis tick labels with LaTeX and custom font size
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(matrix.shape[1]),
            ticktext=x_ticks,
            tickangle=90,
            tickfont=dict(size=20, color='black', family="STIXGeneral, Times New Roman, serif", weight=1000),
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=np.arange(matrix.shape[0]),
            ticktext=y_ticks,
            tickangle=0,
            tickfont=dict(size=20, color='black', family="STIXGeneral, Times New Roman, serif", weight=1000),
        ),
        height=matrix.shape[0] * 30 + max(100, 10 * max(map(len, x_ticks))),
        width=matrix.shape[1] * 25 + max(100, 15 * max(map(len, y_ticks))),
    )

    return fig


def is_running_in_pycharm() -> bool:
    """ Return whether a notebook is running in pycharm """
    return "LC_ALL" in os.environ


def add_legend_spacing(fig: go.Figure, legendgroup: str | None, legend: str) -> None:
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(color='rgba(255,255,255,0)', size=0),
            name=' ',  # Add a blank or spacer name
            legendgroup=legendgroup, showlegend=True, legend=legend
        )
    )
