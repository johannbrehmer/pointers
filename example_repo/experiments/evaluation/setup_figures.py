import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec, cm
from mpl_toolkits.mplot3d import Axes3D

# General size settings
TEXTWIDTH = 9.0  # inches
FONTSIZE = 9.0

# Color map
CMAP = cm.plasma
CMAP_R = cm.plasma_r

# Discrete colors matching the color map
COLORS = [CMAP(i / 4.0) for i in range(5)]
COLORS_NEUTRAL = ["0.0", "0.4", "0.7", "1.0"]


def setup():
    """ Generic matplotlib setup """

    matplotlib.rcParams.update({"font.size": FONTSIZE})  # controls default text sizes
    matplotlib.rcParams.update({"axes.titlesize": FONTSIZE})  # fontsize of the axes title
    matplotlib.rcParams.update({"axes.labelsize": FONTSIZE})  # fontsize of the x and y labels
    matplotlib.rcParams.update({"xtick.labelsize": FONTSIZE})  # fontsize of the tick labels
    matplotlib.rcParams.update({"ytick.labelsize": FONTSIZE})  # fontsize of the tick labels
    matplotlib.rcParams.update({"legend.fontsize": FONTSIZE})  # legend fontsize
    matplotlib.rcParams.update({"figure.titlesize": FONTSIZE})  # fontsize of the figure title
    matplotlib.rcParams.update({"figure.dpi": 300})
    matplotlib.rcParams.update({"savefig.dpi": 300})


def figure_by_height(
    height=TEXTWIDTH * 0.5,
    large_margin=0.14,
    small_margin=0.03,
    make3d=False,
    left_margin_large=True,
    right_margin_large=False,
    bottom_margin_large=True,
    top_margin_large=False,
):
    """ Simple plot, size specified by height """

    left = large_margin if left_margin_large else small_margin
    right = large_margin if right_margin_large else small_margin
    top = large_margin if top_margin_large else small_margin
    bottom = large_margin if bottom_margin_large else small_margin

    width = height * (1.0 + right + left - top - bottom)

    fig = plt.figure(figsize=(width, height))
    if make3d:
        ax = Axes3D(fig)
    else:
        ax = plt.gca()
    plt.subplots_adjust(
        left=left,
        right=1.0 - right,
        bottom=bottom,
        top=1.0 - top,
        wspace=0.0,
        hspace=0.0,
    )

    return fig, ax


def figure_by_width(
    width=TEXTWIDTH * 0.5,
    large_margin=0.14,
    small_margin=0.03,
    make3d=False,
    left_margin_large=True,
    right_margin_large=False,
    bottom_margin_large=True,
    top_margin_large=False,
):
    """ Simple plot, size specified by width """

    left = large_margin if left_margin_large else small_margin
    right = large_margin if right_margin_large else small_margin
    top = large_margin if top_margin_large else small_margin
    bottom = large_margin if bottom_margin_large else small_margin

    height = width / (1.0 + right + left - top - bottom)

    fig = plt.figure(figsize=(width, height))
    if make3d:
        ax = Axes3D(fig)
    else:
        ax = plt.gca()
    plt.subplots_adjust(
        left=left,
        right=1.0 - right,
        bottom=bottom,
        top=1.0 - top,
        wspace=0.0,
        hspace=0.0,
    )

    return fig, ax


def figure_with_cbar_by_height(
    height=TEXTWIDTH * 0.5,
    large_margin=0.14,
    small_margin=0.03,
    cbar_sep=0.02,
    cbar_width=0.04,
    make3d=False,
    left_margin_large=True,
    right_margin_large=False,
    bottom_margin_large=True,
    top_margin_large=False,
):
    """ Figure with color bar, size specified by height """

    top = large_margin if top_margin_large else small_margin
    bottom = large_margin if bottom_margin_large else small_margin
    left = large_margin if left_margin_large else small_margin
    right = (large_margin if right_margin_large else small_margin) + cbar_width + cbar_sep

    width = height * (1.0 + cbar_sep + cbar_width + right + left - top - bottom)

    cleft = 1.0 - (large_margin + cbar_width) * height / width
    cbottom = bottom
    cwidth = cbar_width * height / width
    cheight = 1.0 - top - bottom

    fig = plt.figure(figsize=(width, height))
    if make3d:
        ax = Axes3D(fig)
    else:
        ax = plt.gca()
    plt.subplots_adjust(
        left=left * height / width,
        right=1.0 - right * height / width,
        bottom=bottom,
        top=1.0 - top,
        wspace=0.0,
        hspace=0.0,
    )
    cax = fig.add_axes([cleft, cbottom, cwidth, cheight])

    plt.sca(ax)

    return fig, (ax, cax)


def grid_by_height(
    nx=4,
    ny=2,
    height=0.5 * TEXTWIDTH,
    aspect_ratio=1.0,
    large_margin=0.14,
    small_margin=0.03,
    sep=0.02,
    left_margin_large=True,
    right_margin_large=False,
    bottom_margin_large=True,
    top_margin_large=False,
):
    """ Grid of panels, no colorbars, size specified by height """

    left = large_margin if left_margin_large else small_margin
    right = large_margin if right_margin_large else small_margin
    top = large_margin if top_margin_large else small_margin
    bottom = large_margin if bottom_margin_large else small_margin

    panel_size = (1.0 - top - bottom - (ny - 1) * sep) / ny
    width = height * aspect_ratio * (left + nx * panel_size + (nx - 1) * sep + right)

    # wspace and hspace are complicated beasts
    avg_width_abs = (height * panel_size * nx * ny) / (nx * ny + ny)
    avg_height_abs = height * panel_size
    wspace = sep * height / avg_width_abs
    hspace = sep * height / avg_height_abs

    # Set up figure
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(ny, nx, width_ratios=[1.0] * nx, height_ratios=[1.0] * ny)
    plt.subplots_adjust(
        left=left * height / width,
        right=1.0 - right * height / width,
        bottom=bottom,
        top=1.0 - top,
        wspace=wspace,
        hspace=hspace,
    )
    return fig, gs


def grid_by_width(
    nx=4,
    ny=2,
    width=TEXTWIDTH,
    aspect_ratio=1.0,
    large_margin=0.14,
    small_margin=0.03,
    sep=0.02,
    left_margin_large=True,
    right_margin_large=False,
    bottom_margin_large=True,
    top_margin_large=False,
):
    """ Grid of panels, no colorbars, size specified by width """

    left = large_margin if left_margin_large else small_margin
    right = large_margin if right_margin_large else small_margin
    top = large_margin if top_margin_large else small_margin
    bottom = large_margin if bottom_margin_large else small_margin

    panel_size = (1.0 - top - bottom - (ny - 1) * sep) / ny
    height = width / (left + nx * panel_size + (nx - 1) * sep + right) / aspect_ratio

    # wspace and hspace are complicated beasts
    avg_width_abs = (height * panel_size * nx * ny) / (nx * ny + ny)
    avg_height_abs = height * panel_size
    wspace = sep * height / avg_width_abs
    hspace = sep * height / avg_height_abs

    # Set up figure
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(ny, nx, width_ratios=[1.0] * nx, height_ratios=[1.0] * ny)
    plt.subplots_adjust(
        left=left * height / width,
        right=1.0 - right * height / width,
        bottom=bottom,
        top=1.0 - top,
        wspace=wspace,
        hspace=hspace,
    )
    return fig, gs
