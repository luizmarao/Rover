import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap

from matplotlib.patches import PathPatch
from matplotlib.path import Path


def draw_band(ax, x, y, err, **kwargs):
    # Calculate normals via centered finite differences (except the first point
    # which uses a forward difference and the last point which uses a backward
    # difference).
    dx = np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]])
    dy = np.concatenate([[y[1] - y[0]], y[2:] - y[:-2], [y[-1] - y[-2]]])
    l = np.hypot(dx, dy)
    nx = np.divide(dy, l, out=np.zeros_like(dy), where=l!=0)
    ny = np.divide(-dx, l, out=np.zeros_like(dy), where=l!=0)
    for i in range(len(nx)):
        if np.isnan(nx[i]):
            nx[i] = 0.0
        if np.isnan(ny[i]):
            ny[i] = 0.0

    # end points of errors
    xn = x - nx * err
    yn = y - ny * err

    vertices = np.block([[x, xn[::-1]],
                         [y, yn[::-1]]]).T
    codes = np.full(len(vertices), Path.LINETO)
    codes[0] = codes[1] = Path.MOVETO
    path = Path(vertices, codes)
    ax.add_patch(PathPatch(path, **kwargs))


def multicolor_line_plot(x, y, line_intensity, band=None, band_size_limit=None, background=None, cmap='PuOr',
                         facecolor='black', edgecolor='white', title=None, colorbar_label='bar', savepath=None,
                         show=False):

    # Create a set of line segments so that we can color them individually
    # This creates the points as an N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, axs = plt.subplots(1, 1)

    if background is not None:
        axs.imshow(background, extent=[0, 44, 0, 25])

    if band_size_limit is not None:
        # squash line_intensity to the band_size limit
        max_intensity = abs(band).max()
        band = band * band_size_limit / max_intensity

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(line_intensity.min(), line_intensity.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    # Set the values used for colormapping
    lc.set_array(line_intensity)
    lc.set_linewidth(4)
    line = axs.add_collection(lc)
    cbar = fig.colorbar(line, ax=axs)
    cbar.set_label(colorbar_label)
    if band is not None:
        draw_band(axs, x, y, band, facecolor=facecolor, edgecolor=edgecolor, alpha=.5)
    if title is not None:
        plt.title(title)
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        plt.show()
