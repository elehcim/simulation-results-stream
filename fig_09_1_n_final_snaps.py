import os

import matplotlib
import matplotlib.pylab as plt
import numpy as np

from simulation.data_handler import DataHandler
from simulation.akucm import AkuCM
from simulation.simdata import get_styles_from_peri

from common import labels, get_figsize, LEGEND_FONT_SIZE
plt.style.use('./MNRAS.mplstyle')

def plot_n_final(last_df):
    cm = AkuCM('n')
    print(cm.filename)

    # cmap = matplotlib.cm.get_cmap('jet_r', 50)
    cmap = matplotlib.cm.get_cmap('jet', 50)
    size = 10

    fig, ax = plt.subplots(
         #figsize=(12, 8),
        figsize = (size*1.2, size * cm.img.shape[0]/cm.img.shape[1]),
                           # constrained_layout=True,
                           )

    ax2 = ax.twinx()
    ax2.imshow(cm.img, extent=cm.extent, aspect=cm.aspect, alpha=0.2)
    ax2.set_yticks([], [])

    color_by = 'mass_star'
    vmin = last_df[color_by].min()
    vmax = last_df[color_by].max()
    # vmin = np.inf
    # vmax = 0
    groups = last_df.groupby('pericenter')
    for pericenter, g in groups:
        # print(pericenter)
        color_column = g[color_by]
        sc = ax.scatter(g.mag_sdss_r,
                    g.n,
                    c=np.log10(g[color_by]),
                    # marker='x',
                    marker=get_styles_from_peri(str(pericenter), scatter=True),
                    s=100,
                    alpha=0.9,
                    label=pericenter,
                    cmap=cmap)
        # if color_column.min() < vmin:
        #     vmin = np.min(color_column.values[np.nonzero(color_column.values)])
        # if color_column.max() > vmax:
        #     vmax = color_column.max()
    ax.set_yscale('log')
    ax.set_ylim( cm.extent[2:])
    ax.set_xlim( cm.extent[:2])

    ax.grid(linestyle=":")
    ax.legend()

    ax.set_ylabel(r"Sérsic index")
    ax.set_xlabel(labels["mag_sdss_r"])


    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable)

    cbar.ax.set_ylabel(labels['log_mass_star'] + ' final')
    return fig
