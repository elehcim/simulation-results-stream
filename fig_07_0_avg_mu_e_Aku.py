import os

import matplotlib
import matplotlib.pylab as plt
import numpy as np

from simulation.akucm import AkuCM
from common import labels, get_figsize

plt.style.use('./MNRAS.mplstyle')

# d = DataHandler().data()

_extrema = {'sfr': {41: (0.0004326997212649507, 0.03681058373613701),
                    62: (0.0004326997212649507, 0.03681058373613701),
                    'default': (1e-5, 2e-3),
                    },
            'ssfr': {41: (2.313699463943961e-11, 1.2062309021873976e-09),
                     62: (8.77977968105377e-11, 6.750326315134827e-09),
                     'default': (5e-12, 2e-9),
                     },
            }


def plot_avg_mu_e_aku(d, sim_n: int, color_by, how_many_snaps=15, show_aku=True):
    """
    color_by can be 'sfr' or 'ssfr'"""
    orbits = [k for k in d.keys() if k.startswith(str(sim_n))]
    assert len(orbits) != 0

    nrows = 4
    ncols = len(orbits) // nrows

    figsize = 6 * ncols, 3 * nrows
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize,  # get_figsize(ncols, nrows),
                            # constrained_layout=True,
                            )

    cmap = matplotlib.cm.get_cmap('jet_r', 50)

    cm = AkuCM('mue')

    vmin = np.inf
    vmax = 0

    norm = matplotlib.colors.LogNorm(*_extrema[color_by][sim_n])
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    for i, (k, ax) in enumerate(zip(orbits, axs.flatten())):
        if show_aku:
            ax.imshow(cm.img, extent=cm.extent, aspect=cm.aspect, alpha=0.2)

        slicer = np.linspace(0, len(d[k]) - 1, how_many_snaps, dtype=int)
        print(slicer)
        df = d[k].copy().iloc[slicer]
        df.loc[:, 'ssfr'] = df['sfr'] / df['mass_star']
        # if color_by == 'ssfr':
        #     color_column = df['sfr'] / df['mass_star']
        # else:
        color_column = df[color_by]
        # print(d[k][0:10].sfr)

        # Since log10 of zero is -inf, this way I'm not plotting elements with nan color.
        # df.plot('mag_sdss_r', 'avg_mu_e', c=np.log10(color_column),
        #         ax=ax, kind='scatter', marker='s',
        #         legend=False, s=30, alpha=0.8, colorbar=False, cmap=cmap)

        # Here since to_rgba(0) gives a finite result, the bottom color level.
        ax.scatter(df.mag_sdss_r, df.avg_mu_e,
                   s=30, marker='s', alpha=0.4,
                   label=k[:2],
                   c=mappable.to_rgba(color_column),
                   # c='k',
                   )
        ax.plot(df.mag_sdss_r, df.avg_mu_e,
                alpha=0.8,
                )

        cbar = fig.colorbar(mappable, ax=ax)
        cbar.ax.set_ylabel(labels[color_by])

        ax.grid(linestyle=":")
        # Fix extrema
        # ax.set_ylim(cm.extent[2:])
        ax.set_ylim(30, 18)
        # ax.set_xlim(None, -16)

        # Print extrema
        vals = color_column.values
        # print(vals)
        nonzero = vals[np.nonzero(vals)]
        if nonzero.size == 0:
            minval = 0
        else:
            minval = np.min(vals[np.nonzero(vals)])
        # print(minval, color_column.max())
        ax.set_title(k)
        ax.set_xlabel(labels['mag_sdss_r'])
        ax.set_ylabel(labels['avg_mu_e'])
        if minval < vmin:
            vmin = minval
        if color_column.max() > vmax:
            vmax = color_column.max()
        # ax.set_ylim(df.avg_mu_e.max())

    print(f"Color extrema: vmin={vmin}, vmax={vmax}")
    # cbar = fig.colorbar(mappable, ax=axs.ravel().tolist())
    # cbar.ax.set_ylabel(labels[color_by])
    return fig
# file_stem = os.path.splitext(os.path.basename(__file__))[0]
# plt.savefig(f'{file_stem}.pdf')
