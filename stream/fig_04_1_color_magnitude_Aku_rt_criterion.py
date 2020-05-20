import os
import numpy as np

import matplotlib
import matplotlib.pylab as plt

from simulation.akucm import AkuCM
from simulation.simdata import get_styles_from_sim

plt.style.use('./MNRAS.mplstyle')

# d = DataHandler('data_d_orbit_sideon_20191219.pkl').data_rt()


def plot_color_magnitude_aku(d, color_col='ssfr'):

    # norm = matplotlib.colors.LogNorm(vmin=1e-5, vmax=2e-3)
    if color_col in ('ssfr', 'ssfr_mean'):
        norm = matplotlib.colors.LogNorm(vmin=5e-12, vmax=2e-9)
        cbar_label = 'sSFR [1/yr]'
    elif color_col == 'sfr':
        norm = matplotlib.colors.LogNorm(vmin=1e-5, vmax=2e-3)
        cbar_label = 'SFR [M$_\odot$/yr]'
    elif color_col == 'r':
        cbar_label = 'r [kpc]'
        norm = matplotlib.colors.Normalize(vmin=50, vmax=800)
    elif color_col == 't_period':
        cbar_label = 't/T$_r$'
        norm = matplotlib.colors.Normalize(vmin=-0.25, vmax=1.8)
    else:
        cbar_label = ''
        norm = matplotlib.colors.LogNorm()#vmin=1e-5, vmax=2e-3)

    if color_col == 'r':
        cmap_name = 'jet'
    else:
        cmap_name = 'jet_r'

    cmap = matplotlib.cm.get_cmap(cmap_name, 50)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    slicer = slice(None, None, 10)
    cm = AkuCM('gr')

    perilist = (50, 100, 200, 300)
    ncols = len(perilist)//2

    fig, axs = plt.subplots(ncols=ncols, nrows=len(perilist)//2, figsize=(15,8),
                            # constrained_layout=True,
                            )

    for ax, peri in zip(axs.flat, perilist):
        # orbits = [k for k in d.keys() if k.startswith(str(sim_n)) or k.startswith(str(68))]
        orbits = [k for k in d.keys() if k.endswith('p'+str(peri))]
        # fig, ax = plt.subplots(ncols=len(orbits), figsize=(7* len(orbits), 5))

        ax.imshow(cm.img, extent=cm.extent, aspect=cm.aspect, alpha=0.25);
        print(orbits)
        for i, k in enumerate(orbits):
            # print(k)
            window_size = 30

            df1 = d[k]
            df1['ssfr'] = df1['sfr'] / df1['mass_star']
            df1['ssfr_mean'] = df1['ssfr'].rolling(window_size, min_periods=1, center=True).mean()
            df1['color_sdss_g_r'] = df1['mag_sdss_g'] - df1['mag_sdss_r']
            df1['color_sdss_g_r_mean'] = df1['color_sdss_g_r'].rolling(window_size, min_periods=1, center=True).mean()
            df1['mag_sdss_r_mean'] = df1['mag_sdss_r'].rolling(window_size, min_periods=1, center=True).mean()

            n = len(d[k]) - 1  # -1 because linspace include the limits
            indexes = np.linspace(0, n, 15, dtype=int)
            print(k, n, indexes)
            df = df1.iloc[indexes]

            # Print extrema
            # vals = df.ssfr.values
            # minval = np.min(vals[np.nonzero(vals)])
            # print(minval, df.ssfr.max())
            # x = df.mag_sdss_r.rolling(window_size, min_periods=1, center=True).mean()
            # y = df.color_sdss_g_r.rolling(window_size, min_periods=1, center=True).mean()
            # c = df[color_col].rolling(window_size, min_periods=1, center=True).mean()
            x = df.mag_sdss_r_mean
            y = df.color_sdss_g_r_mean
            c = df[color_col]
            print(k, c)
            ax.scatter(x,y,
               s=60, marker=get_styles_from_sim(k, scatter=True), alpha=0.8,
               label=k[:2],
               c=mappable.to_rgba(c),
               # c='k',
              )
            ax.plot(x,y, ls=":", alpha=0.8)

        if ax is axs.flatten()[0]:
            ax.legend()

        ax.grid(linestyle=":")
        ax.set_xlim(-19, -8)
        ax.set_ylim(0, 1)
        ax.set_title(f'p{peri}')
        ax.set_xlabel("M$_{r'}$ [mag]");
        ax.set_ylabel("g'-r'")


    cbar = fig.colorbar(mappable, ax=axs.ravel().tolist())#, orientation='horizontal')#fraction=0.045*im_ratio, pad=0.04)
    cbar.ax.set_ylabel(cbar_label);
    return fig
# file_stem = os.path.splitext(os.path.basename(__file__))[0]
# fig.savefig(f'{file_stem}.pdf', dpi=300)
