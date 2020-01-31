import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib

from common import labels, get_figsize, LEGEND_FONT_SIZE
from simulation.akucm import AkuCM

from simulation.simdata import get_styles_from_peri, get_styles_from_sim
import cycler

plt.style.use('./MNRAS.mplstyle')

EXTREMA = {'sfr': {41: (0.0004326997212649507, 0.03681058373613701),
                    62: (0.0004326997212649507, 0.03681058373613701),
                    'default': (1e-5, 2e-3),
                    },
            'ssfr': {41: (2.313699463943961e-11, 1.2062309021873976e-09),
                     62: (8.77977968105377e-11, 6.750326315134827e-09),
                     'default': (5e-12, 2e-9),
                     },
            }


def plot_r_eff3d(big_df, rolling_mean=False, window_size=20):
    n = 5
    color = plt.cm.copper(np.linspace(0, 1, n))
    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    nrows = 5
    ncols = 1
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=get_figsize(nrows), dpi=200)
    slicer = slice(None, None, 1)

    for (full_name, sim), ax_s in zip(big_df.groupby('name', sort=False), ax.flat):
        name = full_name[:2]
        for (peri, group) in sim.groupby('pericenter', sort=False):

            # k = f'{name}p{peri}'
            if rolling_mean:
                group['r_eff3d_mean'] = group['r_eff3d'].rolling(window_size,
                                                                 min_periods=1,
                                                                 center=True).mean()

                ax_s.plot(group.t_period, group.r_eff3d_mean, label=peri)
            else:
                ax_s.plot(group.t_period, group.r_eff3d, label=peri)
        if ax_s is not ax[-1]: ax_s.set_xticklabels([])

        ax_s.grid(ls=':')
        ax_s.set_title(name)
        ax_s.set_ylabel(labels['r_eff3d'])
        ax_s.set_ylim(0, 10)
        # ax_s.set_yscale('log')
        ax_s.set_xlim(-0.25, 2)

    # ax_s.legend(, ncol=1)
    ax[0].legend(prop={'size': LEGEND_FONT_SIZE}, ncol=1)
    ax[-1].set_xlabel("t/T$_r$")
    return fig

def plot_sigma(big_df, rolling_mean=True, window_size=20):
    n = 5
    color = plt.cm.copper(np.linspace(0, 1, n))
    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    nrows = 5
    ncols = 1
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=get_figsize(nrows), dpi=200)
    slicer = slice(None, None, 1)

    for (full_name, sim), ax_s in zip(big_df.groupby('name', sort=False), ax.flat):
        name = full_name[:2]
        for (peri, group) in sim.groupby('pericenter', sort=False):
            if rolling_mean:
                group['sigma_star_mean'] = group['sigma_star'].rolling(window_size,
                                                                 min_periods=1,
                                                                 center=True).mean()
                plottable = 'sigma_star_mean'
            else:
                plottable = 'sigma_star'
            ax_s.plot(group.t_period, group[plottable], label=peri)

        if ax_s is not ax[-1]: ax_s.set_xticklabels([])

        ax_s.grid(ls=':')
        ax_s.set_title(name)
        ax_s.set_ylabel(labels['sigma_star'])
        ax_s.set_ylim(0, None)
        # ax_s.set_yscale('log')
        ax_s.set_xlim(-0.25, 2)

    # ax_s.legend(, ncol=1)
    ax[0].legend(prop={'size': LEGEND_FONT_SIZE}, ncol=1)
    ax[-1].set_xlabel("t/T$_r$")
    return fig


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


def plot_avg_mu_e_aku(d, sim_n: int, color_by, how_many_snaps=15, show_aku=True, extrema=EXTREMA):
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

    norm = matplotlib.colors.LogNorm(*extrema[color_by][sim_n])
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

    return fig


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



def plot_ssfr(big_df, cold_gas=False):
    n = 5
    color = plt.cm.copper(np.linspace(0, 1, n))
    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

    nrows = 5
    ncols = 1
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=get_figsize(nrows), dpi=200)
    slicer = slice(None, None, 1)

    for (full_name, sim), ax_s in zip(big_df.groupby('name', sort=False), ax.flat):
        name = full_name[:2]
        for (peri, group) in sim.groupby('pericenter', sort=False):

            k = f'{name}p{peri}'
            group['ssfr'] = group.sfr/group.mass_star

            ax_s.plot(group.t_period, group.ssfr, label=peri, alpha=0.8)

        ax_s.grid(ls=':')
        ax_s.set_title(name)
        if ax_s is not ax[-1]: ax_s.set_xticklabels([])
        ax_s.set_ylim(-0.2e-9, 7e-9)
        # ax_s.set_yscale('log')
        ax_s.set_ylabel(labels['ssfr'])
        ax_s.set_xlim(-0.25, 0.5)

    if cold_gas:
        color = plt.cm.cool(np.linspace(0, 1, n))
        matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

        for (full_name, sim), ax_s in zip(big_df.groupby('name', sort=False), ax.flat):
            ax_cg = ax_s.twinx()
            for (peri, group) in sim.groupby('pericenter', sort=False):
                ax_cg.plot(group.t_period, group.cold_gas, label=peri, alpha=0.8)
            ax_cg.set_ylim(1e5, 6e9)
            ax_cg.set_yscale('log')
            ax_cg.set_ylabel(labels['cold_gas_short'])
            # ax_cg.set_xlim(-0.25, 0.5)
            ax_cg.grid(ls='-.')


    # ax_s.legend(, ncol=1)
    ax[0].legend(prop={'size': LEGEND_FONT_SIZE}, ncol=1)
    ax[-1].set_xlabel("t/T$_r$")
    return fig


def plot_mass(big_df, which, xlim=(None, None), ylim=(1e-1, None), log=False):
    """which is either 'total' or 'star'"""
    n = 5
    color = plt.cm.copper(np.linspace(0, 1, n))
    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

    nrows = 5
    ncols = 1
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=get_figsize(nrows), dpi=200)
    slicer = slice(None, None, 1)
    ylabel = ''
    for (full_name, sim), ax_s in zip(big_df.groupby('name', sort=False), ax.flat):
        name = full_name[:2]
        for (peri, group) in sim.groupby('pericenter', sort=False):

            k = f'{name}p{peri}'
            if which is 'total':
                plottable = group.mass_star + group.dm_mass + group.cold_gas
                ylabel = 'M$_{tot}$ (M$_\odot$)'
            elif which is 'star':
                plottable =  group.mass_star
                ylabel = labels['mass_star']
            elif which is 'm_halo_m_star':
                plottable = group.dm_mass / group.mass_star
                ylabel = labels['m_halo_m_star']
            else:
                raise RuntimeError('Can plot only total or stars or m_halo_m_star')

            ax_s.plot(group.t_period, plottable, label=peri, alpha=0.8)
        if ax_s is not ax[-1]: ax_s.set_xticklabels([])

        # print(ax_s.yaxis.get_major_formatter())
        # ax_s.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

        ax_s.ticklabel_format(style='sci', scilimits=(-3,3), axis='y')
        ax_s.grid(ls=':')
        ax_s.set_title(name)
        ax_s.set_ylim(ylim)
        if log:
            ax_s.set_yscale('log')
        ax_s.set_ylabel(ylabel)
        ax_s.set_xlim(xlim)

    # ax_s.legend(, ncol=1)
    ax[0].legend(prop={'size': LEGEND_FONT_SIZE}, ncol=1)
    ax[-1].set_xlabel("t/T$_r$")
    # From here: https://stackoverflow.com/a/43578952/1611927
    # lgnd = ax.legend(loc='upper center', prop={'size': 5}, ncol=5, scatterpoints=1, fontsize=10)
    # for handle in lgnd.legendHandles:
    #     handle.set_sizes([5]);
    return fig