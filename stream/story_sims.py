import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from common import labels, get_figsize, LEGEND_FONT_SIZE

# from simulation.simdata import get_color_from_name, get_styles_from_peri
from simulation.data_handler import DataHandler
import cycler

plt.style.use('./MNRAS.mplstyle')

my_guys = '69p200', '68p200'

# def plot_example(d):
#     pass

def plot_guys(d, rolling_mean=True, window_size=20):
    # n = 5
    # color = plt.cm.copper(np.linspace(0, 1, n))
    # matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    properties = ['r_eff3d', 'sigma_star', 'n', 'avg_mu_e', 'mass_star', 'sfr', 'ssfr', 'cold_gas', 'm_halo_m_star']
    nrows = len(properties)
    ncols = 1
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=get_figsize(nrows), dpi=200)
    slicer = slice(None, None, 1)
    # print(tuple(d[my_guys[0]].columns))
    for sim in my_guys:
        df = d[sim]
        for prop, ax_s in zip(properties, ax.flat):
            if prop is 'm_halo_m_star':
                df['m_halo_m_star'] = df.dm_mass / df.mass_star
            if prop is 'ssfr':
                df['ssfr'] = df.sfr / df.mass_star

            # if rolling_mean:
            #     group['r_eff3d_mean'] = group['r_eff3d'].rolling(window_size,
            #                                                      min_periods=1,
            #                                                      center=True).mean()
            ax_s.plot(df.t_period, df[prop], label=sim)
            ax_s.set_ylabel(labels[prop])
            if ax_s is not ax[-1]: ax_s.set_xticklabels([])

        ax_s.grid(ls=':')
        # ax_s.set_title(name)
        # ax_s.set_ylim(0, 10)
        # ax_s.set_yscale('log')
        # ax_s.set_xlim(-0.25, 2)

    # ax_s.legend(, ncol=1)
    ax[0].legend(prop={'size': LEGEND_FONT_SIZE}, ncol=1)
    ax[-1].set_xlabel("t/T$_r$")
    return fig

if __name__ == '__main__':
    dh = DataHandler(cache_file='data_d_orbit_sideon_20191219.pkl')
    fig = plot_guys(dh.data())
    file_stem = os.path.splitext(os.path.basename(__file__))[0]
    fig.savefig(f'{file_stem}.pdf', dpi=300)

    plt.show()