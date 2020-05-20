import matplotlib
import matplotlib.pylab as plt
import numpy as np
from simulation.simdata import get_color_from_name, get_styles_from_peri
import cycler

from common import labels, get_figsize, LEGEND_FONT_SIZE

plt.style.use('./MNRAS.mplstyle')
import streamlit as st

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

# file_stem = os.path.splitext(os.path.basename(__file__))[0]
# plt.savefig(f'{file_stem}.pdf')