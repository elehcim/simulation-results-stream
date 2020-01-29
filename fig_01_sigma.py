import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib
import streamlit as st

from common import labels, get_figsize, LEGEND_FONT_SIZE

from simulation.simdata import get_color_from_name, get_styles_from_peri
import cycler

plt.style.use('./MNRAS.mplstyle')

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