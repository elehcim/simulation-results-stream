import os
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import cycler
plt.style.use('./MNRAS.mplstyle')
from common import labels, get_figsize, LEGEND_FONT_SIZE

def plot_mass(big_df, which):
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
        ax_s.set_ylim(0, None)
        # ax_s.set_yscale('log')
        ax_s.set_ylabel(ylabel)
        # ax_s.set_xlim(-0.25, 0.5)

    # ax_s.legend(, ncol=1)
    ax[0].legend(prop={'size': LEGEND_FONT_SIZE}, ncol=1)
    ax[-1].set_xlabel("t/T$_r$")
    return fig
# fig.suptitle('M$_{dm}^c$/M$_{dm}$');
# From here: https://stackoverflow.com/a/43578952/1611927
# lgnd = ax.legend(loc='upper center', prop={'size': 5}, ncol=5, scatterpoints=1, fontsize=10)
# for handle in lgnd.legendHandles:
#     handle.set_sizes([5]);
