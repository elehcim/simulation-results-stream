import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import copy

from simulation.data_handler import DataHandler
from collections import namedtuple

from fig_00_r_eff import plot_r_eff3d
from fig_01_sigma import plot_sigma
from fig_12_4_star_mass import plot_mass
from fig_12_2_sfr import plot_ssfr
from fig_07_0_avg_mu_e_Aku import plot_avg_mu_e_aku
from fig_04_1_color_magnitude_Aku_rt_criterion import plot_color_magnitude_aku
import subprocess

git_version = subprocess.check_output(["git", "describe", "--dirty", "--always", "--tags"]).strip()

# Data source
@st.cache(allow_output_mutation=True)
def get_data_source(cache_file='data_d_orbit_sideon_20191219.pkl'):
    d = DataHandler(cache_file=cache_file).data()
    dl = DataHandler(cache_file=cache_file).data_last()
    big_df = DataHandler(cache_file=cache_file).data_big()
    drt = DataHandler(cache_file=cache_file).data_rt()

    return d, dl, big_df, drt


d, dl, big_df, drt = get_data_source()

fields = 'r_eff sigma star_mass tot_mass m_halo_m_star ssfr avg_mu_e color_magnitude'.split()

# def generate_figs(cold_gas=True):
#     Figs = namedtuple('Fig', fields)
#     figs = Figs(plot_r_eff3d(big_df),
#                 plot_mass(big_df, 'star'),
#                 plot_mass(big_df, 'total'),
#                 plot_mass(big_df, 'm_halo_m_star'),
#                 plot_ssfr(big_df, cold_gas=cold_gas),
#                 plot_avg_mu_e_aku(big_df, cold_gas=cold_gas),
#                 )
#     return figs
# figs = generate_figs()

###

# Adds a button to the sidebar
which_plot = st.sidebar.radio(
    'Which plot',
    fields,
)
st.sidebar.markdown(f'`version: {git_version.decode()}`')

if which_plot == 'r_eff':
    st.markdown('# R_eff')
    rolling_mean = st.checkbox('Rolling mean')
    st.write(plot_r_eff3d(big_df, rolling_mean))

elif which_plot == 'star_mass':
    st.markdown('# Star mass')
    st.write(plot_mass(big_df, 'star'))

elif which_plot == 'tot_mass':
    st.markdown('# Total mass')
    st.write(plot_mass(big_df, 'total'))

elif which_plot == 'sigma':
    st.markdown('# Velocity dispersion')
    st.markdown('This is the central cvelocity dispersion within 250 pc.')
    rolling_mean = st.checkbox('Rolling mean')
    st.write(plot_sigma(big_df, rolling_mean))

elif which_plot == 'm_halo_m_star':
    st.markdown('# M$_h$/M$_\star$ mass')
    st.write(plot_mass(big_df, 'm_halo_m_star'))
    st.markdown('# The final stage of M$_h$/M$_\star$ mass')
    st.markdown('Do I have any galaxy without DM?')
    st.markdown('''Usually ones considers without DM a galaxy which has DM content equal to barionic one.
                 Given this, I don't have enough DM loss.''')
    st.markdown('It is worth noting that the mass is computed within a 10 kpc sphere')

    st.write(plot_mass(big_df, 'm_halo_m_star', xlim=(1, None), log=True))

elif which_plot == 'avg_mu_e':
    st.markdown('# Average surface brightness inside R_eff')
    sim_n = st.selectbox('Which simulation:', [62, 41])
    color_by = st.radio('Color by:', ['sfr', 'ssfr'])
    # how_many_snaps = st.slider('how many points to show',
    #                             min_value=5,
    #                             # max_value=len(d[f'{sim_n}p50']),
    #                             max_value=30,
    #                             value=15,
    #                             step=5)
    st.write(plot_avg_mu_e_aku(copy.deepcopy(d),
                               sim_n=int(sim_n),
                               color_by=color_by,
                               # how_many_snaps=how_many_snaps,
                               show_aku=True,
                               )
             )

elif which_plot == 'ssfr':
    st.markdown('# sSFR')
    cold_gas = st.checkbox('Plot cold gas')
    st.write(plot_ssfr(big_df, cold_gas))

elif which_plot == 'color_magnitude':
    st.markdown('# Evolution on a Color-Magnitude diagram')
    color_by = st.selectbox('Color by:', ['ssfr', 'ssfr_mean', 'sfr', 'r', 't_period', ''])
    st.write(plot_color_magnitude_aku(drt, color_col=color_by))

# fig = plot_ssfr(big_df)
# file_stem = os.path.splitext(os.path.basename(__file__))[0]
# fig.savefig(f'{file_stem}.pdf')

# st.write()
