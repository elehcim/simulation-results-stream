import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os

from common.data_handler import DataHandler
from collections import namedtuple

from fig_00_r_eff import plot_r_eff3d
from fig_12_4_star_mass import plot_mass
from fig_12_2_sfr import plot_ssfr
from fig_07_0_avg_mu_e_Aku import plot_avg_mu_e_aku



# Data source
@st.cache
def get_data_source(cache_file='data_d_orbit_sideon_20191219.pkl'):
    d = DataHandler(cache_file=cache_file).data()
    dl = DataHandler(cache_file=cache_file).data_last()
    big_df = DataHandler(cache_file=cache_file).data_big()
    return d, dl, big_df


d, dl, big_df = get_data_source()

fields = 'r_eff star_mass tot_mass m_halo_m_star ssfr avg_mu_e'.split()

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

# Adds a selectbox to the sidebar
selectbox = st.sidebar.radio(
    'Which plot',
    fields,
)


if selectbox == 'r_eff':
    st.markdown('# R_eff')
    rolling_mean = st.checkbox('Rolling mean')
    st.write(plot_r_eff3d(big_df, rolling_mean))

elif selectbox == 'star_mass':
    st.markdown('# Star mass')
    st.write(plot_mass(big_df, 'star'))

elif selectbox == 'tot_mass':
    st.markdown('# Total mass')
    st.write(plot_mass(big_df, 'total'))

elif selectbox == 'm_halo_m_star':
    st.markdown('# M_h/M_* mass')
    st.write(plot_mass(big_df, 'm_halo_m_star'))

elif selectbox == 'avg_mu_e':
    st.markdown('# Average surface brightness inside R_eff')
    sim_n = st.selectbox('Which simulation:', [62, 41])
    color_by = st.radio('Color by:', ['sfr', 'ssfr'])
    how_many_snaps = st.slider('how many points to show',
                                min_value=5,
                                # max_value=len(d[f'{sim_n}p50']),
                                max_value=30,
                                value=15,
                                step=5)
    st.write(plot_avg_mu_e_aku(d,
                               sim_n=int(sim_n),
                               color_by=color_by,
                               how_many_snaps=how_many_snaps,
                               show_aku=True,
                              )
            )

elif selectbox == 'ssfr':
    st.markdown('# sSFR')
    cold_gas = st.checkbox('Plot cold gas')
    st.write(plot_ssfr(big_df, cold_gas))

# fig = plot_ssfr(big_df)
# file_stem = os.path.splitext(os.path.basename(__file__))[0]
# fig.savefig(f'{file_stem}.pdf')

# st.write()