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

# Data source
@st.cache
def get_data_source(cache_file='data_d_orbit_sideon_20191219.pkl'):
    d = DataHandler(cache_file=cache_file).data_big()
    dl = DataHandler(cache_file=cache_file).data_big()
    big_df = DataHandler(cache_file=cache_file).data_big()
    return d, dl, big_df

# plot_r_eff3d_ = st.cache(plot_r_eff3d)
# plot_mass_ = st.cache(plot_mass)
# plot_ssfr_ = st.cache(plot_ssfr)

d, dl, big_df = get_data_source()

def generate_figs(cold_gas=True):
    fields = 'r_eff star_mass tot_mass ssfr'
    Figs = namedtuple('Fig', fields)

    figs = Figs(plot_r_eff3d(big_df),
                plot_mass(big_df, 'star'),
                plot_mass(big_df, 'total'),
                plot_ssfr(big_df, cold_gas=cold_gas),
                )
    # di = dict(zip(fields.split(), figs))
    return figs
    # return di

###

# Adds a selectbox to the sidebar
selectbox = st.sidebar.selectbox(
    'Which plot',
    ('R_e', 'star_mass', 'total mass', 'sSFR',)
)

# d, dl, big_df = get_data_source()

# figs = generate_figs()
if selectbox == 'R_e':
    st.markdown('# R_eff')
    rolling_mean = st.checkbox('Rolling mean')
    st.write(plot_r_eff3d(big_df, rolling_mean))
elif selectbox == 'star_mass':
    st.markdown('# Star mass')
    st.write(plot_mass(big_df, 'star'))
elif selectbox == 'star_mass':
    st.markdown('# Total mass')
    st.write(plot_mass(big_df, 'total'))
elif selectbox == 'sSFR':
    st.markdown('# sSFR')
    cold_gas = st.checkbox('Plot cold gas')
    st.write(plot_ssfr(big_df, cold_gas))

# fig = plot_ssfr(big_df)
# file_stem = os.path.splitext(os.path.basename(__file__))[0]
# fig.savefig(f'{file_stem}.pdf')

# st.write()