import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import copy

from simulation.data_handler import DataHandler
from collections import namedtuple
import streamlit as st
from plotters import *

@st.cache
def get_data(cache_file='data_d_orbit_sideon_20191219.pkl'):
    print('HITTING CACHE')
    dh = DataHandler(cache_file=cache_file)
    return dh

class Page():
    def __init__(self):
        self.dh = get_data()

    def write(self, *args, **kwargs):
        self._write(*args, **kwargs)

class P_r_eff(Page):
    def get_fig(self):
        self.fig = plot_r_eff3d(self.dh.data_big(), self.rolling_mean)
        return self.fig

    def _write(self):
        print('Writing r_eff')
        st.header('R_eff')
        self.rolling_mean = st.checkbox('Rolling mean')
        st.write(self.get_fig())

class P_star_mass(Page):
    def _write(self):
        st.header('Star mass')
        st.write(plot_mass(self.dh.data_big(), 'star'))

class P_tot_mass(Page):
    def _write(self):
        st.header('Total mass')
        st.write(plot_mass(self.dh.data_big(), 'total'))

class P_sigma(Page):
    def _write(self):
        print('Writing sigma')

        st.header('Velocity dispersion')
        st.markdown('This is the central velocity dispersion within 250 pc.')
        rolling_mean = st.checkbox('Rolling mean', value=True)
        st.write(plot_sigma(self.dh.data_big(), rolling_mean))

class P_m_halo_m_star(Page):
    def _write(self):
        st.header('M$_h$/M$_\star$ mass')
        st.write(plot_mass(self.dh.data_big(), 'm_halo_m_star'))
        st.header('The final stage of M$_h$/M$_\star$ mass')
        st.markdown("""
Do I have any galaxy without DM?
Usually ones considers without DM a galaxy which has DM content equal to barionic one.
Given this, I don't have enough DM loss.
It is worth noting that the mass is computed within a 10 kpc sphere
""")
        st.write(plot_mass(self.dh.data_big(), 'm_halo_m_star', xlim=(1, None), log=True))

class P_avg_mu_e(Page):
    def _write(self):
        st.header('Average surface brightness inside R_eff')
        sim_n = st.selectbox('Which simulation:', [62, 41])
        color_by = st.radio('Color by:', ['sfr', 'ssfr'])
        # how_many_snaps = st.slider('how many points to show',
        #                             min_value=5,
        #                             # max_value=len(d[f'{sim_n}p50']),
        #                             max_value=30,
        #                             value=15,
        #                             step=5)
        st.write(plot_avg_mu_e_aku(self.dh.data(),
                                   sim_n=int(sim_n),
                                   color_by=color_by,
                                   # how_many_snaps=how_many_snaps,
                                   show_aku=True,
                                   )
                 )

class P_ssfr(Page):
    def _write(self):
        st.header('sSFR')
        st.write('Specific star formation around first infall.')
        cold_gas = st.checkbox('Plot cold gas')
        st.write(plot_ssfr(self.dh.data_big(), cold_gas))

class P_CM(Page):
    def _write(self):
        st.header('Evolution on a Color-Magnitude diagram')
        color_by = st.selectbox('Color by:', ['ssfr', 'ssfr_mean', 'sfr', 'r', 't_period', ''])
        st.write(plot_color_magnitude_aku(self.dh.data_rt(), color_col=color_by))
