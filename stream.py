import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import copy
from collections import namedtuple

from pages import *
import subprocess

git_version = subprocess.check_output(["git", "describe", "--dirty", "--always", "--tags"]).strip().decode()


# def write_page(page):  # pylint: disable=redefined-outer-name
#     """Writes the specified page/module
#     Our multipage app is structured into sub-files with a `def write()` function
#     Arguments:
#         page {module} -- A module with a 'def write():' function
#     """
#     # _reload_module(page)
#     page.write()


def get_pages():
    PAGES = {
        'Effective radius': P_r_eff(),
        'Sigma': P_sigma(),
        'Star mass': P_star_mass(),
        'Total mass': P_tot_mass(),
        'M_halo/M_star': P_m_halo_m_star(),
        'sSFR': P_ssfr(),
        'Average mu_e': P_avg_mu_e(),
        'Color Magnitude': P_CM(),
        'Sérsic Index': P_sersic(),
        'Representative galaxies': P_single_sims(),
        'Lambda_R': P_lambda_R(),
        'Angular momentum and Lambda_R': P_compare_angmom(),
        'V/sigma': P_v_over_sigma(),
    }
    return PAGES


# PAGES = get_pages()

# # Adds a button to the sidebar
# which_plot = st.sidebar.radio(
#     'Which plot',
#     list(PAGES.keys()),
# )

# st.sidebar.markdown(f'`version: {git_version}`')


# for page in PAGES.values():
#     page.write()


def main():
    """Main function of the App"""
    st.sidebar.title("Navigation")
    pages = get_pages()
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    page = pages[selection]

    with st.spinner(f"Loading {selection} ..."):
        page.write()
    #     st.sidebar.title("About")
    #     st.sidebar.info(
    #         """
    #         This app is maintained by Michele Mastropietro.
    # """
    #     )

    st.sidebar.subheader("Version")
    st.sidebar.info(git_version)


if __name__ == "__main__":
    main()
