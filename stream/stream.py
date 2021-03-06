import numpy as np
import pandas as pd
import streamlit as st

from pages import *
import subprocess

try:
    git_version = subprocess.check_output(["git", "describe", "--dirty", "--always", "--tags"]).strip().decode()
except:
    git_version = ''

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
        'Star mass': P_star_mass(),
        'Effective radius': P_r_eff(),
        'Sigma': P_sigma(),
        'Total mass': P_tot_mass(),
        'M_halo/M_star': P_m_halo_m_star(),
        'sSFR': P_ssfr(),
        'Lambda_R': P_lambda_R(),
        'Angular momentum and Lambda_R': P_compare_angmom(),
        'V/sigma': P_v_over_sigma(),
        'Representative galaxies': P_single_sims(),
        'Average mu_e (cf. Venhola2019)': P_avg_mu_e(),
        'Color Magnitude (cf. Venhola2019)': P_CM(),
        'Sérsic Index (cf. Venhola2019)': P_sersic(),
        'Lambda vs. mass (cf. Scott2020)': P_lambda_mass_Scott2020(),
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
    st.sidebar.image('https://www.astro.rug.nl/~sundial/images/logo.png',use_column_width=True)
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

    st.sidebar.image('https://www.astro.rug.nl/~sundial/images/EU.png')

    if git_version:
        st.sidebar.subheader("Version")
        st.sidebar.info(git_version)


if __name__ == "__main__":
    main()
