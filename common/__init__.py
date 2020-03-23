

labels = {'avg_mu_e': r"$\bar{\mu}_{e,r'}$ (mag/arcsec$^2$)",
          'cold_gas_full': r"M$_g$ ($T<15000$ K) (M$_\odot$)",
          'cold_gas_short': r"M$_g$ (M$_\odot$)",
          'cold_gas': r"M$_g$ (M$_\odot$)",
          'mass_star': r'M$_\star$ (M$_\odot)$',
          'log_mass_star': r'log$_{10}$(M$_\star$/M$_\odot$)',
          'sfr': r'SFR (M$_\odot$/yr)',
          'ssfr': r'sSFR (1/yr)',
          'r': 'r [kpc)',
          'mag_sdss_r': r"M$_{r'}$",
          'r_eff3d': r"R$_{e}^{3d}$ (kpc)",
          'sigma_star': r"$\sigma_\star$ (km/s)",
          'm_halo_m_star': r'$M_h/M_\star$',
          'n': r"Sérsic index",
          'lambda_r': r"$\lambda_R$",
          # 'js': r"Angmom (kpc km/s)",
          'js_c_y': r"$j_s$ (kpc km/s)",
          }

FIGSIZE = (5, 1.2)

LEGEND_FONT_SIZE = 6

def get_figsize(nrows):
    return FIGSIZE[0], FIGSIZE[1]*nrows

def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )