'''

line_shape.py is a module that manages line radiation
broadening physics one wishes to include

cjperks
Aug. 04, 2023

'''

# Modules
import numpy as np
from scipy.constants import e, m_p, c

__all__ = [
    'get_line_broaden',
    ]



########################################################
#
#                   Main
#
#########################################################

def get_line_broaden(
    dbroad=None,
    ):

    # Various physics options
    if 'Doppler' in dbroad.keys():
        line_shape = _get_Doppler(
            Ti_eV=None,
            )

   
    return line_shape


########################################################
#
#                   Utilities
#
#########################################################

def _get_Doppler(
    Ti_eV=None.
    ):

     # Doppler broadening
    mass = constants.m_p * ion_A
    dnu_g = (
        np.sqrt(2.0 * (Ti_eV * constants.e) / mass)
        * (constants.c / wave_A)
        / constants.c
    )

    # set a variable delta lambda based on the width of the broadening
    _dlam_A = wave_A**2 / constants.c * dnu_g * 5  # 5 standard deviations

    lams_profs_A = np.linspace(wave_A - _dlam_A, wave_A + _dlam_A, 100, axis=1)

    # Gaussian profiles of the lines
    theta = np.exp(
        -(((constants.c / lams_profs_A - c / wave_A[:, None]) / dnu_g[:, None]) ** 2)
    )

    # Normalize Gaussian profile
    theta /= np.sqrt(np.pi) * dnu_g[:, None] * wave_A[:, None] ** 2 / constants.c

    # non-equally spaced wavelenght
    wave_final_A = np.unique(lams_profs_A)

    if (plot_all_lines or plot_spec_tot) and ax is None:
        fig, ax = plt.subplots()

    # contributions to spectrum
    spec = {}
    spec_tot = np.zeros_like(wave_final_A)
    colors = {"ioniz": "r", "excit": "b", "recom": "g", "drsat": "m", "chexc": "c"}
    for ii in np.arange(lams_profs_A.shape[0]):
        line_shape = interp1d(
            lams_profs_A[ii], theta[ii], bounds_error=False, fill_value=0.0
        )(wave_final_A)
        for typ, intens in line_emiss[ii].items():
            comp = intens * line_shape
            if typ not in spec:
                spec[typ] = np.zeros_like(comp)

            if plot_all_lines and intens > np.max([l[typ] for l in line_emiss]) / 1000:
                ax.plot(lams_profs_A[ii] + dlam_A, intens * theta[ii], c=colors[typ])

            spec[typ] += comp
            spec_tot += comp

    return line_shape