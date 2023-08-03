'''

line_shape.py is a module that manages line radiation
broadening physics one wishes to include

cjperks
Aug. 04, 2023

'''

# Modules
import numpy as np
import scipy.constants as cnt

__all__ = [
    'get_line_broaden',
    ]



########################################################
#
#                   Main
#
#########################################################

def get_line_broaden(
    # Settings
    dbroad=None,        # Dictionary storing various broadening mechs
    # ADF15 file
    wave_A=None,        # [AA], dim(trs,), central wavelengths
    ):

    # Initializes ouput
    line_shape = 0

    # Initializes storage
    dshape = {}

    # Loop over various physics options
    for brd in dbroad.keys():
        # If one wishes to include Doppler broadening
        if brd == ' Doppler':
            dshape[brd] = _get_Doppler(
                dphysics=dbroad[brd],
                wave_A=wave_A,
                )

        # If one wishes to inlcude Natural broadening
        elif brd == 'Natural':
            dshape[brd] = _get_Natural(
                dphysics=dbroad[brd],
                wave_A=wave_A,
                )

        # If one wishes to include Instrumental broadening
        elif brd == 'Instrumental':
            dshape[brd] = _get_Instru(
                dphyscis=dbroad[brd],
                wave_A=wave_A,
                )

   
    return line_shape


########################################################
#
#                   Utilities
#
#########################################################

def _get_Doppler(
    # Settings
    dpyhsics=None,         # Dictionary of neccessary phyiscs
    # ADF15 file
    wave_A=None,
    ):
    '''
    INPUTS: dphysics -- [dict], necessary physics information
                i) 'Ti_eV' -- [float], [eV], local ion temperature
                ii) 'ion_A' -- [float], [amu], species mass

            wave_A -- [list], dim(trs,), [AA], 
                transition central wavelength

    OUTPUTS: [dict], line shape
                i) 'lam_profs_A' -- [AA], dim(trs,nlamb), 
                        wavelength mesh for each transition
                ii) 'theta' -- [1/AA], dim(trs, nlamb),
                        line shape for each transition


    '''

     # Doppler broadening FWHM
    mass = cnt.m_p * dphysics['ion_A'] # [kg]
    dnu_g = (
        np.sqrt(2.0 * (dphysics['Ti_eV'] * cnt.e) / mass)
        / cnt. c
        * (cnt.c / wave_A)
        ) # [Hz], dim(trs,)

    # set a variable delta lambda based on the width of the broadening
    _dlam_A = wave_A**2 / cnt.c * dnu_g * 5  # [AA], dim(trs,), 5 standard deviations

    # Wavelength mesh
    lams_profs_A = np.linspace(
        wave_A - _dlam_A, 
        wave_A + _dlam_A, 
        100, axis=1) # [AA], dim(trs, nlamb)

    # Gaussian profiles of the lines
    theta = np.exp(
        -(
            ((cnt.c / lams_profs_A - cnt.c / wave_A[:, None]) 
            / dnu_g[:, None]) ** 2
            )
        ) # [], dim(trs, nlamb)

    # Normalize Gaussian profile
    theta /= (
        np.sqrt(np.pi) * dnu_g[:, None] 
        * wave_A[:, None] ** 2 / cnt.c
        ) # [1/AA], dim(trs,nlamb)

    # Output line shape and wavelength mesh
    return {
        'lams_profs_A': lams_profs_A,   # [AA], dim(trs,nlamb)
        'theta': theta,                 # [1/AA], dim(trs,nlamb)
        }


    # non-equally spaced wavelenght
    #wave_final_A = np.unique(lams_profs_A)

    #if (plot_all_lines or plot_spec_tot) and ax is None:
    #    fig, ax = plt.subplots()

    # contributions to spectrum
    #spec = {}
    #spec_tot = np.zeros_like(wave_final_A)
    #colors = {"ioniz": "r", "excit": "b", "recom": "g", "drsat": "m", "chexc": "c"}
    #for ii in np.arange(lams_profs_A.shape[0]):
    #    line_shape = interp1d(
    #        lams_profs_A[ii], theta[ii], bounds_error=False, fill_value=0.0
    #    )(wave_final_A)
    #    for typ, intens in line_emiss[ii].items():
    #        comp = intens * line_shape
    #        if typ not in spec:
    #            spec[typ] = np.zeros_like(comp)
    #
    #        if plot_all_lines and intens > np.max([l[typ] for l in line_emiss]) / 1000:
    #            ax.plot(lams_profs_A[ii] + dlam_A, intens * theta[ii], c=colors[typ])
    #
    #        spec[typ] += comp
    #        spec_tot += comp
    #
    #return line_shape