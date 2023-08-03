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
#             Broadening Mechanisms
#
#########################################################

# Calculates Doppler broadening
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
    dnu = (
        np.sqrt(2.0 * (dphysics['Ti_eV'] * cnt.e) / mass)
        / cnt. c
        * (cnt.c*1e10 / wave_A)
        ) # [Hz], dim(trs,)

    # Calculates general Gaussian shape
    lams_profs_A, theta = _get_Gaussian(
        dnu = dnu,
        wave_A=wave_A,
        )

    # Output line shape and wavelength mesh
    return {
        'lams_profs_A': lams_profs_A,   # [AA], dim(trs,nlamb)
        'theta': theta,                 # [1/AA], dim(trs,nlamb)
        }

# Calculates Natural broadening
def _get_Natural(
    # Settings
    dphysics=None,
    # ADF15 file
    wave_A=None,
    ):
    '''
    INPUTS: dphysics -- [dict], necessary physics information
                NOTE: The expected form is keys for each central
                wavelength of interst with its associated Einstein
                coefficient as a float. This is so that a user won't
                need to provide a data for every transition in the
                ADF15 file.

                !!! It is assumed you know, at least very closely,
                the exact wavelength stored in your ADF15 file

                i.e., If one wishes to explore the impact of Natural
                broadening on the w, x, y, z lines for He-like Kr, the
                input dictionary would look like --
                dphysics = {
                    '0.9454': 1.529e15, # [1/s], w line
                    '0.9471': 9.327e10, # [1/s], x line
                    '0.9518': 3.945e14, # [1/s], y line
                    '0.9552': 5.715e9,  # [1/s], z line
                    }
                    ref: K.M. Aggarwal and F.P. Keenan, 2012 Phys. Scr. 86 035302

            wave_A -- [list], dim(trs,), [AA], 
                transition central wavelength

    OUTPUTS: [dict], line shape
                i) 'lam_profs_A' -- [AA], dim(trs,nlamb), 
                        wavelength mesh for each transition
                ii) 'theta' -- [1/AA], dim(trs, nlamb),
                        line shape for each transition


    '''

    # Initializes output
    lams_profs_A = np.zeros((len(wave_A), 100)) # [AA], dim(trs, nlamb)
    theta = np.zeros((len(wave_A), 100)) # [1/AA], dim(trs, nlamb)

    # Tolerance on wavelength match
    tol = 1e-4 # [AA]

    # Loop over transitions of interest
    for lmb in dphysics.keys():
        # Finds transitions of interst in ADF15 file
        ind = np.where(
            (wave_A >= lmb -tol)
            & (wave_A <= lamb+tol)
            )[0]

        # Error check
        if len(ind) == 0:
            print('NO TRANSITION FOUND FOR LAMBDA= '+str(lmb))
            continue

        # Characteristic time for transitions
        tau = 2/dphysics[lmb] # [s]

        # FWHM
        dnu = 1/(np.pi*tau) # [Hz]

        # Loop over transitions
        for ii in ind:
            # Calculate Lorentzian shape
            lams_profs_A[ii,:], theta[ii,:] = _get_Lorentzian(
                dnu = dnu,
                wave_A=wave_A[ii],
                )

    # Output line shape and wavelength mesh
    return {
        'lams_profs_A': lams_profs_A,   # [AA], dim(trs,nlamb)
        'theta': theta,                 # [1/AA], dim(trs,nlamb)
        }

# Calculates Instrumental broadening
def _get_Instru(
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



########################################################
#
#             Line Shapes
#
#########################################################

# General Gaussian shape calculator
def _calc_Gaussian(
    dnu = None, # [Hz], [float], FWHM
    wave_A=None, # [AA], dim(trs,), central wavelength
    ):
    '''
    Note this function is meant to be a general-purpose 
    Gaussian shape calculator to use when considering
        1) Doppler broadening
        2) Suprathermal ions
        3) Instrumental broadening

    '''

    # set a variable delta lambda based on the width of the broadening
    _dlam_A = (
        wave_A**2 / (cnt.c*1e10) 
        * dnu * 5
        )  # [AA], dim(trs,), 5 standard deviations

    # Wavelength mesh
    lams_profs_A = np.linspace(
        wave_A - _dlam_A, 
        wave_A + _dlam_A, 
        100, axis=1) # [AA], dim(trs, nlamb)

    # Gaussian profiles of the lines
    theta = np.exp(
        -(
            ((1 / lams_profs_A - 1 / wave_A[:, None])
            *cnt.c *1e10
            / dnu[:, None]) ** 2
            )
        ) # [], dim(trs, nlamb)

    # Normalize Gaussian profile
    theta /= (
        np.sqrt(np.pi) * dnu[:, None] 
        * wave_A[:, None] ** 2 / (cnt.c*1e10)
        ) # [1/AA], dim(trs,nlamb)

    # Output
    return lams_profs_A, theta

# General Lorentzian shape calculator
def _calc_Lorentzian(
    dnu = None, # [Hz], [float], FWHM
    wave_A=None, # [AA], [float], central wavelength
    ):
    '''
    Note this function is meant to be a general-purpose 
    Lorentzian shape calculator to use when considering
        1) Natural broadening
        2) Pressure broadening

    '''

    # set a variable delta lambda based on the width of the broadening
    _dlam_A = (
        wave_A**2 / (cnt.c*1e10) 
        * dnu * 5  
        )# [AA], dim(trs,), 5 standard deviations

    # Wavelength mesh
    lams_profs_A = np.linspace(
        wave_A - _dlam_A, 
        wave_A + _dlam_A, 
        100) # [AA], dim(,nlamb)

    # Lorentz profile
    theta = 1/(
        1+ (
            (1/lam_profs_A - 1/ wave_A)
            * cnt.c*1e10
            * 2 / dnu
            )**2
        ) # [], dim(,nlamb)
                
    # Normalization
    theta *= 2 / (np.pi * dnu) # [1/Hz]

    # Fixes units
    theta *= (cnt.c*1e10)/wave_A**2 # [1/AA]

    # Output
    return lams_profs_A, theta