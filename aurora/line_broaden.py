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
        if brd == 'Doppler':
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
                dphysics=dbroad[brd],
                wave_A=wave_A,
                )

    # If only considering one broadening mech, skips convolution
    #if len(dshape.keys()) == 1:
    #    brd = list(dshape.keys())[0]
    #    lams_profs_A = dshape[brd]['lams_profs_A'] # [AA], dim(trs,nlamb)
    #    theta = dshape[brd]['theta'] # [1/AA], dim(trs,nlamb)

    # Convolutes line shapes
    #else:
    #    lams_profs_A, theta = _convolve(dshape=dshape) # dim(trs,nlamb)


    #return line_shape

    return dshape


########################################################
#
#             Broadening Mechanisms
#
#########################################################

# Calculates Doppler broadening
def _get_Doppler(
    # Settings
    dphysics=None,         # Dictionary of neccessary phyiscs
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
    lams_profs_A, theta = _calc_Gaussian(
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
                    '0.9454': 1.529e15, # [Hz], w line
                    '0.9471': 9.327e10, # [Hz], x line
                    '0.9518': 3.945e14, # [Hz], y line
                    '0.9552': 5.715e9,  # [Hz], z line
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
            (wave_A >= float(lmb) -tol)
            & (wave_A <= float(lmb)+tol)
            )[0]

        # Error check
        if len(ind) == 0:
            print('NO TRANSITION FOUND FOR LAMBDA= '+str(lmb))
            continue

        # FWHM
        dnu = dphysics[lmb]/(2*np.pi) # [Hz]

        # Loop over transitions
        for ii in ind:
            # Calculate Lorentzian shape
            lams_profs_A[ii,:], theta[ii,:] = _calc_Lorentzian(
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
    dphysics=None,         # Dictionary of neccessary phyiscs
    # ADF15 file
    wave_A=None,
    ):
    '''
    INPUTS: dphysics -- [dict], necessary physics information
                i) 'width_A' -- [float], [AA], characteristic FWHM
                                    on detector surface

                NOTE: the philosophy here is that the user wishes to
                quickly scope Instrumental broadening per a simple 
                characteristic Gaussian spread model. 

                i.e., if one considers a highly collimated, monoenergetic
                beam of photons Bragg diffracted on a crystal spectrometer,
                the reflected beam would be spread via the reflection curve
                (rocking curve). We can therefore characterize instrumental 
                broadening, assuming the rocking curve is a Gaussian, by the
                arc length on a detection surface normal to the line-of-sight
                subtended by the rocking curve FWHM. Therefore,
                        width_A = spec_rng * (omega_rc * L_cd / w_d)

                        where, omega_rc is the rocking curve FWHM in [rad],
                        and L_cd is the distance between the crystal and 
                        detector in [m], spec_rng is the spectral range 
                        imaged in [AA], and w_d is the width of the detector
                        image in [m]

                        NOTE: the value in the parentheses is really the fraction
                        of pixels the broadening subtends

                !!! To properly quantify instrumental broadening to account for
                effects such as finite aperture size, defocusing, misalignment, etc.
                it is heavily recommended to use a dedicated ray-tracing code

            wave_A -- [list], dim(trs,), [AA], 
                transition central wavelength

    OUTPUTS: [dict], line shape
                i) 'lam_profs_A' -- [AA], dim(trs,nlamb), 
                        wavelength mesh for each transition
                ii) 'theta' -- [1/AA], dim(trs, nlamb),
                        line shape for each transition


    '''

    # Converts units of FWHM
    dnu = dphysics['width_A'] * cnt.c*1e10 /wave_A**2 # [Hz], dim(trs,)

    # Calculates general Gaussian shape
    lams_profs_A, theta = _calc_Gaussian(
        dnu = dnu,
        wave_A=wave_A,
        )

    # Output line shape and wavelength mesh
    return {
        'lams_profs_A': lams_profs_A,   # [AA], dim(trs,nlamb)
        'theta': theta,                 # [1/AA], dim(trs,nlamb)
        }


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
        * dnu * 4
        )  # [AA], dim(trs,), 4 standard deviations

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
        * dnu * 20
        )# [AA], dim(trs,), 20 standard deviations

    # Wavelength mesh
    lams_profs_A = np.linspace(
        wave_A - _dlam_A, 
        wave_A + _dlam_A, 
        100) # [AA], dim(,nlamb)

    # Lorentz profile
    theta = (1/np.pi) * (
        dnu/2
        /(
            (1/lams_profs_A - 1/wave_A)**2
            * (cnt.c*1e10)**2
            + (dnu/2)**2
            )
        ) # [1/Hz], dim(,nlamb)

    # Fixes units
    theta *= (cnt.c*1e10)/wave_A**2 # [1/AA]

    # Output
    return lams_profs_A, theta