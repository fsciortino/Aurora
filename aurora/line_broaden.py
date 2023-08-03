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
        ) # [m/AA * Hz], dim(trs,)

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
            # set a variable delta lambda based on the width of the broadening
            _dlam_A = (
                wave_A[ii]**2 / (cnt.c*1e10) 
                * dnu * 5  
                )# [AA], dim(trs,), 5 standard deviations

            # Wavelength mesh
            lams_profs_A[ii,:] = np.linspace(
                wave_A[ii] - _dlam_A, 
                wave_A[ii] + _dlam_A, 
                100) # [AA], dim(,nlamb)

            # Lorentz profile
            theta[ii,:] = 1/(
                1+ (
                    (1/lam_profs_A[ii,:] - 1/ wave_A[ii])
                    * cnt.c*1e10
                    * 2*np.pi*tau
                    )**2
                ) # [], dim(,nlamb)
                
            # Normalization
            theta[ii,:] *= 2* tau # [1/Hz]

            # Fixes units
            theta[ii,:] *= (cnt.c*1e10)/wave_A[ii]**2 # [1/AA]

    # Output line shape and wavelength mesh
    return {
        'lams_profs_A': lams_profs_A,   # [AA], dim(trs,nlamb)
        'theta': theta,                 # [1/AA], dim(trs,nlamb)
        }
