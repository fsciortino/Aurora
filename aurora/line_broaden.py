'''

line_shape.py is a module that manages line radiation
broadening physics one wishes to include

cjperks
Aug. 04, 2023

'''

# Modules
import numpy as np
import scipy.constants as cnt
import scipy
from scipy.interpolate import interp1d

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
        # If one wishes to model Doppler+Natural broadening
        if brd == 'Voigt':
            dhsape[brd] = _get_pVoigt(
                dphysics = dbroad[brd],
                wave_A = wave_A,
                )

        # If one wishes to include Doppler broadening
        elif brd == 'Doppler':
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

        # If one wishes to scope Suprathermal IOns
        elif brd == 'Suprathermal_Ions':
            dshape[brd] = _get_Supra(
                dphysics=dbroad[brd],
                wave_A=wave_A,
                )

    # If only considering one broadening mech, skips convolution
    if len(dshape.keys()) == 1:
        brd = list(dshape.keys())[0]
        lams_profs_A = dshape[brd]['lams_profs_A'] # [AA], dim(trs,nlamb)
        theta = dshape[brd]['theta'] # [1/AA], dim(trs,nlamb)

    # Convolutes line shapes
    else:
        lams_profs_A, theta = _convolve(dshape=dshape) # dim(trs,nlamb)

    # Output
    return lams_profs_A, theta


########################################################
#
#             Broadening Mechanisms
#
#########################################################

# Calculates Doppler broadening from Suprathermal Ions
def _get_Supra(
    # Settings
    dphysics=None,         # Dictionary of neccessary phyiscs
    # ADF15 file
    wave_A=None,
    ):
    '''
    NOTE: It is assumed that the user won't use the Suprathermal
    ions and the Doppler (due to Maxwellian ions) options
    together !!!!!

    INPUTS: dphysics -- [dict], necessary physics information
                i) 'model' -- [str], Suprathermal ion model
                    !!! Currently accepted models and the
                        necessary inputs

                    1) 'Bi-Maxwellain'
                ii) 'Ti_eV' -- [float], [eV], local ion temperature
                iii) 'Ti_fast_eV' -- [float], [eV], local fast
                        ion temperature
                iv) 'f_fast' -- [float], [fraction], fraction
                        of fast ion Maxwellian population
                v) 'ion_A' -- [float], [amu], species mass

                    2) NOT YET IMPLEMENTED
                        TBD: ASCOT, TRANSP, etc. files

            wave_A -- [list], dim(trs,), [AA], 
                transition central wavelength

    OUTPUTS: [dict], line shape
                i) 'lam_profs_A' -- [AA], dim(trs,nlamb), 
                        wavelength mesh for each transition
                ii) 'theta' -- [1/AA], dim(trs, nlamb),
                        line shape for each transition


    '''

    # If using the Bi-Maxwellian fast ion model
    if dphysics['model'] == 'Bi-Maxwellian':
        # Ion mass
        mass = cnt.m_p * dphysics['ion_A'] # [kg]

        # Slow population variance
        dnu_slow = (
            np.sqrt(2.0 * (dphysics['Ti_eV'] * cnt.e) / mass)
            / cnt. c
            * (cnt.c*1e10 / wave_A)
            ) # [Hz], dim(trs,)

        # Fast population variance
        dnu_fast = (
            np.sqrt(2.0 * (dphysics['Ti_fast_eV'] * cnt.e) / mass)
            / cnt. c
            * (cnt.c*1e10 / wave_A)
            ) # [Hz], dim(trs,)

        # Calculates general Gaussian shape for slow population
        lams_slow_A, theta_slow = _calc_Gaussian(
            dnu = dnu_slow,
            wave_A=wave_A,
            nstd = 5,
            ) # dim(ntrs, nlamb)

        # Calculates general Gaussian shape for fast population
        lams_fast_A, theta_fast = _calc_Gaussian(
            dnu = dnu_fast,
            wave_A=wave_A,
            nstd = 2,
            ) # dim(ntrs, nlamb)

        # Interpolates the slow pollluation shape onto (larger) fast mesh
        slow_tmp = np.zeros(lams_fast_A.shape)
        for trs in np.arange(lams_fast_A.shape[0]):
            slow_tmp[trs,:] = interp1d(
                lams_slow_A[trs,:],
                theta_slow[trs,:],
                bounds_error=False,
                fill_value=0.0
                )(lams_fast_A[trs,:])

        # Combines population
        lams_profs_A = lams_fast_A # 
        theta = (
            (1-dphysics['f_fast']) * slow_tmp
            + dphysics['f_fast'] * theta_fast
            )

    # Output line shape and wavelength mesh
    return {
        'lams_profs_A': lams_profs_A,   # [AA], dim(trs,nlamb)
        'theta': theta,                 # [1/AA], dim(trs,nlamb)
        }

# Calculates pseudo-Voigt broadening profiles
def _get_pVoigt(
    # Settings
    dphysics=None,      # Dictionary of neccessary physics
    # ADF15 file
    wave_A=None,
    ):
    '''
    NOTE: Here we implement the pseudo-Voigt method as described in:
    https://en.wikipedia.org/wiki/Voigt_profile#Pseudo-Voigt_approximation
        Quoted as accurate within 1%

    INPUTS: dphysics -- [dict], necessary physics information
                i) 'Ti_eV' -- [float], [eV], local ion temperature
                ii) 'ion_A' -- [float], [amu], species mass
                iii) 'key_options' -- [string], defines how Einstein
                    coefficient data is indexed
                    options: 'wavelegth', 'isel'
                iv) rest of the keys are assumed to relate a float
                    for the Einstein coefficient to the transition

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

                NOTE: Einstein coefficients are decay rates, but measured
                        in terms of angular frequencies

                NOTE: Transitions missing an Einstein coefficient will 
                    be treated just as Gaussian from Doppler broadening

            wave_A -- [list], dim(trs,), [AA], 
                transition central wavelength

    OUTPUTS: [dict], line shape
                i) 'lam_profs_A' -- [AA], dim(trs,nlamb), 
                        wavelength mesh for each transition
                ii) 'theta' -- [1/AA], dim(trs, nlamb),
                        line shape for each transition
                iii) 'FWHM' -- [Hz], dim(trs,)
                        full-width, half-max of line shape


    '''

    # Calculates Gaussian part
    out_G = _get_Doppler(
        dphysics = dphysics,
        wave_A = wave_A
        )

    # Calculates Lorentzian part
    out_L = _get_Natural(
        dphysics = dphysics,
        wave_A = wave_A
        )

    # Initializes output
    lams_profs_A = np.zeros((len(wave_A), 101)) # [AA], dim(trs, nlamb)
    theta = np.zeros((len(wave_A), 101)) # [1/AA], dim(trs, nlamb)
    FWHM = np.zeros(len(wave_A)) # [Hz], dim(trs,)

    # Loop over transitions
    for tr in np.arange(len(wave_A)):
        # If no Natural broadening data
        if np.max(out_L['FWHM'][tr]) <= 0:
            # Output
            lams_profs_A[tr,:] = out_G['lams_profs_A'][tr,:]
            theta[tr,:] = out_G['theta'][tr,:]
            FWHM[tr] = out_G['FWHM'][tr]

        # If Dopler+Natural
        else:
            print('xxx')

    # Output line shape and wavelength mesh
    return {
        'lams_profs_A': lams_profs_A,   # [AA], dim(trs,nlamb)
        'theta': theta,                 # [1/AA], dim(trs,nlamb)
        'FWHM': FWHM,                   # [Hz], dim(trs,)
        }

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
                iii) 'FWHM' -- [Hz], dim(trs,)
                        full-width, half-max of line shape
    '''

    # Doppler broadening variance
    mass = cnt.m_p * dphysics['ion_A'] # [kg]
    dnu = (
        np.sqrt(2.0 * (dphysics['Ti_eV'] * cnt.e) / mass)
        / cnt. c
        * (cnt.c*1e10 / wave_A)
        ) # [Hz], dim(trs,)
    FWHM = 2 *dnu *np.sqrt(2 *np.log(2)) # FWHM, [Hz], dim(trs,)

    # Calculates general Gaussian shape
    lams_profs_A, theta = _calc_Gaussian(
        dnu = dnu,
        wave_A=wave_A,
        )

    # Output line shape and wavelength mesh
    return {
        'lams_profs_A': lams_profs_A,   # [AA], dim(trs,nlamb)
        'theta': theta,                 # [1/AA], dim(trs,nlamb)
        'FWHM': FWHM,                   # [Hz], dim(trs,)
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
                i) 'key_options' -- [string], defines how Einstein
                    coefficient data is indexed
                    options: 'wavelegth', 'isel'
                ii) rest of the keys are assumed to relate a float
                    for the Einstein coefficient to the transition 

                If 'key_options' == 'wavelength'  -->   
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

                    NOTE: Einstein coefficients are decay rates, but measured
                        in terms of angular frequencies

            wave_A -- [list], dim(trs,), [AA], 
                transition central wavelength

    OUTPUTS: [dict], line shape
                i) 'lam_profs_A' -- [AA], dim(trs,nlamb), 
                        wavelength mesh for each transition
                ii) 'theta' -- [1/AA], dim(trs, nlamb),
                        line shape for each transition
                iii) 'FWHM' -- [Hz], dim(trs,)
                        full-width, half-max of line shape

    '''

    # Initializes output
    lams_profs_A = np.zeros((len(wave_A), 101)) # [AA], dim(trs, nlamb)
    theta = np.zeros((len(wave_A), 101)) # [1/AA], dim(trs, nlamb)
    FWHM = np.zeros((len(wave_A))) # [Hz]

    # Tolerance on wavelength match
    tol = 1e-4 # [AA]

    # Loop over transitions of interest
    for lmb in dphysics.keys():
        # Error check
        if lmb == 'key_options':
            continue

        # If transition defined by central wavelength
        if dphysics['key_options'] == 'wavelength':
            # Finds transitions of interst in ADF15 file
            ind = np.where(
                (wave_A >= float(lmb) -tol)
                & (wave_A <= float(lmb)+tol)
                )[0]
        # If transition defined by isel index within ADF15 file
        elif dphysics['key_options'] == 'isel':
            ind = int(float(lmb) - 1)

        # Error check
        if len(ind) == 0:
            print('NO TRANSITION FOUND FOR LAMBDA= '+str(lmb))
            continue

        # Loop over transitions
        for ii in ind:
            # FWHM
            FWHM[ii] = dphysics[lmb]/(2*np.pi) # [Hz]

            # Calculate Lorentzian shape
            lams_profs_A[ii,:], theta[ii,:] = _calc_Lorentzian(
                dnu = FWHM[ii],
                wave_A=wave_A[ii],
                )

    # Output line shape and wavelength mesh
    return {
        'lams_profs_A': lams_profs_A,   # [AA], dim(trs,nlamb)
        'theta': theta,                 # [1/AA], dim(trs,nlamb)
        'FWHM': FWHM,                   # [Hz], dim(trs,)
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
                iii) 'FWHM' -- [Hz], dim(trs,)
                        full-width, half-max of line shape

    '''

    # Converts units of variance
    dnu = dphysics['width_A'] * cnt.c*1e10 /wave_A**2 # [Hz], dim(trs,)

    # Calculates FWHM
    FWHM = 2*dnu *np.sqrt(2 *np.log(2)) # [Hz], dim(trs,)

    # Calculates general Gaussian shape
    lams_profs_A, theta = _calc_Gaussian(
        dnu = dnu,
        wave_A=wave_A,
        )

    # Output line shape and wavelength mesh
    return {
        'lams_profs_A': lams_profs_A,   # [AA], dim(trs,nlamb)
        'theta': theta,                 # [1/AA], dim(trs,nlamb)
        'FWHM': FWHM,                   # [Hz], dim(trs,)
        }


########################################################
#
#             Line Shapes
#
#########################################################

# General Gaussian shape calculator
def _calc_Gaussian(
    dnu = None, # [Hz], [float], variance
    wave_A=None, # [AA], dim(trs,), central wavelength
    # Wavelength mesh controls
    nlamb=101, # [scalar], number of wavelength points
    nstd = 5,    # number of standard deviations in wavelength mesh
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
        * dnu * nstd
        )  # [AA], dim(trs,), 5 standard deviations

    # Wavelength mesh
    lams_profs_A = np.linspace(
        wave_A - _dlam_A, 
        wave_A + _dlam_A, 
        nlamb, axis=1) # [AA], dim(trs, nlamb)

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
    # Wavelength mesh controls
    nlamb=101,  # number of wavelength points
    nstd = 20, # number of standard deviations in wavelength mesh
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
        * dnu * nstd
        )# [AA], dim(trs,), 20 standard deviations

    # Wavelength mesh
    lams_profs_A = np.linspace(
        wave_A - _dlam_A, 
        wave_A + _dlam_A, 
        nlamb) # [AA], dim(,nlamb)

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

########################################################
#
#             Utilities
#
#########################################################

# Convolves together line profiles
def _convolve(
    dshape=None,
    ):
    '''
    NOTE: THis function has been benchmarked 
    against analytic euqations for 
        i) Gaussian * Delta function
        ii) Gaussian * Gaussian
    to a <1% error

    '''

    # Broadening mechanisms
    mechs = list(dshape.keys())

    # Initializes output
    lams_profs_A = np.zeros(dshape[mechs[0]]['lams_profs_A'].shape) # [AA], dim(trs, nlamb)
    theta = np.zeros(dshape[mechs[0]]['lams_profs_A'].shape) # [1/AA], dim(trs,nlamb)

    # Number of wavelength points
    nlamb = dshape[mechs[0]]['lams_profs_A'].shape[1]

    # Loop over transitions
    for trs in np.arange(dshape[mechs[0]]['lams_profs_A'].shape[0]):
        # Handling if Natural broadening wasn't included for this transition
        mechs_tmp = mechs.copy()

        if 'Natural' in mechs_tmp:
            # Removes Natural broadening entry from consideration
            if sum(dshape['Natural']['theta'][trs,:]) == 0:
                mechs_tmp.remove('Natural')

        # If now only one mechanism left
        if len(mechs_tmp) == 1:
            lams_profs_A[trs,:] = dshape[mechs_tmp[0]]['lams_profs_A'][trs,:]
            theta[trs,:] = dshape[mechs_tmp[0]]['theta'][trs,:]

        # Convolute if more than one mechanism
        else: 
            # Initializes wavelength mesh
            lams_min = 1e20
            lams_max = 1e-20

            # Calculates wavelength mesh
            for bb in mechs_tmp:
                lams_min = np.min((lams_min, np.min(dshape[bb]['lams_profs_A'][trs,:]))) 
                lams_max = np.max((lams_max, np.max(dshape[bb]['lams_profs_A'][trs,:])) )

                nlamb = np.max((nlamb, dshape[bb]['lams_profs_A'].shape[1]))

            # Assured to be symmetric -> contains lambda_0
            lams_profs_A[trs,:] = np.linspace(lams_min,lams_max,nlamb)

            # Interpolates broadening profiles onto the same mesh
            theta_tmp = {}
            for bb in mechs_tmp:
                theta_tmp[bb] = interp1d(
                    dshape[bb]['lams_profs_A'][trs,:],
                    dshape[bb]['theta'][trs,:],
                    bounds_error=False,
                    fill_value = 0.0,
                    )(lams_profs_A[trs,:])

            # Initializes convolution
            theta[trs,:] = theta_tmp[mechs_tmp[0]]

            # Calculates convolution
            for bb in mechs_tmp[1:]:
                theta[trs,:] = scipy.signal.convolve(
                    theta[trs,:], 
                    theta_tmp[bb], 
                    mode='same'
                    )/sum(theta_tmp[bb])

    # Output
    return lams_profs_A, theta