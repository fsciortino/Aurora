'''

line_shape.py is a module that manages line radiation
broadening physics one wishes to include

cjperks
Aug. 04, 2023

'''

# Modules
import sys
import numpy as np
import scipy.constants as cnt
import scipy
from scipy.interpolate import interp1d, splev, splrep
from scipy.special import voigt_profile as vp

# Non-standard modules
try:
    import ChiantiPy.tools.io as io
    Chianti_avail = True
except:
    Chianti_avail = False

__all__ = [
    'get_line_broaden',
    ]

# Number of wavelength mesh points
nlamb=101

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
    # TMP FOR DEBUG!!!!!
    use_scipy = True,
    use_pseudo = False,
    ):
    '''
    Key options for broadening mechanisms:
        i) '2-photon', ii) 'Voigt', iii) 'Doppler', 
        iv) 'Natural', v) 'Instrumental', vi) 'Suprathermal_Ions'

    '''

    # Initializes ouput
    line_shape = 0

    # Initializes storage
    dshape = {}

    # If relevant, finds 2-photon emission first
    if '2-photon' in dbroad.keys():
        dshape['2-photon'] = _get_2photon(
            dphysics = dbroad['2-photon'],
            wave_A = wave_A,
            )

    # Loop over various physics options
    for brd in dbroad.keys():
        if brd == '2-photon':
            continue

        # If one wishes to model Doppler+Natural broadening
        if brd == 'Voigt':
            dshape[brd] = _get_Voigt(
                dphysics = dbroad[brd],
                wave_A = wave_A,
                # TMP FOR DEBUGGING !!!!
                use_scipy=use_scipy,
                use_pseudo=use_pseudo,
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

        # Error check
        else:
            print('Undefined desired broadening mechanism')
            sys.exit(1)

    # Broadening mechanisms
    brds = list(dshape.keys())
    flag_2q = False
    if '2-photon' in brds:
        brds.remove('2-photon')
        flag_2q = True

    # If only considering one broadening mech, skips convolution
    if len(brds) == 1:
        lams_profs_A = dshape[brds[0]]['lams_profs_A'] # [AA], dim(trs,nlamb)
        theta = dshape[brds[0]]['theta'] # [1/AA], dim(trs,nlamb)

    # Convolutes line shapes
    elif len(brds) > 1:
        lams_profs_A, theta = _convolve(dshape=dshape) # dim(trs,nlamb)

    # Error check
    elif flag_2q:
        lams_profs_A = np.zeros(dshape['2-photon']['theta'].shape)
        theta = np.zeros(dshape['2-photon']['theta'].shape)
    else:
        print('No Broadening mechanisms defined')
        sys.exit(1)

    # Overrides line shapes for 2-photon shape
    if flag_2q:
        for ii, tr in enumerate(dshape['2-photon']['ind_2photon']):
            lams_profs_A[tr,:] = dshape['2-photon']['lams_profs_A'][ii,:]
            theta[tr,:] = dshape['2-photon']['theta'][ii,:]

    # Output, [AA], [1/AA], dim(ntrs, nlambda)
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

# Calculates Voigt broadening profiles
def _get_Voigt(
    # Settings
    dphysics=None,      # Dictionary of neccessary physics
    # ADF15 file
    wave_A=None,
    # Data source options
    use_scipy = True,
    use_pseudo = False,
    ):
    '''
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
                    'Ti_eV': 10e3,
                    'ion_A': 83.798,
                    'key_options': 'wavelength',
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
    lams_profs_A = np.zeros((len(wave_A), nlamb)) # [AA], dim(trs, nlamb)
    theta = np.zeros((len(wave_A), nlamb)) # [1/AA], dim(trs, nlamb)
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
            # Use scipy's Voigt function
            if use_scipy:
                (
                    lams_profs_A[tr,:],
                    theta[tr,:],
                    FWHM[tr] 
                    ) = _calc_Voigt_scipy(
                        FWHM_G = out_G['FWHM'][tr],
                        FWHM_L = out_L['FWHM'][tr],
                        lambda0 = wave_A[tr]
                        )

            # Use pseudo-Voigt weighted sum function
            elif use_pseudo:
                (
                    lams_profs_A[tr,:],
                    theta[tr,:],
                    FWHM[tr]
                    ) = _calc_Voigt_pseudo(
                        FWHM_G = out_G['FWHM'][tr],
                        lams_G = out_G['lams_profs_A'][tr,:],
                        theta_G = out_G['theta'][tr,:],
                        FWHM_L = out_L['FWHM'][tr],
                        lams_L = out_L['lams_profs_A'][tr,:],
                        theta_L = out_L['theta'][tr,:],
                        lambda0 = wave_A[tr]
                        )
 
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
    # NOTE: Factor of sqrt(2) already accounted for in dnu
    FWHM = 2*dnu *np.sqrt(np.log(2)) # FWHM, [Hz], dim(trs,)

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
                    'key_options': 'wavelength,
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
    lams_profs_A = np.zeros((len(wave_A), nlamb)) # [AA], dim(trs, nlamb)
    theta = np.zeros((len(wave_A), nlamb)) # [1/AA], dim(trs, nlamb)
    FWHM = np.zeros((len(wave_A))) # [Hz]

    # Tolerance on wavelength match
    tol = 1e-4 # [AA]

    # Loop over transitions of interest
    for lmb in dphysics.keys():
        # Error check
        if lmb in ['key_options', 'Ti_eV', 'ion_A']:
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
            ind = [int(float(lmb) -1)]

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
                i) 'width_A' -- [float] or [array], [AA], characteristic
                                FWHM on detector surface
                    NOTE: if [array] -> then assuming to be modeling
                        instrumental broadening as a sum of Gaussians
                        WITH NO SYSTEMATIC SHIFT!!!!!

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
    if isinstance(dphysics['width'], float):
        dnu = dphysics['width_A'] * cnt.c*1e10 /wave_A**2 # [Hz], dim(trs,)
    else:
        tmp = (
            dphysics['width_A'][None,: ] 
            * cnt.c*1e10 /wave_A[:,None]**2 
            ) # [Hz], dim(trs, nGaussians)
        dnu = np.sqrt(
            np.sum(tmp**2, axis=1)
            ) # [Hz], dim(trs,)

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

# Voigt profile calculator using scipy
def _calc_Voigt_scipy(
    FWHM_G = None,      # [Hz], Gaussian-part full-width, half-max
    FWHM_L = None,      # [Hz], Lorentzian-part full-width, half-max
    lambda0 = None,     # [AA], Central wavelength
    ):

    # Estimate for Voigt full-width, half-max
    FWHM_V = (
        0.5346*FWHM_L 
        + np.sqrt(
            0.2166*FWHM_L**2
            + FWHM_G**2
            )
        ) # [Hz] 

    # Non-dimensionalized mesh
    mesh = np.linspace(3,-3,nlamb)

    # Non-dimensionalized Gaussian-part variance
    sig = FWHM_G/FWHM_V /(2*np.sqrt(2*np.log(2)))

    # Non-dimensionalized Lorentzian-part half-width, half-max
    gam = FWHM_L/FWHM_V /2

    # Voigt porfile
    theta = (
        vp(mesh,sig,gam) # []
        /FWHM_V
        * cnt.c*1e10/lambda0**2
        ) # [1/AA], dim(nlamb,)

    # Dimensionalized mesh
    mesh_A = (
        cnt.c * 1e10
        /(
            FWHM_V*mesh
            +cnt.c*1e10/lambda0
            )
        ) # [AA]

    # Ensure normalization
    theta /= np.trapz(theta,mesh_A)

    # Output
    return mesh_A, theta, FWHM_V

# Voigt profile calculator using pseudo-Voigt weighted sum method
def _calc_Voigt_pseudo(
    FWHM_G = None,      # [Hz], Gaussian-part full-width, half-max
    lams_G = None,      # [AA], Gaussian-part wavelength mesh
    theta_G = None,     # [1/AA], Gaussian-part distribution
    FWHM_L = None,      # [Hz], Lorentzian-part full-width, half-max
    lams_L = None,      # [AA], Lorentzian-part wavelength mesh
    theta_L = None,     # [1/AA], Lorentzian-part distribution
    lambda0 = None,     # [AA], Central wavelength
    ):
    '''
    NOTE: Here we implement the pseudo-Voigt method as described in:
    https://en.wikipedia.org/wiki/Voigt_profile#Pseudo-Voigt_approximation
        Quoted as accurate within 1%

    '''

     # Calculates Voigt FWHM
    FWHM_V = (
        FWHM_G**5
        + 2.69269 *FWHM_G**4 *FWHM_L
        + 2.42843 *FWHM_G**3 *FWHM_L**2
        + 4.47163 *FWHM_G**2 *FWHM_L**3
        + 0.07842 *FWHM_G *FWHM_L**4
        + FWHM_L**5
        )**(1/5) # [Hz]

    # Calculates weighting factor
    eta = (
        1.36603 * FWHM_L/FWHM_V
        - 0.47719 *(FWHM_L/FWHM_V)**2
        + 0.11116 *(FWHM_L/FWHM_V)**3
        )

    # Calculates wavelength mesh
    lams_min = np.min(
        (
            np.min(lams_G), 
            np.min(lams_L)
            )
        ) 
    lams_max = np.max(
        (
            np.max(lams_G), 
            np.max(lams_L)
            )
        )

    # Assured to be symmetric -> contains lambda_0
    lams_V = np.linspace(lams_min,lams_max,nlamb)

    # Profile shape
    theta= (
        eta * interp1d(
            lams_L,
            theta_L,
            bounds_error =False,
            fill_value = 0.0
            )(lams_V)
        + (1-eta) * interp1d(
            lams_G,
            theta_G,
            bounds_error =False,
            fill_value = 0.0
            )(lams_V)
        )    

    # Output, [AA], [1/AA], [Hz], dim(nlambda,)
    return lams_V, theta, FWHM_V 

# General Gaussian shape calculator
def _calc_Gaussian(
    dnu = None, # [Hz], dim(ntrs,), variance
    wave_A=None, # [AA], dim(ntrs,), central wavelength
    # Wavelength mesh controls
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

    # Output, dim(nlambda,)
    return lams_profs_A, theta

########################################################
#
#             2-Photon Line Shape
#
#########################################################

# Get 2-photon energy distribution
def _get_2photon(
    dphysics = None,
    wave_A = None,
    # Data source options
    use_Chianti = True,
    use_fit = True,
    ):
    '''
    INPUTS: dphysics -- [dict], necessary physics information
                i) either 'wavelength' or 'isel', [list]
                    if 'wavelength' -> Minimum photon wavelength for 2-photon emission
                        This is want would be stored in the ADF15 file, h*c/ \Delta E_ki
                    elif 'isel' -> Index for the 2-photon transition PECs as stored in
                        the ADF15 file

                ii) (optional), 'Znuc' -> Ion nuclear charge
                    Needed if using ChiantiPy for photon energy distributions
                iii) (optional), 'nele' -> Number of electrons
                    Needed if using ChiantiPy for photon energy distributions
                    NOTE: Only nele = 1 or 2 supported

            wave_A -- [list], dim(trs,), [AA], 
                transition central wavelength

    OUTPUTS: [dict], line shape
                i) 'lam_profs_A' -- [AA], dim(trs,nlamb), 
                        wavelength mesh for each transition
                ii) 'theta' -- [1/AA], dim(trs, nlamb),
                        line shape for each transition
                iii) 'inds' -- dim(trs,)
                        indices of 2-photon emission PECs in 
                        ADF15 file

    '''

    # Tolerance on wavelength match
    tol = 1e-4 # [AA]

    # If searching for the transition by its wavelength
    if 'wavelength' in dphysics.keys():
        inds = []
        for ii in np.arange(len(dphysics['wavelength'])):
            tmp = np.where(
                (wave_A >= dphysics['wavelength'][ii] - tol)
                & (wave_A <= dphysics['wavelength'][ii] +tol)
                )[0]
            for tt in tmp:
                inds.append(tt) # dim(ntrs,)

    # If searching for the transition by its indices
    elif 'isel' in dphysics.keys():
        inds = [int(xx-1) for xx in dphysics['isel']] # dim(ntrs,)

    # Error check
    if len(inds) == 0:
        print('NO TRANSITION FOUND FOR LAMBDA= '+str(lmb))
        sys.exit(1)

    # Initializes output
    lams_profs_A = np.zeros((len(inds),nlamb)) # [AA], dim(ntrs,nlamb)
    theta = np.zeros((len(inds),nlamb))# [1/AA], dim(ntrs,nlamb)

    # Fit grid, [], dim(nlamb,)
    y_grid = np.linspace(0.99, 0.01, int(nlamb-1))

    # Loop over transitions
    for ind_t, ind_y in enumerate(inds):
        # Get phton energy distribution from ChiantiPy
        if use_Chianti and Chianti_avail:
            print('Getting 2-photon distribution from ChiantiPy')
            lams_profs_A[ind_t,1:], theta[ind_t,1:] = _2photon_Chianti(
                lamb0 = wave_A[ind_y],
                y_grid = y_grid,
                nele = dphysics['nele'],
                Znuc = dphysics['Znuc'],
                ) # [AA], [1/AA], dim(nlmabda,)

        # Get photon energy distribution for analytic fit
        elif use_fit:
            print('Getting 2-photon distribution from analytic fit')
            lams_profs_A[ind_t,1:], theta[ind_t,1:] = _2photon_fit(
                lamb0 = wave_A[ind_y],
                y_grid = y_grid,
                ) # [AA], [1/AA], dim(nlmabda,)

        # Error check
        else:
            print('No database for 2-photon distribution selected')
            sys.exit(1)

        # Adds bound at lambda = lambda_0
        lams_profs_A[ind_t,0] = wave_A[ind_y]

    # Output
    return {
        'lams_profs_A': lams_profs_A,     # [AA], dim(ntrs, nlambda)
        'theta': theta,                 # [1/AA]. dim(ntrs, nlambda)
        'ind_2photon': inds,
        }

# Get 2-photon energy distribution from analytic fit
def _2photon_fit(
    # Wavelength data
    lamb0 = None,   # Minimum wavelength of a photon
    y_grid = None,  # Reduced wavelength grid, []
    ):
    '''
    DISCLAIMER:
        What's in this function is heavily inspired by the open-source code
            PyAtomDB, specficially `pyatomdb.calc_two_phot()`
            https://github.com/AtomDB/pyatomdb
            http://www.atomdb.org/

        Utilized for the photon energy distribution is the analytic fit
        presented for H-like 2-photon emission in
            H. Nussbaumer & W. Schmutz -- Astron. & Astrophys. 1984, 138, 495-496
        The reported accuracy of the fit is <0.6% for 0.01 < \nu /\nu_0 < 0.99
            \nu being the photon frequency
            \nu_0 being the maximum photon frequency = \Delta E_ki/ h

            We assume emission outside this bounds is effectively zero

        Note that this function can also be used for He-like 2-photon emission
        per the results of: (see Figure 5)
            A. Derevianko & W.R. Johnson -- Phys. Rev. A 1997, vol 56, num 2
        The authors show that the He-like photon distribution can be well
        approximated using the H-like distribution, particularly for Z>20

    '''

    # Fit coefficients
    alpha = 0.88
    beta = 1.53
    gamma = 0.8
    CC = 202.0 # [1/s]
    A_2qH = 8.2249 # [1/s], neutral hydrogen 2-photon emission rate

    # Calculates photon distribution
    xx = y_grid*(1-y_grid)
    zz = (4*xx)**gamma
    A_y = CC *(
        xx*(1-zz)
        + alpha *xx**beta *zz
        ) # [1/s]

    # Normalizes photon distribution
    # NOTE: Technical also need R_Z/R_H, but even for Z=26,
    #   this is a 1.0005% error, so we'll ignore it
    A_y /= A_2qH # []

    # Output, [AA], [1/AA], dim(nlambda,)
    return lamb0/y_grid, y_grid**2/lamb0 *A_y

# Get 2-photn energy distribution from ChiantiPy
def _2photon_Chianti(
    # Wavelength data
    lamb0 = None,   # Minimum wavelength of a photon
    y_grid = None,  # Reduced wavelength grid, []
    # Ion data
    nele = None,    # Number of electrons
    Znuc = None,    # Nuclear charge
    ):
    '''
    DISCLAIMER:
        What's in this function is heavily inspired by the open-source code
            ChiantiPy, specifically `chiantipy.core.Ion.twoPhoton()`

        Included in the Chianti database is data tables to interpolate the
        photon energy distribution as well as Einstein coefficients for Z <=30

        The reference for the 2-photon implementation in Chianti suggests that
        the fits are high-accuracy and extrapolation to higher-Z should be fine
        as there little change to the H-like distribution for Z>28 and the He-like
        distribution for Z>10
            Ref -- P.R. Young et al, ApJSS, 144, 135-152, 2003

    '''
    # Loads H-like data table
    if nele == 1:
        data = io.twophotonHRead()
    # Loads He-like data table
    elif nele == 2:
        data = io.twophotonHeRead()

    # Error check
    if Znuc > len(data['psi0']):
        print('Nuclear charge outside data table range, using distribution for Z=%0d'%(len(data['psi0'])))
        Znuc = len(data['psi0'])

    # Interpolates data, []
    distr1 = splrep(data['y0'], data['psi0'][Znuc-1], s=0) 

    # Calcaultes photon distribution, # [1/AA]
    distr = y_grid**2/lamb0 *splev(y_grid, distr1)
    if nele == 1:
        distr /= data['asum'][Znuc-1]

    # Output, [AA], [1/AA], dim(nlambda,)
    return lamb0/y_grid, distr


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
    if '2-photon' in mechs:
        mechs.remove('2-photon')

    # Initializes output
    lams_profs_A = np.zeros(dshape[mechs[0]]['lams_profs_A'].shape) # [AA], dim(trs, nlamb)
    theta = np.zeros(dshape[mechs[0]]['lams_profs_A'].shape) # [1/AA], dim(trs,nlamb)

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