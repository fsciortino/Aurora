# -*- coding: utf-8 -*-

import jax.numpy as jnp
from jax.scipy import integrate
import jax
from functools import partial
from IPython import embed
import jax.numpy as jnp
from jax.scipy import integrate
from collections import namedtuple
#jax.config.update("jax_disable_jit", True)

# Define constants
mp = 1.672623e-27   # mass of the proton   [kg]
me = 9.1096e-31     # mass of the electron [kg]
qe = 1.60217653e-19 # elementary charge    [-]
eps0 = 8.8542e-12   # vacuum permittivity  [F/m]
eps_pi_fac = 3 * eps0**2 * (2 * jnp.pi / qe)**1.5 / qe  # common factor in collision times

# Define a named tuple to store the results
FACITResult = namedtuple('FACITResult', [
    'rho', 'Ti', 'Ni', 'Nimp', 'Machi', 'Mzstar', 'Zeff', 'Te', 'gradTi', 'gradNi', 'gradNimp', 'qmag',
    'R', 'Z', 'J', 'B', 'Dz_PS', 'Kz_PS', 'Hz_PS', 'Vrz_PS', 'Vconv_PS', 'Dz_BP', 'Kz_BP', 'Hz_BP',
    'Vrz_BP', 'Vconv_BP', 'Dz_CL', 'Kz_CL', 'Hz_CL', 'Vrz_CL', 'Vconv_CL', 'Dz', 'Kz', 'Hz', 'Vconv',
    'Vrz', 'Flux_z', 'horiz_asym', 'vert_asym', 'nn'
])

@partial(jax.jit, static_argnames=('nat_asym','full_geom', 'nth', 'rotation_model', 'fsaout'))
def FACIT(rho: (jnp.ndarray, float), \
        Zimp: (jnp.ndarray, float),
        Aimp: (int, float), \
        Zi: (jnp.ndarray, float),
        Ai: (int, float), \
        Ti: (jnp.ndarray, float), \
        Ni: (jnp.ndarray, float), 
        Nimp: (jnp.ndarray, float), \
        Machi: (jnp.ndarray, float), 
        Zeff: (jnp.ndarray, float), \
        gradTi: (jnp.ndarray, float), 
        gradNi: (jnp.ndarray, float),
        gradNimp: (jnp.ndarray, float), \
        invaspct: (float), 
        B0: (float),
        R0: (float),
        qmag: (jnp.ndarray, float),  \
        rotation_model = 0,
        Te_Ti = 1.0,\
        RV = None,
        ZV = None, 
        BV = None, 
        FV = None, 
        dpsidx = None,\
        fsaout = True, 
        full_geom = False, 
        nat_asym = True,\
        nth = 20, 
        regulopt = [1e-2,0.5,1e-5,1e2], \
        fH = 0.,
        bC = 1., 
        sigH = 1., 
        TperpTpar_axis = 1.):



    '''
    Calculation of collisional impurity transport coefficients
    ----------------------------------------------------------
    References:  Maget et al 2020 Plasma Phys. Control. Fusion 62 105001
                 Fajardo et al 2022 Plasma Phys. Control. Fusion 64 055017
                 Maget et al 2022 Plasma Phys. Control. Fusion 64 069501
                 Fajardo et al 2023 Plasma Phys. Control. Fusion 65 035021 
          
    License statement:
    Software name : FACIT
    Authors : P. Maget and D. Fajardo, C. Angioni, P. Manas
    Copyright holders : Commissariat Ã  lâ€™Energie Atomique et aux Energies Alternatives (CEA), France, 
                        Max-Planck Institut fÃ¼r Plasmaphysik, Germany
    CEA and IPP authorize the use of the FACIT software under the CeCILL-C open source license 
    https://cecill.info/licences/Licence_CeCILL-C_V1-en.html  
    The terms and conditions of the CeCILL-C license are deemed to be accepted upon downloading 
    the software and/or exercising any of the rights granted under the CeCILL-C license.
    
    Parameters
    ----------
    rho : 1D array (nr) or float
        mid-plane radius normalized to minor radius [-], :math:`r/a`
    Zimp : 1D array (nr) or float or int
        impurity charge [-]
    Aimp : int or float
        impurity mass [-]
    Zi : int or float
        main ion charge [-]
    Ai : int or float
        main ion mass [-]
    Ti : nD array (...,nr) or float
        main ion temperature [:math:`eV`]
    ne : nD array (...,nr) or float
        electron density [:math:`1/m^3`]
    Ni : nD array (...,nr) or float
        main ion density [:math:`1/m^3`]
    Nimp : nD array (...,nr) or float
        FSA impurity density [:math:`1/m^3`]
    Machi : nD array (...,nr) or float
        main ion Mach number [-]
    Zeff: nD array (...,nr) or float
        effective charge [-]
    gradTi : nD array (...,nr) or float
        main ion temperature gradient [:math:`1/m^3/m`]
    gradNi : nD array (...,nr) or float
        main ion density gradient [:math:`1/m^3/m`]
    gradNimp : nD array (...,nr) or float
        FSA impurity density gradient [:math:`1/m^3/m`]
    invaspct : float
        inverse aspect ratio [-], :math:`a/R0`
    B0 : float
        magnetic field at magnetic axis [:math:`T`]
    R0 : float
        major radius at magnetic axis [:math:`m`]
    qmag : 1D array (nr) or float
        safety factor [-]
    rotation_model : int, optional
        if 0, use non-rotating limit from Fajardo PPCF (2022)
        if 1, use PS model from Maget PPCf (2020)
        if 2, use BP and PS model from Fajardo (subm. to PPCF)
        Default is 0. 0 is recommended for low-Z impurities (e.g. B, N, Ne)
    Te_Ti : nD array (...,nr) or float, optional
        electron to ion temperature ratio [-]
        default is 1.0
    RV : 2D array (nr, nth), optional
        R contours of flux surfaces, 2D distribution on (r,theta) [:math:`m`]
        used if rotation_model == 2 or rotation_model == 1 and full_geom
        default is None, which calculates it in circular geometry
    ZV : 2D array (nr, nth), optional
        Z contours of flux surfaces, 2D distribution on (r,theta) [:math:`m/T`]
        used if rotation_model == 2 or rotation_model == 1 and full_geom
        default is None, which calculates it in circular geometry
    BV : 2D array (nr, nth), optional
        magnetic field, poloidal distribution on (r,theta) [:math:`T`]
        used if rotation_model == 1 and full_geom
        default is None, which calculates it in circular geometry
    FV : 1D array (nr), optional
        poloidal current flux function [:math:`T*m`]
        required if rotation_model == 1 and full_geom
        default is None, which makes it R0*B0
    dpsidx : 1D array (nr), optional
        radial coordinate transformation from flux psi to rho [:math:`T*m^2`]
        required if rotation_model == 1 and full_geom
        default is None
    fsaout : bool, optional
        if True, return FSA transport coefficients
        only relevant if rotation_model == 2
        default is True
    full_geom : bool, optional
        only relevant if rotation_model == 1
        if True, use full flux surface shaped geometry in the calculation of
        the asymmetries, which means an iterative solution. This would be the
        most computationally-expensive setting of FACIT.
        if False, use circular geometry with large aspect ratio, such that
        the calculation of the asymmetries can be completely analytical
        default is False
    nat_asym : bool, optional
        if True, consider friction-induced "natural" poloidal asymmetry
        only relevant if pol_asym
        default is True
    nth : int, optional
        number of poloidal grid points [-]
        if RV, BV, ZV are given, it is the size of axis 1,
        else default is 20
    regulopt : list (4), optional
        parameters for the convergence of the iterative calculation of the
        asymmetries
        only relevant if rotation_model = 1 and full_geom
        default is [1e-2,0.5,1.e-5,1e2]
    fH : float, optional
        resonant hydrogen minority fraction [-]
        for ICRH-induced asymmetries
        only relevant if pol_asym
        default is 0.0
    bC : float, optional
        Bres/B0 [-], where Bres is the field where the ICRH frequency matches the 
        fundamental cyclotron resonance of the minority ion: 2*pi*f = Zi*e*Bres/mi
        for ICRH-induced asymmetries
        only relevant if pol_asym
        default is 1.0
    sigH : float, optional
        std. dev. for radial exponential decay of temperature anisotropy
        for ICRH-induced asymmetries
        only relevant if pol_asym
        default is 1.0
    TperpTpar_axis : float, optional
        main ion temperature anisotropy at magnetic axis [-]
        for ICRH-induced asymmetries
        only relevant if pol_asym
        default is 1.0
        
    Returns (as attributes)
    -------
    Dz_* : 1D array (nr)
        diffusion coefficient for each flux component [:math:`m^2/s`]
        * = PS, BP or CL (for Pfirsch-SchlÃ¼ter, Banana-Plateau and Classical)
    Kz_* : 1D array (nr)
        coefficient of the main ion density gradient for each flux component [:math:`m^2/s`]
    Hz_* : 1D array (nr)
        coefficient of the main ion temperature gradient for each flux component [:math:`m^2/s`]
    Vrz_* : 1D array (nr)
        radial flux per particle (Flux_z/nz)_* for each flux component [:math:`m/s`]
    Vconv_* : 1D array (nr)
        radial convective velocity for each flux component [:math:`m/s`]
    Dz : 1D array (nr)
        total diffusion coefficient [:math:`m^2/s`]
    Kz : 1D array (nr)
        total coefficient of the main ion density gradient [:math:`m^2/s`]
    Hz : 1D array (nr)
        total coefficient of the main ion temperature gradient [:math:`m^2/s`]
    Vconv : 1D array (nr)
        total convective velocity [:math:`m/s`]
    Vrz : 1D array (nr)
        total radial flux per particle (Flux_z/nz) [:math:`m/s`]
    Flux_z : 1D array (nr)
        total radial flux [:math:`1/(m^2 s)`]
    horiz_asym : 1D array (nr)
        horizontal asymmetry of the impurity density [-]
        in the representation n = nz/<nz> = 1 + horiz_asym*cos(theta) + vert_asym*sin(theta)
    vert_asym : 1D array (nr)
        vertical asymmetry of the impurity density [-]
    nn : 2D array (nr, nth)
        poloidal distribution of the impurity density in (r,theta) [-]
        nn = nz(r,theta)/<nz(r,theta)>(r)
    '''
    
    
    #---------------------------------------------------------------------
    #-------------- Definition of physical quantities --------------------
    #---------------------------------------------------------------------
    
    mi = Ai * mp   # main ion mass [kg]
    mimp = Aimp * mp # impurity mass [kg]
    
    rho = jnp.maximum(1e-6, rho) # to avoid dividing by zero
    
    nr = rho.size # number of radial grid points
    
    eps = jnp.maximum(0.005, rho * invaspct)  # local inverse aspect ratio [-]
    eps2 = eps**2 # auxiliary
    
    amin = invaspct * R0 # minor radius [m]

    
    # logarithmic gradients: dlnQ/dr = (dQ/dr)/Q = dlnQ/d(amin*rho)
    grad_ln_Ni = gradNi / Ni     # [1/m]
    grad_ln_Ti = gradTi / Ti     # [1/m]
    grad_ln_Nimp = gradNimp / Nimp # [1/m]
    
    #this will force V on axis to be zero. 
    grad_ln_Ni = jnp.where(rho < 1e-6, 0, grad_ln_Ni)
    grad_ln_Ti = jnp.where(rho < 1e-6, 0, grad_ln_Ti)
    grad_ln_Nimp = jnp.where(rho < 1e-6, 0, grad_ln_Nimp)

    
    ft = ftrap(eps) # trapped particle fraction [-]
    
    # Coulomb logarithms, from NRL formulary
    LniiNRL = 23. - jnp.log(Zi**3 * jnp.sqrt(2 * (Ni / 1e6))) + 1.5 * jnp.log(Ti)
    LnimpiNRL = 23. - jnp.log(Zi * Zimp * jnp.sqrt((Ni / 1e6) * Zi**2 + (Nimp / 1e6) * Zimp**2)) + 1.5 * jnp.log(Ti)
    LnimpimpNRL = 23. - jnp.log(Zimp**3 * jnp.sqrt(2 * (Nimp / 1e6))) + 1.5 * jnp.log(Ti)
    
    # Braginskii collision times
    Tauii = (eps_pi_fac * jnp.sqrt(mi) * Ti**1.5) / (Zi**4 * Ni * LniiNRL)             # ion-ion collision time [s]
    Tauimpi = jnp.sqrt(Aimp / Ai) * ((Zi**2 * LniiNRL) / (Zimp**2 * LnimpiNRL)) * Tauii           # impurity-ion collision time [s]
    Tauiimp = ((Zi**2 * Ni * LniiNRL) / (Zimp**2 * Nimp * LnimpiNRL)) * Tauii                    # ion-impurity collision time [s]
    Tauimpimp = jnp.sqrt(Aimp / Ai) * ((Zi**4 * Ni * LniiNRL) / (Zimp**4 * Nimp * LnimpimpNRL)) * Tauii # impurity-impurity collision time [s]
    
    # collisionalities
    nuz = 1 / (jnp.sqrt(1 + Aimp / Ai) * Tauimpi)           # impurity collision frequency [1/s]
    g = (qmag * R0) / (jnp.sqrt(2 * (qe * Ti) / mi) * Tauii) # main collisionality parameter [-]
    nuistar = g / eps**1.5                                 # main ion collisionality [-]
    
    alpha = Zimp**2 * Nimp / (Zi**2 * Ni) # impurity strength parameter [-]
    
    # ion-electron heat exchange term (Fulop-Helander PoP 2001) [-]
    mu_ie = (96 * jnp.sqrt(2) / 125) * (1 / Zi**2) * jnp.sqrt(me / mi) * ((1 / Te_Ti)**1.5)
    
    if rotation_model == 0:
        Machi = Machi * 0.0
    
    Machi2 = Machi**2 # auxiliary
    
    # Effective impurity Mach number
    Mzstar = jnp.sqrt(Aimp / Ai - (Zimp / Zi) * Zeff / (Zeff + 1 / Te_Ti)) * Machi
    
    # fitted factors
    f1, f2, f3, fG, fU, yy, fv, fdps, fhbp = facs(Zimp, Aimp, ft, Mzstar, rotation_model)

    
    # new friction coefficient of the main ion parallel heat flow [-]
    C0z = C2(alpha, g, Ai, Aimp, f1, f2) / (1 + f3 * mu_ie * g**2)
    
    ki = ki_f(nuistar, ft, Zeff, Machi) # neoclassical main ion flow coefficient [-]
    
    wcz = Zimp * qe * B0 / mimp         # impurity cyclotron frequency [1/s]
    rhoLz2 = (2 * (qe * Ti) / mimp) / wcz**2 # impurity Larmor radius squared [m^2]
    
    # viscosity coefficients
    K11i, K12i, K11z, K12z = KVISC(Nimp, Ni, Ti, Ai, Aimp, Zi, Zimp, Tauii, Tauimpimp, Tauiimp, Tauimpi, eps, ft, R0, qmag, yy)
    
    #---------------------------------------------------------------------
    #------------------ Poloidal asymmetry terms -------------------------
    #---------------------------------------------------------------------
    
    if RV is not None:
        if nr == 1:
            RV = jnp.reshape(RV, (1, RV.size))
            JV = jnp.reshape(RV, (1, RV.size))
            BV = jnp.reshape(RV, (1, RV.size))
        nth = RV.shape[1]
    
    if FV is None:
        FV = R0 * B0 * jnp.ones(nr)
    
    if dpsidx is not None: # consistent q when full equilibrium is available
        if nr == 1:
            dpsidx = dpsidx * jnp.ones(nr)
        qmag = amin * eps * FV / dpsidx
    
    theta = jnp.linspace(0, 2 * jnp.pi, nth)
    
    if RV is None or ZV is None:
        # make circular geometry
        RV = R0 * (1 + eps[:, None] * jnp.cos(theta)[None, :])
        JV = B0 / (qmag[:, None] * RV)
    
    if RV is not None and ZV is not None:
        JV = Jacobian(RV, ZV, amin * rho, theta)
    
    if BV is None:
        BV = B0 / (1 + eps[:, None] * jnp.cos(theta)[None, :])
    
    if rotation_model == 0: # poloidally symmetric limit
        
        # geometric coefficient in flux equations
        CgeoG = 2 * eps2 * (0.96 * (1 - 0.54 * ft**4.5)) #CG0
        CgeoU = 0.0
        
        CclG = 1 + 2 * eps2
        
        deltan = jnp.zeros(nr)       # horizontal asymmetry of impurity density
        Deltan = jnp.zeros(nr)       # vertical asymmetry of impurity density
        nn = jnp.ones((nr, nth)) # poloidal distribution of the impurity density
        
        e0imp = 1.0
    
    elif rotation_model == 1:
        
        # common gradient terms
        UU = -(Zimp / Zi) * (C0z + ki) * (grad_ln_Ti * amin)
        GG = (grad_ln_Nimp * amin) - (Zimp / Zi) * (grad_ln_Ni * amin) + (1 + (Zimp / Zi) * (C0z - 1.0)) * (grad_ln_Ti * amin)
        
        # rotation and ICRH asymmetries input
        AsymPhi, AsymN = polasym_input(rho, eps, Zeff, Zi, Te_Ti, Machi2, fH, bC, TperpTpar_axis, sigH)
        
        if full_geom: # ...in full flux surface shaped geometry
            
            b2 = BV**2 / fluxavg(BV**2, JV)[:, None] # asymmetry of magnetic field: B^2/<B^2>
            
            deltan, Deltan, nn, b2sNNavg, NV = asymmetry_iterative(regulopt, nr, theta, GG, UU, Ai, Aimp, Zi, Zimp, Te_Ti, Machi2, R0, nuz, BV, RV, JV, FV, dpsidx, AsymPhi, AsymN, b2, nat_asym)
            
            b2snnavg = fluxavg(b2 / nn, JV) # <b^2/n>
            nnsb2avg = fluxavg(nn / b2, JV) # <n/b^2>
            
            # geometric coefficients in flux equations
            CgeoG = nnsb2avg - 1 / b2snnavg
            CgeoU = fluxavg(nn / NV, JV) - b2sNNavg / b2snnavg
            CclG = nnsb2avg
        
        else: # ... in circular geometry with large aspect ratio
            
            deltaM = 2 * (Aimp / Ai) * Machi2 * eps # rotation strength parameter [-]
            
            dminphia = Zimp * (Te_Ti) * AsymPhi[0] # horizontal asym. in ES potential
            dmajphia = Zimp * (Te_Ti) * AsymPhi[1] # vertical asym. in ES potential
            
            dNH = AsymN[0] # horizontal asym. in main ion density
            dNV = AsymN[1] # vertical asym. in main ion density
            
            deltan, Deltan, nn = asymmetry_analytical(rho, theta, GG, UU, eps, invaspct, qmag, nuz / wcz, deltaM, Ai, Aimp, Zi, Zimp, dNH, dNV, dminphia, dmajphia, nat_asym)
            
            dD2 = 0.5 * (deltan**2 + Deltan**2) # auxiliary
            
            # geometric coefficients in flux equations
            CgeoG = 2.0 * eps * deltan + dD2 + 2.0 * eps2
            CgeoU = -(eps * (dNH - deltan) - dD2 + 0.5 * (deltan * dNH + Deltan * dNV))
            CclG = 1.0 + eps * deltan + 2 * eps2
        
        e0imp = 1.0
    
    elif rotation_model == 2:
        
        CG0 = 2 * eps2 * (0.96 * (1 - 0.54 * ft**4.5))
        CgeoG = fG * CG0
        CgeoU = fU * CG0
        CclG = 1 + 2 * eps2
        
        if fsaout:
            e0imp = fluxavg(jnp.exp(Mzstar[..., None]**2 * ((RV**2 - (RV[:, 0]**2)[:, None]) / R0**2)), JV)
        else:
            e0imp = jnp.ones(nr)
        
        deltan = 1 / e0imp - 1                              # horizontal asymmetry of impurity density
        Deltan = jnp.zeros(nr)                             # vertical asymmetry of impurity density
        nn = 1 + deltan[..., None] * jnp.cos(theta) # poloidal distribution of the impurity density
    
    #---------------------------------------------------------------------
    #------------- Collisional transport coefficients --------------------
    #---------------------------------------------------------------------
    
    # Pfirsch-Schuter (PS)
    Dz_PS = fdps * qmag**2 * rhoLz2 * nuz * (CgeoG / (2 * eps2)) / e0imp # PS diffusion coefficient [m^2/s]
    Kz_PS = (Zimp / Zi) * Dz_PS # PS coefficient of the main ion density gradient [m^2/s]
    Hz_PS = (-(1.0 + (Zimp / Zi) * (C0z - 1.0)) + (CgeoU / CgeoG) * (Zimp / Zi) * (C0z + ki)) * Dz_PS # PS coefficient of the main ion temperature gradient [m^2/s]
    
    Vrz_PS = -Dz_PS * grad_ln_Nimp + Kz_PS * grad_ln_Ni + Hz_PS * grad_ln_Ti # PS radial flux per particle [m/s]
    Vconv_PS = Kz_PS * grad_ln_Ni + Hz_PS * grad_ln_Ti # PS convective velocity [m/s]
    
    # Banana-Plateau (BP)
    Dz_BP = (1.5 * (qe * Ti) / (Zimp**2 * qe**2 * FV**2 * Nimp)) * (1 / (1 / K11i + 1 / K11z)) / e0imp # BP diffusion coefficient [m^2/s]
    Kz_BP = (Zimp / Zi) * Dz_BP # BP coefficient of the main ion density gradient [m^2/s]
    Hz_BP = fhbp * ((Zimp / Zi) * (K12i / K11i - fv) - (K12z / K11z - fv)) * Dz_BP # BP coefficient of the main ion temperature gradient [m^2/s]
    
    Vrz_BP = -Dz_BP * grad_ln_Nimp + Kz_BP * grad_ln_Ni + Hz_BP * grad_ln_Ti  # BP radial flux per particle [m/s]
    Vconv_BP = Kz_BP * grad_ln_Ni + Hz_BP * grad_ln_Ti # BP convective velocity [m/s]
    
    # Classical
    Dz_CL = (CclG * 2 * eps2 / CgeoG) * Dz_PS / (2 * qmag**2) # CL diffusion coefficient [m^2/s]
    Kz_CL = (Zimp / Zi) * Dz_CL # CL coefficient of the main ion density gradient [m^2/s]
    Hz_CL = -(1.0 + (Zimp / Zi) * (C0z - 1.0)) * Dz_CL # CL coefficient of the main ion temperature gradient [m^2/s]
    
    Vrz_CL = -Dz_CL * grad_ln_Nimp + Kz_CL * grad_ln_Ni + Hz_CL * grad_ln_Ti # CL radial flux per particle [m/s]
    Vconv_CL = Kz_CL * grad_ln_Ni + Hz_CL * grad_ln_Ti # CL convective velocity [m/s]
    
    # Total
    Dz = Dz_PS + Dz_BP + Dz_CL # collisional diffusion coefficient [m^2/s]
    Kz = Kz_PS + Kz_BP + Kz_CL # coefficient of the main ion density gradient [m^2/s]
    Hz = Hz_PS + Hz_BP + Hz_CL # coefficient of the main ion temperature gradient [m^2/s]
    
    Vconv = Kz * grad_ln_Ni + Hz * grad_ln_Ti # convective velocity [m/s]
    Vrz = Vrz_PS + Vrz_BP + Vrz_CL   # radial flux per particle [m/s]
    
    Flux_z = Vrz * Nimp # radial collisional flux [1/(m s)]
    
    # Return the results as a named tuple
    return FACITResult(
        rho=rho, Ti=Ti, Ni=Ni, Nimp=Nimp, Machi=Machi, Mzstar=Mzstar, Zeff=Zeff, Te=Te_Ti * Ti,
        gradTi=gradTi, gradNi=gradNi, gradNimp=gradNimp, qmag=qmag, R=RV, Z=ZV, J=JV, B=BV,
        Dz_PS=Dz_PS, Kz_PS=Kz_PS, Hz_PS=Hz_PS, Vrz_PS=Vrz_PS, Vconv_PS=Vconv_PS,
        Dz_BP=Dz_BP, Kz_BP=Kz_BP, Hz_BP=Hz_BP, Vrz_BP=Vrz_BP, Vconv_BP=Vconv_BP,
        Dz_CL=Dz_CL, Kz_CL=Kz_CL, Hz_CL=Hz_CL, Vrz_CL=Vrz_CL, Vconv_CL=Vconv_CL,
        Dz=Dz, Kz=Kz, Hz=Hz, Vconv=Vconv, Vrz=Vrz, Flux_z=Flux_z,
        horiz_asym=deltan, vert_asym=Deltan, nn=nn
    )



def ftrap(eps):
    '''
    Trapped particle fraction as a function of the local inverse aspect ratio
    
    Parameters
    ----------
    eps : float or 1D array
        local inverse aspect ratio r/R0 [-]
    
    Returns
    -------
    ftrap : float or 1D array
        trapped particle fraction [-]
    '''
    return 1. - (1. - eps)**1.5 / (jnp.sqrt(1. + eps) * (1 + 1.46 * jnp.sqrt(eps)))

def C2(alpha, g, Ai, A, f1, f2):
    '''
    Function related to the friction coefficient of the main ion parallel
    heat flow in the impurity-main ion parallel heat flow.
    New parametrization of HSW result.
    
    Refs: Hirshman-Sigmar NF (1981), https://doi.org/10.1088/0029-5515/21/9/003
          Wenzel-Sigmar NF (1990),   https://doi.org/0029-5515/30/6/013
          Fajardo PPCF (2022), https://doi.org/10.1088/1361-6587/ac5b4d
    
    Parameters
    ----------
    alpha : float or 1D array
        impurity strength parameter, nz*Z**2/(ni*Zi**2) [-]
    g : float or 1D array
        collisionality parameter, ratio of main ion transit time to ion-ion
        collision time, q*R0/(vthi*Tauii) [-]
    Ai : int or float
        main ion mass [-]
    A  : int or float
        impurity mass [-]
    f1 : float or 1D array
        fitted factor [-]
        obtained from facs
    f2 : float or 1D array
        fitted factor [-]
        obtained from facs
    
    Returns
    -------
    C2 : float or 1D array
        value in C0z coefficient [-]
    '''
    return 1.5 / (1 + f1 * (Ai / A)) - (0.29 + 0.68 * alpha) / (0.59 + alpha + (1.34 + f2) * g**(-2))

def ki_f(nui_star, ft, Zeff, Machi):
    '''
    Analytical expression for the neoclassical main ion flow coefficient
    
    Ref: Fajardo (subm. to PPCF)
    
    Parameters
    ----------
    nui_star : float or 1D array
        main ion collisionality [-]
    ft : float or 1D array
        trapped particle fraction [-]
    Zeff : float or 1D array
        effective plasma charge [-]
    Machi: float or 1D array
        main ion Mach number [-]
    
    Returns
    -------
    ki : float or 1D array
        neoclassical main ion flow coefficient [-]
    '''
    c01 = 0.53 * (1 + 0.646 * Machi**1.5)
    c02 = 1.158 * (1 - 0.968 * Machi**1.56)
    c03 = -0.98 * (1 - 1.228 * Machi**1.7)
    
    l1k = 5.7 * (1 - ft)**6.7 + 0.38
    l2k = (-1.52 + 38.4 * (1 - ft)**3.02 * ft**2.07) + (-1 + 2.6 * ft) * Machi**(2.5 * (1 - 0.6 * ft))
    l3k = 0.25 + 1.2 * (1 - ft)**3.65
    l5k = 0.1 * (1 - ft)**1.46 * ft**4.33 + 0.051 * (1 - 0.82 * ft) * Machi**2.5
    l4k = 0.8 + (1.25 * ft + 0.585) * Machi
    l6k = (-0.05 + 1.95 * ft**2.5) / (1 + 2.55 * ft**17) + -((0.217 + 14.57 * ft**6.3) / (1 + 5.62 * ft**5.72)) * Machi**1.5
    
    ki0 = -(c01 + 0.055 * (Zeff - 1)) * (1 - ft) / ((0.53 + 0.17 * (Zeff - 1)) * \
          (1 - (c02 - 0.065 * (Zeff - 1)) * ft - c03 * ft**2))
    
    kiv = ((ki0 + l1k * Zeff * jnp.sqrt(ft * nui_star) + l2k * nui_star**0.25) / (1 + l3k * jnp.sqrt(nui_star)) - \
          l4k * l5k * nui_star**2 * ft**6 + l6k * nui_star**0.25) / (1 + l5k * nui_star**2 * ft**6)
    
    return kiv

def facs(Z, A, ft, Mzstar, rotation_model):
    '''
    Fitted factors for FACIT model
    
    Ref: Fajardo PPCF (2022), https://doi.org/10.1088/1361-6587/ac5b4d
         Fajardo (subm. to PPCF)
    
    Parameters
    ----------
    Z : float or 1D array
        impurity charge [-]
    A : float
        impurity mass number [-]
    ft : float or 1D array
        trapped particle fraction [-]
    Mzstar : float or 1D array
        effective impurity Mach number [-]
    rotation_model : int
        if 0, non-rotating limit
        if 1, PS model of Maget PPCF 2020
        if 2, PS model of Fajardo (subm. to PPCF)
    
    Returns
    -------
    f1 : float or 1D array
        factor in the C2 function [-]
    f2 : float or 1D array
        factor in the C2 function [-]
    f3 : float or 1D array
        factor of the ion-electron collisional heat exchange term in C0z [-]
    fG : float or 1D array
        factor of CgeoG [-]
    fU : float or 1D array
        factor of CgeoU [-]
    yy : list
        factors of the viscosity coefficients [-]
    fv : float or 1D array
        factor in expression for BP H coefficient [-]
    fdps : float or 1D array
        factor of the PS diffusion coefficient [-]
    fhbp : float or 1D array
        factor of the BP H coefficient [-]
    '''
    Z, Mzstar = jnp.broadcast_arrays(Z, Mzstar)
    
    f1 = (1.74 * (1 - 0.028 * A) + 10.25 / (1 + A / 3.0)**2.8) - 0.423 * (1 - 0.136 * A) * ft**(5 / 4)
    f2 = (88.28389935 + 10.50852772 * Z) / (1 + 0.2157175 * Z**2.57338463)
    f3 = (-4.45719179e+06 + 2.72813878e+06 * Z) / (1 + 5.26716920e+06 * Z**8.33610108e-01)
    
    if rotation_model == 2:
        f2 = f2 * jnp.exp(-10 * Mzstar**2)
        f3 = f3 * (1 + (1 + 1.86e6 * ft**11.07 * (1 - ft)**7.36) * Mzstar**4) * jnp.exp(-0.8 * Mzstar**2)
    
    fg1 = -1.4 * ft**7 + 2.23 * (1 - 0.31 * ft)
    fg2 = 2.8 * (1 - 0.63 * ft)
    fg3 = 3.5 * (1 - ft) / fg2
    fg4 = 4 * ft
    fg5 = 0.38 * ft**4
    fg6 = 3.95 * (1 + 0.424 * ft * (1 - 0.65 * ft))
    
    fG = (1 + fg1 * Mzstar**fg2)**fg3 * (1 + 0.2 * Mzstar**fg4) / (1 + fg5 * Mzstar**fg6)
    
    c1f = 2.72 * (1 - 0.91 * ft)
    c2f = 2.98 * (1 - 0.471 * ft)
    c3f = 0.95 * (1 - ft)**4
    c4f = 4 * ft
    c5f = 0.1314 * ft**2.84 + 3.178 * ft**11.4
    c6f = -9.38 * (ft - 0.5)**2 + 4.64
    
    fU = (c1f * Mzstar**c2f) * (1 + c3f * Mzstar**c4f) / (1 + c5f * Mzstar**c6f)
    
    y11zb = (8 * (1 - ft)**20 - 0.66 * ft**4.9 + 0.94) * ((1.085e4 + 9.3e3 * 14 / Z**(5 / 3)) / (1 + 14 / Z**(5 / 3))) * ft**4.6 * (1 - ft) / \
            (1 + 9.44e3 * (1 - 2e-3 * Z) * ft**4.16)
    y11zp = 1.0
    y11zps = 1.0
    
    y12zb = (1.8e3 + 7.54 * Z**1.8) * ft**(4.05 * (1 + 0.0039 * Z)) * (1 - ft)**(0.7 * (1 + 0.015 * Z)) / (1 + 1276 * (1 + 0.053 * Z) * ft**3.6)
    y12zp = 1.0
    y12zps = 1.0
    
    y11ib = ((5.91e-5 * Z + 0.812 + 0.806 / Z**0.44) + (0.013 * Z + 0.098 - 7.03 / Z**1.04) * ft + \
            (-0.047 * Z + 0.79 + 13.8 / Z**1.26) * ft**2 + (0.04 * Z - 0.575 - 9.64 / Z**1.31) * ft**3) * \
            (571.6 * (1 + 0.84 * Z + 7.8e-7 * Z**5.15) * ft**(3.43 * (1 + 0.012 * Z)) * \
            (1 - ft)**(1.2 * (1 + 1.6e-8 * Z**5.1)) / (1 + 500 * ft**(8 / 3)) + 1e-3)
    y11ip = 1.0
    y11ips = 1 / ((1 + (99 / 44**6) * Z**6))
    
    y12ib = (571.6 * (1 + 0.84 * Z + 7.8e-7 * Z**5.15) * ft**(3.43 * (1 + 0.012 * Z)) * \
            (1 - ft)**(1.2 * (1 + 1.6e-8 * Z**5.1)) / (1 + 500 * ft**(8 / 3)) + 1e-3)
    
    
    y12ip = 1.0
    y12ips = 1.0
    l7f = 1.0
    
    if rotation_model == 2:
        c11f = 1 + 14.86 * (1 - ft)**16.45 + 15.27 * ft**7.4
        c12f = 0.77 * (1 + 4.11 * ft)
        c13f = 0.01 * (1 + 359 * ft**2.5 + 1078 * ft**12)
        l1f = (1 + c11f * Mzstar**c12f) * jnp.exp(-c13f * Mzstar**2)
        
        c21f = 6.39 * (1 - ft)**15.8 + 0.1
        c22f = 0.943 * (1 + 3.5 * ft)
        l2f = (1 + c21f * Mzstar**0.5) / (1 + c22f * Mzstar**(10 / 3))
        
        l3f = 1 / (1 + 2 * ft * Mzstar)
        
        c1f = 6.13 + 28.18 * ft**2.13 + 336.25 * (1 - ft)**11.65
        c2f = 0.5 + 9.55 * ft**1.14 * (1 - ft)**1.42
        c3f = (0.0087 + 4.49 * ft**3.48) / (1 + 0.873 * ft**3.48)
        c4f = 3.6 * (1 - 0.36 * ft)
        
        l4f = (1 + c1f * Mzstar**c2f) / (1 + c3f * Mzstar**c4f)
        
        c1f = (1 - ft)**8
        c2f = 113.5 * ft**8.46
        c3f = 11 * (1 - ft)
        
        l5f = (1 + c1f * c2f * Mzstar**c3f) / (1 + c2f * Mzstar**c3f)
        
        l6f = (1 + 0.035 * 10 * Mzstar**4) / (1 + 10 * Mzstar**4)
        
        l7f = jnp.exp(-10 * Mzstar**2)
        
        # Update y11zb, y11zp, y11zps
        y11zb = y11zb * l1f
        y11zp = y11zp * (l1f / l2f)
        y11zps = y11zps * (l1f / (l2f * l3f))

        # Update y12zb, y12zp, y12zps
        y12zb = y12zb * l4f
        y12zp = y12zp * (l4f / l5f)
        y12zps = y12zps * (l4f / (l5f * l6f))

        # Update y11ib, y11ip, y11ips
        y11ib = y11ib * l1f
        y11ip = y11ip * l1f
        y11ips = y11ips * l1f

     

            
    
    yy = [[y11zb, y11zp, y11zps], [y12zb, y12zp, y12zps], \
         [y11ib, y11ip, y11ips], [y12ib, y12ip, y12ips, l7f]]
    
    if rotation_model == 2:
        fv = 1.5 * jnp.exp(-10 * Mzstar**2)
    else:
        fv = 1.5
    
    fdps = (0.711 + 2.08e-3 * Z**1.26) / (1 + 1.06e-11 * Z**5.78)
    
    if rotation_model == 2:
        fhbp = (0.135 + 2.647e-3 * Z**1.464 + 3.478e-10 * Z**5.347) * \
               ((1 + (3 / (1 + 1e-7 * Z**6)) * (1 / (1 + 1.2e5 * ft**12)) * Mzstar) / \
               (1 + (3 / (1 + 1e-7 * Z**6)) * (1.208 - 4.46 * ft + 4.394 * ft**2) * Mzstar**2))
    else:
        fhbp = (1.01579172e+00 + -1.78923911e-03 * Z) / (1 + 6.60170647e-13 * Z**6.66398825e+00)
    
    return f1, f2, f3, fG, fU, yy, fv, fdps, fhbp





def KVISC(nimp, ni, ti, Ai, Aimp, Zi, Zimp,
          Tauii, Tauimpimp, Tauiimp, Tauimpi,
          eps, ft, R0, qmag, yy):
    '''
    Calculates the viscosity coefficients for the BP flux analytically
    
    Refs: Hishman-Sigmar NF (1981),           https://doi.org/10.1088/0029-5515/21/9/003
          Wenzel-Sigmar NF (1990) Appendix A, https://doi.org/0029-5515/30/6/013
    
    Parameters
    ----------
    nimp : 1D array
        impurity density [1/m^3]
    ni : 1D array
        main ion density [1/m^3]
    ti : 1D array
        main ion temperature [eV]
    Ai : int or float
        main ion mass [-]
    Aimp : int or float
        impurity mass [-]
    Zi : int or float
        main ion charge [-]
    Zimp : int or float or 1D array
        impurity charge [-]
    Tauii : 1D array
        ion-ion collision time [s]
    Tauimpimp : 1D array
        impurity-impurity collision time [s]
    Tauiimp : 1D array
        ion-impurity collision time [s]
    Tauimpi : 1D array
        impurity-ion collision time [s]
    eps : 1D array
        local inverse aspect ratio [-]
    ft : 1D array
        trapped particle fraction [-]
    R0 : float
        major radius at magnetic axis [m]
    qmag : 1D array
        safety factor [-]
    yy : list
        fitted factors, obtained from facs [-]
    qe : float
        elementary charge [C]
    mp : float
        proton mass [kg]
    
    Returns
    -------
    K11i : 1D array
        (1,1) viscosity coefficient of the main ion [kg/(m s)]
    K12i : 1D array
        (1,2) viscosity coefficient of the main ion [kg/(m s)]
    K11z : 1D array
        (1,1) viscosity coefficient of the impurity [kg/(m s)]
    K12z : 1D array
        (1,2) viscosity coefficient of the impurity [kg/(m s)]
    '''
    
    # Transit frequencies
    wii = jnp.sqrt(2 * (qe * ti) / (Ai * mp)) / (qmag * R0)  # main ion transit frequency [1/s]
    wimpimp = jnp.sqrt(Ai / Aimp) * wii                      # impurity transit frequency [1/s]
    
    # Plateau regime
    fac_a_P = nimp * (qe * ti) * jnp.sqrt(jnp.pi) / (3 * wimpimp)
    fac_i_P = ni * (qe * ti) * jnp.sqrt(jnp.pi) / (3 * wii)
    
    K11aP = fac_a_P * 2
    K12aP = fac_a_P * 2 * 3
    
    K11iP = fac_i_P * 2
    K12iP = fac_i_P * 2 * 3
    
    # Pfirsch-Schluter regime
    r00 = 1 / jnp.sqrt(2)
    r01 = 1.5 / jnp.sqrt(2)
    r11 = 3.75 / jnp.sqrt(2)
    
    # Thermal velocity fractions
    xai = jnp.sqrt(Aimp / Ai)
    xia = 1 / xai
    
    x2ai = xai**2
    x2ia = xia**2
    
    xfac_ai = jnp.sqrt(1 + x2ai)
    xfac_ia = jnp.sqrt(1 + x2ia)
    
    qaa00 = qii00 = 8 / 2**1.5
    qaa01 = qii01 = 15 / 2**2.5
    qaa11 = qii11 = 132.5 / 2**3.5
    
    qai00 = (3 + 5 * x2ai) / xfac_ai**3
    qia00 = (3 + 5 * x2ia) / xfac_ia**3
    
    qai01 = 1.5 * (3 + 7 * x2ai) / xfac_ai**5
    qia01 = 1.5 * (3 + 7 * x2ia) / xfac_ia**5
    
    qai11 = (35 * x2ai**3 + 38.5 * x2ai**2 + 46.25 * x2ai + 12.75) / xfac_ai**7
    qia11 = (35 * x2ia**3 + 38.5 * x2ia**2 + 46.25 * x2ia + 12.75) / xfac_ia**7
    
    fac_qai_PS = (ni * Zi**2) / (nimp * Zimp**2)
    fac_qia_PS = 1 / fac_qai_PS
    
    qa00 = fac_qai_PS * qai00 + qaa00 - r00
    qi00 = fac_qia_PS * qia00 + qii00 - r00
    
    qa01 = fac_qai_PS * qai01 + qaa01 - r01
    qi01 = fac_qia_PS * qia01 + qii01 - r01
    
    qa11 = fac_qai_PS * qai11 + qaa11 - r11
    qi11 = fac_qia_PS * qia11 + qii11 - r11
    
    Qa = 0.4 * (qa00 * qa11 - qa01**2)
    Qi = 0.4 * (qi00 * qi11 - qi01**2)
    
    la11 = qa11 / Qa
    la12 = 3.5 * (qa11 + qa01) / Qa
    
    li11 = qi11 / Qi
    li12 = 3.5 * (qi11 + qi01) / Qi
    
    fac_imp_PS = nimp * (qe * ti) * Tauimpimp
    fac_ion_PS = ni * (qe * ti) * Tauii

    K11aPS = fac_imp_PS * la11
    K12aPS = fac_imp_PS * la12
    
    K11iPS = fac_ion_PS * li11
    K12iPS = fac_ion_PS * li12
    
    # Banana regime
    fac_B = (ft / (1 - ft)) * (2 * R0**2 * qmag**2 / (3 * eps**2))

    nuDai_int = (xfac_ai + x2ai * jnp.log(xai / (1 + xfac_ai))) / Tauimpi
    nuD2ai_int = 1 / (xfac_ai * Tauimpi)
    
    nuDia_int = (xfac_ia + x2ia * jnp.log(xia / (1 + xfac_ia))) / Tauiimp
    nuD2ia_int = 1 / (xfac_ia * Tauiimp)
    
    nuDaa_int = (jnp.sqrt(2) + jnp.log(1 / (1 + jnp.sqrt(2)))) / Tauimpimp
    nuD2aa_int = 1 / (jnp.sqrt(2) * Tauimpimp)
    
    nuDii_int = (jnp.sqrt(2) + jnp.log(1 / (1 + jnp.sqrt(2)))) / Tauii
    nuD2ii_int = 1 / (jnp.sqrt(2) * Tauii)
    
    K11aB = fac_B * nimp * (Aimp * mp) * (nuDai_int + nuDaa_int)
    K12aB = fac_B * nimp * (Aimp * mp) * (nuD2ai_int + nuD2aa_int)
    
    K11iB = fac_B * ni * (Ai * mp) * (nuDia_int + nuDii_int)
    K12iB = fac_B * ni * (Ai * mp) * (nuD2ia_int + nuD2ii_int)
    
    # Fitted factors
    y11zb, y11zp, y11zps = yy[0]
    y12zb, y12zp, y12zps = yy[1]
    y11ib, y11ip, y11ips = yy[2]
    y12ib, y12ip, y12ips, l7f = yy[3]  #underflow!!
    
    # Total: rational interpolation across collisionality regimes
    K11a = y11zb * K11aB / ((1 + y11zb * K11aB / (y11zp * K11aP)) * (1 + y11zp * K11aP / (y11zps * K11aPS)))
    K12a = y12zb * K12aB / ((1 + y12zb * K12aB / (y12zp * K12aP)) * (1 + y12zp * K12aP / (y12zps * K12aPS)))
    
    K11i = y11ib * K11iB / ((1 + y11ib * K11iB / (y11ip * K11iP)) * (1 + y11ip * K11iP / (y11ips * K11iPS)))
    K12i = l7f * y12ib * K12iB / ((1 + y12ib * K12iB / (y12ip * K12iP)) * (1 + y12ip * K12iP / (y12ips * K12iPS)))

    
    return K11i, K12i, K11a, K12a






def fluxavg(QV, JV):
    '''
    Calculates the flux surface average (FSA) of a quantity QV(r, theta) 
    in a coordinate system with a Jacobian JV(r,theta)
    
    Parameters
    ----------
    QV : 1D array (nth) or 2D array (nr, nth)
        quantity to take the FSA of
    JV : 1D array (nth) or 2D array (nr, nth)
        Jacobian of the coordinate system
    
    Returns
    -------
    Qavg : float or 1D array
        averaged QV, dimension of input QV is lowered by 1
    '''
    denom = integrate.trapezoid(JV, axis=-1)
    Qavg  = integrate.trapezoid(QV*JV, axis=-1)/denom
    return Qavg

def polasym_input(rho, eps, Zeff, Zi, Te_Ti, Machi2,
                  fH, bC, TperpTpar_axis, sigH):
    '''
    Poloidal asymmetries driven by rotation or ICRH-induced temperature anisotropies
    on the electrostatic potential and main ion density (N = ni/<ni>).
    Input to FACIT
    
    Ref: Maget PPCF (2020), https://doi.org/10.1088/1361-6587/aba7f9
    
    Parameters
    ----------
    rho : 1D array
        normalized radial coordinate [-], mid-plane radius over minor radius r/a
    eps : 1D, 2D array
        local inverse aspect ratio [-]
    Zeff : 1D, 2D array
        effective plasma charge [-]
    Zi : int or float
        main ion charge [-]
    Te_Ti : 1D, 2D array
        electron to main ion temperature ratio [-]
    Machi2 : 1D, 2D array
        main ion Mach number squared [-]
    fH : float
        resonant hydrogen minority fraction [-]
    bC : float
        Bres/B0 [-], where Bres is the field where the ICRH frequency matches the 
        fundamental cyclotron resonance of the minority ion: 2*pi*f = Zi*e*Bres/mi
    TperpTpar_axis : float
        main ion temperature anisotropy at magnetic axis [-]
    sigH : float
        [-]
    
    Returns
    -------
    AsymPhi : 2D array (2, nr)
        horizontal (index 0) and vertical (index 1) in electrostatic potential [-], phi/<phi>
    AsymN : 2D array (2, nr)
        horizontal (index 0) and vertical (index 1) in main ion density [-], ni/<ni>
    '''
    # Broadcast inputs to get output shape
    broadcasted = jnp.broadcast_arrays(Zeff, Te_Ti, Machi2)
    out_shape = broadcasted[0].shape
    
    # Initialize arrays with JAX-compatible operations
    AsymPhi = jnp.zeros((2,) + out_shape)
    AsymN = jnp.zeros((2,) + out_shape)
    
    # Calculate temperature anisotropy profile
    TperpTpar = (TperpTpar_axis - 1.) * jnp.exp(-(rho/sigH)**2) + 1.
    
    # Calculate horizontal asymmetry components
    phi0 = eps/(1. + Zeff*(Te_Ti)) * (
        fH*(TperpTpar - 1.)*bC/(bC + TperpTpar*(1. - bC)) + 2*Machi2
    )
    AsymPhi = AsymPhi.at[0].set(phi0)
    
    # Calculate density asymmetries
    n0 = -Zi*(Te_Ti)*phi0 + 2*eps*Machi2
    AsymN = AsymN.at[0].set(n0)
    
    # Vertical components remain zero as per original implementation
    # AsymPhi[1] remains 0, AsymN[1] gets calculated from zero-valued AsymPhi[1]
    AsymN = AsymN.at[1].set(-Zi*(Te_Ti)*AsymPhi[1])
    
    return AsymPhi, AsymN





@partial(jax.jit, static_argnames=('nat_asym',))
def asymmetry_analytical(rho, theta, GG, UU, eps, invaspct, qmag, nuswcz,
                         deltaM, Ai, Aimp, Zi, Zimp, dNH, dNV, dminphia, dmajphia, nat_asym):
    '''
    Calculate poloidal asymmetries of the impurity density self-consistently,
    including the friction-induced natural asymmetry and rotation and ICRH-
    induced asymmetries.
    This is done assuming a circular, large aspect ratio geometry, such that
    the solution is completely analytical.
    
    Refs: Maget PPCF (2020), https://doi.org/10.1088/1361-6587/ab53ab
          Maget PPCF (2020), https://doi.org/10.1088/1361-6587/aba7f9
    
    Parameters
    ----------
    rho : 1D array (nr)
        normalized radial coordinate [-], mid-plane radius over minor radius r/a
    theta : 1D array (nth)
        poloidal grid [-]
    GG : 1D array (nr)
        gradient term [-], see Maget PPCF (2020) (first one) eq. 10
    UU : 1D array (nr)
        gradient term [-], see Maget PPCF (2020) (first one) eq. 11
    eps : 1D array (nr)
        local inverse aspect ratio [-]
    invaspct: float
        inverse aspect ratio [-]
    qmag : 1D array (nr)
        safety factor [-]
    nuswcz : 1D array (nr)
        impurity collision frequency to impurity cyclotron frequency ratio [-]
    deltaM : 1D array (nr)
        impurity rotation strength [-], see text above Maget PPCF (2020) 
        (second one) eq. 15
    Ai : int or float
        main ion mass [-]
    Aimp : int or float
        impurity mass [-]
    Zi : int or float
        main ion charge [-]
    Zimp : int or float or 1D array (nr)
        impurity charge [-]
    dNH : 1D array (nr)
        horizontal asymmetry of the main ion density [-]
        from polasym_input method
    dNV : 1D array (nr)
        vertical asymmetry of the main ion density [-]
    dminphia : 1D array (nr)
        horizontal asymmetry of the electrostatic potential [-]
        see Maget PPCF (2020) (second one), eq. 13 and following text
    dmajphia : 1D array (nr)
        vertical asymmetry of the electrostatic potential [-]
    nat_asym : bool
        if True, include friction-induced natural poloidal asymmetry
    
    Returns
    -------
    deltan : 1D array (nr)
        horizontal asymmetry of the impurity density [-]
        in the representation n = nz/<nz> = 1 + deltan*cos(theta) + Deltan*sin(theta)
    Deltan : 1D array (nr)
        vertical asymmetry of the impurity density [-]
    nn : 2D array (nr, nth)
        poloidal distribution of the impurity density in (r,theta) [-]
        nn = nz(r,theta)/<nz(r,theta)>(r)
    '''
    
    UG = 1 + UU/GG
    
    # Conditionally compute Ae based on nat_asym (static argument)
    Ae = jax.lax.cond(
        nat_asym,
        lambda: nuswcz * qmag**2 / invaspct,
        lambda: jnp.zeros_like(nuswcz)
    )

    AGe = Ae * GG
    HH = 1.0
    CD0 = -eps / UG
    QQ = CD0 * (dNV / eps) * (UG - 1.0)
    FF = CD0 * (1 - 0.5 * dNH * (UG - 1.0) / eps)
    KK = 1.0
    
    CD = FF - 0.5 * (dminphia - deltaM)
    CDV = -0.5 * (dmajphia + QQ)
    RD = jnp.sqrt((FF + 0.5 * (dminphia - deltaM))**2 + 0.25 * (dmajphia - QQ)**2)
    DD = RD**2 + AGe**2 * (RD / CD0)**2
    
    num = ((AGe / CD0)**2 - 1) * (FF / CD0 + 0.5 * (dminphia - deltaM) / CD0) + \
          (AGe / CD0) * (0.5 * dNV * (UG - 1.0) / eps - 0.5 * dmajphia / CD0)
    cosa = RD * CD0 * num / DD
    
    num = 2 * AGe * (FF / CD0 + 0.5 * (dminphia - deltaM) / CD0) + \
          ((AGe / CD0)**2 - 1) * (0.5 * dmajphia - 0.5 * dNV * CD0 * (UG - 1.0) / eps)
    sina = RD * num / DD
    
    deltan = CD + RD * cosa
    Deltan = CDV + RD * jnp.sqrt(KK / HH) * sina
    
    nn = 1 + jnp.outer(deltan, jnp.cos(theta)) + jnp.outer(Deltan, jnp.sin(theta))
    
    return deltan, Deltan, nn




def Jacobian(R, Z, r, theta):
    '''
    Calculates the Jacobian of the coordinate transformation (r,theta) <--> (R,Z)
    Based on GKW manual, eqs (2.104-2.107)

    Parameters
    ----------
    R : 2D array (nr, nth)
        major radius contours of flux surfaces [m]
    Z : 2D array (nr, nth)
        vertical coordinate contours of flux surfaces [m]
    r : 1D array
        radial coordinate
    theta : 1D array
        poloidal angle coordinate

    Returns
    -------
    J : 2D array (nr, nth)
        Jacobian of coordinate transformation
    '''
    
    # Compute partial derivatives using JAX's gradient
    dRdr = jnp.gradient(R, r, axis=0)
    dRdth = jnp.gradient(R, theta, axis=1)
    dZdr = jnp.gradient(Z, r, axis=0)
    dZdth = jnp.gradient(Z, theta, axis=1)

    # Calculate metric tensor components
    grr = dRdr**2 + dZdr**2
    grth = dRdr*dRdth + dZdr*dZdth
    gthth = dRdth**2 + dZdth**2

    # Compute Jacobian determinant
    return R * jnp.sqrt(grr*gthth - grth**2)






def fluxavg(QV, JV):
    """JAX-compatible flux surface average"""
    denom = integrate.trapezoid(JV, axis=-1)
    Qavg = integrate.trapezoid(QV * JV, axis=-1) / denom
    return Qavg


@partial(jax.jit, static_argnames=('nr', 'nat_asym'))
def asymmetry_iterative(regulopt, nr, theta, GG, UU, Ai, Aimp, Zi, Zimp, Te_Ti,
                       Machi2, R0, nuz, BV, RV, JV, FV, dpsidx, AsymPhi, AsymN,
                       b2, nat_asym):
    # Unpack regularization parameters
    err, prog, regulweight, ierrmax = regulopt
    
    # Initialize theta grid
    theta = jnp.linspace(0, 2*jnp.pi, theta.size + 1)[:-1]
    
    # Calculate rotation factor
    Factrot0 = (Aimp/Ai) * Machi2 / R0**2

    # Calculate Apsi with JAX conditional
    def true_fun(args):
        nuz, JV, FV, Aimp, mp, Zimp, qe, dpsidx = args
        return JV * FV[:,None] * (Aimp * mp) * nuz[:,None] / \
               (Zimp[:,None] * qe * (dpsidx[:,None]**2 + 1e-33))

    def false_fun(args):
        return jnp.zeros_like(JV)
    
    Apsi = jax.lax.cond(
        nat_asym,
        true_fun,
        false_fun,
        (nuz, JV, FV, Aimp, mp, Zimp, qe, dpsidx)
    )

    # Calculate potential and density perturbations
    PhiV = jnp.outer(AsymPhi[0], jnp.cos(theta)) + jnp.outer(AsymPhi[1], jnp.sin(theta))
    NV = 1 + jnp.outer(AsymN[0], jnp.cos(theta)) + jnp.outer(AsymN[1], jnp.sin(theta))
    
    # Calculate flux averages
    b2sNNavg = fluxavg(b2/NV, JV)
    
    # Initialize impurity density array
    nn = jnp.ones_like(BV)
    dtheta = theta[1] - theta[0]

    # Radial loop using JAX scan
    def radial_step(carry, ix):
        nn, asym_err = carry
        
        # Extract current radial slice
        bv_ix = BV[ix]
        rv_ix = RV[ix]
        jv_ix = JV[ix]
        fv_ix = FV[ix]
        dpsidx_ix = dpsidx[ix]
        gg_ix = GG[ix]
        uu_ix = UU[ix]
        te_ti_ix = Te_Ti[ix]
        zimp_ix = Zimp[ix]
        machi2_ix = Machi2[ix]
        factrot0_ix = Factrot0[ix]
        phiv_ix = PhiV[ix]
        nv_ix = NV[ix]
        b2_ix = b2[ix]
        apsi_ix = Apsi[ix]

        # Iterative solver state
        def while_cond(state):
            _, _, _, erreur, ierr = state
            return (erreur > err) & (ierr < ierrmax)

        def while_body(state):
            nnx, nnp, progx, erreur, ierr = state
            
            # Calculate flux average
            b2snavg = fluxavg(b2_ix/nnx, jv_ix)

            # Calculate coefficients
            FFF = apsi_ix * (gg_ix + (b2_ix/nv_ix)*uu_ix)
            GGG = -zimp_ix*te_ti_ix*(phiv_ix - phiv_ix[0]) + \
                   factrot0_ix*(rv_ix**2 - rv_ix[0]**2)
            HHH = apsi_ix * (b2_ix/b2snavg) * (gg_ix + b2sNNavg[ix]*uu_ix)

            # Matrix construction
            AA = jnp.zeros((theta.size+1, theta.size+1))
            LL = jnp.zeros((theta.size+1, theta.size+1))
            BB = jnp.zeros((theta.size+1, 1))

            # Middle rows
            def update_middle(i, mats):
                AA, LL, BB = mats
                AA = AA.at[i,i-1].set(-0.5/dtheta)
                AA = AA.at[i,i].set(-FFF[i] - (0.5/dtheta)*(GGG[i+1]-GGG[i-1]))
                AA = AA.at[i,i+1].set(0.5/dtheta)
                LL = LL.at[i,i-1].set(1.0)
                LL = LL.at[i,i].set(-2.0)
                LL = LL.at[i,i+1].set(1.0)
                BB = BB.at[i,0].set(-HHH[i])
                return (AA, LL, BB)

            AA, LL, BB = jax.lax.fori_loop(1, theta.size-1, update_middle, (AA, LL, BB))

            # Boundary rows
            AA = AA.at[0,0].set(-FFF[0] - (0.5/dtheta)*(GGG[1]-GGG[-2]))
            AA = AA.at[0,1].set(0.5/dtheta)
            AA = AA.at[0,-2].set(-0.5/dtheta)
            LL = LL.at[0,0].set(-2.0)
            LL = LL.at[0,1].set(1.0)
            LL = LL.at[0,-2].set(1.0)
            BB = BB.at[0,0].set(-HHH[0])

            AA = AA.at[-2,-3].set(-0.5/dtheta)
            AA = AA.at[-2,-2].set(-FFF[-1] - (0.5/dtheta)*(GGG[0]-GGG[-2]))
            AA = AA.at[-2,-1].set(0.5/dtheta)
            LL = LL.at[-2,-3].set(1.0)
            LL = LL.at[-2,-2].set(-2.0)
            LL = LL.at[-2,-1].set(1.0)
            BB = BB.at[-2,0].set(-HHH[-1])

            AA = AA.at[-1,-1].set(-1.0)
            AA = AA.at[-1,0].set(1.0)
            BB = BB.at[-1,0].set(0.0)

            # Solve linear system
            CC = AA.T @ AA + regulweight * LL.T @ LL
            nnya = jnp.linalg.inv(CC) @ AA.T @ BB
            nny = nnya[:-1,0] / fluxavg(nnya[:-1,0], jv_ix)
            nny = jnp.clip(nny, 1e-5, None)

            # Update state
            new_nnx = progx*nnp + (1-progx)*nny
            new_err = jnp.max(jnp.abs(jnp.gradient(jnp.log(new_nnx)-GGG, theta) - FFF + HHH/new_nnx))
            
            return (new_nnx, nnx, progx, new_err, ierr+1)

        # Initial state
        init_state = (jnp.ones_like(theta), jnp.ones_like(theta), prog, jnp.inf, 0)
        
        # Run iterations
        final_state = jax.lax.while_loop(while_cond, while_body, init_state)
        
        # Update results
        nnx, _, _, erreur, _ = final_state
        nn = nn.at[ix].set(nnx)
        asym_err = asym_err.at[ix].set(erreur)
        
        return (nn, asym_err), None

    # Execute radial scan
    (nn, asym_err), _ = jax.lax.scan(radial_step, (nn, jnp.zeros(nr)), jnp.arange(nr))

    # Calculate final asymmetries
    deltan = 2 * jnp.mean((nn-1) * jnp.cos(theta), axis=1)
    Deltan = 2 * jnp.mean((nn-1) * jnp.sin(theta), axis=1)

    return deltan, Deltan, nn, b2sNNavg, NV
















if __name__ == '__main__':
   

    import matplotlib.pyplot as plt
    import numpy as np
    # simple example
    
    rotation_model = 1 
    
    rho = np.linspace(0,1,150)
    invaspct = 0.5/1.65
    R0 = 1.65
    B0 = 2.5
    
    Zimp = 35*(1-rho**2) + 15
    Aimp = 184
    
    Zi = 1
    Ai = 2
    
    Ti = 4000*(1 - rho**2)**2 + 100
    Ni = 5e19*(1 - rho**2) + 5e18
    
    Nimp = 1e-7*Ni
    
    Machi = 0.35*(1 - rho**2) + 0.05
    
    
    Zeff = 1.5*np.ones_like(rho)
    
    gradTi = np.gradient(Ti, rho*invaspct*R0)
    gradNi = np.gradient(Ni, rho*invaspct*R0)
    gradNimp = 1e-7*gradNi
    
    qmag = 1.0 + 2.5*rho**2
    
    # circular geometry
    RV = None
    ZV = None
   
    #200us, 5x speed-up compared to original python 
    fct = FACIT(rho, 
                Zimp, Aimp, \
                Zi, Ai, \
                Ti, Ni, Nimp, Machi, Zeff, \
                gradTi, gradNi, gradNimp, \
                invaspct, B0, R0, qmag,  \
                rotation_model = rotation_model, Te_Ti = 1.0,\
                RV = RV, ZV = ZV)

        
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
    
    ax1.plot(rho, fct.Dz_CL, label = 'CL component')
    ax1.plot(rho, fct.Dz_BP, label = 'BP component')
    ax1.plot(rho, fct.Dz_PS, label = 'PS component')
    ax1.plot(rho, fct.Dz, label = ' Total')
    
    ax1.legend(frameon=False)
    ax1.set_xlabel(r'$\rho = r/a$')
    ax1.set_ylabel('Diffusion coefficient [m$^2$/s]')
    ax1.tick_params(which = 'both', direction = 'in', axis = 'both', top = True, right = True)
    
    
    ax2.plot(rho, fct.Vconv_CL, label = 'CL component')
    ax2.plot(rho, fct.Vconv_BP, label = 'BP component')
    ax2.plot(rho, fct.Vconv_PS, label = 'PS component')
    ax2.plot(rho, fct.Vconv, label = ' Total')
    ax2.tick_params(which = 'both', direction = 'in', axis = 'both', top = True, right = True)
    
    #ax2.legend(frameon=False)
    ax2.set_xlabel(r'$\rho = r/a$')
    ax2.set_ylabel('Convective velocity [m/s]')
    
    fig.tight_layout()
    plt.show()
    
