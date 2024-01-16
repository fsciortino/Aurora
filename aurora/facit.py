#-----------------------------------------------------------------------------
# Python standalone of FACIT routine for collisional impurity transport
# Daniel Fajardo (daniel.fajardo@ipp.mpg.de), January 2023
#-----------------------------------------------------------------------------

import numpy as np


class FACIT:
    '''
    Calculation of collisional impurity transport coefficients
    ----------------------------------------------------------
    References:  Maget et al 2020 Plasma Phys. Control. Fusion 62 105001
                 Fajardo et al 2022 Plasma Phys. Control. Fusion 64 055017
                 Maget et al 2022 Plasma Phys. Control. Fusion 64 069501
                 Fajardo et al 2023 Plasma Phys. Control. Fusion 65 
          
    License statement:
    Software name : FACIT
    Authors : P. Maget and D. Fajardo, C. Angioni, P. Manas
    Copyright holders : Commissariat à l’Energie Atomique et aux Energies Alternatives (CEA), France, 
                        Max-Planck Institut für Plasmaphysik, Germany
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
    gradNi : nD array (...,nr) or float
        main ion density gradient [:math:`1/m^3/m`]
    gradTi : nD array (...,nr) or float
        main ion temperature gradient [:math:`1/m^3/m`]
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
        * = PS, BP or CL (for Pfirsch-Schlüter, Banana-Plateau and Classical)
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
    
    # some constants
    mp   = 1.672623e-27   # mass of the proton   [kg]
    me   = 9.1096e-31     # mass of the electron [kg]
    qe   = 1.60217653e-19 # elementary charge    [-]
    eps0 = 8.8542e-12     # vacuum permittivity  [F/m]
    eps_pi_fac = 3*eps0**2*(2*np.pi/qe)**1.5/qe # common factor in collision times
    
    
    def __init__(self, rho: (np.ndarray, float), \
                       Zimp: (np.ndarray, float), Aimp: (int, float), \
                       Zi: (np.ndarray, float), Ai: (int, float), \
                       Ti: (np.ndarray, float), Ni: (np.ndarray, float), Nimp: (np.ndarray, float), \
                       Machi: (np.ndarray, float), Zeff: (np.ndarray, float), \
                       gradTi: (np.ndarray, float), gradNi: (np.ndarray, float), gradNimp: (np.ndarray, float), \
                       invaspct: (float), B0: (float), R0: (float), qmag: (np.ndarray, float),  \
                       
                       rotation_model = 0, Te_Ti = 1.0,\
                       RV = None, ZV = None, BV = None, FV = None, dpsidx = None,\
                       fsaout = True, full_geom = False, nat_asym = True,\
                       nth = 20, regulopt = [1e-2,0.5,1e-5,1e2], \
                       fH = 0., bC = 1., sigH = 1., TperpTpar_axis = 1.):
        
        #---------------------------------------------------------------------
        #-------------- Definition of physical quantities --------------------
        #---------------------------------------------------------------------
        
        mi   = Ai*self.mp   # main ion mass [kg]
        mimp = Aimp*self.mp # impurity mass [kg]
        
        rho = np.maximum(1e-6, rho) # to avoid dividing by zero
        
        if type(rho) is float or type(rho) is np.float64:
            nr  = 1 # number of radial grid points
            
            rho      = np.atleast_1d(rho)
            Ti       = np.atleast_1d(Ti)
            Ni       = np.atleast_1d(Ni)
            Nimp     = np.atleast_1d(Nimp)
            Machi    = np.atleast_1d(Machi)
            Zeff     = np.atleast_1d(Zeff)
            gradTi   = np.atleast_1d(gradTi)
            gradNi   = np.atleast_1d(gradNi)
            gradNimp = np.atleast_1d(gradNimp)
            qmag     = np.atleast_1d(qmag)
            
        else:
            nr  = rho.size # number of radial grid points
        
        eps   = np.maximum(0.005, rho*invaspct)  # local inverse aspect ratio [-]
        eps2  = eps**2 # auxiliary
        
        
        amin = invaspct*R0 # minor radius [m]
        
        if type(Zimp) is int or type(Zimp) is float or type(Zimp) is np.float64:
            Zimp  = np.atleast_1d(Zimp)
        if type(Te_Ti) is int or type(Te_Ti) is float or type(Te_Ti) is np.float64:
            Te_Ti = np.atleast_1d(Te_Ti)
        
        if Nimp.mean() < 1:
            Nimp = np.ones_like(Ni)
        
        # logarithmic gradients: dlnQ/dr = (dQ/dr)/Q = dlnQ/d(amin*rho)
        grad_ln_Ni   = gradNi/Ni     # [1/m]
        grad_ln_Ti   = gradTi/Ti     # [1/m]
        grad_ln_Nimp = gradNimp/Nimp # [1/m]
        
        if abs(rho[0]) <= 1e-6:
            grad_ln_Ni[...,0]   = 0.0 # [1/m]
            grad_ln_Ti[...,0]   = 0.0 # [1/m]
            grad_ln_Nimp[...,0] = 0.0 # [1/m]
        
        
        ft = self.ftrap(eps) # trapped particle fraction [-]
        
        # Coulomb logarithms, from NRL formulary
        LniiNRL     = 23. - np.log(Zi**3*np.sqrt(2*(Ni/1e6))) + 1.5*np.log(Ti)
        LnimpiNRL   = 23. - np.log(Zi*Zimp*np.sqrt((Ni/1e6)*Zi**2+(Nimp/1e6)*Zimp**2)) + 1.5*np.log(Ti)
        LnimpimpNRL = 23. - np.log(Zimp**3*np.sqrt(2*(Nimp/1e6))) + 1.5*np.log(Ti)
        
        # Braginskii collision times
        
        Tauii     = (self.eps_pi_fac*np.sqrt(mi)*Ti**1.5)/(Zi**4*Ni*LniiNRL)             # ion-ion collision time [s]
        # set other collision times with respect to Tauii:
        Tauimpi   = np.sqrt(Aimp/Ai)*((Zi**2*LniiNRL)/(Zimp**2*LnimpiNRL))*Tauii           # impurity-ion collision time [s]
        Tauiimp   = ((Zi**2*Ni*LniiNRL)/(Zimp**2*Nimp*LnimpiNRL))*Tauii                    # ion-impurity collision time [s]
        Tauimpimp = np.sqrt(Aimp/Ai)*((Zi**4*Ni*LniiNRL)/(Zimp**4*Nimp*LnimpimpNRL))*Tauii # impurity-impurity collision time [s]
        
        # collisionalities
        nuz     = 1/(np.sqrt(1 + Aimp/Ai)*Tauimpi)           # impurity collision frequency [1/s]
        g       = (qmag*R0)/(np.sqrt(2*(self.qe*Ti)/mi)*Tauii) # main collisionality parameter [-]
        nuistar = g/eps**1.5                                 # main ion collisionality [-]
        
        
        alpha = Zimp**2*Nimp/(Zi**2*Ni) # impurity strength parameter [-]
        
        # ion-electron heat exchange term (Fülöp-Helander PoP 2001) [-]
        mu_ie = (96*np.sqrt(2)/125)*(1/Zi**2)*np.sqrt(self.me/mi)*((1/Te_Ti)**1.5)
        
        
        if rotation_model == 0:
            Machi = np.zeros_like(Machi)
            
        Machi2 = Machi**2 # auxiliary
        
        # Effective impurity Mach number
        Mzstar = np.sqrt(Aimp/Ai - (Zimp/Zi)*Zeff/(Zeff + 1/Te_Ti))*Machi
        
        
        # fitted factors
        f1, f2, f3, fG, fU, yy, fv, fdps, fhbp = self.facs(Zimp, Aimp, ft, Mzstar, rotation_model)
        
        # new friction coefficient of the main ion parallel heat flow [-]
        C0z = self.C2(alpha, g, Ai, Aimp, f1, f2)/(1 + f3*mu_ie*g**2)
        
        ki = self.ki_f(nuistar, ft, Zeff, Machi) # neoclassical main ion flow coefficient [-]
        
        
        wcz = Zimp*self.qe*B0/mimp         # impurity cyclotron frequency [1/s]
        rhoLz2 = (2*(self.qe*Ti)/mimp)/wcz**2 # impurity Larmor radius squared [m^2]
        
        
        
        # viscosity coefficients
        K11i, K12i, K11z, K12z = self.KVISC(Nimp, Ni, Ti, Ai, Aimp, Zi, Zimp, \
                                            Tauii, Tauimpimp, Tauiimp, Tauimpi, \
                                            eps, ft, R0, qmag, yy)
        
        
        
        #---------------------------------------------------------------------
        #------------------ Poloidal asymmetry terms -------------------------
        #---------------------------------------------------------------------
        
        if RV is not None:
            RV = np.atleast_2d(RV)
            BV = np.atleast_2d(BV)
            nth = RV.shape[1]
            
        if FV is None:
            FV = np.atleast_1d(R0*B0)
            
        if dpsidx is not None: # consistent q when full equilibrium is available
            dpsidx = np.atleast_1d(dpsidx)
            qmag = amin*eps*FV/dpsidx
        
        theta = np.linspace(0,2*np.pi, nth)
        
        if RV is None or ZV is None:
            # make circular geometry
            RV = R0*(1 + eps[:,None]*np.cos(theta)[None,:])
            JV = B0/(qmag[:,None]*RV)
                
        else:
            JV = self.Jacobian(RV, ZV, amin*rho, theta)
            
            
        if BV is None:
            BV = B0/(1 + eps[:,None]*np.cos(theta)[None,:])
            
        
        if rotation_model == 0: # poloidally symmetric limit
            
            # geometric coefficient in flux equations
            
            CgeoG = 2*eps2*(0.96*(1 - 0.54*ft**4.5)) #CG0
            CgeoU = 0.
            
            CclG = 1 + 2*eps2
            
            deltan = np.zeros(nr)       # horizontal asymmetry of impurity density
            Deltan = np.zeros(nr)       # vertical asymmetry of impurity density
            nn     = np.ones((nr, nth)) # poloidal distribution of the impurity density
            
            
            e0imp = 1.0
            
            
        elif rotation_model == 1:
            
            # common gradient terms
            UU = -(Zimp/Zi)*(C0z + ki)*(grad_ln_Ti*amin)
            GG = (grad_ln_Nimp*amin) - (Zimp/Zi)*(grad_ln_Ni*amin) +\
                 (1 + (Zimp/Zi)*(C0z - 1.0))*(grad_ln_Ti*amin)
                 
            # rotation and ICRH asymmetries input
            AsymPhi, AsymN = self.polasym_input(rho, eps, Zeff, Zi, Te_Ti, Machi2, \
                                                fH, bC, TperpTpar_axis, sigH)
            
            
            if full_geom: # ...in full flux surface shaped geometry
            
                b2        = BV**2/self.fluxavg(BV**2, JV)[:,None] # asymmetry of magnetic field: B^2/<B^2>
        
                deltan, Deltan, nn, b2sNNavg, NV = self.asymmetry_iterative(regulopt, nr, theta, GG, UU, \
                                                                            Ai, Aimp, Zi, Zimp, Te_Ti, \
                                                                            Machi2, R0, nuz, BV, RV, JV, FV, dpsidx, \
                                                                            AsymPhi, AsymN, b2, nat_asym)
                                                              

                b2snnavg = self.fluxavg(b2/nn, JV) # <b^2/n>
                nnsb2avg = self.fluxavg(nn/b2, JV) # <n/b^2>
                
                # geometric coefficients in flux equations
                
                CgeoG = nnsb2avg - 1/b2snnavg
                CgeoU = self.fluxavg(nn/NV, JV) - b2sNNavg/b2snnavg
                CclG  = nnsb2avg
                
            else: # ... in circular geometry with large aspect ratio
                
                deltaM = 2*(Aimp/Ai)*Machi2*eps # rotation strength parameter [-]
                
                dminphia = Zimp*(Te_Ti)*AsymPhi[0] # horizontal asym. in ES potential
                dmajphia = Zimp*(Te_Ti)*AsymPhi[1] # vertical asym. in ES potential
                
                dNH =  AsymN[0] # horizontal asym. in main ion density
                dNV =  AsymN[1] # vertical asym. in main ion density
                
                deltan, Deltan, nn = self.asymmetry_analytical(rho, theta, GG, UU, eps, \
                                                               invaspct, qmag, nuz/wcz, \
                                                               deltaM, Ai, Aimp, Zi, Zimp,\
                                                               dNH, dNV, dminphia, dmajphia, nat_asym)
                
                
                dD2 = 0.5*(deltan**2 + Deltan**2) # auxiliary

                # geometric coefficients in flux equations
            
                CgeoG = 2.0*eps*deltan + dD2 + 2.0*eps2
                CgeoU = -(eps*(dNH - deltan) - dD2 + 0.5*(deltan*dNH + Deltan*dNV))
                CclG = 1.0 + eps*deltan + 2*eps2
                
            e0imp = 1.0
        
            
            
        elif rotation_model == 2:
            
            CG0 = 2*eps2*(0.96*(1 - 0.54*ft**4.5))
            CgeoG = fG*CG0
            CgeoU = fU*CG0
            CclG = 1 + 2*eps2
            
            if fsaout:
                e0imp = self.fluxavg(np.exp(Mzstar[...,None]**2*((RV**2 - (RV[:,0]**2)[:,None])/R0**2)), JV)
            else:
                e0imp = np.ones(nr)
            
            
            
            deltan = 1/e0imp - 1                              # horizontal asymmetry of impurity density
            Deltan = np.zeros(nr)                             # vertical asymmetry of impurity density
            nn     = 1 + deltan[...,None]*np.cos(theta) # poloidal distribution of the impurity density
        
        #---------------------------------------------------------------------
        #------------- Collisional transport coefficients --------------------
        #---------------------------------------------------------------------
        
        # inputs
        self.rho = rho
        self.Ti = Ti
        self.Ni = Ni
        self.Nimp = Nimp
        self.Machi = Machi
        self.Mzstar = Mzstar
        self.Zeff = Zeff
        self.Te = Te_Ti*Ti
        self.gradTi = gradTi
        self.gradNi = gradNi
        self.gradNimp = gradNimp
        self.qmag = qmag
        self.R = RV
        self.Z = ZV
        self.J = JV
        self.B = BV
        
        # Pfirsch-Schlüter (PS)
        
        self.Dz_PS = fdps*qmag**2*rhoLz2*nuz*(CgeoG/(2*eps2))/e0imp # PS diffusion coefficient [m^2/s]
        self.Kz_PS = (Zimp/Zi)*self.Dz_PS # PS coefficient of the main ion density gradient [m^2/s]
        self.Hz_PS = (-(1.0 + (Zimp/Zi)*(C0z - 1.0)) + (CgeoU/CgeoG)*(Zimp/Zi)*(C0z + ki))*\
                        self.Dz_PS # PS coefficient of the main ion temperature gradient [m^2/s]
        
        self.Vrz_PS = -self.Dz_PS*grad_ln_Nimp + self.Kz_PS*grad_ln_Ni + \
                       self.Hz_PS*grad_ln_Ti # PS radial flux per particle [m/s]
        self.Vconv_PS = self.Kz_PS*grad_ln_Ni + self.Hz_PS*grad_ln_Ti # PS convective velocity [m/s]
        
        # Banana-Plateau (BP)
        
        self.Dz_BP = (1.5*(self.qe*Ti)/(Zimp**2*self.qe**2*FV**2*Nimp))*(1/(1/K11i + 1/K11z))/e0imp # BP diffusion coefficient [m^2/s]
        self.Kz_BP = (Zimp/Zi)*self.Dz_BP # BP coefficient of the main ion density gradient [m^2/s]
        self.Hz_BP = fhbp*((Zimp/Zi)*(K12i/K11i - fv) - (K12z/K11z - fv))*self.Dz_BP # BP coefficient of the main ion temperature gradient [m^2/s]
        
        self.Vrz_BP = -self.Dz_BP*grad_ln_Nimp + self.Kz_BP*grad_ln_Ni + \
                       self.Hz_BP*grad_ln_Ti  # BP radial flux per particle [m/s]
        
        self.Vconv_BP = self.Kz_BP*grad_ln_Ni + self.Hz_BP*grad_ln_Ti # BP convective velocity [m/s]
        
        # Classical
        
        self.Dz_CL = (CclG*2*eps2/CgeoG)*self.Dz_PS/(2*qmag**2) # CL diffusion coefficient [m^2/s]
        self.Kz_CL = (Zimp/Zi)*self.Dz_CL # CL coefficient of the main ion density gradient [m^2/s]
        self.Hz_CL = -(1.0 + (Zimp/Zi)*(C0z - 1.0))*self.Dz_CL # CL coefficient of the main ion temperature gradient [m^2/s]
        
        self.Vrz_CL = -self.Dz_CL*grad_ln_Nimp + self.Kz_CL*grad_ln_Ni + \
                       self.Hz_CL*grad_ln_Ti # CL radial flux per particle [m/s]
        self.Vconv_CL  = self.Kz_CL*grad_ln_Ni + self.Hz_CL*grad_ln_Ti # CL convective velocity [m/s]
        
        
        # Total
        
        self.Dz = self.Dz_PS + self.Dz_BP + self.Dz_CL # collisional diffusion coefficient [m^2/s]
        self.Kz = self.Kz_PS + self.Kz_BP + self.Kz_CL # coefficient of the main ion density gradient [m^2/s]
        self.Hz = self.Hz_PS + self.Hz_BP + self.Hz_CL # coefficient of the main ion temperature gradient [m^2/s]
       
        self.Vconv = self.Kz*grad_ln_Ni + self.Hz*grad_ln_Ti # convective velocity [m/s]
        self.Vrz = self.Vrz_PS + self.Vrz_BP + self.Vrz_CL   # radial flux per particle [m/s]
        
        self.Flux_z = self.Vrz*Nimp # radial collisional flux [1/(m s)]
        
        
        # asymmetries
        
        self.horiz_asym = deltan # horizontal asymmetry of the impurity density [-]
        self.vert_asym  = Deltan # vertical asymmetry of the impurity density [-]
        self.nn         = nn # poloidal distribution of the impurity density, nn = Nimp/<Nimp> [-]
        
        
        
        #---------------------------------------------------------------------
        #----------------- end of FACIT, methods next ------------------------
        #---------------------------------------------------------------------
        
        
    def ftrap(self, eps):
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
        return 1. - (1. - eps)**1.5/ (np.sqrt(1. + eps)*(1 + 1.46*np.sqrt(eps)))
    
    def C2(self, alpha, g, Ai, A, f1, f2):
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
        
        return 1.5/(1+f1*(Ai/A)) - (0.29 + 0.68*alpha)/(0.59 + alpha + (1.34 + f2)*g**(-2))
    
    
    
    def ki_f(self, nui_star, ft, Zeff, Machi):
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
        
        c01 = 0.53*(1 + 0.646*Machi**1.5)
        c02 = 1.158*(1-0.968*Machi**1.56)
        c03 = -0.98*(1-1.228*Machi**1.7)
        
        l1k = 5.7*(1-ft)**6.7 + 0.38
        l2k = (-1.52 + 38.4*(1-ft)**3.02*ft**2.07) + (-1 + 2.6*ft)*Machi**(2.5*(1 - 0.6*ft))
        l3k = 0.25 + 1.2*(1-ft)**3.65
        l5k = 0.1*(1-ft)**1.46*ft**4.33 + 0.051*(1 - 0.82*ft)*Machi**2.5
        l4k = 0.8 + (1.25*ft + 0.585)*Machi
        l6k = (-0.05 + 1.95*ft**2.5)/(1 + 2.55*ft**17) + -((0.217 + 14.57*ft**6.3)/(1 + 5.62*ft**5.72))*Machi**1.5
    
        ki0 = -(c01 + 0.055*(Zeff-1))*(1-ft)/((0.53 + 0.17*(Zeff-1))*\
                  (1-(c02 - 0.065*(Zeff-1))*ft - c03*ft**2))
        
        kiv = ((ki0 + (l1k)*Zeff*np.sqrt(ft*nui_star) + l2k*nui_star**(0.25))/(1+l3k*np.sqrt(nui_star))-\
                  l4k*l5k*nui_star**2*ft**6 + l6k*nui_star**(0.25))/(1+l5k*nui_star**2*ft**6)
        
        return kiv
    
    def facs(self, Z, A, ft, Mzstar, rotation_model):
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
        
        #---------------------- PS facs ---------------------------------------
        # f1, f2, f3 factors in C0z
        #broadcast in the same shape
        Z,Mzstar = np.broadcast_arrays(Z,Mzstar)
        
        f1 = (1.74*(1-0.028*A) + 10.25/(1 + A/3.0)**2.8) - 0.423*(1-0.136*A)*ft**(5/4)
        f2 = (88.28389935 + 10.50852772*Z)/( 1 + 0.2157175*Z**2.57338463)
        f3 = (-4.45719179e+06 + 2.72813878e+06*Z)/(1+5.26716920e+06*Z**8.33610108e-01)
        
        if rotation_model == 2:
            f2 *= np.exp(-10*Mzstar**2)
            f3 *= (1 + (1 + 1.86e6*ft**11.07*(1-ft)**7.36)*Mzstar**4)*np.exp(-0.8*Mzstar**2)
            
            
        fg1 = -1.4*ft**7 + 2.23*(1-0.31*ft)
        fg2 = 2.8*(1-0.63*ft)
        fg3 = 3.5*(1 - ft)/fg2
        fg4 = 4*ft
        fg5 = 0.38*ft**4
        fg6 = 3.95*(1 + 0.424*ft*(1 - 0.65*ft))
        
        fG = (1 + fg1*Mzstar**fg2)**(fg3)*(1 + 0.2*Mzstar**fg4)/(1 + fg5*Mzstar**fg6)
        
        
        c1f = 2.72*(1-0.91*ft)
        c2f = 2.98*(1-0.471*ft)
        c3f = 0.95*(1-ft)**4
        c4f = 4*ft
        c5f = 0.1314*ft**2.84 + 3.178*ft**11.4
        c6f = -9.38*(ft-0.5)**2 + 4.64
        
        fU = (c1f*Mzstar**c2f)*(1 + c3f*Mzstar**c4f)/(1 + c5f*Mzstar**c6f)
            
        #---------------------- BP facs ---------------------------------------
        
        # factors in viscosity coefficients
        
        y11zb  = (8*(1-ft)**20 - 0.66*ft**4.9 + 0.94)*((1.085e4 + 9.3e3*14/Z**(5/3))/(1 + 14/Z**(5/3)))*ft**4.6*(1-ft)/\
                 (1 + 9.44e3*(1 - 2e-3*Z)*ft**4.16)
        y11zp  = 1.0
        y11zps = 1.0
        
        y12zb  = (1.8e3 + 7.54*Z**1.8)*ft**(4.05*(1 + 0.0039*Z))*(1-ft)**(0.7*(1 + 0.015*Z))/(1 + 1276*(1 + 0.053*Z)*ft**3.6)
        y12zp  = 1.0
        y12zps = 1.0
        
        
        y11ib  = ((5.91e-5*Z + 0.812 + 0.806/Z**0.44) + (0.013*Z + 0.098 - 7.03/Z**1.04)*ft + \
                 (-0.047*Z + 0.79 + 13.8/Z**1.26)*ft**2 + (0.04*Z - 0.575 - 9.64/Z**1.31)*ft**3)*\
                 (571.6*(1 + 0.84*Z + 7.8e-7*Z**5.15)*ft**(3.43*(1 + 0.012*Z))*\
                 (1-ft)**(1.2*(1 + 1.6e-8*Z**5.1))/(1 + 500*ft**(8/3)) + 1e-3)
        y11ip  = 1.0
        y11ips = 1/((1+(99/44**6)*Z**6))
        
        y12ib  = (571.6*(1 + 0.84*Z + 7.8e-7*Z**5.15)*ft**(3.43*(1 + 0.012*Z))*\
                 (1-ft)**(1.2*(1 + 1.6e-8*Z**5.1))/(1 + 500*ft**(8/3)) + 1e-3)
        y12ip  = 1.0
        y12ips = 1.0
        
        
        if rotation_model == 2:
            
            c11f = 1 + 14.86*(1 - ft)**16.45 + 15.27*ft**7.4
            c12f = 0.77*(1 + 4.11*ft)
            c13f = 0.01*(1 + 359*ft**2.5 + 1078*ft**12)
            l1f = (1 + c11f*Mzstar**c12f)*np.exp(-c13f*Mzstar**2)
            
            c21f = 6.39*(1 -ft)**15.8 + 0.1
            c22f = 0.943*(1 + 3.5*ft)
            l2f  = (1 + c21f*Mzstar**0.5)/(1 + c22f*Mzstar**(10/3))
            
            l3f = 1/(1 + 2*ft*Mzstar)
            
            c1f = 6.13 + 28.18*ft**2.13 + 336.25*(1-ft)**11.65
            c2f = 0.5 + 9.55*ft**1.14*(1-ft)**1.42
            c3f = (0.0087 + 4.49*ft**3.48)/(1 + 0.873*ft**3.48)
            c4f = 3.6*(1 - 0.36*ft)
            
            l4f =  (1 + c1f*Mzstar**c2f)/(1 + c3f*Mzstar**(c4f))
            
            c1f = (1-ft)**8
            c2f = 113.5*ft**8.46
            c3f = 11*(1-ft)
            
            l5f = (1 + c1f*c2f*Mzstar**(c3f))/(1 + c2f*Mzstar**(c3f))
            
            l6f = (1 + 0.035*10*Mzstar**4)/(1 + 10*Mzstar**4)
            
            l7f = np.exp(-10*Mzstar**2)
            
            y11zb  *= l1f
            y11zp  *= l1f/l2f
            y11zps *= l1f/(l2f*l3f)
            
            y12zb  *= l4f
            y12zp  *= l4f/l5f
            y12zps *= l4f/(l5f*l6f)
            
            y11ib  *= l1f
            y11ip  *= l1f
            y11ips *= l1f
            
            y12ib  *= l7f
            y12ip  *= l7f
            y12ips *= l7f
            
            
        yy = [[y11zb, y11zp, y11zps],[y12zb, y12zp, y12zps], \
              [y11ib, y11ip, y11ips],[y12ib, y12ip, y12ips]]
        
        if rotation_model == 2:
            fv = 1.5*np.exp(-10*Mzstar**2)
        else:
            fv = 1.5
        
        # factor of the PS diffusion coefficient
        fdps = ((0.711 + 2.08e-3*Z**1.26)/(1 + 1.06e-11*Z**5.78))
        
        # factor of the BP H coefficient
        
        if rotation_model == 2:
            fhbp = (0.135 + 2.647e-3*Z**1.464 + 3.478e-10*Z**5.347)*\
                   ((1 + (3/(1 + 1e-7*Z**6))*(1/(1 + 1.2e5*ft**12))*Mzstar)/\
                   (1 + (3/(1 + 1e-7*Z**6))*(1.208 - 4.46*ft + 4.394*ft**2)*Mzstar**2))
                       
        else:
            fhbp = (1.01579172e+00 + -1.78923911e-03*Z)/(1 + 6.60170647e-13*Z**6.66398825e+00)
        
        
        return f1, f2, f3, fG, fU, yy, fv, fdps, fhbp
    
    
    
    def KVISC(self, nimp, ni, ti, Ai, Aimp, Zi, Zimp, \
                    Tauii, Tauimpimp, Tauiimp, Tauimpi, \
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
        
        # transit frequencies
        
        wii     = np.sqrt(2*(self.qe*ti)/(Ai*self.mp))/(qmag*R0) # main ion transit frequency [1/s]
        wimpimp = np.sqrt(Ai/Aimp)*wii                           # impurity transit frequency [1/s]
        
        # Plateau regime
        
        fac_a_P = nimp*(self.qe*ti)*np.sqrt(np.pi)/(3*wimpimp)
        fac_i_P = ni*(self.qe*ti)*np.sqrt(np.pi)/(3*wii)
        
        K11aP = fac_a_P*2
        K12aP = fac_a_P*2*3
        
        K11iP = fac_i_P*2
        K12iP = fac_i_P*2*3
        
        # Pfirsch-Schlüter regime
        
        r00 = 1/np.sqrt(2)
        r01 = 1.5/np.sqrt(2)
        r11 = 3.75/np.sqrt(2)
        
        
        # thermal velocity fractions
        xai = np.sqrt(Aimp/Ai)
        xia = 1/xai
        
        x2ai = xai**2
        x2ia = xia**2
        
        xfac_ai = (1+x2ai)**0.5
        xfac_ia = (1+x2ia)**0.5
        
        qaa00 = qii00 = 8/2**1.5
        qaa01 = qii01 = 15/2**2.5
        qaa11 = qii11 = 132.5/2**3.5
        
        qai00 = (3+5*x2ai)/xfac_ai**3
        qia00 = (3+5*x2ia)/xfac_ia**3
        
        qai01 = 1.5*(3+7*x2ai)/xfac_ai**5
        qia01 = 1.5*(3+7*x2ia)/xfac_ia**5
        
        qai11 = (35*x2ai**3 + 38.5*x2ai**2 + 46.25*x2ai + 12.75)/xfac_ai**7
        qia11 = (35*x2ia**3 + 38.5*x2ia**2 + 46.25*x2ia + 12.75)/xfac_ia**7
        
        fac_qai_PS = (ni*Zi**2/(nimp*Zimp**2))
        fac_qia_PS = 1/fac_qai_PS #(nimp*Zimp**2/(ni*Zi**2))
        
        
        qa00 = fac_qai_PS*qai00 +  qaa00 - r00
        qi00 = fac_qia_PS*qia00 + qii00 - r00
        
        qa01 = fac_qai_PS*qai01 + qaa01 - r01
        qi01 = fac_qia_PS*qia01 + qii01 - r01
        
        qa11 = fac_qai_PS*qai11 + qaa11 - r11
        qi11 = fac_qia_PS*qia11 + qii11 - r11
        
        
        Qa = 0.4*(qa00*qa11-qa01*qa01)
        Qi = 0.4*(qi00*qi11-qi01*qi01)
        
        
        la11 = qa11/Qa
        la12 = 3.5*(qa11+qa01)/Qa
        
        li11 = qi11/Qi
        li12 = 3.5*(qi11+qi01)/Qi
        
        
        fac_imp_PS = nimp*(self.qe*ti)*Tauimpimp
        fac_ion_PS = ni*(self.qe*ti)*Tauii

            
        K11aPS = fac_imp_PS*la11
        K12aPS = fac_imp_PS*la12
        
        K11iPS = fac_ion_PS*li11
        K12iPS = fac_ion_PS*li12
        
        # Banana regime
        fac_B = (ft/(1-ft))*(2*R0**2*qmag**2/(3*eps**2))

        nuDai_int  = (xfac_ai + x2ai*np.log(xai/(1+xfac_ai)))/Tauimpi
        nuD2ai_int = 1/(xfac_ai*Tauimpi)
        
        nuDia_int  = (xfac_ia + x2ia*np.log(xia/(1+xfac_ia)))/Tauiimp
        nuD2ia_int = 1/(xfac_ia*Tauiimp)
        
        nuDaa_int  = (np.sqrt(2) + np.log(1/(1+np.sqrt(2))))/Tauimpimp
        nuD2aa_int = 1/(np.sqrt(2)*Tauimpimp)
        
        nuDii_int  = (np.sqrt(2) + np.log(1/(1+np.sqrt(2))))/Tauii
        nuD2ii_int = 1/(np.sqrt(2)*Tauii)
        
        K11aB = fac_B*nimp*(Aimp*self.mp)*(nuDai_int + nuDaa_int)
        K12aB = fac_B*nimp*(Aimp*self.mp)*(nuD2ai_int + nuD2aa_int)
        
        K11iB = fac_B*ni*(Ai*self.mp)*(nuDia_int + nuDii_int)
        K12iB = fac_B*ni*(Ai*self.mp)*(nuD2ia_int + nuD2ii_int)
        
        # fitted factors
        
        y11zb  = yy[0][0]
        y11zp  = yy[0][1]
        y11zps = yy[0][2]
        
        y12zb  = yy[1][0]
        y12zp  = yy[1][1]
        y12zps = yy[1][2]
        
        y11ib  = yy[2][0]
        y11ip  = yy[2][1]
        y11ips = yy[2][2]
        
        y12ib  = yy[3][0]
        y12ip  = yy[3][1]
        y12ips = yy[3][2]
        
        
        # total: rational interpolation across collisionality regimes
        
        K11a = y11zb*K11aB/((1 + y11zb*K11aB/(y11zp*K11aP))*(1 + y11zp*K11aP/(y11zps*K11aPS)))
        K12a = y12zb*K12aB/((1 + y12zb*K12aB/(y12zp*K12aP))*(1 + y12zp*K12aP/(y12zps*K12aPS)))
        
        K11i = y11ib*K11iB/((1 + y11ib*K11iB/(y11ip*K11iP))*(1 + y11ip*K11iP/(y11ips*K11iPS)))
        K12i = y12ib*K12iB/((1 + y12ib*K12iB/(y12ip*K12iP))*(1 + y12ip*K12iP/(y12ips*K12iPS)))
    
        return K11i, K12i, K11a, K12a
    
    
    
    def fluxavg(self, QV, JV):
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
        
        denom = np.trapz(JV, axis=-1)
        Qavg  = np.trapz(QV*JV, axis=-1)/denom
    
        return Qavg
    
    
    def polasym_input(self, rho, eps, Zeff, Zi, Te_Ti, Machi2, \
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
        TperpTpar_axis : float, optional
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
        #TODO use broadcast_shapes in new numpy
        out_shape = np.broadcast(Zeff, Te_Ti, Machi2).shape
        AsymPhi = np.zeros((2,)+out_shape) # asymmetry in electrostatic potential [-]
        AsymN   = np.zeros((2,)+out_shape) # asymmetry in main ion density [-]
    
        TperpTpar = (TperpTpar_axis - 1.)*np.exp(-(rho/sigH)**2) + 1.
        
        AsymPhi[0] = eps/(1. + Zeff*(Te_Ti))*\
                     (fH*(TperpTpar - 1.)*bC/(bC + TperpTpar*(1. - bC)) + 2*Machi2)
                        
        AsymN[0] = -Zi*(Te_Ti)*AsymPhi[0] + 2*eps*Machi2
        AsymN[1] = -Zi*(Te_Ti)*AsymPhi[1]
        
        
        return AsymPhi, AsymN#, TperpTpar
    
    
    
    def asymmetry_analytical(self, rho, theta, GG, UU, eps, invaspct, qmag, nuswcz, \
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
        invaspect: float
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
        dminphia : 1D array (nr)
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
        
        if nat_asym:
            Ae = nuswcz*qmag**2/invaspct
        else:
            Ae = 0
        
        AGe = Ae*GG
        HH = 1.0
        CD0 = -eps/UG
        QQ = CD0*(dNV/(eps))*(UG-1.0)
        FF = CD0*(1-0.5*dNH*(UG-1.0)/(eps) )
        KK = 1.
        
        CD = FF - 0.5*(dminphia - deltaM)
        CDV = -0.5*(dmajphia + QQ)
        RD = np.sqrt((FF + 0.5*(dminphia-deltaM))**2 + 0.25*(dmajphia - QQ)**2)
        DD = RD**2 + AGe**2*(RD/CD0)**2
        
        num  = ((AGe/CD0)**2 - 1)*(FF/(CD0) + 0.5*(dminphia-deltaM)/CD0) + \
               (AGe/CD0)*(0.5*dNV*(UG-1.0)/(eps) - 0.5*dmajphia/CD0)
        cosa = RD*CD0*num/DD
      
        num  = 2*AGe*(FF/CD0 + 0.5*(dminphia-deltaM)/CD0)+((AGe/CD0)**2-1)*\
               (0.5*dmajphia - 0.5*dNV*CD0*(UG-1.0)/((eps)))
        sina = RD*num/DD
        
        deltan = CD + RD*cosa
        Deltan = CDV + RD*np.sqrt(KK/HH)*sina
        
        nn = 1 + np.outer(deltan, np.cos(theta)) + np.outer(Deltan, np.sin(theta))
        
        return deltan, Deltan, nn
    
    
    
    def asymmetry_iterative(self, regulopt, nr, theta, GG, UU, Ai, Aimp, Zi, Zimp, Te_Ti, \
                                  Machi2, R0, nuz, BV, RV, JV, FV, dpsidx, AsymPhi, AsymN, \
                                  b2, nat_asym):
        '''
        Calculate poloidal asymmetries of the impurity density self-consistently,
        including the friction-induced natural asymmetry and rotation and ICRH-
        induced asymmetries.
        This is done in the full flux surface shaped geometry, such that the 
        solution is iterative.
        
        Refs: Maget PPCF (2020), https://doi.org/10.1088/1361-6587/ab53ab
              Maget PPCF (2020), https://doi.org/10.1088/1361-6587/aba7f9
        
        Parameters
        ----------
        regulopt : list (4)
            parameters for the convergence of the iterative calculation of the
            asymmetries
            can be set to [1e-2,0.5,1.e-5,1e2]
        nr : int
            number of radial grid points
        theta : 1D array (nth)
            poloidal grid [-]
        GG : 1D array (nr)
            gradient term [-], see Maget PPCF (2020) (first one) eq. 10
        UU : 1D array (nr)
            gradient term [-], see Maget PPCF (2020) (first one) eq. 11
        Ai : int or float
            main ion mass [-]
        Aimp : int or float
            impurity mass [-]
        Zi : int or float
            main ion charge [-]
        Zimp : int or float or 1D array (nr)
            impurity charge [-]
        Te_Ti : 1D array (nr)
            electron to main ion temperature ratio [-]
        Machi2 : 1D array (nr)
            main ion Mach number squared [-]
        R0 : float
            major radius at magnetic axis [m]
        nuz : 1D array (nr)
            impurity collision frequency [1/s]
        BV : 2D array (nr, nth)
            magnetic field, poloidal distribution on (r,theta) [T]
        RV : 2D array (nr, nth)
            major radius, poloidal distribution on (r,theta) [m]
        JV : 2D array (nr, nth)
            Jacobian of coordinate system, poloidal distribution on (r,theta) [m/T]
        FV : 1D array (nr)
            poloidal current flux function [T*m]
        dpsidx : 1D array (nr)
            radial coordinate transformation from flux psi to rho [T*m^2]
        AsymPhi : 2D array (2, nr)
            horizontal (index 0) and vertical (index 1) in electrostatic potential [-], phi/<phi>
            from polasym_input method
        AsymN : 2D array (2, nr)
            horizontal (index 0) and vertical (index 1) in main ion density [-], ni/<ni>
        b2 : 2D array (nr, nth)
            poloidal distribution of the magnetic field [-], b^2 = B^2/<B^2>
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
        b2sNNavg : 1D array (nr)
            <b^2/N> [-]
        NV : 2D array (nr, nth)
            poloidal distribution of the main ion density in (r,theta) [-]
            NV = ni(r,theta)/<ni(r,theta)>(r)
        '''
        
        # Parameters for convergence on parallel momentum equation
        err         = regulopt[0]
        prog        = regulopt[1]
        regulweight = regulopt[2]
        ierrmax     = regulopt[3]
        
        Error       = np.zeros((nr, int(ierrmax)))
        asym_err    = np.zeros(nr)
        
        #GG0 = GG - grad_ln_Nimp
    
        theta = np.linspace(0, 2*np.pi, theta.size + 1)[:-1] #poloidal coordinate grid without repeating 0 at 2pi
        #thetalong = np.concatenate((theta-2*np.pi,theta,theta+2*np.pi))

        Factrot0 = (Aimp/Ai)*Machi2/R0**2
            
        if nat_asym:
            Apsi = JV*FV[:,None]*(Aimp*self.mp)*nuz[:,None]/\
                   (Zimp[:,None]*self.qe*(dpsidx[:,None]**2 + 1.e-33)) # as defined after eq. 9 in Maget (2020)
        else:
            Apsi = np.zeros_like(JV)
        
        PhiV = np.outer(AsymPhi[0], np.cos(theta)) + np.outer(AsymPhi[1], np.sin(theta))
        NV   = 1 + np.outer(AsymN[0], np.cos(theta)) + np.outer(AsymN[1], np.sin(theta))
        
        b2sNNavg  = self.fluxavg(b2/NV , JV)  # <b**2/N>
        
        nn = np.ones_like(BV) # poloidal distribution of the impurity density: n = nz/<nz>
        
        dtheta = theta[1]-theta[0] # step size in poloidal coordinate grid
        
        
        
        for ix in range(nr): #this cannot be vectorized because in the while, the local nn[ix] is set each time
        
            nnp = 2        # 
            nnx = nn[ix]   # local impurity density, poloidal dependence
            
            ierr   = 1     # 
            progx  = prog  # 
            Erreur = 2*err # 
            
            
            AA = np.zeros((theta.size + 1, theta.size + 1))
            LL = np.zeros((theta.size + 1, theta.size + 1))
            BB = np.zeros((theta.size + 1, 1))
            
            while (Erreur>err and ierr<ierrmax): # iterative calculation of poloidal asymmetry
                
                b2snavg = self.fluxavg(b2[ix]/nnx, JV[ix]) # <b**2/n>

                FFF  = Apsi[ix]*( GG[ix] + (b2[ix]/NV[ix])*UU[ix])
                GGG  = -Zimp[ix]*(Te_Ti[ix])*(PhiV[ix]-PhiV[ix,0]) + \
                        Factrot0[ix]*(RV[ix]**2-RV[ix,0]**2)
                HHH  = Apsi[ix]*(b2[ix]/b2snavg)*(GG[ix] + b2sNNavg[ix]*UU[ix])
                
                
                for ii in range(1,theta.size-1): 
                    
                    AA[ii,ii-1]  = -0.5/dtheta
                    AA[ii,ii]    = -FFF[ii]-(0.5/dtheta)*(GGG[ii+1]-GGG[ii-1])
                    AA[ii,ii+1]  = 0.5/dtheta
                    
                    LL[ii, ii-1] = 1.
                    LL[ii, ii]   = -2.
                    LL[ii, ii+1] = 1.
                    
                    BB[ii,0]     = -HHH[ii]

                
                AA[0,0]  = -FFF[0]-(0.5/dtheta)*(GGG[1]-GGG[theta.size-2])
                AA[0,1]  = 0.5/dtheta
                AA[0,-2] = -0.5/dtheta 
                LL[0,0]  = -2.
                LL[0,1]  = 1.
                LL[0,-2] = 1. 
                BB[0,0]  = -HHH[0]
                
                
                AA[-2,-3] = -0.5/dtheta
                AA[-2,-2] = -FFF[theta.size-1]-(0.5/dtheta)*(GGG[0]-GGG[theta.size-2])
                AA[-2,-1] = 0.5/dtheta
                LL[-2,-3] = 1.
                LL[-2,-2] = -2.
                LL[-2,-1] = 1.
                BB[-2,0]  = -HHH[theta.size-1]
                
                
                AA[-1,-1] = -1.
                AA[-1,0]  = 1
                BB[-1,0]  = 0.
                
                CC   = AA.T @ AA + regulweight * LL.T @ LL
                nnya = np.linalg.inv(CC) @ AA.T @ BB
                nny  = nnya[:-1,0]/self.fluxavg(nnya[:-1,0],JV[ix])
                nny  = np.maximum(nny, 1e-5)
                
                nnp = nnx
                nnx = progx*nnp + (1-progx)*nny
                
                Err            = np.max(np.abs(np.gradient(np.log(nnx)-GGG,theta)-FFF+HHH/nnx))
                Error[ix,ierr] = Erreur
                
                ierr = ierr+1
                #print('r/a=', cdat.rho[ix],' - ierr=',ierr,' - err=', Erreur)
                    
                nn[ix]       = nnx
                asym_err[ix] = Err
    
    
        deltan = 2.*np.mean((nn-1)*np.cos(theta), axis = 1) # horizontal asymmetry of the impurity density
        Deltan = 2.*np.mean((nn-1)*np.sin(theta), axis = 1) # vertical asymmetry of the impurity density
        
        return deltan, Deltan, nn, b2sNNavg, NV
    
    def Jacobian(self, R, Z, r, theta):
        '''
        Calculates the Jacobian of the coordinate transformation (r,theta) <--> (R,Z)
        Based on GKW manual, eqs (2.104-2.107)

        Parameters
        ----------
        R : 2D array (nr, nth)
            major radius countours of flux surfaces [m].
        Z : 2D array (nr, nth)
            vertical coordinate contours of flux surfaces [m].
        r : TYPE
            DESCRIPTION.
        theta : TYPE
            DESCRIPTION.

        Returns
        -------
        J : 2D array (nr, nth)
            Jacobian of coordinate transformation

        '''
        
        dRdr  = np.gradient(R, r, axis = 0)
        dRdth = np.gradient(R, theta, axis = 1)
        dZdr  = np.gradient(Z, r, axis = 0)
        dZdth = np.gradient(Z, theta, axis = 1)

        grr   = dRdr**2 + dZdr**2
        grth  = dRdr*dRdth + dZdr*dZdth
        gthth = dRdth**2 + dZdth**2

        return R*np.sqrt(grr*gthth - grth**2)
        
        
    

#%%


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    # simple example
    
    rho = np.linspace(0,1,101)
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
   
    
    fct = FACIT(rho, \
                Zimp, Aimp, \
                Zi, Ai, \
                Ti, Ni, Nimp, Machi, Zeff, \
                gradTi, gradNi, gradNimp, \
                invaspct, B0, R0, qmag,  \
                rotation_model = 2, Te_Ti = 1.0,\
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
    
