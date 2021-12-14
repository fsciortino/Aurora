#-----------------------------------------------------------------------------
# Python standalone of FACIT routine for collisional impurity transport
# Daniel Fajardo (daniel.fajardo@ipp.mpg.de), November 2021
#-----------------------------------------------------------------------------

import numpy as np

class FACIT:
    '''
    Calculates collisional impurity transport coefficients and the poloidal
    asymmetry of the impurity density.
    
    Refs: Maget PPCF (2020), https://doi.org/10.1088/1361-6587/ab53ab
          Maget PPCF (2020), https://doi.org/10.1088/1361-6587/aba7f9
          new paper...
    
    Parameters
    ----------
    rho : 1D array (nr)
        mid-plane radius normalized to minor radius [-], r/a
    Zimp : int or float or 1D array (nr)
        impurity charge [-]
    Aimp : int or float
        impurity mass [-]
    Zi : int or float
        main ion charge [-]
    Ai : int or float
        main ion mass [-]
    Te : 1D array (nr)
        electron temperature [eV]
    Ti : 1D array (nr)
        main ion temperature [eV]
    Ne : 1D array (nr)
        electron density [1/m^3]
    Ni : 1D array (nr)
        main ion density [1/m^3]
    Nimp : 1D array (nr)
        impurity density [1/m^3]
    Machi : float or 1D array (nr)
        main ion Mach number [-]
        set to zero to perform calculations in the non-rotating limit
    invaspct : float
        inverse aspect ratio [-], a/R0
    B0 : float
        magnetic field at magnetic axis [T]
    R0 : float
        major radius at magnetic axis [m]
    qmag : 1D array (nr)
        safety factor [-]
    pol_asym : bool, optional
        if True, calculate the poloidal asymmetry of the impurity density
        default is False
    full_geom : bool, optional
        if True, use full flux surface shaped geometry in the calculation of
        the asymmetries, which means an iterative solution. This would be the
        most computationally-expensive setting of FACIT.
        if False, use circular geometry with large aspect ratio, such that
        the calculation of the asymmetries can be completely analytical
        default is False
        if pol_asym is False, full_geom matters only in the calculation of
        some geometric coefficients
    nth : int, optional
        number of poloidal grid points [-]
        default is 20
    regulopt : list (4), optional
        parameters for the convergence of the iterative calculation of the
        asymmetries
        only relevant if pol_asym and full_geom
        default is [1e-2,0.5,1.e-5,1e2]
    dpsidx : 1D array (nr), optional
        radial coordinate transformation from flux psi to rho [T*m^2]
        only relevant if pol_asym and full_geom
        default is None
    FV : 1D array (nr), optional
        poloidal current flux function [T*m]
        only relevant if pol_asym and full_geom
        default is None
    BV : 2D array (nr, nth), optional
        magnetic field, poloidal distribution on (r,theta) [T]
        only relevant if pol_asym and full_geom
        default is None
    RV : 2D array (nr, nth), optional
        major radius, poloidal distribution on (r,theta) [m]
        only relevant if pol_asym and full_geom
        default is None
    JV : 2D array (nr, nth), optional
        Jacobian of coordinate system, poloidal distribution on (r,theta) [m/T]
        only relevant if pol_asym and full_geom
        default is None
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
        
        for ICRH-induced asymmetries
        only relevant if pol_asym
        default is 1.0
    TperpTpar_axis : float, optional
        main ion temperature anisotropy at magnetic axis [-]
        for ICRH-induced asymmetries
        only relevant if pol_asym
        default is 1.0
    nat_asym : bool, optional
        if True, consider friction-induced "natural" poloidal asymmetry
        only relevant if pol_asym
        default is True
        
    Returns
    -------
    Dz_* : 1D array (nr)
        diffusion coefficient from each flux component [m^2/s]
        * = PS, BP or CL (for Pfirsch-Schlüter, Banana-Plateau and Classical)
    Kz_* : 1D array (nr)
        coefficient of the main ion density gradient from each flux component [m^2/s]
    Hz_* : 1D array (nr)
        coefficient of the main ion temperature gradient from each flux component [m^2/s]
    Vrot_* : 1D array (nr)
        centrifugal velocity from each flux component [m^2/s]
    Vrz_* : 1D array (nr)
        radial flux per particle (Flux_z/nz)_* from each flux component [m/s]
    Dz : 1D array (nr)
        total diffusion coefficient [m^2/s]
    Kz : 1D array (nr)
        total coefficient of the main ion density gradient [m^2/s]
    Hz : 1D array (nr)
        total coefficient of the main ion temperature gradient [m^2/s]
    Vconv : 1D array (nr)
        total convective velocity [m/s]
    Vrot : 1D array (nr)
        total centrifugal velocity [m^2/s]
    Vrz : 1D array (nr)
        total radial flux per particle (Flux_z/nz) [m/s]
    Flux_z : 1D array (nr)
        total radial flux [1/(m^2 s)]
    horiz_asym : 1D array (nr)
        horizontal asymmetry of the impurity density [-]
        in the representation n = nz/<nz> = 1 + horiz_asym*cos(theta) + vert_asym*sin(theta)
    vert_asym : 1D array (nr)
        vertical asymmetry of the impurity density [-]
    nn : 2D array (nr, nth)
        poloidal distribution of the impurity density in (r,theta) [-]
        nn = nz(r,theta)/<nz(r,theta)>(r)
    '''
    
    # variables shared by all instances
    mp   = 1.672623e-27   # mass of the proton
    me   = 9.1096e-31     # mass of the electron
    qe   = 1.60217653e-19 # elementary charge
    eps0 = 8.8542e-12     # vacuum permittivity
    
    eps_pi_fac = 3*eps0**2*(2*np.pi/qe)**1.5/qe # common factor in collision times
    
    
    def __init__(self, rho, \
                       Zimp, Aimp, Zi, Ai, \
                       Te, Ti, Ne, Ni, Nimp, Machi, \
                       invaspct, B0, R0, qmag, \
                       
                       pol_asym = False, full_geom = False, \
                       nth = 20, regulopt = [1e-2,0.5,1e-5,1e2], \
                       dpsidx = None, FV = None, BV = None, RV = None, JV = None, \
                       fH = 0., bC = 1., sigH = 1., TperpTpar_axis = 1., nat_asym = True):
        
        #---------------------------------------------------------------------
        #-------------- Definition of physical quantities --------------------
        #---------------------------------------------------------------------
        
        
        mi   = Ai*self.mp   # main ion mass [kg]
        mimp = Aimp*self.mp # impurity mass [kg]
        
        rho = np.maximum(1e-6, rho) # to avoid dividing by zero
        nr  = rho.size # number of radial grid points
        
        eps = rho*invaspct  # local inverse aspect ratio [-]
        eps2  = eps**2 # auxiliary
        
        
        amin = invaspct*R0 # minor radius [m]
        
        if type(Zimp) is int: # impurity charge must always be a radial array
            Zimp = Zimp*np.ones(nr)
        
        Zeff = (Zimp**2*Nimp + Zi*Ni)/Ne # effective charge [-]
        
        # logarithmic gradients: dlnQ/dr = (dQ/dr)/Q = dlnQ/d(amin*rho)
        grad_ln_Ni   = np.gradient(Ni, amin*rho)/Ni     # [1/m]
        grad_ln_Ti   = np.gradient(Ti, amin*rho)/Ti     # [1/m]
        grad_ln_Nimp = np.gradient(Nimp, amin*rho)/Nimp # [1/m]
        
        
        ft = self.ftrap(eps) # trapped particle fraction [-]
        
        # Coulomb logarithms, from NRL formulary
        LniiNRL     = 23. - np.log(Zi**3*np.sqrt(2*(Ni/1e6))) + 1.5*np.log(Ti)
        LnimpiNRL   = 23. - np.log(Zi*Zimp*np.sqrt((Ni/1e6)*Zi**2+(Nimp/1e6)*Zimp**2)) + 1.5*np.log(Ti)
        LnimpimpNRL = 23. - np.log(Zimp**3*np.sqrt(2*(Nimp/1e6))) + 1.5*np.log(Ti)
        
        # Braginskii collision times
        
        Ti15    = Ti**1.5 # auxiliary
        
        Tauii     = (self.eps_pi_fac*np.sqrt(mi)*Ti15)/(Zi**4*Ni*LniiNRL)             # ion-ion collision time [s]
        #Tauimpi   = (self.eps_pi_fac*np.sqrt(mimp)*Ti15)/(Zi**2*Zimp**2*Ni*LnimpiNRL) # impurity-ion collision time [s]
        #Tauiimp   = (self.eps_pi_fac*np.sqrt(mi)*Ti15)/(Zi**2*Zimp**2*Nimp*LnimpiNRL) # ion-impurity collision time [s]
        #Tauimpimp = (self.eps_pi_fac*np.sqrt(mimp)*Ti15)/(Zimp**4*Nimp*LnimpimpNRL)   # impurity-impurity collision time [s]
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
        mu_ie = (96*np.sqrt(2)/125)*np.sqrt(self.me/mi)*(Ti15/Te**1.5)
        
        
        # fitted factors
        f1, f2, f3, y1, y2, y3, y4, adps, ahbp = self.facs(Zimp, ft)
        
        # new friction coefficient of the main ion parallel heat flow [-]
        C0z = self.C2(alpha, g, Zimp, Ai, Aimp, f1, f2)/(1 + f3*mu_ie*g**2)
        
        ki = self.ki_Redl(nuistar, ft, Zeff) # neoclassical main ion flow coefficient [-]
        
        
        wcz = Zimp*self.qe*B0/mimp         # impurity cyclotron frequency [1/s]
        rhoLz2 = (2*(self.qe*Ti)/mimp)/wcz**2 # impurity Larmor radius squared [m^2]
        
        Machi2 = Machi**2 # auxiliary
        
        # viscosity coefficients
        K11i, K12i, K11z, K12z = self.KVISC(Nimp, Ni, Ti, Ai, Aimp, Zi, Zimp, \
                                            Tauii, Tauimpimp, Tauiimp, Tauimpi, \
                                            eps, ft, R0, qmag, y1, y2, y3, y4)
        
        
        
        #---------------------------------------------------------------------
        #------------------ Poloidal asymmetry terms -------------------------
        #---------------------------------------------------------------------
        
        theta, b2, invb2avg, gradR2, gradR2avg = self.GEOM(full_geom, nth, rho, eps, B0, BV, RV, JV)
        
        
        if pol_asym: # calculate poloidal asymmetries...
        
            # common gradient terms
            UU = -(Zimp/Zi)*(C0z + ki)*(grad_ln_Ti*amin)
            GG = (grad_ln_Nimp*amin) - (Zimp/Zi)*(grad_ln_Ni*amin) +\
                 (1 + (Zimp/Zi)*(C0z - 1.0))*(grad_ln_Ti*amin)
                 
            Te_Ti = Te/Ti # auxiliary
                 
            # rotation and ICRH asymmetries input
            AsymPhi, AsymN = self.polasym_input(rho, eps, Zeff, Zi, Te_Ti, Machi2, \
                                                fH, bC, TperpTpar_axis, sigH)
        
            if full_geom: # ...in full flux surface shaped geometry
        
                deltan, Deltan, nn, b2sNNavg, NV = self.asymmetry_iterative(regulopt, nr, theta, GG, UU, \
                                                                            Ai, Aimp, Zi, Zimp, Te_Ti, \
                                                                            Machi2, R0, nuz, BV, RV, JV, FV, dpsidx, \
                                                                            AsymPhi, AsymN, b2, gradR2, gradR2avg, nat_asym)
                                                              

                b2snnavg = self.fluxavg(b2/nn, JV) # <b^2/n>
                nnsb2avg = self.fluxavg(nn/b2, JV) # <n/b^2>
                
                # geometric coefficients in flux equations
                # CHECK SIGNS
                CgeoG = nnsb2avg - 1/b2snnavg
                CgeoU = self.fluxavg(nn/NV, JV) - b2sNNavg/b2snnavg
                CgeoR = self.fluxavg(gradR2/(b2/nn), JV) - gradR2avg/b2snnavg
            
                CclG  = nnsb2avg
                CclR  = self.fluxavg(gradR2/(b2/nn), JV)
                
                
            else: # ... in circular geometry with large aspect ratio
                
                deltaM = 2*(Aimp/Ai)*Machi**2*eps # rotation strength parameter [-]
                
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
                CgeoU = eps*(dNH - deltan) - dD2 + 0.5*(deltan*dNH + Deltan*dNV)
                CgeoR = R0*(2.0*eps + deltan)
            
                CclG = 1.0 + eps*deltan + 2*eps2
                CclR = R0*(1.0 + 2.0*eps + deltan - eps*deltan - dD2)
                
                
                

        else: # poloidally symmetric limit
            
            # geometric coefficient in flux equations
            
            CgeoG = invb2avg - 1
            CgeoU = 0.
            CgeoR = 2*eps*R0
            
            CclG = 1 + 2*eps2
            CclR = R0*(1 + 2*eps)
            
            deltan = np.zeros(nr)       # horizontal asymmetry of impurity density
            Deltan = np.zeros(nr)       # vertical asymmetry of impurity density
            nn     = np.ones((nr, nth)) # poloidal distribution of the impurity density
        
        
        #---------------------------------------------------------------------
        #------------- Collisional transport coefficients --------------------
        #---------------------------------------------------------------------
        
        
        # Pfirsch-Schlüter (PS)
        
        self.Dz_PS = adps*qmag**2*rhoLz2*nuz*(CgeoG/(2*eps2)) # PS diffusion coefficient [m^2/s]
        self.Kz_PS = (Zimp/Zi)*self.Dz_PS # PS coefficient of the main ion density gradient [m^2/s]
        self.Hz_PS = -((1.0 + (Zimp/Zi)*(C0z - 1.0)) + \
                       (CgeoU/CgeoG)*(Zimp/Zi)*(C0z + ki))*\
                        self.Dz_PS # PS coefficient of the main ion temperature gradient [m^2/s]
        
        self.Vrot_PS = (CgeoR/CgeoG)*(Aimp/Ai)*(Machi2/R0**2)*\
                        (1.0-(Ai*Zimp)/(Aimp*Zi))*self.Dz_PS/amin # PS centrifugal velocity [m/s]
        
        self.Vrz_PS = -self.Dz_PS*grad_ln_Nimp + self.Kz_PS*grad_ln_Ni + \
                       self.Hz_PS*grad_ln_Ti + self.Vrot_PS # PS radial flux per particle [m/s]
        
        
        # Banana-Plateau (BP)
        
        self.Dz_BP = (1.5*(self.qe*Ti)/(Zimp**2*self.qe**2*B0**2*R0**2*Nimp))*(1/(1/K11i + 1/K11z))# BP diffusion coefficient [m^2/s]
        self.Kz_BP = (Zimp/Zi)*self.Dz_BP # BP coefficient of the main ion density gradient [m^2/s]
        self.Hz_BP = ahbp*((Zimp/Zi)*(K12i/K11i - 1.5) - (K12z/K11z - 1.5))*self.Dz_BP # BP coefficient of the main ion temperature gradient [m^2/s]
        
        self.Vrot_BP = 0. # no centrifugal effects in BP, perhaps in the future?
        
        self.Vrz_BP = -self.Dz_BP*grad_ln_Nimp + self.Kz_BP*grad_ln_Ni + \
                       self.Hz_BP*grad_ln_Ti + self.Vrot_BP # BP radial flux per particle [m/s]
        
        
        # Classical
        
        self.Dz_CL = (CclG*2*eps2/CgeoG)*self.Dz_PS/(2*qmag**2) # CL diffusion coefficient [m^2/s]
        self.Kz_CL = (Zimp/Zi)*self.Dz_CL # CL coefficient of the main ion density gradient [m^2/s]
        self.Hz_CL = -(1.0 + (Zimp/Zi)*(C0z - 1.0))*self.Dz_CL # CL coefficient of the main ion temperature gradient [m^2/s]
        
        self.Vrot_CL = (CclR/CclG)*(Aimp/Ai)*(Machi2/R0**2)*\
                        (1.0-(Ai*Zimp)/(Aimp*Zi))*self.Dz_CL/amin # CL centrifugal velocity [m/s]
        
        self.Vrz_CL = -self.Dz_CL*grad_ln_Nimp + self.Kz_CL*grad_ln_Ni + \
                       self.Hz_CL*grad_ln_Ti + self.Vrot_CL # CL radial flux per particle [m/s]
        
        
        # Total
        
        self.Dz = self.Dz_PS + self.Dz_BP + self.Dz_CL # collisional diffusion coefficient [m^2/s]
        self.Kz = self.Kz_PS + self.Kz_BP + self.Kz_CL # coefficient of the main ion density gradient [m^2/s]
        self.Hz = self.Hz_PS + self.Hz_BP + self.Hz_CL # coefficient of the main ion temperature gradient [m^2/s]
        
        self.Vconv = self.Kz*grad_ln_Ni + self.Hz*grad_ln_Ti # convective velocity [m/s]
        
        self.Vrot = self.Vrot_PS + self.Vrot_BP + self.Vrot_CL # centrifugal velocity [m/s]
        
        self.Vrz = self.Vrz_PS + self.Vrz_BP + self.Vrz_CL
        
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
        return 1. - (1. - eps)**2/ (np.sqrt(1. - eps**2)*(1 + 1.46*np.sqrt(eps)))
        #return np.sqrt(2*eps)
    
    
    
    def C2(self, alpha, g, Za, Ai, Aa, f1, f2):
        '''
        Function related to the friction coefficient of the main ion parallel
        heat flow in the impurity-main ion parallel heat flow.
        New parametrization of HSW result.
        
        Refs: Hirshman-Sigmar NF (1981), https://doi.org/10.1088/0029-5515/21/9/003
              Wenzel-Sigmar NF (1990),   https://doi.org/0029-5515/30/6/013
              new paper...
        
        Parameters
        ----------
        alpha : float or 1D array
            impurity strength parameter, nz*Z**2/(ni*Zi**2) [-]
        g : float or 1D array
            collisionality parameter, ratio of main ion transit time to ion-ion
            collision time, q*R0/(vthi*Tauii) [-]
        Za : float or 1D array
            impurity charge [-]
        Ai : int or float
            main ion mass [-]
        Aa : int or float
            impurity mass [-]
        f1 : float or 1D array
            fitted factor (function of Za) [-]
            obtained from facs
        f1 : float or 1D array
            fitted factor (function of Za) [-]
            obtained from facs
        
        Returns
        -------
        C2 : float or 1D array
            value in C0z coefficient [-]
        '''

        return 1.5/(1+f1*(Ai/Aa)) - (0.29 + 0.68*alpha)/(0.59 + alpha + (1.34 + f2)*g**(-2))
    
    
    
    def ki_Redl(self, nui_star, ft, Zeff):
        '''
        Analytical expression for the neoclassical main ion flow coefficient
        
        Ref: Redl PoP (2021), https://doi.org/10.1063/5.0012664
        
        Parameters
        ----------
        nui_star : float or 1D array
            main ion collisionality [-]
        ft : float or 1D array
            trapped particle fraction [-]
        Zeff : float or 1D array
            effective plasma charge [-]
        
        Returns
        -------
        ki : float or 1D array
            neoclassical main ion flow coefficient [-]
        '''

        alpha0 = -(0.62+0.055*(Zeff-1))*(1-ft)/((0.53+0.17*(Zeff-1))*\
                  (1-(0.31-0.065*(Zeff-1))*ft - 0.25*ft**2))
            
        return ((alpha0 + 0.7*Zeff*np.sqrt(ft*nui_star))/(1+0.18*np.sqrt(nui_star))-\
                 0.002*nui_star**2*ft**6)/(1+0.004*nui_star**2*ft**6)
    
    
    
    
    def facs(self, Z, ft):
        '''
        Fitted factors for FACIT model
        
        Ref: new paper...
        
        Parameters
        ----------
        Z : float or 1D array
            impurity charge [-]
        ft : float or 1D array
            trapped particle fraction [-]
        
        Returns
        -------
        f1 : float or 1D array
            factor in the C2 function [-]
        f2 : float or 1D array
            factor in the C2 function [-]
        f3 : float or 1D array
            factor of the ion-electron collisional heat exchange term in C0z [-]
        y1 : float or 1D array
            factor of the (1,1) banana viscosity coefficient of the impurity [-]
        y2 : float or 1D array
            factor of the (1,2) banana plateau coefficient of the impurity [-]
        y3 : float or 1D array
            factor of the (1,2) Pfirsch-Schlüter viscosity coefficient of the impurity [-]
        y4 : float or 1D array
            factor of the (1,2) banana viscosity coefficient of the main ion [-]
        adps : float or 1D array
            factor of the PS diffusion coefficient [-]
        ahbp : float or 1D array
            factor of the BP diffusion coefficient [-]
        '''
        
        # f1, f2, f3 factors in C0z
        
        f1 = (-6.83808564e+05 + 2.46855534e+06*Z)/( 1 + 6.04692708e+05*Z**1.61470425)
        f2 = (88.28389935 + 10.50852772*Z)/( 1 + 0.2157175*Z**2.57338463)
        f3 = (-4.45719179e+06 + 2.72813878e+06*Z)/(1+5.26716920e+06*Z**8.33610108e-01)
        
        # y1, y2, y3, y4 factors in viscosity coefficients
        
        # auxiliary coefficients for y1
        w11 = 1.23805214e-05*Z**3 - 1.03611576e-03*Z**2 + 1.85221287e-02*Z + 1.29758029
        w12 = -5.84996779e-05*Z**3 + 4.79623045e-03*Z**2 -1.01924030e-01*Z - 5.81816797
        w13 = 5.98152997e-05*Z**3 - 4.84255587e-03*Z**2 + 1.03585208e-01*Z + 5.71139227
        w14 = -1.33253954e-05*Z**3 + 1.04416741e-03*Z**2 - 1.94973421e-02*Z - 1.10061492
        
        y1 = w11*ft**2 + w12*ft + w13*ft**0.5 + w14
        
        y2 = 21.31*ft**3 - 21.88*ft**2 + 7.316*ft + 0.6264
        
        y3 = ((4.2857e5 - 4.4978e5*Z)/(1 - 1.3557e5*Z**1.9))*\
             ((-9.09204093e+06 + 8.15802759e+06*Z)/(1 + 3.25263641e+06*Z**1.07640221))
             
        # auxiliary coefficients for y4
        wp1 = ( 0.11603574 + 0.47297835*Z**0.94456671)/( 1 + 0.1245426*Z**1.20221441)
        wp2 = ( 0.6327602 - 2.92116611*Z**1.06310182)/( 1 + 0.30469549*Z**1.16495283)
        wp3 = ( -0.69318477 + 2.85619511*Z**1.11765611)/( 1 + 0.34127361*Z**1.1952383)
        wp4 = ( 1.80217558 - 1.0080554*Z**0.96345934)/( 1 + 0.51677339*Z**1.11131896)
        
        yp = wp1*ft**2 + wp2*ft + wp3*ft**0.5 + wp4
        
        y4 = yp/y1
        
        # factor of the PS diffusion coefficient
        adps = (-110378.34909954367 + 753838.9265706746*Z**1.0610783384110476)/(1+ 1007273.2273681393*Z)
        
        # factor of the BP diffusion coefficient
        ahbp = (1.01579172e+00 + -1.78923911e-03*Z)/(1 + 6.60170647e-13*Z**6.66398825e+00)
        
        return f1, f2, f3, y1, y2, y3, y4, adps, ahbp
    
    
    
    def KVISC(self, nimp, ni, ti, Ai, Aimp, Zi, Zimp, \
                    Tauii, Tauimpimp, Tauiimp, Tauimpi, \
                    eps, ft, R0, qmag, y1, y2, y3, y4):
        '''
        Calculates the viscosity coefficients for the BP flux analytically
        
        Refs: Hishman-Sigmar NF (1981),           https://doi.org/10.1088/0029-5515/21/9/003
              Wenzel-Sigmar NF (1990) Appendix A, https://doi.org/0029-5515/30/6/013
              new paper...
        
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
        y1 : 1D array
            factor of the (1,1) banana viscosity coefficient of the impurity [-]
            from facs(Z,ft) method
        y2 : 1D array
            factor of the (1,2) banana plateau coefficient of the impurity [-]
        y3 : 1D array
            factor of the (1,2) Pfirsch-Schlüter viscosity coefficient of the impurity [-]
        y4 : 1D array
            factor of the (1,2) banana viscosity coefficient of the main ion [-]
        
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
        
        
        # total: rational interpolation across collisionality regimes
        
        K11a = y1*K11aB/((1+y1*K11aB/(K11aP))*(1+K11aP/(K11aPS)))
        K12a = K12aB/((1+K12aB/(y2*K12aP))*(1+y2*K12aP/(y3*K12aPS)))
        
        K11i = K11iB/((1+K11iB/(K11iP))*(1+K11iP/(K11iPS)))
        K12i = y4*K12iB/((1+y4*K12iB/(K12iP))*(1+K12iP/(K12iPS)))
    
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
    
    
    
    def GEOM(self, full_geom, nth, rho, eps, B0, BV, RV, JV):
        '''
        Calculates some geometric parameters used in FACIT
        
        Parameters
        ----------
        full_geom : bool
            if True, use full flux surface shaped geometry
            if False, use cicular geometry with large aspect ratio
        nth : int
            number of poloidal grid points
        rho : 1D array (nr)
            normalized radial coordinate [-], mid-plane radius over minor radius r/a
        eps : 1D array (nr)
            local inverse aspect ratio [-]
        B0 : float
            magnetic field at magnetic axis [T]
        BV : 2D array (nr, nth)
            magnetic field, poloidal distribution on (r,theta) [T]
            only relevant if full_geom
            else, it can be set to None
        RV : 2D array (nr, nth)
            major radius, poloidal distribution on (r,theta) [m]
            only relevant if full_geom
            else, it can be set to None
        JV : 2D array (nr, nth)
            Jacobian of coordinate system, poloidal distribution on (r,theta) [m/T]
            only relevant if full_geom
            else, it can be set to None
        
        Returns
        -------
        theta : 1D array (nth)
            poloidal grid [-]
        b2 : 2D array (nr, nth)
            poloidal distribution of the magnetic field [-], b^2 = B^2/<B^2>
        invb2avg : 1D array (nr)
            <1/b^2> [-]
        gradR2 : 2D array (nr, nth)
            gradient of major radius squared [m^2]
        gradR2avg : 1D array (nr)
            <grad R^2>
        '''
        
        theta = np.linspace(0, 2*np.pi, nth) # poloidal grid
        
        if full_geom: # full flux surface shaped geometry
            
            b2 = BV**2/self.fluxavg(BV**2, JV)[:,None] # asymmetry of magnetic field: B^2/<B^2>
            
            invb2avg = self.fluxavg(1/b2, JV) # <1/b2>
            
            gradR2    = np.gradient(RV**2, rho, axis=0) # gradient of major radius squared
            gradR2avg = self.fluxavg(gradR2, JV)          # <dr R^2>
            
        else: # circular geometry in large aspect ratio
            
            eps2 = eps**2 # auxiliary
            
            b2 = (1 - eps[:,None]*np.cos(theta)[None,:])**2/\
                  (1 + 0.5*eps2[:,None]) # asymmetry of magnetic field: B^2/<B^2>
        
            invb2avg = 1 + 2*eps2 # <1/b2>
            
            gradR2    = 0.0
            gradR2avg = 0.0
            
        return theta, b2, invb2avg, gradR2, gradR2avg
    
    
    
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
        eps : 1D array
            local inverse aspect ratio [-]
        Zeff : 1D array
            effective plasma charge [-]
        Zi : int or float
            main ion charge [-]
        Te_Ti : 1D array
            electron to main ion temperature ratio [-]
        Machi2 : 1D array
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
        
        AsymPhi = np.zeros((2, rho.size)) # asymmetry in electrostatic potential [-]
        AsymN   = np.zeros((2, rho.size)) # asymmetry in main ion density [-]
    
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
        
        RR = 0.5*(1 - (Ai*Zimp)/(Aimp*Zi))/rho
        UG = 1 + UU/GG
        
        if nat_asym:
            Ae = nuswcz*qmag**2/invaspct # ?: nuswcz*qmag2/eps
        else:
            Ae = 0
        
            
        CD0 = -eps/UG
        #CD0 = -invaspct/UG
        
        AGeC = Ae*GG*eps/CD0
        #AGeC = Ae*GG*invaspct/CD0
        
        HH = 1. + deltaM*CD0*RR/GG
        
        QQ = CD0*(dNH/eps)*(UG-1)
        #QQ = CD0*(dNH/invaspct)*(UG-1)
        
        FF = CD0*(1 - 0.5*(dNV/eps)*(UG-1) - (deltaM/eps)*RR/GG)
        #FF = CD0*(1 - 0.5*(dNV/invaspct)*(UG-1) - (deltaM/invaspct)*RR/GG)
        
        KK = 1.
        hh_kk = HH * KK * AGeC**2

        CD = FF/KK - 0.5*(dminphia - deltaM)
        CDV = -0.5*(dmajphia + QQ/HH)

        RD = np.sqrt((FF/KK + 0.5*(dminphia - deltaM))**2 + 0.25*(HH/KK)*(dmajphia - QQ/HH)**2)
        #DD = RD**2 + hh_kk*RD**2
        DD = RD**2*(1 + hh_kk)

        num = (hh_kk - 1.)*(FF/(KK*CD0) + 0.5*(dminphia - deltaM)/CD0) +\
              AGeC*(0.5*dNV*(UG-1)/(eps) - 0.5*HH*dmajphia/CD0)
        #num = (hh_kk - 1.)*(FF/(KK*CD0) + 0.5*(dminphia - deltaM)/CD0) +\
        #      AGeC*(0.5*dNV*(UG-1)/(invaspct) - 0.5*HH*dmajphia/CD0)
        
        cosa = RD*CD0*num/DD

        #num = 2.*AGeC*(FF + 0.5*KK*(deltaPhia-deltaM)) + \
        #      (hh_kk - 1.)*(0.5*DeltaPhia - 0.5*DeltaN*CD0*UU/(cdat.epsilon*GG*HH))
              
        num = 2.*AGeC*(FF + 0.5*KK*(dminphia-deltaM)) + \
              (hh_kk - 1.)*(0.5*dmajphia - 0.5*dNV*(UG-1)*CD0/(eps*HH))
        #num = 2.*AGeC*(FF + 0.5*KK*(dminphia-deltaM)) + \
        #      (hh_kk - 1.)*(0.5*dmajphia - 0.5*dNV*(UG-1)*CD0/(invaspct*HH))
        
        sina = RD*num*np.sqrt(HH/KK)/DD
        
        deltan = CD + RD*cosa
        Deltan = CDV + RD*np.sqrt(KK/HH)*sina
        
        nn = 1 + np.outer(deltan, np.cos(theta)) + np.outer(Deltan, np.sin(theta))
        
        return deltan, Deltan, nn
    
    
    
    def asymmetry_iterative(self, regulopt, nr, theta, GG, UU, Ai, Aimp, Zi, Zimp, Te_Ti, \
                                  Machi2, R0, nuz, BV, RV, JV, FV, dpsidx, AsymPhi, AsymN, \
                                  b2, gradR2, gradR2avg, nat_asym):
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
            from GEOM method
        gradR2 : 2D array (nr, nth)
            gradient of major radius squared [m^2]
        gradR2avg : 1D array (nr)
            <grad R^2>
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
        Factrot  = Factrot0*(1-(Ai*Zimp)/(Aimp*Zi)) #pre-factor to gradR2 in toroidal rotation term
            
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

                FFF  = Apsi[ix]*( GG[ix] + (b2[ix]/NV[ix])*UU[ix] - Factrot[ix]*gradR2[ix])
                GGG  = -Zimp[ix]*(Te_Ti[ix])*(PhiV[ix]-PhiV[ix,0]) + \
                        Factrot0[ix]*(RV[ix]**2-RV[ix,0]**2)
                HHH  = Apsi[ix]*(b2[ix]/b2snavg)*(GG[ix] + b2sNNavg[ix]*UU[ix] - Factrot[ix]*gradR2avg[ix])
                
                
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










if __name__ == '__main__':
    
    action = input('[c]ompare , [t]ime analysis: ')
    
    if action == 'c':
        
        import sys
        sys.path.append('/afs/ipp/home/d/dfajardo/coll_transp')
        
        import matplotlib.pyplot as plt
    
        from data_colltr import DATA_IMPFLUX
        
        cdat = DATA_IMPFLUX(exp='AUG', nshot=38910, time=2.15, fimp = 1e-7, Zimp = 'W', Aimp = 184)
        impf = FACIT(cdat.rho, cdat.Zimp, cdat.Aimp, cdat.Zi, cdat.Ai, \
                     cdat.te, cdat.ti, cdat.ne, cdat.ni, cdat.nimp, 0.0, \
                     cdat.invaspct, cdat.B0, cdat.R0, cdat.q)
            
        plt.plot(cdat.rho, impf.Vrz)
            
            
    elif action == 't':
        
        import time
        
        # array of poloidal points for convergence
        thconv = np.array([2,4,8,16,20,26,38,48,96,192])
        # array of poloidal points for execution time
        thexec = np.array([2,4,8,16,20,26,32,48])
        
        dminvec = np.zeros(thconv.size)
        timexec = np.zeros(thexec.size)
        
        # build profiles myself
        
        Te0 = 3.0e3
        tedge = 10.
        tau = 0.5
        Ni0 = 4.0e19
        nedge = 1.0e17
        invaspct = 1/3
        B0 = 3.5851
        R0 = 2.5852
        Ai = 2.
        Zi = 1.
        Aa = 184.
        Za = 44.*np.ones(101)
        qa = 4.
        cimp = 1.0e-4
        amin  = R0*invaspct
    
        fH = 0.05
        bC = 1.0
        Zeff = Zi
        sigH = 0.3
        nsigH = 2.
        TperpsTpar_axis = 10.
    
        Machi_axis = 0.7
        
        nx = 101
    
        xn = np.maximum(np.linspace(0,1,nx),1e-6)
        Te = (Te0-tedge)*(1-xn**2)**2 + tedge
        Ti = tau*Te

       	Ni = (Ni0-nedge)*(1-xn**2)+nedge
       	Na = cimp*Ni
       	Ne = Zi*Ni + Za*Na
  
       	Machi = Machi_axis*np.ones_like(xn)
        qmag = 1+(qa-1)*xn**2
           
       	dpsidx = amin**2*B0*xn/qmag
       	
       	FV = R0*B0*np.ones_like(xn)
        
        
        for i, nth in enumerate(thconv):
            
            theta = np.linspace(0,2*np.pi,nth)

            RV = R0*(1.+invaspct*xn[:,None]*np.cos(theta[None,:]))
            BV = B0/(1.+invaspct*xn[:,None]*np.cos(theta[None,:]))
            
            jacob = amin**2*R0*xn[:,None]*np.ones_like(theta)[None,:]
        
            impf = FACIT(xn, Za, Aa, Zi, Ai, \
                         Te, Ti, Ne, Ni, Na, Machi, \
                         invaspct, B0, R0, qmag,\
                         pol_asym = True, full_geom=True, \
                         nth=nth, dpsidx=dpsidx, FV=FV,\
                         BV=BV, RV=RV, JV=jacob, fH=fH, bC=bC, \
                         sigH=sigH, TperpTpar_axis = TperpsTpar_axis,
                         nat_asym=True)
                
            dminvec[i] = impf.horiz_asym[50]
            
            print('\n convergence, nth = %d \n' %nth)
                
                
        for i, nth in enumerate(thexec):
            
            theta = np.linspace(0,2*np.pi,nth)
            
            RV = R0*(1.+invaspct*xn[:,None]*np.cos(theta[None,:]))
            BV = B0/(1.+invaspct*xn[:,None]*np.cos(theta[None,:]))
            
            jacob = amin**2*R0*xn[:,None]*np.ones_like(theta)[None,:]
            
            starttime = time.time()
            
            impf = FACIT(xn, Za, Aa, Zi, Ai, \
                         Te, Ti, Ne, Ni, Na, Machi, \
                         invaspct, B0, R0, qmag,\
                         pol_asym = True, full_geom=True, \
                         nth=nth, dpsidx=dpsidx, FV=FV,\
                         BV=BV, RV=RV, JV=jacob, fH=fH, bC=bC, \
                         sigH=sigH, TperpTpar_axis = TperpsTpar_axis,
                         nat_asym=True)
            
            timexec[i] = time.time()-starttime
            
            print('\n execution, nth = %d \n' %nth)
            
        # other FACIT setups
        
        # circ geom
        starttime = time.time()
        
        impf = FACIT(xn, Za, Aa, Zi, Ai, \
                     Te, Ti, Ne, Ni, Na, Machi, \
                     invaspct, B0, R0, qmag,\
                     pol_asym = True, full_geom=False, \
                     fH=fH, bC=bC, \
                     sigH=sigH, TperpTpar_axis = TperpsTpar_axis,
                     nat_asym=True)
            
        time_circ = time.time()-starttime
        
        
        #polsym
        
        starttime = time.time()
        
        impf = FACIT(xn, Za, Aa, Zi, Ai, \
                     Te, Ti, Ne, Ni, Na, Machi, \
                     invaspct, B0, R0, qmag,\
                     pol_asym = False, full_geom=False)
            
        time_sym = time.time()-starttime
        
        
        
        print('dminvec: ', dminvec, '\n', 'timexec: ', timexec, '\n', 'time_circ: ', time_circ, '\n', 'time_sym: ', time_sym)
    
