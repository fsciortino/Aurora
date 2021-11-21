"""
Functionality to create Plasma OPeration CONtour (POPCON) plots, based on the work of the 22.63 MIT/Princeton class of fall 2020.
"""
import sys
import numpy as np
from scipy import constants
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import aurora

class POPCON():
    def __init__(self, R=3.7, a=1.14, kappa=1.75, B0=12.3, Ip=17.4, H=1.15, M_i=2.5,
                 fHe=0.02, imp='Xe', impfrac=0.0005, imcharg=42,
                 f_LH=0.9, fixed_quantity='impfrac', Psol_target=20, 
                 T_edge=0.1, Talpha1=2., Talpha2 = 1.5,
                 n_frac_edge = 0.4, nalpha1 = 2, nalpha2 = 1.5,
                 jalpha1 = 2, jalpha2 = 1.5,
                 override = False, volavgcurr = False, maxit = 150,
                 relax = 0.9, nmin_frac = 0.05, nmax_frac = 1, Ti_min = 1.,
                 Ti_max = 30, M = 40):
        '''Initialization of the POPCON class.

        Parameters
        ----------
        R : float
            Major Radius (m)
        a : float
            Minor Radius (m)
        kappa : float
            Elongation
        B0 : float
            Toroidal magnetic field on axis (T)
        Ip : float
            Plasma Current (MA)
        H : float
            H-factor (confinement quality)
        M_i : float
            Reduced ion mass, in atomic units (2.5 for D-T)
        fHe : float
            Helium Ash Fraction
        imp : str
            Impurity atomic symbol, e.g. 'Xe'
        impfrac : float
            Impurity fraction.
        imcharg : int
            Average charge of chosen impurity. TODO: generalize to local charge states.
        f_LH : float
            Target f_LH = P_sol/P_LH
        fixed_quantity : str
            Quantity to maintain fixed in POPCON, one of 'impfrac', 'f_LH', 'P_SOL'
        Psol_target : float
            Target power going into the SOL

        Parameters for kinetic profiles:
        T_edge : float
            Edge temperature [keV]
        Talpha1 : float
            alpha1 parameter for the temperature profile
        Talpha2 : float
            alpha2 parameter for the temperature profile
        n_frac_edge : float
            n_edge/n_center
        nalpha1 : float
            alpha1 parameter for the density profile
        nalpha2 : float
            alpha2 parameter for the density profile
        jalpha1 : float
            alpha1 parameter for the current density profile
        jalpha2 : float
            alpha2 parameter for the current density profile

        Debug/algorithm settings:
        override : bool
            Set to True to allow violation of stability conditions
        volavgcurr : bool
            Set to True to assume constant current
        maxit : int
            Maximum number of iterations for solver
        relax : float
            Relaxation parameter in power balance solver, e.g. 0.9
        nmin_frac : float
            Min density, as a fraction of n_GW
        nmax_frac : float
            Max density, as a fraction of n_GW
        Ti_min : float
            Min temperature [keV]
        Ti_max : float
            Max temperature [keV]
        M : int
            Number of n, T points for which the POPCON solver is run
        
        Notes
        -----
        Kinetic profiles are parametrized with alpha1, alpha2, such that
        f = (f_center-f_edge)*(1-rho**alpha1)**alpha2 + f_edge.

        '''
        self.R       = float(R)
        self.a       = float(a)
        self.kappa   = float(kappa)
        self.B0      = float(B0)
        self.Ip      = float(Ip)
        self.H       = float(H)
        self.M_i     = float(M_i)
        self.fHe     = float(fHe)
        self.imp     = str(imp)
        self.impfrac = float(impfrac)
        self.imcharg = int(imcharg)
        self.f_LH    = float(f_LH)
        self.fixed_quantity = str(fixed_quantity)
        if self.fixed_quantity not in ['impfrac', 'f_LH', 'P_SOL']:
            raise ValueError('Unrecognized option for fixed_quantity input')
        self.Psol_target = float(Psol_target)
        
        # f = (f_center-f_edge)*(1-rho**alpha1)^alpha2 + f_edge
        self.T_edge  = float(T_edge)
        self.Talpha1 = float(Talpha1)
        self.Talpha2 = float(Talpha2)
        self.n_frac_edge = float(n_frac_edge)
        self.nalpha1 = float(nalpha1)
        self.nalpha2 = float(nalpha2)
        self.jalpha1 = float(jalpha1)
        self.jalpha2 = float(jalpha2)

        # debug/algorithm settings
        self.override   = bool(override)
        self.volavgcurr = bool(volavgcurr)
        self.maxit      = int(maxit)
        self.relax      = float(relax)
        self.nmin_frac  = float(nmin_frac)
        self.nmax_frac  = float(nmax_frac)
        self.Ti_min      = float(Ti_min)
        self.Ti_max      = float(Ti_max)
        self.M          = int(M)

        # cache ADAS atomic data - effective ionization and recombination rates
        self.atom_data = aurora.get_atom_data(self.imp)
        
    # Externally interfacing routines
    #--------------------------------------------------------------------------
    def make_popcons(self):
        ''' 
        Externally called routine that holds master logic for reading inputs
        and making POPCONS
        '''
        self._process_inputs()
        self._print_parameters()

        self.auxp   = np.empty([self.M,self.M])   # auxiliary power (RF, NBI, ...)
        self.pamat  = np.empty([self.M,self.M])   # alpha power
        self.prmat  = np.empty([self.M,self.M])   # radiated power
        self.tratio = np.empty([self.M,self.M])   # TODO: allow Ti != Te
        self.plmat  = np.empty([self.M,self.M])   # power loss, i.e. alpha + auxiliary powers
        self.qpmat  = np.empty([self.M,self.M])   # Q=P_fus/(P_aux+P_oh)
        self.pohmat = np.empty([self.M,self.M])   # Ohmic power
        self.beta   = np.empty([self.M,self.M])   # TODO compute 2*mu0*p/B^2
        self.flh    = np.empty([self.M,self.M])   # LH fraction, i.e. (P_loss - P_rad) / P_{LH threshold}
        self.impfrc = np.empty([self.M,self.M])   # impurity fraction
        self.Vloop  = np.empty([self.M,self.M])   # Vloop
        self.BetaN  = np.empty([self.M,self.M])   # Normalized Beta

        self.xv     = np.empty(self.M)            # array to save T_i vals for plotting
        self.yv     = np.empty(self.M)            # array to save <n>/n_GR vals for plotting

        # set up list of Ti and n for which POPCON will evaluate power balance
        n_norm  = integrate.quad(self._nfun,0,1,args=(1))[0] # defined s.t. n_avg = n_peak*n_norm
        n20list = np.linspace(self.nmin_frac*self.n_g/n_norm, self.nmax_frac*self.n_g/n_norm, self.M)
        Tilist  = np.linspace(self.Ti_min,self.Ti_max,self.M)

        print("\n### Solve power balance at each (n,T)")
        for i in np.arange(self.M): #T_i loop
            Ti = Tilist[i]
            self.xv[i]=Ti
            
            for j in np.arange(self.M): #n_e loop
                n20 = n20list[j]
                self.yv[j] = n20

                # compute normalised Prad for our impurity to quickly obtain impurity fraction from Prad
                Prad_norm = self.dvolfac*integrate.quad(self._P_rad,0,1,args=(n20, Ti, 1, self.B0, self.a, self.R),limit=200)[0]

                # call solver
                _out = self._auxpowIt(n20,Ti,Prad_norm)
                self.auxp[i,j],self.plmat[i,j],self.pamat[i,j], self.prmat[i,j],self.qpmat[i,j], self.pohmat[i,j] = _out

                # L-H power threshold
                self.flh[i,j]= (self.plmat[i,j]-self.prmat[i,j]) / get_P_LH_threshold(n20, self.bs)

                # impurity fraction
                self.impfrc[i,j] = self.prmat[i,j] / Prad_norm

                # Loop voltage
                self.Vloop[i,j] = self._vloopfun(n20, Ti, self.impfrc[i,j])
                
                # Normalize beta [%]
                self.BetaN[i,j] = 4 * (4*np.pi*1e-7) * (n20*1e20) * (Ti*1000*1.6021e-19) \
                    * self.a * 100 / self.B0 / self.Ip

                # set ignition curve
                if(self.qpmat[i,j]>=10000.0 or self.qpmat[i,j]<0.0):
                    self.qpmat[i,j]=10000

                # debug prints
                if(self.M < 4):
                    print('For n20= ', n20,', Ti= ',Ti)
                    print('P_aux= ', self.auxp[i,j], ' P_alpha= ', self.pamat[i,j], \
                          'P_rad= ', self.prmat[i,j], ' P_ohm = ',self.pohmat[i,j])

            # progress bar for overview
            printProgressBar(i+1, self.M, prefix = 'Progress:', length = 50)

    def plot_popcons(self, save_filename='popcon.pdf'):
        ''' Plot the popcons and, optionally, save them to a PDF file. 
        
        Parameters
        ----------
        save_filename : str or None
            If set to a string, the POPCON is saved to a file named according to this argument.
        '''
        plt.rcParams.update({'font.size': 18})
        if self.M > 3:
            self.plot_contours()
        else:
            self.plot_profiles()

        if save_filename is not None:
            self.fig.savefig(save_filename)

    def plot_contours(self, plot_Paux=True, plot_Pfus=True, plot_Q=True, plot_Prol=True,
                      plot_Psol=True, plot_Pohm=True, plot_f_LH=True, plot_impf=True,
                      plot_Vloop=True, plot_BetaN=True, plot_Pload_fus=True, plot_Pload_rad=True):
        ''' Plot popcon contours. Various boolean flags allow specification of which contours 
        should be plotted.
        '''
        n_norm = integrate.quad(self._nfun,0,1,args=(1))[0] # defined s.t. n_avg = n_peak*n_norm
        xx, yy = np.meshgrid(self.xv, self.yv/self.n_g*n_norm)
        
        self.fig, ax = plt.subplots(figsize=(16,9))
        
        Pauxlevels = [0,1,3,5,7,10,15,20,25]
        Pfuslevels = [30,50,100,300,500,1000,2000,3000]
        Qlevels    = [10]
        Prad_oloss_levels = [0.1, 0.5, 0.9, 0.99]
        Psollevels = [10, 20, 30, 40]
        Pohmlevels = [1]
        flh_levels = [0.1, 0.2, 0.5]
        Vlooplevels = [1e-2, 1.3e-2, 1.5e-2, 2e-2, 3e-2, 4e-2, 5e-2, 7e-2, 9e-2, 1.2e-1, 1.5e-1]
        Pload_fus_levels = ""
        Pload_rad_levels = ""
        BetaN_levels = [3]
        
        Igshade=plt.contourf(xx,yy,np.transpose(self.qpmat),
                             levels=[6000,10000],colors='r',alpha=0.25)
        
        if plot_Paux: plot_single_contour(ax, "$P_\mathrm{aux}$",
                                          xx, yy, self.auxp, Pauxlevels, 'r', 1.0, fmt='%1.0f')
        if plot_Pfus: plot_single_contour(ax, "$P_\mathrm{fus}$",
                                          xx, yy, self.pamat*5.03, Pfuslevels, 'k', fmt='%1.0f')
        if plot_Q: plot_single_contour(ax, "Q",
                                       xx,yy,self.qpmat, Qlevels, 'lime', fmt='%1.0f')
        if plot_Prol: plot_single_contour(ax, "$P_\mathrm{rad}/P_{loss}$",
                                          xx,yy,self.prmat/self.plmat,Prad_oloss_levels, 'c', 1.0, fmt='%1.2f')
        if plot_Psol: plot_single_contour(ax, "$P_\mathrm{sol}$",
                                          xx, yy, self.prmat-self.plmat, Psollevels, 'darkviolet', 1.0, fmt='%1.0f')
        if plot_Pohm: plot_single_contour(ax, "$P_\mathrm{ohm}$",
                                        xx, yy, self.pohmat, Pohmlevels, 'orange', 1.0, fmt='%1.0f')
        if plot_f_LH: plot_single_contour(ax, "$f_\mathrm{LH}$",
                                          xx,yy,self.flh,flh_levels, 'forestgreen', 1.0, fmt='%1.2f')
        if plot_impf: plot_single_contour(ax, "$n_\mathrm{imp}/n_e$",
                                          xx,yy,self.impfrc,"", 'magenta', 1.0, fmt='%0.1e')
        if plot_Vloop: plot_single_contour(ax, "$V_\mathrm{loop}$",
                                           xx, yy, self.Vloop,Vlooplevels, 'mediumvioletred', 1.0, fmt = '%0.2e')
        if plot_BetaN: plot_single_contour(ax, "$\u03B2_\mathrm{N}$",
                                           xx, yy, self.BetaN,BetaN_levels, 'darkslategrey', 1.0, fmt = '%0.2e')
        
        # Plot power loads (MW/m^2) on wall. For area, use plasma surface area as rough estimate
        area = 2*np.pi*self.a * np.sqrt((1+self.kappa**2)/2) * 2*np.pi*self.R
        if plot_Pload_fus: plot_single_contour(ax, "$P_\mathrm{n}/A$",
                                               xx, yy, self.pamat*4/area, Pload_fus_levels, 'peru', 1.0, fmt = '%0.2e')
        if plot_Pload_rad: plot_single_contour(ax, "$P_\mathrm{rad}/A$",
                                               xx, yy, self.prmat/area, Pload_rad_levels, 'dodgerblue', 1.0, fmt = '%0.2e')
        
        ax.set_ylabel(r"$\bar{n} / n_{GR}$")
        ax.set_xlabel('$T^i_\mathrm{keV}$')
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left',fontsize = 12)
        plt.tight_layout()
        

    def plot_profiles(self):
        ''' Plot 1D profiles
        '''
        self.fig = plt.figure(figsize=(12,9))
        for i_temp in np.arange(self.M):
             T0 = self.xv[i_temp]
             for i_dens in np.arange(self.M):
                  n20 = self.yv[i_dens]

                  ax1 = plt.subplot(self.M, self.M, i_dens*self.M+i_temp+1)

                  plt.subplots_adjust(wspace=None, hspace=None)

                  plt.title(r"$n_0/n_{GR} =$ %.1f, $T^0_\mathrm{keV} =$ %.0f" % (n20/get_n_GR(self.Ip, self.a), T0), pad=-100)

                  rho = np.linspace(1e-10, 1, 100)
                  Prad = np.empty(len(rho))
                  Pfus = np.empty(len(rho))
                  Pohm = np.empty(len(rho))
                  for i_rho in np.arange(len(rho)):
                      Prad[i_rho] = self._P_rad(rho[i_rho], n20, T0, self.impfrc[i_temp, i_dens], self.B0, self.a, self.R)/rho[i_rho]
                      Pfus[i_rho] = self._P_alpha(rho[i_rho], n20, T0, self.impfrc[i_temp, i_dens])/rho[i_rho]
                      Pohm[i_rho] = self._P_OH(rho[i_rho], n20, T0, self.impfrc[i_temp,i_dens])/rho[i_rho]

                  plt.plot(rho, Prad, label="$P_\mathrm{rad}$")
                  plt.plot(rho, Pfus, label="$P_\mathrm{fus}$")
                  plt.plot(rho, Pohm, label="$P_\mathrm{ohm}$")

                  plt.grid()
                  plt.xlim([0,1])

                  if i_temp == 0:
                      plt.ylabel(r"$P$ in MW/m$^3$")
                  if i_dens == self.M-1:
                      plt.xlabel(r"$\rho$")
                  if i_dens == 0 and i_temp == 0:
                      plt.legend()


    # Internal Routines
    #--------------------------------------------------------------------------
    def _process_inputs(self):
        ''' Calculate some derived quantities including:
        - Plasma Volume (vol)
        - Plasma Dilution (dil)
        - Z Effective (Zeff)
        - Greenwald Limit (n_g)
        - Spitzer Resistivity Coefficient (Cspitz)
        - B0*Surface Area (bs) with exponential scalings
        '''
        # plasma volume
        self.vol = get_plasma_vol(self.a, self.R, self.kappa)

	# volume of drho element for integration
        self.dvolfac = get_plasma_dvolfac(self.a, self.R, self.kappa)

        # plasma dilution
        self.dil = get_plasma_dilution(self.impfrac, self.imcharg, self.fHe)

        # Z effective, effective ion charge
        self.Zeff = get_Zeff(self.fz_imp, self.impfrac, self.imcharg, self.fHe)

        # n_g, Greenwald density (10^20 m^-3)
        self.n_g = get_n_GR(self.Ip, self.a)

        # q_a edge safety factor
        self.q_a = get_q_a(self.a, self.B0, self.kappa, self.R, self.Ip)
        if (self.q_a <= 2.0 and self.override == False):
            sys.exit('q_a too low. External kink unstable')

        # Spitzer resistivity coef for ohmic heating power
        self.Cspitz = get_Cspitz(self.Zeff, self.Ip, self.q_a, self.a, self.kappa, self.volavgcurr)

        # BS from Martin H Mode Scaling (Martin et al J. Phys 2008)
        self.bs = get_bs_factor(self.B0, self.R, self.a, self.kappa)

        # H89_base = H89/n20**(0.1) for convenience as n20 varies in POPCON
        self.H89_base = get_ITER_H89_base(self.H,self.M_i,self.Ip,self.R,self.a,self.kappa,self.B0)

        # Peak current density, from integral(j) = Ip
        self.j0 = self.Ip*1e6 / (2*np.pi*self.a**2*self.kappa) / integrate.quad(self._ijfun,0,1,args=(1))[0]


    # Iterative solvers for power balance.
    #--------------------------------------------------------------------------
    def _auxpowIt(self, n20, Ti, Prad_norm):
        ''' Iterate to get the aux power needed for power balance
        '''
        accel = 1.7
        pderror = 1e-5

        # Alpha power density in units of 10^20keV
        Ptalpha = self.dvolfac*integrate.quad(self._P_alpha,0,1,args=(n20,Ti,self.impfrac))[0]

        # Total thermal energy in plasma
        Wtot = self.dvolfac*integrate.quad(self._Wtot,0,1,args=(Ti,n20,self.impfrac))[0]

        # Volume averaged Ptoh
        if (self.volavgcurr == True):
            navg = self.dvolfac*integrate.quad(self._infun,0,1,args=(n20))[0]/self.vol
            Tavg = self.dvolfac*integrate.quad(self._iTfun,0,1,args=(Ti))[0]/self.vol
            Ptoh = self.vol*self._P_OH_avg(navg,Tavg)
        else:
            Ptoh = self.dvolfac*integrate.quad(self._P_OH,0,1,args=(n20,Ti,self.impfrac))[0]

        # iterative solve for equilibrium solution to power balance
        Ph = 0.0
        for i in np.arange(self.maxit):
            # Total heating power
            Pheat = Ptoh + Ph + Ptalpha

            # Radiated power
            if (self.fixed_quantity == "impfrac"):
                Prad = self.dvolfac*integrate.quad(self._P_rad,0,1,args=(n20,Ti,self.impfrac, self.B0, self.a, self.R))[0]

            elif (self.fixed_quantity == "f_LH" or self.fixed_quantity == "Psol"):
                if self.fixed_quantity == "f_LH":
                    # Set Prad to satisfy f_LH target specified, but keep > 0.
                    Prad = max(0, Pheat - self.f_LH * get_P_LH_threshold(n20, self.bs))

                elif self.fixed_quantity == "Psol":
                    # Set Prad to bring Psol < Psol_target
                    Prad = max(0, Pheat - self.Psol_target)

                # Infer impurity fraction necessary to get this Prad
                self.impfrac = Prad / Prad_norm
                
                # Update dilution and Zeff with the new impurity fraction
                self.dil = get_plasma_dilution(self.impfrac, self.imcharg, self.fHe)
                self.Zeff = get_Zeff(self.fz_imp, self.impfrac, self.imcharg, self.fHe)

            else:
                sys.exit("Choose fixed quantity for radiation: impfrac, f_LH, Psol.")

            # Recalculate powers to account for changed dilution, Zeff, ...
            Ptalpha = self.dvolfac*integrate.quad(self._P_alpha,0,1,args=(n20,Ti,self.impfrac))[0]
            Wtot = self.dvolfac*integrate.quad(self._Wtot,0,1,args=(Ti,n20,self.impfrac))[0]
            if (self.volavgcurr == True):
                Ptoh = self.vol*self._P_OH_avg(navg, Tavg)
            else:
                Ptoh = self.dvolfac*integrate.quad(self._P_OH,0,1,args=(n20,Ti,self.impfrac))[0]

            # Total power
            Ptot  = Pheat - Prad

            # Calculate confinement time and W/tau_E
            tauE  = self.H89_base * n20**(0.1) * Pheat**(-0.5)

            # add neoclassical loss term, avoids problems at low density
            tauNeo = self._neoTauCorrect(n20, Ti)
            tauEtot = 1./(1./tauE + 1./tauNeo)
            WoTau = Wtot/tauEtot

            # Adjust auxiliary power to get towards power balance W/tau_E = P_heat
            dPh = WoTau - Pheat

            # ensure self consistency in power balance
            if Pheat + dPh - Prad < 0:
                dPh = Prad - Pheat

            # overrelaxation
            reld = abs(dPh/max(Ph,1.0))
            Ph = Ph + ((reld*self.relax + 0.2*accel)/(reld + 0.2))*dPh

            # check for convergence
            if abs(dPh)/max(abs(Ph), 1.) <= pderror:
                break

        Paux = Ph
        if i+1 == self.maxit:
            print("Auxpow did not converge. Ph = %.3f, Ti = %.3f, n = %.3f" % (Paux, Ti, n20))

        Ploss  = Ptalpha + Paux + Ptoh
        Q = 5.03 * Ptalpha/(Paux + Ptoh)

        return Paux, Ploss, Ptalpha, Prad, Q, Ptoh



    # Power Calculations, in units of MW/m^3
    #--------------------------------------------------------------------------
    def _P_OH(self, rho, n20, Te, impfrac):
        ''' Ohmic heating power density calculation, assuming constant current.
        '''
        return rho * self._etafun(rho,n20,Te,impfrac) * self._jfun(rho,self.j0)**2 * 1e-6

    def _P_OH_avg(self,n20,Te):
        ''' Ohmic heating power density calculation, assuming constant current
        '''
        loglam = 24-np.log(np.sqrt(n20)*1.0e20)/(Te*1e3)
        return self.Cspitz / (Te*1e3)**(1.5)*0.016*loglam/15.0

    def _P_rad(self, rho, n20, Te, impfrac, B0, a, R):
        ''' Radiated power, sum of continuum (brems), line radiation and synchrotron radiation,
        using ADAS rates.
        '''
        line_rad, cont_rad = aurora.get_cooling_factors(
            self.imp,
            np.array([self._nfun(rho, n20)*14,]),  # cm^-3
            np.array([self._Tfun(rho,Te)*1e3,]),   # eV
            plot=False)
        
        P_line = rho * line_rad/1e6 * (self._nfun(rho, n20)*1e20)**2 * impfrac
        P_cont = rho * cont_rad/1e6 * (self._nfun(rho, n20)*1e20)**2 * impfrac

        # Calculates synchrotron radiation using formula in Zohm JNE 2019, eq.6, [MW/m^3]
        P_sync = rho * aurora.sync_rad(B0,   # T
                                       self._nfun(rho, n20)*1e14,  # cm^-3
                                       self._Tfun(rho,Te)*1e3,     # eV
                                       a, R)   # m
        
        return P_line + P_cont + P_sync

    def _P_alpha(self, rho, n20, Ti, impfrac):
        ''' Fusion power from alpha particles, using Hale-Bosch reactivity fits.
        '''
        # import omfit_classes here to avoid import during regression tests
        from omfit_classes import utils_fusion
        
        sigmav = utils_fusion.reactivity(1e4, model='D-T')    # m^3/s
        return rho * 0.016 * 3.5e3 * 1e20 * sigmav * (self._nfun(rho,n20) * self.dil)**2 / 4.
    

    def _Wtot(self, rho, T0, n0, impfrac):
        ''' Total stored energy. NB: dilution can have large effects on POPCON.
        '''
        return rho * 0.016 * 1.5 * self._nfun(rho,n0) * (1+self.dil) * self._Tfun(rho,T0)

    def _fionalp(self, Te):
        ''' Calculate fraction of alpha energy to ions at electron temp Te.
        '''
        Ei = 3500
        Tfac = 31.90
        twosqrt3 = 1.1547
        atsq3 = -0.52360

        x=Ei/(Tfac*Te)
        return (-0.333*np.log((1.+np.sqrt(x))**2/(1.-np.sqrt(x)+x)) \
                  +twosqrt3*(np.arctan((np.sqrt(x)-0.5)*twosqrt3)-atsq3))/x

    # Profile functions and differential elements of profile functions for vol. integrals
    #--------------------------------------------------------------------------
    def _Tfun(self,rho,T0):
        return (T0-self.T_edge)*(1-rho**self.Talpha1)**self.Talpha2 + self.T_edge

    def _iTfun(self,rho,T0):
        return rho*self._Tfun(rho,T0)

    def _nfun(self,rho,n0):
        return n0 * ( (1-self.n_frac_edge)*(1-rho**self.nalpha1)**self.nalpha2 + self.n_frac_edge)

    def _infun(self,rho,n0):
        return rho*self._nfun(rho,n0)

    def _jfun(self,rho,j0):
        return j0 * (1-rho**self.jalpha1)**self.jalpha2

    def _ijfun(self,rho,j0):
        return rho*self._jfun(rho,j0)

    def _get_ioniz_eqm(self, rho, T0):
        ''' Cache charge state fractional abundances of selected impurity.
        '''
        _, self.fz_imp = aurora.get_frac_abundances(self.atom_data,
                                                    1e14, # dummy
                                                    self._Tfun(rho,T0)*1e3,  # eV
                                                    plot=False)

    def _imchargfun(self, rho, T0):
        ''' Average charge state of chosen impurity at given temperature.
        '''
        if not hasattr(self, 'fz_imp'):
            self._get_ioniz_eqm(rho,T0)

        return np.arange(self.fz_imp.shape[1])*self.fz_imp[0]

    def _etafun(self,rho,n20,T0,impfrac):
        ''' Neoclassical resistivity from Jardin et al. 1993, in Ohm*m
        '''
        Zeff = get_Zeff(self.fz_imp, impfrac, self._imchargfun(rho,T0), self.fHe)
        Lambda = max(5, (self._Tfun(rho,T0)*1e3 / np.sqrt(self._nfun(rho,n20)*1e20))*np.exp(17.1))
        Lambda_E = 3.40/Zeff * (1.13+Zeff)/(2.67+Zeff)
        C_R = 0.56/Zeff * (3.0-Zeff)/(3.0+Zeff)
        xi  = 0.58 + 0.20*Zeff
        
        # inverse aspect ratio
        delta = self.a/self.R
        
        # trapped particle fraction
        f_T = np.sqrt(2*rho*delta)   # TODO: correct for elongation
        
        # safety factor
        q = rho * get_q_a(self.a, self.B0, self.kappa, self.R, self.Ip)
        
        # electron collisionality
        nu_star_e = 1./10.2e16 * self.R * q * (self._nfun(rho,n20)*1e20) *\
                    Lambda / f_T * delta * (self._Tfun(rho,T0)*1e3)**2

        factor_NC = Lambda_E * (1.-f_T/(1.+xi*nu_star_e))*(1.-C_R*f_T/(1.+xi*nu_star_e))

        eta_C = 1.03e-4 * np.log(Lambda) * (self._Tfun(rho,T0)*1e3)**(-3/2)

        return eta_C * factor_NC

    def _neoTauCorrect(self,n20,T0):
        ''' Calculate neoclassical correction to confinement time using 
        K2 neoclassical factor from Chang-Hinton PoF 1982 and Chang-Hazeltine Rev. Mod. Phys. 1978
        '''
        gmn = 0.5 # profile factor, 0.5 seems to be a decent guess
        IpAmps = self.Ip*1.0e6 # [A]
        ntmp = n20*1.0e20 # [m^-3]
        aor = 0.7*self.a/self.R
        
        # Neoclassical factors calculation
        nuis = n20*(self.R/self.a)**(1.5)*self.q_a*self.R*0.03*self.Zeff**2
        #nues = 0.01*((n20)/(T0*T0))*(self.q_a*self.R/self.a)**1.5*\
        #          self.R*self.Zeff*(1-1/(self.q_a**2))**(0.5-4)
        
        nutot = nuis
        k2 = 1.77*nuis*(aor)**3/(1+0.74*nutot*aor**1.5)
        k2 += (1+2.85*np.sqrt(aor)-2.33*aor)/(1+1.03*np.sqrt(nutot)+0.32*nutot)
        k2 *= 0.66
        
        # calculate loss term using mu=2.5 since this code is intended for D-T fusion
        tauineo = 1e7*(IpAmps**2/ntmp)*np.sqrt(T0*self.R/self.a)*(gmn/2+1)/(20*np.sqrt(2.5/2))
        tauineo /= k2
        
        return tauineo

    def _vspecific(self, rho, n20, Te, impfrac):
        return self._etafun(rho,n20,Te,impfrac) * self._jfun(rho,self.j0)*rho

    def _vloopfun(self, n20, Te, impfrac):
        return self.dvolfac*integrate.quad(self._vspecific,0,1,args=(n20,Te,impfrac),limit=200)[0]/get_plasma_area(self.a,self.kappa)


    # output functions
    #--------------------------------------------------------------------------
    def _print_parameters(self):
        print("### POPCON parameters:")
        print('\n'.join("  %-15s: %s" % item for item in vars(self).items()))

#----------------------------------------------------------
# Compendium of useful functions

def get_P_LH_threshold(n20, bs):
    ''' L-H power transition scaling, from Martin et al J. Phys 2008. '''
    return 0.049*n20**(0.72)*bs

def get_bs_factor(B0, R, a, kappa):
    ''' BS from Martin H Mode Scaling (Martin et al J. Phys 2008) '''
    return B0**(0.8)*(2.*np.pi*R * 2*np.pi*a * np.sqrt((kappa**2+1)/2))**(0.94)

def get_plasma_vol(a, R, kappa):
    ''' Plasma volume '''
    return 2.*np.pi**2*a**2*R*kappa

def get_plasma_area(a, kappa):
    ''' Plasma cross section area '''
    return np.pi*a**2*kappa

def get_plasma_dvolfac(a, R, kappa):
    ''' Volume of drho element, where rho=0 at magnetic axis and rho=1 at separatrix '''
    return 4*np.pi**2*a**2*R*kappa

def get_plasma_dilution(impfrac, imcharg, fHe):
    ''' Plasma dilution. 
    NB: this is very rough, only using an average charge for an average impurity species.
    '''
    return 1/(1 + impfrac*imcharg + 2*fHe)

def get_Zeff(fz_imp, impfrac, imcharg, fHe):
    ''' Get Zeff from chosen concentration of a certain impurity and fusion ash.
    '''
    Zvals = np.arange(fz_imp.shape[1])
    return 1. + 2. * fHe + np.sum((Zvals**2 - Zvals) * fz_imp[0,:]) * impfrac

def get_n_GR(Ip, a):
    ''' n_GR, Greenwald density in 10^20/m^3 '''
    return Ip/(np.pi*a**2)

def get_q_a(a, B0, kappa, R, Ip):
    ''' Edge safety factor. TODO: allow one to extract this from geqdsk data.
    '''
    return 2*np.pi*a**2*B0*(kappa**2+1)/(2*R*constants.mu_0*Ip*10**6)

def get_Cspitz(Zeff, Ip, q_a, a, kappa, volavgcurr):
    ''' Spitzer resistivity for Ohmic power calculation. '''
    Fz    = (1+1.198*Zeff + 0.222*Zeff**2)/(1+2.966*Zeff + 0.753*Zeff**2)
    eta1  = 1.03e-4*Zeff*Fz
    j0avg = Ip/(np.pi*a**2*kappa)*1.0e6

    Cspitz = eta1*q_a*j0avg**2 if volavgcurr == True else eta1

    Cspitz /= 1.6e-16*1.0e20  # unit conversion to keV 10^20 m^-3
    return Cspitz

def get_ITER_H89_base(H_fac, M_i, Ip, R, a, kappa, B0):
    ''' H89/n20**(0.1), for convenience as n20 varies in the POPCON '''
    return 0.048*H_fac*M_i**(0.5)*Ip**(0.85)*R**(1.2)*a**(0.3)*kappa**(0.5)*B0**(0.2)

def plot_single_contour(ax, name, xx, yy, quantity,
                        levels, color, linewidth=1.5, fmt='%1.3f', fontsize=10):
    ''' Function to plot a POPCON contour '''
    
    if levels == "":
        contour=ax.contour(xx,yy,np.transpose(quantity),
                           colors=color,linewidths=linewidth)
        rect = patches.Rectangle((0,0),0,0,fc = color,label = name)
    else:
        contour=ax.contour(xx,yy,np.transpose(quantity),
                           levels=levels,colors=color,linewidths=linewidth)
        rect = patches.Rectangle((0,0),0,0,fc = color,label = name)

    ax.add_patch(rect)
    ax.clabel(contour,inline=1,fmt=fmt,fontsize=fontsize)


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    #print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
    
