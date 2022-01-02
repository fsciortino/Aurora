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
from IPython import embed

class POPCON:
    def __init__(self, R=3.7, a=1.14, kappa=1.75, B0=12.3, Ip=17.4, H=1.15, M_i=2.5,
                 c_He=0.02, imp='Xe', c_imp=0.0005, Z_imp=42,
                 f_LH=0.9, fixed_quantity='c_imp', Psol_target=20, 
                 T_edge=0.1, Talpha1=2., Talpha2=1.5,
                 n_frac_edge=0.4, nalpha1=2, nalpha2=1.5,
                 jalpha1=2, jalpha2=1.5,
                 override=False, volavgcurr=False, maxit=150,
                 relax=0.9, nmin_frac=0.05, nmax_frac=1, Ti_keV_min=1.,
                 Ti_keV_max=30, M=50):
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
        c_He : float
            Helium ash on-axis concentration
        imp : str
            Impurity atomic symbol, e.g. 'Xe'
        c_imp : float
            Impurity ion concentration.
        Z_imp : int
            Average charge of chosen impurity. TODO: generalize to local charge states.
        f_LH : float
            Target f_LH=P_sol/P_LH
        fixed_quantity : str
            Quantity to maintain fixed in POPCON, one of 'c_imp', 'f_LH', 'P_SOL'
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
        Ti_keV_min : float
            Min temperature [keV]
        Ti_keV_max : float
            Max temperature [keV]
        M : int
            Number of n, T points for which the POPCON solver is run
        
        Notes
        -----
        Kinetic profiles are parametrized with alpha1, alpha2, such that
        f=(f_center-f_edge)*(1-rho**alpha1)**alpha2 + f_edge.

        '''
        self.R       = float(R)
        self.a       = float(a)
        self.kappa   = float(kappa)
        self.B0      = float(B0)
        self.Ip      = float(Ip)
        self.H       = float(H)
        self.M_i     = float(M_i)
        self.c_He     = float(c_He)
        self.imp     = str(imp)
        self.c_imp = float(c_imp)
        self.Z_imp = int(Z_imp)
        self.f_LH    = float(f_LH)
        self.fixed_quantity = str(fixed_quantity)
        if self.fixed_quantity not in ['c_imp', 'f_LH', 'P_SOL']:
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
        self.Ti_keV_min = float(Ti_keV_min)
        self.Ti_keV_max = float(Ti_keV_max)
        self.M          = int(M)

        # cache ADAS atomic data - effective ionization and recombination rates
        self.atom_data = aurora.get_atom_data(self.imp)

        # quantities that can be plotted in POPCON
        self.plot_fields = {'Paux': True,
                            'Pfus': True,
                            'Q': True,
                            'Prad_frac': True,
                            'Psol': True,
                            'Pohm': False,
                            'f_LH': True,
                            'c_imp': True,
                            'Vloop': False,
                            'betaN': True,
                            'Pload_fus': True,
                            'Pload_rad': True}

    def make_popcons(self):
        ''' 
        Externally called routine that holds master logic for reading inputs
        and making POPCONS
        '''
        self._process_inputs()
        self._print_parameters()

        self.auxp   = np.empty([self.M,self.M])   # auxiliary power (RF, NBI, ...)
        self.pamat  = np.empty([self.M,self.M])   # alpha power
        self.prmat  = np.empty([self.M,self.M])   # radiated power by impurities
        self.psmat  = np.empty([self.M,self.M])   # radiated synchrotron power
        self.tratio = np.empty([self.M,self.M])   # TODO: allow Ti != Te
        self.plmat  = np.empty([self.M,self.M])   # power loss, i.e. alpha + auxiliary power
        self.qpmat  = np.empty([self.M,self.M])   # Q=P_fus/(P_aux+P_oh)
        self.pohmat = np.empty([self.M,self.M])   # Ohmic power
        self.flh    = np.empty([self.M,self.M])   # LH fraction, i.e. (P_loss - P_rad) / P_{LH threshold}
        self.f_imp = np.empty([self.M,self.M])   # impurity fraction (when set as free parameter)
        self.Vloop  = np.empty([self.M,self.M])   # Vloop
        self.betaN  = np.empty([self.M,self.M])   # Normalized beta

        #self.xv     = np.empty(self.M)            # array to save T_i vals for plotting
        #self.yv     = np.empty(self.M)            # array to save <n>/n_GR vals for plotting

        # set up list of Ti and n for which POPCON will evaluate power balance
        n_norm  = integrate.quad(self._nfun,0,1,args=(1))[0] # defined s.t. n_avg = n_peak*n_norm
        self.n20_arr = np.linspace(self.nmin_frac*self.n_g/n_norm, self.nmax_frac*self.n_g/n_norm, self.M)
        self.T_keV_arr  = np.linspace(self.Ti_keV_min,self.Ti_keV_max,self.M)

        NE,TE=np.meshgrid(self.n20_arr*1e14,self.T_keV_arr*1e3) 
        # line and continuum radiation at all values of ne and Te
        _line_rad, _cont_rad = aurora.get_cooling_factors(
            self.imp,
            NE.flatten(),
            TE.flatten(),
            plot=False)  # W.m^3
        self.line_rad = np.reshape(_line_rad, (len(self.n20_arr),len(self.T_keV_arr)))
        self.cont_rad = np.reshape(_cont_rad, (len(self.n20_arr),len(self.T_keV_arr)))
                    
        print("\n### Solve power balance at each (n,T)")
        for i in np.arange(self.M): # T_i loop
            Ti_keV = self.T_keV_arr[i]
            
            for j in np.arange(self.M): # n_e loop
                n20 = self.n20_arr[j]
            
                # compute normalised Prad for our impurity to quickly obtain impurity fraction from Prad
                Prad_norm = self.dvolfac*integrate.quad(self._P_rad, 0, 1,
                                                        args=(n20, Ti_keV, 1, self.a, self.R),
                                                        limit=200)[0]

                # synchrotron radiation does not depend on impurity density, so calculate it separately
                P_sync = self.dvolfac*integrate.quad(self._P_sync, 0, 1,
                                                     args=(n20, Ti_keV, self.B0, self.a, self.R),
                                                     limit=200)[0]

                # call solver
                _out = self._auxpowIt(n20, Ti_keV, Prad_norm, P_sync)
                self.auxp[i,j],self.plmat[i,j],self.pamat[i,j], self.prmat[i,j],self.qpmat[i,j], self.pohmat[i,j] = _out

                if i==20 and j==20:
                    embed()
                    
                # L-H power threshold
                self.flh[i,j]= (self.plmat[i,j]-self.prmat[i,j]-self.psmat[i,j]) / get_P_LH_threshold(n20, self.bs)

                # impurity fraction
                self.f_imp[i,j] = self.prmat[i,j] / Prad_norm

                # Loop voltage
                self.Vloop[i,j] = self._vloopfun(n20, Ti_keV, self.f_imp[i,j])
                
                # Normalize beta [%]
                self.betaN[i,j] = 4 * (4*np.pi*1e-7) * (n20*1e20) * (Ti_keV*1000*1.6021e-19) \
                    * self.a * 100 / self.B0 / self.Ip

                # set ignition curve
                if self.qpmat[i,j]>=10000.0 or self.qpmat[i,j]<0.0:
                    self.qpmat[i,j]=10000

                # debug prints
                if self.M <= 3:
                    print(f'For n20={n20:.2f}, Ti_keV={Ti_keV:.2f}')
                    print(f'P_aux={self.auxp[i,j]:.2f}, P_alpha={self.pamat[i,j]:.2f}, '+\
                          'P_rad={self.prmat[i,j]:.2f}, P_sync={self.psmat[i,j]:.2f}, '+\
                          'P_ohm={self.pohmat[i,j]:.2f}')

                # progress bar for overview
                ascii_progress_bar(i+1, b=self.M, mess='of POPCON map completed')

    def plot_contours(self):
        ''' Plot popcon contours. The `plot_fields' dictionary contains boolean flags for each of the allowed 
        quantities to plot contours for. Modifying these allows visualization of different plot_fields in the 
        POPCON. 
        '''
        n_norm = integrate.quad(self._nfun,0,1,args=(1))[0] # defined s.t. n_avg = n_peak*n_norm
        xx, yy = np.meshgrid(self.T_keV_arr, self.n20_arr/self.n_g*n_norm)
        #xx, yy = np.meshgrid(self.xv, self.yv/self.n_g*n_norm)
        
        self.fig, ax = plt.subplots(figsize=(16,9))
        
        Pauxlevels = [0,1,3,5,7,10,15,20,25,50,100]
        Pfuslevels = [30,50,100,300,500,1000,2000,3000]
        Qlevels    = [1,5,10,20,50]
        Prad_oloss_levels = [0.1, 0.5, 0.9, 0.99]
        Psollevels = [10, 20, 30, 40]
        Pohmlevels = [1, 2, 5]
        flh_levels = [0.5, 1.]
        f_imp_levels = None
        Vlooplevels = [1e-2, 1.3e-2, 1.5e-2, 2e-2, 3e-2, 4e-2, 5e-2, 7e-2, 9e-2, 1.2e-1, 1.5e-1]
        Pload_fus_levels = None
        Pload_rad_levels = None
        betaN_levels = [3]
        
        #Igshade = plt.contourf(xx, yy, np.transpose(self.qpmat),
        #                      levels=[6000,10000], colors='r', alpha=0.25)
        
        if self.plot_fields['Paux']: self._plot_single_contour(
                ax, "$P_\mathrm{aux}$", xx, yy, self.auxp, 'r', Pauxlevels, 1.0, fmt='%1.0f')
        if self.plot_fields['Pfus']: self._plot_single_contour(
                ax, "$P_\mathrm{fus}$", xx, yy, self.pamat*5.03, 'k', Pfuslevels, fmt='%1.0f')
        if self.plot_fields['Q']: self._plot_single_contour(
                ax, "Q", xx, yy, self.qpmat, 'lime', Qlevels, fmt='%1.0f')
        if self.plot_fields['Prad_frac']: self._plot_single_contour(
                ax, "$P_\mathrm{rad}/P_{loss}$", xx, yy, self.prmat/self.plmat, 'c', Prad_oloss_levels, 1.0, fmt='%1.2f')
        if self.plot_fields['Psol']: self._plot_single_contour(
                ax, "$P_\mathrm{sol}$", xx, yy, self.prmat-self.plmat, 'darkviolet', Psollevels, 1.0, fmt='%1.0f')
        if self.plot_fields['Pohm']: self._plot_single_contour(
                ax, "$P_\mathrm{ohm}$", xx, yy, self.pohmat, 'orange', Pohmlevels, 1.0, fmt='%1.0f')
        if self.plot_fields['f_LH']: self._plot_single_contour(
                ax, "$f_\mathrm{LH}$", xx, yy, self.flh, 'forestgreen', flh_levels, 1.0, fmt='%1.2f')
        if self.plot_fields['c_imp']: self._plot_single_contour(
                ax, "$n_\mathrm{imp}/n_e$", xx, yy, self.f_imp, 'magenta', f_imp_levels, 1.0, fmt='%0.1e')
        if self.plot_fields['Vloop']: self._plot_single_contour(
                ax, "$V_\mathrm{loop}$", xx, yy, self.Vloop, 'mediumvioletred', Vlooplevels, 1.0, fmt = '%0.2e')
        if self.plot_fields['betaN']: self._plot_single_contour(
                ax, "$\u03B2_\mathrm{N}$", xx, yy, self.betaN, 'darkslategrey', betaN_levels, 1.0, fmt = '%0.2e')
        
        # Plot power loads (MW/m^2) on wall. For area, use plasma surface area as rough estimate
        area = 2*np.pi*self.a * np.sqrt((1+self.kappa**2)/2) * 2*np.pi*self.R
        if self.plot_fields['Pload_fus']: self._plot_single_contour(
                ax, "$P_\mathrm{fus}/A$", xx, yy, self.pamat*4/area, 'peru', Pload_fus_levels, 1.0, fmt = '%0.2e')
        if self.plot_fields['Pload_rad']: self._plot_single_contour(
                ax, "$P_\mathrm{rad}/A$", xx, yy, self.prmat/area, 'dodgerblue', Pload_rad_levels, 1.0, fmt = '%0.2e')
        
        ax.set_ylabel(r"$\bar{n} / n_{GR}$")
        ax.set_xlabel('$T^i_\mathrm{keV}$')
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left',fontsize = 12)
        plt.tight_layout()
        
    def plot_profiles(self):
        ''' Plot 1D profiles
        '''
        self.fig = plt.figure(figsize=(12,9))
        for i_temp in np.arange(self.M):
            T_keV = self.T_keV_arr[i_temp]
            for i_dens in np.arange(self.M):
                n20 = self.n20_arr[i_dens]
                
                ax1 = plt.subplot(self.M, self.M, i_dens*self.M+i_temp+1)
                
                plt.subplots_adjust(wspace=None, hspace=None)
                
                plt.title(r"$n_0/n_{GR} =$ %.1f, $T^0_\mathrm{keV} =$ %.0f" % (n20/get_n_GR(self.Ip, self.a), T_keV), pad=-100)
                
                rho = np.linspace(1e-10, 1, 100)
                Prad = np.empty(len(rho))
                Psync = np.empty(len(rho))
                Pfus = np.empty(len(rho))
                Pohm = np.empty(len(rho))
                for i_rho in np.arange(len(rho)):
                    Prad[i_rho] = self._P_rad(rho[i_rho], n20, T_keV,
                                              self.f_imp[i_temp, i_dens], self.a, self.R)/rho[i_rho]
                    Psync[i_rho] = self._P_sync(rho[i_rho], n20, T_keV, self.B0, self.a, self.R)/rho[i_rho]
                    Pfus[i_rho] = self._P_alpha(rho[i_rho], n20, T_keV, self.f_imp[i_temp, i_dens])/rho[i_rho]
                    Pohm[i_rho] = self._P_OH(rho[i_rho], n20, T_keV, self.f_imp[i_temp,i_dens])/rho[i_rho]

                plt.plot(rho, Prad, label="$P_\mathrm{rad}$")
                plt.plot(rho, Psync, label="$P_\mathrm{sync}$")
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

    def _plot_single_contour(self, ax, name, xx, yy, quantity,
                             color, levels=None, linewidth=1.5, fmt='%1.3f', fontsize=10):
        '''Plot a single POPCON contour '''

        contour = ax.contour(xx,yy,np.transpose(quantity),
                           levels=levels,colors=color,linewidths=linewidth)
        rect = patches.Rectangle((0,0),0,0,fc = color,label = name)
        ax.add_patch(rect)
        ax.clabel(contour,inline=1,fmt=fmt,fontsize=fontsize)

    def _process_inputs(self):
        ''' Calculate some derived quantities including:
        - Plasma Volume (vol)
        - Plasma Dilution (dil)
        - Z Effective (Zeff)
        - Greenwald Limit (n_g)
        - Spitzer Resistivity Coefficient (nu_Spitzer)
        - B0*Surface Area (bs) with exponential scalings
        '''
        # plasma volume
        self.vol = get_plasma_vol(self.a, self.R, self.kappa)

	# volume of drho element for integration
        self.dvolfac = get_plasma_dvolfac(self.a, self.R, self.kappa)

        # plasma dilution
        self.dil = get_plasma_dilution(self.c_imp, self.Z_imp, self.c_He)

        # Z effective, effective ion charge
        #self.Zeff = get_Zeff_simple(self.fz_imp, self.c_imp, self.Z_imp, self.c_He)
        self.Zeff = get_Zeff_simple(self.c_imp, self.Z_imp, self.c_He)

        # n_g, Greenwald density (10^20 m^-3)
        self.n_g = get_n_GR(self.Ip, self.a)

        # q_a edge safety factor
        self.q_a = get_q_a(self.a, self.B0, self.kappa, self.R, self.Ip)
        if self.q_a <= 2.0 and self.override == False:
            sys.exit('q_a too low. External kink unstable')

        # Spitzer resistivity coefficient for ohmic heating power
        self.nu_Spitzer = get_nu_Spitzer(self.Zeff, self.Ip, self.q_a, self.a, self.kappa, self.volavgcurr)

        # BS from Martin H Mode Scaling (Martin et al J. Phys 2008)
        self.bs = get_bs_factor(self.B0, self.R, self.a, self.kappa)

        # H89_base = H89/n20**(0.1)/P**(-0.5) for convenience as n20 and P vary in POPCON
        self.H89_base = self.H * get_ITER_H89(self.M_i,self.Ip,self.R,self.a,self.kappa, 1., self.B0, 1.)
        self.H98_base = self.H * get_IPB98y2(self.Ip, self.B0, 1., 1., self.M_i, self.R, self.a, self.kappa)
        # Peak current density, from integral(j) = Ip
        self.j0 = self.Ip*1e6 / (2*np.pi*self.a**2*self.kappa) / integrate.quad(self._ijfun,0,1,args=(1))[0]


    # Iterative solvers for power balance.
    #--------------------------------------------------------------------------
    def _auxpowIt(self, n20, Ti_keV, Prad_norm, P_sync):
        ''' Iterate to get the aux power needed for power balance
        '''
        accel = 1.7
        pderror = 1e-5

        # Alpha power density in units of 10^20keV
        Ptalpha = self.dvolfac*integrate.quad(self._P_alpha,0,1,args=(n20,Ti_keV,self.c_imp))[0]

        # Total thermal energy in plasma
        Wtot = self.dvolfac*integrate.quad(self._Wtot,0,1,args=(Ti_keV,n20,self.c_imp))[0]

        # Volume averaged Ptoh
        if self.volavgcurr == True:
            navg = self.dvolfac*integrate.quad(self._infun,0,1,args=(n20))[0]/self.vol
            Tavg = self.dvolfac*integrate.quad(self._iTfun,0,1,args=(Ti_keV))[0]/self.vol
            Ptoh = self.vol*self._P_OH_avg(navg,Tavg)
        else:
            Ptoh = self.dvolfac*integrate.quad(self._P_OH,0,1,args=(n20,Ti_keV,self.c_imp))[0]

        # iteratively solve for equilibrium solution to power balance
        Ph = 0.0
        for i in np.arange(self.maxit):
            # Total heating power
            Pheat = Ptoh + Ph + Ptalpha

            # Radiated power
            if self.fixed_quantity == "c_imp":
                Prad = self.dvolfac*integrate.quad(self._P_rad, 0, 1,
                                                   args=(n20, Ti_keV, self.c_imp, self.a, self.R))[0]

            elif self.fixed_quantity == "f_LH" or self.fixed_quantity == "P_SOL":
                if self.fixed_quantity == "f_LH":
                    # Set Prad to satisfy f_LH target specified, but keep > 0
                    Prad = max(0, Pheat - P_sync - self.f_LH * get_P_LH_threshold(n20, self.bs))

                elif self.fixed_quantity == "P_SOL":
                    # Set Prad to bring Psol < Psol_target
                    Prad = max(0, Pheat - P_sync - self.Psol_target)

                # Infer impurity fraction necessary to get this Prad
                self.c_imp = Prad / Prad_norm
                
                # Update dilution and Zeff with the new impurity fraction
                self.dil = get_plasma_dilution(self.c_imp, self.Z_imp, self.c_He)
                #self.Zeff = get_Zeff(self.fz_imp, self.c_imp, self.Z_imp, self.c_He)
                self.Zeff = get_Zeff_simple(self.c_imp, self.Z_imp, self.c_He)

            else:
                sys.exit("Choose fixed quantity for radiation: c_imp, f_LH, Psol.")

            # Recalculate powers to account for changed dilution and Zeff
            Ptalpha = self.dvolfac*integrate.quad(self._P_alpha,0,1,args=(n20,Ti_keV,self.c_imp))[0]
            Wtot = self.dvolfac*integrate.quad(self._Wtot,0,1,args=(Ti_keV,n20,self.c_imp))[0]
            if self.volavgcurr:
                Ptoh = self.vol*self._P_OH_avg(navg, Tavg)
            else:
                Ptoh = self.dvolfac*integrate.quad(self._P_OH,0,1,args=(n20,Ti_keV,self.c_imp))[0]

            # Total power
            Ptot  = Pheat - P_sync - Prad

            # Calculate confinement time and W/tau_E
            tauE  = self.H89_base * n20**(0.1) * Pheat**(-0.5)

            # add neoclassical loss term, avoids problems at low density
            tauNeo = self._neoTauCorrect(n20, Ti_keV)
            tauEtot = 1./(1./tauE + 1./tauNeo)
            WoTau = Wtot/tauEtot

            # Adjust auxiliary power to get towards power balance W/tau_E = P_heat
            dPh = WoTau - Pheat

            # ensure self consistency in power balance
            if Pheat + dPh - P_sync - Prad < 0:
                dPh = Prad + P_sync - Pheat

            # overrelaxation
            reld = abs(dPh/max(Ph,1.0))
            Ph = Ph + ((reld*self.relax + 0.2*accel)/(reld + 0.2))*dPh

            # check for convergence
            if abs(dPh)/max(abs(Ph), 1.) <= pderror:
                break

        Paux = Ph
        if i+1 == self.maxit:
            print(f'Auxpow did not converge. Ph = {Paux:.3f}, Ti_keV = {Ti_keV:.3f}, n = {n20:.3f}')

        Ploss = Ptalpha + Paux + Ptoh
        Q = 5.03 * Ptalpha/(Paux + Ptoh)

        return Paux, Ploss, Ptalpha, Prad, Q, Ptoh



    # Power Calculations, in units of MW/m^3
    #--------------------------------------------------------------------------
    def _P_OH(self, rho, n20, Te_keV, c_imp):
        ''' Ohmic heating power density calculation, assuming constant current.
        Units of :math:`MW/m$^3$`.'''
        return rho * self._etafun(rho,n20,Te_keV,c_imp) * self._jfun(rho,self.j0)**2 * 1e-6

    def _P_OH_avg(self,n20,Te_keV):
        ''' Ohmic heating power density calculation, assuming constant current.
        Units of :math:`MW/m$^3$`. '''
        loglam = 24 - np.log(np.sqrt(n20)*1e20)/(Te_keV*1e3)
        return self.nu_Spitzer / (Te_keV*1e3)**(1.5)*0.016*loglam/15.0

    def _P_rad(self, rho, n20, Te_keV, c_imp, a, R):
        ''' Radiated power, sum of continuum (brems), line radiation and synchrotron radiation
        using ADAS rates. Units of :math:`MW/m$^3$`. '''
        i_dens = np.argmin(np.abs(self.n20_arr - n20))
        i_temp = np.argmin(np.abs(self.T_keV_arr - Te_keV))
        line_rad = self.line_rad[i_temp, i_dens]
        cont_rad = self.cont_rad[i_temp, i_dens]
        
        P_line = rho * line_rad/1e6 * (self._nfun(rho, n20)*1e20)**2 * c_imp
        P_cont = rho * cont_rad/1e6 * (self._nfun(rho, n20)*1e20)**2 * c_imp

        return P_line + P_cont # MW/m^3

    def _P_sync(self, rho, n20, Te_keV, B0, a, R):
        ''' Calculate synchrotron radiation, which does not scale with impurity density.
        Units of :math:`MW/m$^3$`. '''       
        return rho * aurora.sync_rad(B0,   # T
                                     self._nfun(rho, n20)*1e14,  # cm^-3
                                     self._Tfun(rho,Te_keV)*1e3,     # eV
                                     a, R)/1e6   # m

    def _P_alpha(self, rho, n20, Ti_keV, c_imp):
        ''' Fusion power from alpha particles, using Hale-Bosch reactivity fits.
        Units of :math:`MW/m$^3$`.'''
        # import omfit_classes here to avoid import during regression tests
        from omfit_classes import utils_fusion
        
        sigmav = utils_fusion.reactivity(Ti_keV*1e3, model='D-T')    # m^3/s
        return rho * 0.016 * 3.5e3 * 1e20 * sigmav * (self._nfun(rho,n20) * self.dil)**2 / 4.
    

    def _Wtot(self, rho, T_keV, n0, c_imp):
        ''' Total stored energy. NB: dilution can have large effects on POPCON.
        Units of :math:`MW/m$^3$`. '''
        return rho * 0.016 * 1.5 * self._nfun(rho,n0) * (1+self.dil) * self._Tfun(rho,T_keV)

    def _fionalp(self, Te_keV):
        ''' Calculate fraction of alpha energy to ions at electron temperature Te_keV. '''
        Ei = 3500
        Tfac = 31.90
        twosqrt3 = 1.1547
        atsq3 = -0.52360

        x=Ei/(Tfac*Te_keV)
        return (-0.333*np.log((1.+np.sqrt(x))**2/(1.-np.sqrt(x)+x)) \
                  +twosqrt3*(np.arctan((np.sqrt(x)-0.5)*twosqrt3)-atsq3))/x

    # Profile functions and differential elements of profile functions for vol. integrals
    #--------------------------------------------------------------------------
    def _Tfun(self,rho,T_keV):
        return (T_keV-self.T_edge)*(1-rho**self.Talpha1)**self.Talpha2 + self.T_edge

    def _iTfun(self,rho,T_keV):
        return rho*self._Tfun(rho,T_keV)

    def _nfun(self,rho,n0):
        return n0 * ( (1-self.n_frac_edge)*(1-rho**self.nalpha1)**self.nalpha2 + self.n_frac_edge)

    def _infun(self,rho,n0):
        return rho*self._nfun(rho,n0)

    def _jfun(self,rho,j0):
        return j0 * (1-rho**self.jalpha1)**self.jalpha2

    def _ijfun(self,rho,j0):
        return rho*self._jfun(rho,j0)

    def _get_ioniz_eqm(self, rho, T_keV):
        ''' Cache charge state fractional abundances of selected impurity.
        '''
        _, self.fz_imp = aurora.get_frac_abundances(self.atom_data,
                                                    1e14, # dummy
                                                    self._Tfun(rho,T_keV)*1e3,  # eV
                                                    plot=False)

    def _Z_imp_fun(self, rho, T_keV):
        ''' Average charge state of chosen impurity at given temperature.
        '''
        if not hasattr(self, 'fz_imp'):
            self._get_ioniz_eqm(rho,T_keV)

        return np.arange(self.fz_imp.shape[1])*self.fz_imp[0]

    def _etafun(self,rho,n20,T_keV,c_imp):
        ''' Neoclassical resistivity from Jardin et al. 1993, in Ohm*m
        '''
        #Zeff = get_Zeff(self.fz_imp, c_imp, self._Z_imp_fun(rho,T_keV), self.c_He)
        Zeff = get_Zeff_simple(self.c_imp, self.Z_imp, self.c_He)
        Lambda = max(5, (self._Tfun(rho,T_keV)*1e3 / np.sqrt(self._nfun(rho,n20)*1e20))*np.exp(17.1))
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
                    Lambda / f_T * delta * (self._Tfun(rho,T_keV)*1e3)**2

        factor_NC = Lambda_E * (1.-f_T/(1.+xi*nu_star_e))*(1.-C_R*f_T/(1.+xi*nu_star_e))

        eta_C = 1.03e-4 * np.log(Lambda) * (self._Tfun(rho,T_keV)*1e3)**(-3/2)

        return eta_C * factor_NC

    def _neoTauCorrect(self,n20,T_keV):
        ''' Calculate neoclassical correction to confinement time using 
        K2 neoclassical factor from Chang-Hinton PoF 1982 and Chang-Hazeltine Rev. Mod. Phys. 1978
        '''
        gmn = 0.5 # profile factor, 0.5 seems to be a decent guess
        IpAmps = self.Ip*1e6 # [A]
        ntmp = n20*1e20 # [m^-3]
        aor = 0.7*self.a/self.R
        
        # Neoclassical factors calculation
        nuis = n20*(self.R/self.a)**(1.5)*self.q_a*self.R*0.03*self.Zeff**2
        #nues = 0.01*((n20)/(T_keV*T_keV))*(self.q_a*self.R/self.a)**1.5*\
        #          self.R*self.Zeff*(1-1/(self.q_a**2))**(0.5-4)
        
        nutot = nuis
        k2 = 1.77*nuis*(aor)**3/(1+0.74*nutot*aor**1.5)
        k2 += (1+2.85*np.sqrt(aor)-2.33*aor)/(1+1.03*np.sqrt(nutot)+0.32*nutot)
        k2 *= 0.66
        
        # calculate loss term using mu=2.5 since this code is intended for D-T fusion
        tauineo = 1e7*(IpAmps**2/ntmp)*np.sqrt(T_keV*self.R/self.a)*(gmn/2+1)/(20*np.sqrt(2.5/2))
        tauineo /= k2
        
        return tauineo

    def _vspecific(self, rho, n20, Te_keV, c_imp):
        return self._etafun(rho, n20, Te_keV, c_imp) * self._jfun(rho,self.j0)*rho

    def _vloopfun(self, n20, Te_keV, c_imp):
        return self.dvolfac*integrate.quad(self._vspecific,0,1,args=(n20,Te_keV,c_imp),limit=200)[0]/\
            get_plasma_area(self.a,self.kappa)

    def _print_parameters(self):
        print("--- POPCON parameters:")
        print('\n'.join("  {:15s}: {}".format(key,val) for key,val in vars(self).items() if key!='atom_data'))

#----------------------------------------------------------
# Compendium of useful functions

def get_P_LH_threshold(n20, bs):
    ''' L-H power transition scaling, from Martin et al J. Phys 2008. '''
    return 0.049*n20**(0.72)*bs

def get_bs_factor(B0, R, a, kappa):
    ''' BS from Martin H-mode scaling (Martin et al J. Phys 2008) '''
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

def get_plasma_dilution(c_imp, Z_imp, c_He):
    ''' Plasma dilution. 
    NB: this is very rough, only using an average charge for an average impurity species.
    '''
    return 1/(1 + c_imp*Z_imp + c_He*2)

def get_Zeff_simple(c_imp, Z_imp, c_He):
    '''Get Zeff from chosen concentration of a certain impurity and fusion ash.

    Parameters
    ----------
    c_imp : float
        Concentration of lumped impurity.
    Z_imp : float
        Modelled impurity charge 
    c_He : float
        Concentration of He ash.
    '''
    #return ((1-c_imp-c_He) + c_imp*Z_imp**2 + c_He*2**2)*dil
    return (1-c_imp-c_He) + c_imp*Z_imp**2 + c_He*2**2

def get_Zeff(fz_imp, c_imp, c_He):
    '''Get Zeff from chosen concentration of a certain impurity and fusion ash.

    Parameters
    ----------
    fz_imp : 1D array
        Impurity charge state fractional abundances, normalized to 1.
    c_imp : float
        Concentration of lumped impurity.
    c_He : float
        Concentration of He ash.
    '''
    Zvals = np.arange(fz_imp.shape[1])
    return (1 - c_He - c_imp) + 2. * c_He + np.sum((Zvals**2 - Zvals) * fz_imp[0,:]) * c_imp

def get_n_GR(Ip, a):
    ''' n_GR, Greenwald density in 10^20/m^3 '''
    return Ip/(np.pi*a**2)

def get_q_a(a, B0, kappa, R, Ip):
    ''' Edge safety factor. TODO: allow one to extract this from geqdsk data.
    '''
    return 2*np.pi*a**2*B0*(kappa**2+1)/(2*R*constants.mu_0*Ip*10**6)

def get_nu_Spitzer(Zeff, Ip, q_a, a, kappa, volavgcurr):
    ''' Spitzer resistivity for Ohmic power calculation. '''
    Fz    = (1+1.198*Zeff + 0.222*Zeff**2)/(1+2.966*Zeff + 0.753*Zeff**2)
    eta1  = 1.03e-4*Zeff*Fz
    j0avg = Ip/(np.pi*a**2*kappa)*1e6

    nu_Spitzer = eta1*q_a*j0avg**2 if volavgcurr == True else eta1

    nu_Spitzer /= constants.e * 1e3 * 1e20  # unit conversion to keV 10^20 m^-3
    return nu_Spitzer


def get_ITER_H89(M_i, Ip, R, a, kappa, n20, B0, P):
    '''ITER89-P scaling law, Eq. 3 of 
    P.N. Yushmanov et al 1990 Nucl. Fusion 30 1999
    '''
    return 0.048*M_i**(0.5)*Ip**(0.85)*R**(1.2)*a**(0.3)*kappa**(0.5)*n20**(0.1)*B0**(0.2)*P**(-0.5)

def get_IPB98y2(Ip, B0, P, n20, M_i, R, a, kappa):
    '''ITER IPB98(y,2) scaling law, Eq. 20 of
    ITER Physics Expert Group on Confinement and Transport et al 1999 Nucl. Fusion 39 2175
    '''
    n19 = n20*10.
    epsilon = a/R
    return 0.0562 * Ip**0.93 * B0**0.15 * P**(-0.69) * n19**0.41 * M_i**0.19 * R**1.97 * epsilon**0.58 * kappa**0.78
    
def get_reinke_fz(B0, f_LH, qstar, R, a, f_sep, f_gw, kappa, lhat, m_L):
    '''Divertor impurity fraction predicted to lead to detachment from Eq. 10 in
    M.L. Reinke 2017 Nucl. Fusion 57 034004

    Note
    ----
    In Reinke's paper, `fz` stands for an impurity concentration, rather than charge state fractional abundances.
    Here, to avoid confusion, we use c_z rather than f_z for concentrations.
    '''
    qstar = get_qstar(a, kappa, B0, R, Ip)
    epsilon = a/R
    return 0.014 * (B0**0.88 * f_LH**1.14 * qstar**0.32 * R**1.33 * epsilon**0.59)/\
        (f_sep**2 * f_gw**1.18 * (1 + kappa**2)**0.64 * lhat**0.86 * m_L)

def get_qstar(a, kappa, B0, R, Ip):
    '''Definition of qstar, see near Eq. 3 in 
    M.L. Reinke 2017 Nucl. Fusion 57 034004
    '''
    return np.pi*a**2 * (1+kappa**2)*B0/(constants.mu_0 * R * Ip)
    
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = ' ', printEnd = "\r"):
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
    



def ascii_progress_bar(
    n,
    a=0,
    b=100,
    mess='',
    newline=False,
    clean=False,
    width=20,
    fill='#',
    void='-',
    style=' [{sfill}{svoid}] {perc:3.2f}% {mess}',
    tag='INFO',
    quiet=False,
):
    """
    Displays an ASCII progress bar

    :param n: current value OR iterable

    :param a: default 0, start value (ignored if n is an iterable)

    :param b: default 100, end value  (ignored if n is an iterable)

    :param mess: default blank, message to be displayed

    :param fill: default '#', filled progress bar character

    :param void: default '-', empty progress bar character

    :param style: default ' [{sfill}{svoid}] {perc:3.2f}% {mess}' full format string

    :param tag: default 'HELP', see tag_print()

    :param quiet: do not print

    Example::
        for n in ascii_progress_bar(np.linspace(12.34, 56.78, 4), mess=lambda x:f'{x:3.3}'):
            OMFITx.Refresh() # will slow things down
    """

    def ascii_progress_bar_base(n, a, b, mess, newline, clean, width, fill, void, style, tag, quiet, dtime=0):
        # handle manual iteration
        if not (a < b and a <= n and n <= b):
            return
        perc = 100.0 * (n + 1 - a) / (b + 1 - a)
        nfill = int(np.round(width * perc / 100.0))
        sfill = fill * nfill
        svoid = void * (width - nfill)
        if newline:
            buff = ''
        else:
            buff = '\r'
        buff += style.format(**locals())
        if newline or (n == b and not clean):
            buff += '\n'
        elif n == b and clean:
            buff = '\r' + ' ' * len(buff) + '\r'
        if not quiet:
            print(buff) #, tag=tag, end='')
        return n

    def ascii_progress_bar_iterable(n, **kw):
        data = list(n)  # we need to know the length of the data in case `n` was a generator
        kw['a'] = 0
        kw['b'] = len(data) - 1
        messages = [kw['mess']] * len(data)
        if not kw['newline']:
            messages = copy.copy(messages)
            if not len(messages):
                messages.append('')
            messages[-1] = ''
        for n, d in enumerate(data):
            kw['mess'] = messages[n]
            if '{dtime' in style:
                t0 = time.time()
                yield d
                kw['dtime'] = time.time() - t0
                ascii_progress_bar_base(n, **kw)
            else:
                ascii_progress_bar_base(n, **kw)
                yield d
        return

    import numpy as np

    if np.iterable(n):
        return ascii_progress_bar_iterable(
            n, a=a, b=b, mess=mess, newline=newline, clean=clean, width=width, fill=fill, void=void, style=style, tag=tag, quiet=quiet
        )
    else:
        return ascii_progress_bar_base(
            n, a=a, b=b, mess=mess, newline=newline, clean=clean, width=width, fill=fill, void=void, style=style, tag=tag, quiet=quiet
        )



if __name__=='__main__':

    # Initialize POPCON with default parameters
    popcon = aurora.POPCON()

    # Adjust input parameters that won't be scanned
    popcon.R = 1.85
    popcon.a = 0.57
    popcon.kappa = 1.97
    popcon.delta = 0.54  # not used yet
    popcon.B0=12.2
    popcon.Ip=8.7
    popcon.H=1.
    popcon.M_i=2.5
    popcon.c_He=0.02
    popcon.imp='W' #'Xe'
    popcon.c_imp=1e-3 # 1e-5 #0.0005
    popcon.Z_imp=50 #42
    popcon.f_LH=1.0
    popcon.fixed_quantity='P_SOL'
    popcon.Psol_target=20

    # kinetic profile parameters
    popcon.T_edge=0.1
    popcon.Talpha1=2.
    popcon.Talpha2=1.5
    popcon.Ti_keV_min=1.
    popcon.Ti_keV_max=30      
    popcon.n_frac_edge=0.4    
    popcon.nalpha1=2
    popcon.nalpha2=1.5
    popcon.nmin_frac=0.05
    popcon.nmax_frac=1  
    popcon.jalpha1=2
    popcon.jalpha2=1.5

    # numerical parameters
    popcon.override=False
    popcon.volavgcurr=True
    popcon.maxit=150
    popcon.relax=0.9
    popcon.M=50

    # plotting choices
    popcon.plot_fields['Psol']  = not (popcon.fixed_quantity == "P_SOL")
    popcon.plot_fields['c_imp']  = not (popcon.fixed_quantity == "c_imp")
    popcon.plot_fields['f_LH']  = not (popcon.fixed_quantity == "f_LH")

    # create POPCON
    popcon.make_popcons()

    # plot result
    popcon.plot_contours()

    # use only if M is small...
    #popcon.plot_profiles()
