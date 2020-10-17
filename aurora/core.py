#-*-Python-*-
# Created by sciortino & odstrcil

import scipy.io
import copy,os,sys
import numpy as np
from scipy.interpolate import interp1d
from . import _aurora
from . import interp
from . import atomic
from . import grids_utils
from . import source_utils


class aurora_sim:
    '''
    Class to setup and run aurora simulations.
    '''
    def __init__(self, namelist, geqdsk=None, nbi_cxr=None):
        ''' Setup aurora simulation input dictionary from the given namelist.

        INPUTS
        ------
        namelist : dict
            Dictionary containing aurora inputs. See default_nml.py for some defaults, which users should 
            modify for their runs.
        geqdsk : dict, optional
            EFIT gfile as returned after postprocessing by the omfit_eqdsk package (OMFITgeqdsk class). 
            If left to None (default), the geqdsk dictionary is constructed starting from the gfile in the MDS+ tree.
        nbi_cxr : array, optional
            If namelist['nbi_cxr']=True, this array represents the charge exchange rates with NBI neutrals, fast and/or
            thermal, across the entire radius and on the time base of interest. 
            Creating this input is not trivial and must be done externally to aurora. General steps:
            - get density of fast NBI neutrals (both fast and thermal/halo) ---> n0_nbi, n0_halo
            - get total rates (n-unresolved) for CX with NBI neutrals --> _alpha_CX_NBI_rates
            - thermal rates for the halo may be from ADAS CCD files or from the same methods used for fast neutrals
            - sum n0_nbi *  alpha_CX_NBI_rates + n0_halo * alpha_CX_rates
            This method still needs more testing within this class. Please contact author for details. 

        '''
        self.namelist = namelist
        self.kin_profs = namelist['kin_profs']
        self.nbi_cxr = nbi_cxr
        self.imp = namelist['imp']

        if geqdsk is None:
            # Fetch CMOD geqdsk from MDS+ and post-process it using the OMFIT geqdsk format.
            try:
                import omfit_eqdsk
            except:
                raise ValueError('Could not import omfit_eqdsk! Install with pip install omfit_eqdsk')
            geqdsk = omfit_eqdsk.OMFITgeqdsk('').from_mdsplus(device=namelist['device'],shot=namelist['shot'],
                                                              time=namelist['time'], SNAPfile='EFIT01',
                                                              fail_if_out_of_range=False,time_diff_warning_threshold=20)

        # Get r_V to rho_pol mapping
        rho_pol, _rvol = grids_utils.get_rhopol_rV_mapping(geqdsk)
        rvol_lcfs = interp1d(rho_pol,_rvol)(1.0)
        self.rvol_lcfs = self.namelist['rvol_lcfs'] = np.round(rvol_lcfs,3)  # set limit on accuracy

        # create radial grid
        grid_params = grids_utils.create_radial_grid(self.namelist, plot=False)
        self.rvol_grid,self.pro_grid,self.qpr_grid,self.prox_param = grid_params

        # get rho_poloidal grid corresponding to aurora internal (rvol) grid
        self.rhop_grid = interp1d(_rvol,rho_pol)(self.rvol_grid)

        # Save R on LFS and HFS
        self.Rhfs, self.Rlfs = grids_utils.get_HFS_LFS(geqdsk, rho_pol_arb=self.rhop_grid)

        # define time grid ('timing' must be in namelist)
        self.time_grid, self.save_time = grids_utils.create_time_grid(timing=self.namelist['timing'], plot=False)
        self.time_out = self.time_grid[self.save_time]

        # get kinetic profiles on the radial and (internal) temporal grids
        self._ne,self._Te,self._Ti,self._n0 = self.get_aurora_kin_profs()

        # store also kinetic profiles on output time grid
        self.ne = self._ne[self.save_time,:]
        self.Te = self._Te[self.save_time,:]
        self.Ti = self._Ti[self.save_time,:]
        self.n0 = self._n0  # at present, n0 is assumed to be time-indpt
        
        # Get time-dependent parallel loss rate
        self.par_loss_rate = self.get_par_loss_rate()

        # Obtain atomic rates on the computational time and radial grids
        self.S_rates, self.R_rates = self.get_time_dept_atomic_rates()

        # get radial profile of source function
        self.source_rad_prof = source_utils.get_radial_source(self.namelist, self.rvol_grid,
                                                              self.S_rates[:,0],  # take only 1 time slice of S_rates
                                                              self.pro_grid, self._Ti)
        
        # create array of 0's of length equal to self.time_grid, with 1's where sawteeth must be triggered
        self.saw_on = np.zeros_like(self.time_grid)
        self.saw_times = np.array(self.namelist['saw_model']['times'])[self.namelist['saw_model']['times']<self.time_grid[-1]]
        if self.namelist['saw_model']['saw_flag'] and len(self.namelist['saw_model']['times'])>0:
            self.saw_on[self.time_grid.searchsorted(self.saw_times)] = 1

        # source function
        self.source_time_prof = source_utils.get_aurora_source(self.namelist, self.time_grid)

        # Extract other inputs from namelist:
        self.mixing_radius = self.namelist['saw_model']['rmix']
        self.decay_length_boundary = self.namelist['SOL_decay']
        self.Z_imp = int(self.namelist['Z_imp'])
        self.wall_recycling = self.namelist['wall_recycling']
        self.source_div_fraction = self.namelist['divbls']   # change of nomenclature
        self.tau_div_SOL = self.namelist['tau_div_SOL']
        self.tau_pump = self.namelist['tau_pump']
        self.tau_rcl_ret = self.namelist['tau_rcl_ret']
        
        # if recycling flag is set to False, then prevent any divertor return flows
        # To include divertor return flows but no recycling, user should use wall_recycling=0
        if not self.namelist['recycling_flag']:
            self.wall_recycling = -1.0  # no divertor return flows

        self.bound_sep = self.namelist['bound_sep']
        self.lim_sep = self.namelist['lim_sep']
        self.sawtooth_erfc_width = self.namelist['saw_model']['sawtooth_erfc_width']
        self.main_ion_Z = self.namelist['Z']
        self.main_ion_A = self.namelist['a']
        self.cxr_flag = self.namelist['cxr_flag']
        self.nbi_cxr_flag = self.namelist['nbi_cxr_flag']

        
        
    def interp_kin_prof(self, prof): 
        ''' Interpolate the given kinetic profile on the radial and temporal grids [units of s].
        This function extrapolates in the SOL based on input options using the same methods as in STRAHL.
        '''
        times = self.kin_profs[prof]['times']

        r_lcfs = np.interp(1,self.rhop_grid,self.rvol_grid)

        # extrapolate profiles outside of LCFS by exponential decays
        r = interp1d(self.rhop_grid, self.rvol_grid,fill_value='extrapolate')(self.kin_profs[prof]['rhop'])
        if self.kin_profs[prof]['fun'] == 'interp':
            data = interp.interp_quad(r/r_lcfs,self.kin_profs[prof]['vals'],
                                             self.kin_profs[prof]['decay'],r_lcfs,self.rvol_grid)
            data[data < 1.01] = 1
        elif self.kin_profs[prof]['fun'] == 'interpa':
            data = interp.interpa_quad(r/r_lcfs,self.kin_profs[prof]['vals'],r_lcfs,self.rvol_grid)

        # linear interpolation in time
        if len(times) > 1:  # time-dept
            data = interp1d(times,data,axis=0)(np.clip(self.time_grid,*times[[0,-1]]))

        return data


    def get_aurora_kin_profs(self, min_T=1.01, min_ne=1e10):
        ''' Get kinetic profiles on radial and time grids.  '''

        Te = self.interp_kin_prof('Te')
        ne = self.interp_kin_prof('ne')
        if 'Ti' in self.kin_profs:
            Ti = interp_kin_prof('Ti')
        else:
            Ti = Te

        # get neutral background ion density
        if self.namelist.get('cxr_flag',False):
            n0 = self.interp_kin_prof('n0')
        else:
            n0 = None

        # set minima in temperature and density
        Te[Te < min_T] = min_T
        Ti[Ti < min_T] = min_T
        ne[ne < min_ne] = min_ne

        # make sure that Te,ne have the same shape at this stage (allow n0 to be time-indpt)
        ne,Te,Ti = np.broadcast_arrays(ne,Te,Ti)

        return ne,Te,Ti,n0


    def get_time_dept_atomic_rates(self):
        ''' Obtain time-dependent ionization and recombination rates for a simulation run.
        If kinetic profiles are given as time-independent, atomic rates for each time slice
        will be set to be the same.
        '''
        lne = np.log10(self._ne)
        lTe = np.log10(self._Te)

        # get TIME-DEPENDENT atomic rates
        atom_data = atomic.get_all_atom_data(self.imp,['acd','scd'])

        # get electron impact ionization and radiative recombination rates in units of [s^-1]
        S_rates = atomic.interp_atom_prof(atom_data['scd'],lne, lTe)
        alpha_rates = atomic.interp_atom_prof(atom_data['acd'],lne, lTe)

        # define effective recombination rate as R:
        R_rates = copy.deepcopy(alpha_rates)
        if self.namelist['cxr_flag']:
            # include thermal charge exchange recombination
            atom_data = atomic.get_all_atom_data(self.imp,['ccd'])

            lTi = np.log10(Ti)
            alpha_CX_rates = atomic.interp_atom_prof(atom_data['ccd'], lne, lTi, x_multiply=False)

            # change rates from units of [1/s/cm^3] to [1/s] ---> this is the result of STRAHL's `sas' subroutine
            R_rates += self._n0[:,None] * alpha_CX_rates[:,:alpha_rates.shape[1],:]   # select only relevant CCD ion stages (useful for Foster scaling)

        if self.namelist['nbi_cxr_flag']:
            # include charge exchange between NBI neutrals and impurities
            R_rates += self.nbi_cxr.transpose(1,0,2)

        # nz=nion of rates arrays must be filled with zeros - final shape: (nr,nion,nt)
        S_rates = np.append(S_rates, np.zeros_like(S_rates[:,[0]]),axis=1).T
        R_rates = np.append(R_rates, np.zeros_like(R_rates[:,[0]]),axis=1).T

        # broadcast in the requested time-dependent shape
        S_rates,R_rates,_ = np.broadcast_arrays(S_rates,R_rates,self.time_grid[None, None, :])

        # set up as Fortran order in memory for speed
        S_rates = np.asfortranarray(S_rates)
        R_rates = np.asfortranarray(R_rates)
        
        return S_rates, R_rates





    def get_par_loss_rate(self, trust_SOL_Ti=False):
        '''
        Calculate the parallel loss frequency on the radial and temporal grids [1/s].

        trust_SOL_Ti should generally be set to False, unless specific Ti measurements are available
        in the SOL.
        ***
        '''
        # background mass (=2 for D)
        apl = float(self.namelist['a'])

        # factor for v = machnumber * sqrt((3T_i+T_e)k/m)
        vpf = self.namelist['SOL_mach']*np.sqrt(1.602e-19/1.601e-27/apl)  # v[m/s]=vpf*sqrt(T[ev])

        # number of points inside of LCFS
        ids = self.rvol_grid.searchsorted(self.namelist['rvol_lcfs'],side='left')
        idl = self.rvol_grid.searchsorted(self.namelist['rvol_lcfs']+self.namelist['lim_sep'],side='left')

        # Calculate parallel loss frequency using different connection lengths in the SOL and in the limiter shadow
        dv = np.zeros_like(self._Te.T) # space x time

        if not trust_SOL_Ti:
            # Ti may not be reliable in SOL, replace it by Te
            Ti = self._Te
        else:
            Ti = self._Ti

        # open SOL
        dv[ids:idl] = vpf*np.sqrt(3.*Ti.T[ids:idl] + self._Te.T[ids:idl])/self.namelist['clen_divertor']

        # limiter shadow
        dv[idl:] = vpf*np.sqrt(3.*Ti.T[idl:] + self._Te.T[idl:])/self.namelist['clen_limiter']

        dv,_ = np.broadcast_arrays(dv,self.time_grid[None])

        return np.asfortranarray(dv)






    def run_aurora(self, times_DV, D_z, V_z, nz_init=None, method='old',evolneut=False):
        ''' Run a simulation using inputs in the given dictionary and D,v profiles as a function
        of space, time and potentially also ionization state. Users may give an initial state of each
        ion charge state as an input.

        Results can be conveniently visualized with time-slider using

        >> aurora.slider_plot(rhop,time, nz.transpose(1,2,0),   # charge states in first dimension
                                   xlabel=r'$\rho_p$', ylabel='time [s]', zlabel=r'$n_z$ [cm$^{-3}$]', plot_sum=True,
                                   labels=[f'Ca$^{{{str(i)}}}$' for i in np.arange(nz_w.shape[1]])
        INPUTS
        -------------
        times_DV : 1D array
            Array of times at which D_z and V_z profiles are given. (Note that it is assumed that
            D and V profiles are already on the self.rvol_grid radial grid).
        D_z, V_z: arrays, shape of (space, time,nZ) or (space,time)
            Diffusion and convection coefficients, in units of cm^2/s and cm/s, respectively.
            This may be given as a function of (space,time) or (space,nZ, time), where nZ indicates
            the number of charge states. If inputs are found to be have only 2 dimensions, it is
            assumed that all charge states should be set to have the same transport coefficients.
        nz_init: array, shape of (space, nZ)
            Impurity charge states at the initial time of the simulation. If left to None, this is
            internally set to an array of 0's.
        method : str, optional
            If method='linder', use the Linder algorithm for increased stability and accuracy.
        evolneut : bool, optional
            If True, evolve neutral impurities based on their D,V coefficients. Default is False, in
            which case neutrals are only taken as a source and those that are not ionized immediately after
            injection are neglected.

        OUTPUTS:
        -------------
        out : list
            List containing each particle reservoir of a simulation, i.e.
            nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out
        '''
        if nz_init is None:
            # default: start in a state with no impurity ions
            nz_init = np.zeros((len(self.rvol_grid),int(self.Z_imp+1)))

        if D_z.ndim==2:
            # set all charge states to have the same transport
            D_z = np.tile(np.atleast_3d(D_z),(1,1,self.Z_imp+1))  # include elements for neutrals
            # unless specified, D_z for neutrals should be 0
            D_z[:,:,0] = 0.0
        if V_z.ndim==2:
            V_z = np.tile(np.atleast_3d(V_z),(1,1,self.Z_imp+1))  # include elements for neutrals
            # unless specified, V_z for neutrals should be 0
            V_z[:,:,0] = 0.0

        # Use only f_contiguous arrays for speed in Fortran!
        return _aurora.run(
            len(self.time_out), # number of times at which simulation outputs results
            times_DV,
            D_z, # cm^2/s      #(ir,nt_trans,nion)
            V_z, # cm/s
            self.par_loss_rate,  # time dependent
            self.source_rad_prof,# source profile in radius
            self.S_rates, # ioniz_rate,
            self.R_rates, # recomb_rate,
            self.rvol_grid,
            self.pro_grid,
            self.qpr_grid,
            self.mixing_radius,
            self.decay_length_boundary,
            self.time_grid,
            self.saw_on,
            self.source_time_prof, # source profile in time
            self.save_time,
            self.sawtooth_erfc_width, # dsaw width  [cm, circ geometry]
            self.wall_recycling, # rcl   [fraction]
            self.source_div_fraction, # divbls   [fraction of source into divertor]
            self.tau_div_SOL * 1e-3, # taudiv   [s]
            self.tau_pump *1e-3, # taupump  [s]
            self.tau_rcl_ret *1e-3,   # tauwret  [s]
            self.rvol_lcfs,       # rx = rvol_lcfs + dbound
            self.bound_sep,
            self.lim_sep,
            self.prox_param,
            rn_t0 = nz_init,  # if omitted, internally set to 0's
            linder=True if method=='linder' else False,
            evolneut=evolneut
        )





    def run_julia(self, times_DV, D_z, V_z, nz_init=None):
        ''' Run a single simulation using the Julia version.

        INPUTS
        -------------
        times_DV : 1D array
            Array of times at which D_z and V_z profiles are given. (Note that it is assumed that
            D and V profiles are already on the self.rvol_grid radial grid).
        D_z, V_z: arrays, shape of (space, nZ, time)
            Diffusion and convection coefficients, in units of cm^2/s and cm/s, respectively.
            This may be given as a function of (space,time) or (space,nZ, time), where nZ indicates
            the number of charge states. If inputs are found to be have only 2 dimensions, it is
            assumed that all charge states should be set to have the same transport coefficients.
        nz_init: array, shape of (space, nZ)
            Impurity charge states at the initial time of the simulation. If left to None, this is
            internally set to an array of 0's.

        OUTPUTS:
        -------------
        out : list
            List containing each particle reservoir of a simulation, i.e.
            nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out
        '''

        from julia.api import Julia
        import os

        jl = Julia(compiled_modules=False, sysimage=os.path.dirname(os.path.realpath(__file__)) + "/jlib/aurora.so")
        from julia import aurora as aurora_jl

        if nz_init is None:
            # default: start in a state with no impurity ions
            nz_init = np.zeros((len(self.rvol_grid),int(self.Z_imp+1)))

        if D_z.ndim==2:
            # set all charge states to have the same transport
            D_z = np.tile(np.atleast_3d(D_z),(1,1,self.Z_imp+1))  # include elements for neutrals
        if V_z.ndim==2:
            V_z = np.tile(np.atleast_3d(V_z),(1,1,self.Z_imp+1))  # include elements for neutrals

        
        # NOTE: use only f_contiguous arrays for speed in Julia, like in Fortran!
        return aurora_jl.run(len(self.time_out),  # number of times at which simulation outputs results
                             times_DV,
                             D_z,
                             V_z, # cm/s
                             self.par_loss_rate,  # time dependent
                             self.source_rad_prof,# source profile in radius
                             self.S_rates, # ioniz_rate,
                             self.R_rates, # recomb_rate,
                             self.rvol_grid,
                             self.pro_grid,
                             self.qpr_grid,
                             self.mixing_radius,
                             self.decay_length_boundary,
                             self.time_grid,
                             self.saw_on,
                             self.source_time_prof, # source profile in time
                             self.save_time,
                             self.sawtooth_erfc_width, # dsaw width  [cm, circ geometry]
                             self.wall_recycling, # rcl   [fraction]
                             self.source_div_fraction, # divbls   [fraction of source into divertor]
                             self.tau_div_SOL * 1e-3, # taudiv   [s]
                             self.tau_pump *1e-3, # taupump  [s]
                             self.tau_rcl_ret *1e-3,   # tauwret  [s]
                             self.rvol_lcfs,       # rx = rvol_lcfs + dbound
                             self.bound_sep,
                             self.lim_sep,
                             self.prox_param,
                             nz_init)
