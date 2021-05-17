'''This module includes the core class to set up simulations with :py:mod:`aurora`. The :py:class:`~aurora.core.aurora_sim` takes as input a namelist dictionary and a g-file dictionary (and possibly other optional argument) and allows creation of grids, interpolation of atomic rates and other steps before running the forward model.
'''

import os,sys
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.constants import e as q_electron, m_p
import pickle as pkl

from . import interp
from . import atomic
from . import grids_utils
from . import source_utils
from . import particle_conserv
from . import plot_tools
from . import synth_diags
from . import adas_files

class aurora_sim:
    '''Setup the input dictionary for an Aurora ion transport simulation from the given namelist.

    Parameters
    ----------
    namelist : dict
        Dictionary containing aurora inputs. See default_nml.py for some defaults, 
        which users should modify for their runs.
    geqdsk : dict, optional
        EFIT gfile as returned after postprocessing by the :py:mod:`omfit_classes.omfit_eqdsk` 
        package (OMFITgeqdsk class). If left to None (default), the geqdsk dictionary 
        is constructed starting from the gfile in the MDS+ tree indicated in the namelist.
    nbi_cxr : array, optional
        If namelist['nbi_cxr']=True, this array represents the charge exchange rates 
        with NBI neutrals, fast and/or thermal, across the entire radius and on the 
        time base of interest. 
        Creating this input is not trivial and must be done externally to aurora. 
        General steps:
        - get density of fast NBI neutrals (both fast and thermal/halo) ---> n0_nbi, n0_halo
        - get total rates (n-unresolved) for CX with NBI neutrals --> _alpha_CX_NBI_rates
        - thermal rates for the halo may be from ADAS CCD files or from the same methods used 
        for fast neutrals
        - sum n0_nbi *  alpha_CX_NBI_rates + n0_halo * alpha_CX_rates
        This method still needs more testing within this class. Contact sciortino-at-psfc.mit.edu for details. 
          
    '''
    def __init__(self, namelist, geqdsk=None, nbi_cxr=None):

        if namelist is None:
            # option useful for calls like omfit_classes.OMFITaurora(filename)
            # A call like omfit_classes.OMFITaurora('test', namelist, geqdsk=geqdsk) is also possible
            # to initialize the class as a dictionary.
            return
        
        self.namelist = namelist
        self.kin_profs = namelist['kin_profs']
        self.nbi_cxr = nbi_cxr
        self.imp = namelist['imp']


        if geqdsk is None:
            # import omfit_eqdsk here to avoid issues with docs and packaging
            from omfit_classes import omfit_eqdsk
            # Fetch geqdsk from MDS+ (using EFIT01) and post-process it using the OMFIT geqdsk format.
            self.geqdsk = omfit_eqdsk.OMFITgeqdsk('').from_mdsplus(
                device=namelist['device'],shot=namelist['shot'],
                time=namelist['time'], SNAPfile='EFIT01',
                fail_if_out_of_range=False,
                time_diff_warning_threshold=20
            )
        else:
            self.geqdsk = geqdsk

        self.Raxis_cm = self.geqdsk['RMAXIS']*100. # cm
        self.namelist['Baxis'] = self.geqdsk['BCENTR']
        
        # specify which atomic data files should be used -- use defaults unless user specified in namelist
        atom_files = {}
        atom_files['acd'] = self.namelist.get('acd',adas_files.adas_files_dict()[self.imp]['acd'])
        atom_files['scd'] = self.namelist.get('scd',adas_files.adas_files_dict()[self.imp]['scd'])
        if self.namelist['cxr_flag']:
            atom_files['ccd'] = self.namelist.get('ccd', adas_files.adas_files_dict()[self.imp]['ccd'])

        # now load ionization and recombination rates
        self.atom_data = atomic.get_atom_data(self.imp,files=atom_files)

        # set up radial and temporal grids
        self.setup_grids()

        # set up kinetic profiles and atomic rates
        self.setup_kin_profs_depts()

        # import here to avoid issues when building docs or package
        from omfit_classes.utils_math import atomic_element

        # get nuclear charge Z and atomic mass number A
        out = atomic_element(symbol=self.imp)
        spec = list(out.keys())[0]
        self.Z_imp = int(out[spec]['Z'])
        self.A_imp = int(out[spec]['A'])
        
        self.reload_namelist()
        
    def reload_namelist(self, namelist=None):
    
        '''reloadd namelist in the case that any of the input scalar parameters has changed 
        '''
        if namelist is not None:
            self.namelist = namelist
        
        # Extract other inputs from namelist:
        self.mixing_radius = self.namelist['saw_model']['rmix']
        self.decay_length_boundary = self.namelist['SOL_decay']
        self.wall_recycling = self.namelist['wall_recycling']
        self.source_div_fraction = self.namelist['divbls']   # change of nomenclature
        self.tau_div_SOL_ms = self.namelist['tau_div_SOL_ms']
        self.tau_pump_ms = self.namelist['tau_pump_ms']
        self.tau_rcl_ret_ms = self.namelist['tau_rcl_ret_ms']
        
        # if recycling flag is set to False, avoid any divertor return flows
        # To include divertor return flows but no recycling, set wall_recycling=0
        if not self.namelist['recycling_flag']:
            self.wall_recycling = -1.0  # no divertor return flows

        self.bound_sep = self.namelist['bound_sep']
        self.lim_sep = self.namelist['lim_sep']
        self.sawtooth_erfc_width = self.namelist['saw_model']['crash_width']
        self.cxr_flag = self.namelist['cxr_flag']
        self.nbi_cxr_flag = self.namelist['nbi_cxr_flag']

        
    def save(self, filename):
        '''Save state of `aurora_sim` object.
        '''
        with open(filename, 'wb') as f:
            pkl.dump(self, f)

    def load(self, filename):
        '''Load `aurora_sim` object.
        '''
        with open(filename,'rb') as f:
            obj= pkl.load(f)
        self.__dict__.update(obj.__dict__)
    
    def save_dict(self):
        return self.__dict__
    
    def load_dict(self, aurora_dict):
        self.__dict__.update(aurora_dict) 
        
        

        
        
            
    def setup_grids(self):
        '''Method to set up radial and temporal grids given namelist inputs.
        '''
        
        # Get r_V to rho_pol mapping
        rho_pol, _rvol = grids_utils.get_rhopol_rvol_mapping(self.geqdsk)
        rvol_lcfs = interp1d(rho_pol,_rvol)(1.0)
        self.rvol_lcfs = self.namelist['rvol_lcfs'] = np.round(rvol_lcfs,3)  # set limit on accuracy
        
        # create radial grid
        grid_params = grids_utils.create_radial_grid(self.namelist, plot=False)
        self.rvol_grid,self.pro_grid,self.qpr_grid,self.prox_param = grid_params

        # get rho_poloidal grid corresponding to aurora internal (rvol) grid
        self.rhop_grid = interp1d(_rvol,rho_pol, fill_value='extrapolate')(self.rvol_grid)
        self.rhop_grid[0] = 0.0 # enforce on axis
        
        # Save R on LFS and HFS
        self.Rhfs, self.Rlfs = grids_utils.get_HFS_LFS(self.geqdsk, rho_pol=self.rhop_grid)

        # define time grid ('timing' must be in namelist)
        self.time_grid, self.save_time = grids_utils.create_time_grid(timing=self.namelist['timing'], plot=False)
        self.time_out = self.time_grid[self.save_time]

        # create array of 0's of length equal to self.time_grid, with 1's where sawteeth must be triggered
        self.saw_on = np.zeros_like(self.time_grid)
        input_saw_times = self.namelist['saw_model']['times']
        self.saw_times = np.array(input_saw_times)[input_saw_times<self.time_grid[-1]]
        if self.namelist['saw_model']['saw_flag'] and len(self.saw_times)>0:
            self.saw_on[self.time_grid.searchsorted(self.saw_times)] = 1

    def setup_kin_profs_depts(self):
        '''Method to set up Aurora inputs related to the kinetic background from namelist inputs.
        '''        
        # get kinetic profiles on the radial and (internal) temporal grids
        self._ne,self._Te,self._Ti,self._n0 = self.get_aurora_kin_profs()

        # store also kinetic profiles on output time grid
        if len(self._ne) > 1: #all have the same shape now
            save_time = self.save_time
        else:
            save_time = [0]
            
        self.ne = self._ne[save_time,:]
        self.Te = self._Te[save_time,:]
        self.Ti = self._Ti[save_time,:]
        self.n0 = self._n0[save_time,:]
        
            
        # Get time-dependent parallel loss rate
        self.par_loss_rate = self.get_par_loss_rate()

        # Obtain atomic rates on the computational time and radial grids
        self.S_rates, self.R_rates = self.get_time_dept_atomic_rates()
        
        S0 = self.S_rates[:,0,:]
        # get radial profile of source function
        if len(save_time) == 1:  # if time averaged profiles were used
            S0 = S0[:, [0]]  # 0th charge state (neutral)

    
        if np.ndim(self.namelist['explicit_source_vals'])==2:
            # interpolate explicit source values on time and rhop grids of simulation
            source_rad_prof = RectBivariateSpline(self.namelist['explicit_source_rhop'],
                                                  self.namelist['explicit_source_time'],
                                                  self.namelist['explicit_source_vals'].T,
                                                  kx=1,ky=1)(self.rhop_grid, self.time_grid)
            
            # get first ionization stage
            pnorm = np.pi*np.sum(source_rad_prof*S0*(self.rvol_grid/self.pro_grid)[:,None],0)
            self.source_time_history = np.asfortranarray(pnorm)
            
            # neutral density for influx/unit-length = 1/cm
            self.source_rad_prof = np.asfortranarray(source_rad_prof/pnorm)

        else:
            # get time history and radial profiles separately
            self.source_time_history = source_utils.get_source_time_history(
                self.namelist, self.Raxis_cm, self.time_grid
            )
            
            # get radial profile of source function for each time step
            self.source_rad_prof = source_utils.get_radial_source(self.namelist,
                                                                  self.rvol_grid, self.pro_grid,
                                                                  S0,   # 0th charge state (neutral) and 0th time
                                                                  len(self.time_grid),self._Ti)
    
    
    

    def interp_kin_prof(self, prof): 
        ''' Interpolate the given kinetic profile on the radial and temporal grids [units of s].
        This function extrapolates in the SOL based on input options using the same methods as in STRAHL.
        '''
        times = self.kin_profs[prof]['times']

        r_lcfs = np.interp(1,self.rhop_grid,self.rvol_grid)

        # extrapolate profiles outside of LCFS by exponential decays
        r = interp1d(self.rhop_grid, self.rvol_grid,fill_value='extrapolate')(self.kin_profs[prof]['rhop'])
        if self.kin_profs[prof]['fun'] == 'interp':
            if 'decay' not in self.kin_profs[prof]:
                # if decay length in the SOL was not given by the user, assume a decay length of 1cm
                print(f'Namelist did not provide a {prof} decay length for the SOL. Setting it to 1cm.')
                self.kin_profs[prof]['decay'] = np.ones(len(self.kin_profs[prof]['vals']))
                
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
        '''Get kinetic profiles on radial and time grids.
        '''
        # ensure 2-dimensional inputs:
        self.kin_profs['ne']['vals'] = np.atleast_2d(self.kin_profs['ne']['vals'])
        self.kin_profs['Te']['vals'] = np.atleast_2d(self.kin_profs['Te']['vals'])
        
        Te = self.interp_kin_prof('Te')
        ne = self.interp_kin_prof('ne')

        if 'Ti' in self.kin_profs and 'vals' in self.kin_profs['Ti']:
            self.kin_profs['Ti']['vals'] = np.atleast_2d(self.kin_profs['Ti']['vals'])
            Ti = self.interp_kin_prof('Ti')
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

        # make sure that Te,ne,Ti and n0 have the same shape at this stage
        ne,Te,Ti,n0 = np.broadcast_arrays(ne,Te,Ti,n0)

        return ne,Te,Ti,n0


    def get_time_dept_atomic_rates(self):
        '''Obtain time-dependent ionization and recombination rates for a simulation run.
        If kinetic profiles are given as time-independent, atomic rates for each time slice
        will be set to be the same.
        '''
        lne = np.log10(self._ne)
        lTe = np.log10(self._Te)

        # get electron impact ionization and radiative recombination rates in units of [s^-1]

        S_rates = atomic.interp_atom_prof(self.atom_data['scd'],lne, lTe, x_multiply=True)
        R_rates = atomic.interp_atom_prof(self.atom_data['acd'],lne, lTe, x_multiply=True)

        if self.namelist['cxr_flag']:
            # include thermal charge exchange recombination
            lTi = np.log10(self._Ti)
            alpha_CX_rates = atomic.interp_atom_prof(self.atom_data['ccd'], lne, lTi, x_multiply=False)

            # change rates from units of [1/s/cm^3] to [1/s] - select only relevant CCD ion stages (useful for Foster scaling)
            R_rates += self._n0[:,None] * alpha_CX_rates[:,:R_rates.shape[1],:]   

        if self.namelist['nbi_cxr_flag']:
            # include charge exchange between NBI neutrals and impurities
            R_rates += self.nbi_cxr.transpose(1,0,2)

        # nz=nion of rates arrays must be filled with zeros - final shape: (nr,nion,nt)        
        S_rates_t = np.zeros((S_rates.shape[2], S_rates.shape[1] + 1, self.time_grid.size), order='F')
        S_rates_t[:, :-1] = S_rates.T

        R_rates_t = np.zeros((R_rates.shape[2], R_rates.shape[1] + 1, self.time_grid.size), order='F')
        R_rates_t[:, :-1] = R_rates.T

        return S_rates_t, R_rates_t


    def get_par_loss_rate(self, trust_SOL_Ti=False):
        '''Calculate the parallel loss frequency on the radial and temporal grids [1/s].

        trust_SOL_Ti should generally be set to False, unless specific Ti measurements are available
        in the SOL.
        
        '''
        # import here to avoid issues when building docs or package
        from omfit_classes.utils_math import atomic_element

        # background mass number (=2 for D)
        self.main_element = self.namelist['main_element']
        out = atomic_element(symbol=self.namelist['main_element'])
        spec = list(out.keys())[0]
        self.main_ion_A = int(out[spec]['A'])
        self.main_ion_Z = int(out[spec]['Z'])

        # factor for v = machnumber * sqrt((3T_i+T_e)k/m)
        vpf = self.namelist['SOL_mach']*np.sqrt(q_electron/m_p/self.main_ion_A)  
        # v[m/s]=vpf*sqrt(T[ev])

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






    def run_aurora(self, D_z, V_z,
                   times_DV=None, nz_init=None, superstages = [], unstage=False,
                   alg_opt=1, evolneut=False, use_julia=False, plot=False):
        '''Run a simulation using inputs in the given dictionary and diffusion and convection profiles 
        as a function of space, time and potentially also ionization state. Users may give an initial 
        state of each ion charge state as an input.

        Results can be conveniently visualized with time-slider using

        .. code-block:: python

            aurora.slider_plot(rhop,time, nz.transpose(1,2,0),
                               xlabel=r'$\\rho_p$', ylabel='time [s]', 
                               zlabel=r'$n_z$ [cm$^{-3}$]', plot_sum=True,
                               labels=[f'Ca$^{{{str(i)}}}$' for i in np.arange(nz_w.shape[1]])

        Parameters
        ----------
        D_z: array, shape of (space,time,nZ) or (space,time) or (space,)
            Diffusion coefficients, in units of :math:`cm^2/s`.
            This may be given as a function of space only, (space,time) or (space,nZ, time), 
            where nZ indicates the number of charge states. If given with 1 or 2 dimensions, 
            it is assumed that all charge states should have the same diffusion coefficients.
            If given as 1D, it is further assumed that diffusion is time-independent. 
            Note that it is assumed that radial profiles are already on the self.rvol_grid radial grid.
        V_z: array, shape of (space,time,nZ) or (space,time) or (space,)
            Convection coefficients, in units of :math:`cm/s`.
            This may be given as a function of space only, (space,time) or (space,nZ, time), 
            where nZ indicates the number of charge states. If given with 1 or 2 dimensions, 
            it is assumed that all charge states should have the same convection coefficients.
            If given as 1D, it is further assumed that convection is time-independent. 
            Note that it is assumed that radial profiles are already on the self.rvol_grid radial grid.
        times_DV : 1D array, optional
            Array of times at which `D_z` and `V_z` profiles are given. By Default, this is None, 
            which implies that `D_z` and `V_z` are time independent. 
        nz_init: array, shape of (space, nZ)
            Impurity charge states at the initial time of the simulation. If left to None, this is
            internally set to an array of 0's.
        superstages : list or 1D array
            Indices of charge states of chosen ion that should be modeled. If left empty, all ion stages
            are modeled. If only some indices are given, these are modeled as "superstages".
            If `D_z` and `V_z` are given as a function of charge state, only the indices corresponding
            to the superstages are used.
            NB: users must explicitly add the element 0 (neutral stage) to the list of superstages, 
            or else an exception will be raised.
        unstage : bool, optional
            If a list of superstages are provided, this parameter sets whether the output should be "unstaged"
            by multiplying by the appropriate fractional abundances of all charge states at ionization 
            equilibrium. Note that this unstaging process cannot account for transport and is therefore
            only an approximation, to be used carefully.
        alg_opt : int, optional
            If `alg_opt=1`, use the finite-volume algorithm proposed by Linder et al. NF 2020. 
            If `alg_opt=0`, use the older finite-differences algorithm in the 2018 version of STRAHL.
        evolneut : bool, optional
            If True, evolve neutral impurities based on their D,V coefficients. Default is False, in
            which case neutrals are only taken as a source and those that are not ionized immediately after
            injection are neglected.
            This option is NOT CURRENTLY RECOMMENDED, because this method is still under development/
            examination. 
        use_julia : bool, optional
            If True, run the Julia pre-compiled version of the code. Run the julia makefile option to set 
            this up. Default is False (still under development)
        plot : bool, optional
            If True, plot density for each charge state using a convenient slides over time and check 
            particle conservation in each particle reservoir. 

        Returns
        -------
        nz : array, (nr,nZ,nt)
            Charge state densities [:math:`cm^{-3}`] over the space and time grids.
            If a number of superstages are indicated in the input, only charge state densities for
            these are returned.
        N_wall : array (nt,)
            Number of particles at the wall reservoir over time.
        N_div : array (nt,)
            Number of particles in the divertor reservoir over time.
        N_pump : array (nt,)
            Number of particles in the pump reservoir over time.
        N_ret : array (nt,)
             Number of particles temporarily held in the wall reservoirs. 
        N_tsu : array (nt,)
             Edge particle loss [:math:`cm^{-3}`]
        N_dsu : array (nt,)
             Parallel particle loss [:math:`cm^{-3}`]
        N_dsul : array (nt,)
             Parallel particle loss at the limiter [:math:`cm^{-3}`]
        rcld_rate : array (nt,)
             Recycling from the divertor [:math:`s^{-1} cm^{-3}`]
        rclw_rate : array (nt,)
             Recycling from the wall [:math:`s^{-1} cm^{-3}`]
        '''
        
        # D_z and V_z must have the same shape
        assert np.array(D_z).shape == np.array(V_z).shape
        
        if (times_DV is None) and (D_z.ndim>1 or V_z.ndim>1):
            raise ValueError('D_z and V_z given as time dependent, but times were not specified!')
        
        if superstages is None:
            superstages = []
            
        if len(superstages):

            superstages = np.array(superstages)
            
            if 1 not in superstages:
                print('Warning: 1th superstage needs to be included')
                superstages = np.r_[1,superstages]
            if 0 not in superstages:
                print('Warning: 0th superstage for neutral was included')
                superstages = np.r_[0,superstages]
            if np.any(np.diff(superstages)<=0):
                print('Warning: 0th superstage for neutral was included')
                superstages = np.sort(superstages)
            if superstages[-1] > self.Z_imp:
                raise Exception('The higher superstage must be less than Z_imp = %d'%self.Z_imp)

            ##the last ion must have zero ionisation rate
            R_rates = self.R_rates[:,np.r_[superstages[1:]-1,-1]]
            S_rates = self.S_rates[:,np.r_[superstages[1:]-1,-1]]

            _superstages = np.copy(superstages)
            _superstages[-1] = self.Z_imp

            for i in range(len(superstages)-1):
                if superstages[i]+1!= superstages[i+1]:
                    sind = slice(superstages[i]-1, superstages[i+1]-1)
 
                    rate_ratio =  self.S_rates[:,sind]/self.R_rates[:,sind]
                    fz = np.cumprod(rate_ratio, axis=1) 
                    #bundled stages can have a very high values,
                    #which reduces a numerical stability of calculation
                    #np.minimum(rate,1) prevents numerical instability, any better solution???
                    R_rates[:,i] /=  fz[:,-1]/fz.sum(1)
                    S_rates[:,i-1] /= fz[:,0]/fz.sum(1)
                    #R_rates[:,i] = np.minimum(R_rates[:,i],1)
                    #S_rates[:,i] = np.minimum(S_rates[:,i],1)

            num_cs = len(superstages)
   
            if D_z.ndim==3:
                D_z = D_z[:,:,superstages]
                V_z = V_z[:,:,superstages]
                

        else:
            num_cs = int(self.Z_imp+1)
            S_rates = self.S_rates
            R_rates = self.R_rates

        #prevent recombination back to neutral state, these neutrals were lost causing error in particle conservation 
        if not evolneut:
            R_rates[:,0] = 0
        
        #update 1D sources if the namelist has changed
        if np.ndim(self.namelist['explicit_source_vals']) < 2:
            # get time history and radial profiles separately
            self.source_time_history = source_utils.get_source_time_history(
                    self.namelist, self.Raxis_cm, self.time_grid
                )
                
            
        if nz_init is None:
            # default: start in a state with no impurity ions
            nz_init = np.zeros((len(self.rvol_grid),num_cs))


        if D_z.ndim < 3:
            # set all charge states to have the same transport
            # num_cs = Z+1 - include elements for neutrals

            D_z =  np.tile(D_z.T, (num_cs, 1, 1)).T # create fortran contiguous arrays
            V_z =  np.tile(V_z.T, (num_cs, 1, 1)).T
            
            D_z[:,:,0] = 0.0
            V_z[:,:,0] = 0.0
            if np.size(times_DV) == 0:
                times_DV = [1.] # dummy, no time dependence
 
        nt = len(self.time_out)

        # NOTE: for both Fortran and Julia, use f_configuous arrays for speed!
        if use_julia:
            # run Julia version of the code
            from julia.api import Julia
            jl = Julia(compiled_modules=False,
                       sysimage=os.path.dirname(os.path.realpath(__file__)) + "/../aurora.jl/sysimage.so")
            from julia import aurora as aurora_jl

            self.res = aurora_jl.run(nt,  # number of times at which simulation outputs results
                                     times_DV,
                                     D_z, V_z, # cm^2/s & cm/s    #(ir,nt_trans,nion)
                                     self.par_loss_rate,  # time dependent
                                     self.source_rad_prof,# source profile in radius
                                     S_rates, # ioniz_rate,
                                     R_rates, # recomb_rate,
                                     self.rvol_grid, self.pro_grid, self.qpr_grid,
                                     self.mixing_radius, self.decay_length_boundary,
                                     self.time_grid, self.saw_on,
                                     self.source_time_history, # source profile in time
                                     self.save_time, self.sawtooth_erfc_width, # dsaw width  [cm, circ geometry]
                                     self.wall_recycling,
                                     self.source_div_fraction, # divbls [fraction of source into divertor]
                                     self.tau_div_SOL_ms * 1e-3, self.tau_pump_ms *1e-3, self.tau_rcl_ret_ms *1e-3,  #[s] 
                                     self.rvol_lcfs, self.bound_sep, self.lim_sep, self.prox_param,
                                     nz_init, alg_opt, evolneut)
        else:
            # import here to avoid import when building documentation or package (negligible slow down)
            from ._aurora import run as fortran_run
            
            
            self.res =  fortran_run(nt,  # number of times at which simulation outputs results
                                    times_DV,
                                    D_z, V_z, # cm^2/s & cm/s    #(ir,nt_trans,nion)
                                    self.par_loss_rate,  # time dependent
                                    self.source_rad_prof,# source profile in radius
                                    S_rates, # ioniz_rate,
                                    R_rates, # recomb_rate,
                                    self.rvol_grid, self.pro_grid, self.qpr_grid,
                                    self.mixing_radius, self.decay_length_boundary,
                                    self.time_grid, self.saw_on,
                                    self.source_time_history, # source profile in time
                                    self.save_time, self.sawtooth_erfc_width, # dsaw width  [cm, circ geometry]
                                    self.wall_recycling,
                                    self.source_div_fraction, # divbls [fraction of source into divertor]
                                    self.tau_div_SOL_ms * 1e-3, self.tau_pump_ms *1e-3, self.tau_rcl_ret_ms *1e-3,  # [s]  
                                    self.rvol_lcfs, self.bound_sep, self.lim_sep, self.prox_param,
                                    rn_t0 = nz_init,  # if omitted, internally set to 0's
                                    alg_opt=alg_opt,
                                    evolneut=evolneut)
        
        if plot:

            # plot charge state density distributions over radius and time
            plot_tools.slider_plot(self.rvol_grid, self.time_out, self.res[0].transpose(1,0,2),
                                          xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'$n_z$ [$cm^{-3}$]',
                                          labels=[str(i) for i in np.arange(0,self.res[0].shape[1])],
                                          plot_sum=True, x_line=self.rvol_lcfs)
            
            # check particle conservation by summing over simulation reservoirs
            _ = self.check_conservation(plot=True)
        
 
        if unstage and len(superstages):
            # "unstage" superstages to recover estimates for density of all charge states
            nz_unstaged = np.zeros(( len(self.rvol_grid), self.Z_imp+1, nt))
            _superstages = np.copy(superstages)
            _superstages[-1] = self.Z_imp
            
            for i in range(len(superstages)):
                if i > 0 and _superstages[i-1]+1 < _superstages[i]:
                    # fill skipped stages from ionization equilibrium
                    ind = slice(_superstages[i-1],_superstages[i])
                    S = self.S_rates[:,ind,self.save_time] 
                    R = self.R_rates[:,ind,self.save_time]
                    fz = np.cumprod(S/R,axis=1)
                    fz /= np.maximum(1e-5,fz.sum(1))[:,None] # prevents zero division
                    
                    # split the superstage into the separate stages using ionization equilibrium
                    nz_unstaged[:,_superstages[i-1]+1:_superstages[i]+1] = self.res[0][:,[i]]*fz
                else:
                    nz_unstaged[:,_superstages[i]] = self.res[0][:,i]
 
            self.res = nz_unstaged, *self.res[1:]
        
        # nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = self.res
        return self.res


    def run_aurora_steady(self, D_z, V_z, nz_init=None, superstages = [], unstage=False,
                          alg_opt=1, evolneut=False, use_julia=False,
                          tolerance=0.01, max_sim_time = 500e-3, dt=1e-4, dt_increase=1.0, 
                          n_steps = 100, plot=False):
        '''Run an AURORA simulation until reaching steady state profiles. This method calls :py:meth:`~aurora.core.run_aurora`
        checking at every iteration whether profile shapes are still changing within a given fractional tolerance.

        Parameters
        ----------
        D_z: array, shape of (space,nZ) or (space,)
            Diffusion coefficients, in units of :math:`cm^2/s`. This may be given as a function of space only or (space,nZ). 
            No time dependence is allowed in this function. Here, nZ indicates the number of charge states.
            Note that it is assumed that radial profiles are already on the self.rvol_grid radial grid.
        V_z: array, shape of (space,nZ) or (space,)
            Convection coefficients, in units of :math:`cm/s`. This may be given as a function of space only or (space,nZ). 
            No time dependence is allowed in this function. Here, nZ indicates the number of charge states.
        nz_init: array, shape of (space, nZ)
            Impurity charge states at the initial time of the simulation. If left to None, this is
            internally set to an array of 0's.
        superstages : list or 1D array
            Indices of charge states of chosen ion that should be modeled. If left empty, all ion stages
            are modeled. See docs for :py:meth:`~aurora.core.run_aurora` for details.
        unstage : bool, optional
            If a list of superstages are provided, this parameter sets whether the output should be "unstaged".
            See docs for :py:meth:`~aurora.core.run_aurora` for details.
        alg_opt : int, optional
            If `alg_opt=1`, use the finite-volume algorithm proposed by Linder et al. NF 2020. 
            If `alg_opt=0`, use the older finite-differences algorithm in the 2018 version of STRAHL.
        evolneut : bool, optional
            If True, evolve neutral impurities based on their D,V coefficients. Default is False.
            See docs for :py:meth:`~aurora.core.run_aurora` for details.
        use_julia : bool, optional
            If True, run the Julia pre-compiled version of the code. See docs for :py:meth:`~aurora.core.run_aurora` for details.
        tolerance : float
            Fractional tolerance in charge state profile shapes. This method reports charge state density profiles obtained when 
            the discrepancy between normalized profiles at adjacent time steps varies by less than this tolerance fraction. 
        max_sim_time : float
            Maximum time in units of seconds for which simulations should be run if a steady state is not found.        
        dt : float
            Initial time step to apply, in units of seconds. This can be increased by a multiplier given by :param:`dt_increase`
            after each time step.
        dt_increase : float
            Multiplier for time steps.
        n_steps : int
            Number of time steps before convergence is checked. 
        plot : bool
            If True, plot time evolution of charge state density profiles to show convergence.
        '''
        if not self.namelist['source_type']=='const':
            raise ValueError('This method is designed to operate with time-independent sources! Modify namelist["source_type"].')

        if not self.ne.shape[0]==1:
            raise ValueError('This method is designed to operate with time-independent background profiles!')

        # round to avoid rounding errors
        dt = np.round(dt,9)
        
        # number of maximum simulations to be run before possibly giving up
        num_sims = int(max_sim_time/(n_steps*dt))

        # build timing dictionary
        self.namelist['timing'] = {'dt_start': np.array([float(dt), float(dt)]),
                                   'dt_increase': np.array([float(dt_increase), 1.]),
                                   'steps_per_cycle': np.array([1, 1]),
                                   'times' : np.array([0. , n_steps*dt])}

        self.setup_grids()

        # update kinetic profile dependencies to get everything to the right shape (TODO: avoid this to reduce computational cost)
        self.setup_kin_profs_depts()
        
        times_DV = None
        if D_z.ndim==2:
            # make sure that transport coefficients were given as a function of space and nZ
            assert D_z.shape[0]==self.rhop_grid and D_z.shape[1]==self.Z_imp+1
            assert V_z.shape[0]==self.rhop_grid and V_z.shape[1]==self.Z_imp+1
            
            # transport coefficients were given as a function of (space,nZ). Need to add fictitious time dependence
            D_z =  np.asfortranarray(np.tile(D_z, (n_steps, 1, 1)).transpose(1,0,2))
            V_z =  np.asfortranarray(np.tile(V_z, (n_steps, 1, 1)).transpose(1,0,2))

            # mock up time dependence of D and V
            times_DV = self.namelist['timing']['times']
            
        # run first iteration
        nz_new = self.run_aurora(D_z, V_z, times_DV, nz_init=nz_init,
                                     superstages = superstages, unstage=unstage, alg_opt=alg_opt,
                                     evolneut=evolneut, use_julia=use_julia, plot=False)[0]
        nz_all = np.copy(nz_new)
        nz_norm = nz_new[:,:,-1]/np.max(nz_new[:,:,-1])
        time_grid = np.copy(self.time_grid)

        if num_sims<1:
            raise ValueError('The input max_sim_time, n_steps and dt do not allow for any iteration to steady state profiles!')

        for i in np.arange(num_sims-1):
            # Update time array
            t0 = np.round(self.time_grid[-1]+dt,6)
            dt0 = np.diff(self.time_grid)[-1]
            t1 = np.round(self.time_grid[-1] + dt0*(1-dt_increase**(n_steps-1))/(1-dt_increase),6)
            self.setup_grids()

            # update kinetic profile dependencies to get everything to the right shape (TODO: avoid this to reduce computational cost)
            self.setup_kin_profs_depts()
        
            # get charge state densities from latest time step
            nz_init_new = nz_all[:,:,-1]
            nz_new = self.run_aurora(D_z, V_z, times_DV, nz_init=nz_init_new,
                                     superstages = superstages, unstage=unstage, alg_opt=alg_opt,
                                     evolneut=evolneut, use_julia=use_julia, plot=False)[0]
            
            nz_all = np.dstack((nz_all, nz_new))
            time_grid = np.concatenate((time_grid, self.time_grid))
            
            # check if normalized profiles have converged
            nz_norm_new = nz_new[:,:,-1]/np.max(nz_new[:,:,-1]) 
            if np.linalg.norm(nz_norm_new-nz_norm,ord=2)<1e-2:   # tolerance of 1%
                break
            
            nz_norm = np.copy(nz_norm_new)

        # store final time grid
        self.time_grid = time_grid
        
        if plot:
            # plot charge state distributions over radius and time
            plot_tools.slider_plot(self.rhop_grid, self.time_grid, nz_all.transpose(1,0,2),
                                          xlabel=r'$\rho_p$', ylabel='time [s]', zlabel=r'$n_z$ [$cm^{-3}$]',
                                          labels=[str(i) for i in np.arange(0,nz_all.shape[1])],
                                          plot_sum=True)
            
        if i==num_sims-1:
            raise ValueError(f'Could not reach convergence before {max_sim_time:.2f} of simulation time!')

        # compute effective particle confinement time from latest few time steps 
        circ = 2*np.pi*self.Raxis_cm # cm
        zvol = circ * np.pi * self.rvol_grid / self.pro_grid

        wh = (self.rhop_grid <= 1.0)
        zvol = zvol[wh]
        var_volint = np.nansum(nz_new[wh]*zvol[:,None,None],axis=(0,1))

        # compute effective particle confinement time (avoid last time point because source is set to 0 there)
        self.tau_imp = np.mean(var_volint[:-1] / self.source_time_history[:-1])
        
        return nz_norm
    
        
    def calc_Zeff(self):
        '''Compute Zeff from each charge state density, using the result of an Aurora simulation.
        The total Zeff change over time and space due to the simulated impurity can be simply obtained by summing 
        over charge states

        Results are stored as an attribute of the simulation object instance. 
        '''
        # This method requires that a simulation has already been run:
        assert hasattr(self,'res')

        # extract charge state densities from the simulation result
        nz = self.res[0]

        # Compute the variation of Zeff from these charge states
        Zmax = nz.shape[1]-1
        Z = np.arange(Zmax+1)
        self.delta_Zeff = nz*(Z*(Z-1))[None,:,None]   # for each charge state
        self.delta_Zeff/= self.ne.T[:,None,:]


    def plot_resolutions(self):
        '''Convenience function to show time and spatial resolution in Aurora simulation setup. 
        '''
        # display radial resolution
        _ = grids_utils.create_radial_grid(self.namelist, plot=True)
        
        # display time resolution
        _ = grids_utils.create_time_grid(timing=self.namelist['timing'], plot=True)

    def check_conservation(self, plot=True, axs=None, plot_resolutions=False):
        '''Check particle conservation for an aurora simulation.

        Parameters
        -----------------
        plot : bool, optional
            If True, plot time histories in each particle reservoir and display quality of particle conservation.
        axs : matplotlib.Axes instances, optional 
            Axes to pass to :py:meth:`~aurora.particle_conserv.check_particle_conserv`
            These may be the axes returned from a previous call to this function, to overlap 
            results for different runs. 
        
        Returns
        ------------
        out : dict
            Dictionary containing density of particles in each reservoir.
        axs : matplotlib.Axes instances , only returned if plot=True
            New or updated axes returned by :py:meth:`~aurora.particle_conserv.check_particle_conserv`
        '''
        import xarray # import only if necessary
        
        nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = self.res
        nz = nz.transpose(2,1,0)   # time,nZ,space

        if self.namelist['explicit_source_vals'] is None:
            source_time_history = self.source_time_history
        else:
            # if explicit source was provided, all info about the source is in the source_rad_prof array
            srp = xarray.Dataset({'source': ([ 'time','rvol_grid'], self.source_rad_prof.T),
                                 'pro': (['rvol_grid'], self.pro_grid), 
                                 'rhop_grid': (['rvol_grid'], self.rhop_grid)
            },
                                coords={'time': self.time_out, 
                                        'rvol_grid': self.rvol_grid
                                })
            source_time_history = particle_conserv.vol_int(self.Raxis_cm, srp, 'source')

        source_time_history = self.source_time_history
        # Check particle conservation
        ds = xarray.Dataset({'impurity_density': ([ 'time', 'charge_states','rvol_grid'], nz),
                         'source_time_history': (['time'], source_time_history ),
                         'particles_in_divertor': (['time'], N_div), 
                         'particles_in_pump': (['time'], N_pump), 
                         'parallel_loss': (['time'], N_dsu), 
                         'parallel_loss_to_limiter': (['time'], N_dsul), 
                         'edge_loss': (['time'], N_tsu), 
                         'particles_at_wall': (['time'], N_wall), 
                         'particles_retained_at_wall': (['time'], N_ret), 
                         'recycling_from_wall':  (['time'], rclw_rate), 
                         'recycling_from_divertor':  (['time'], rcld_rate), 
                         'pro': (['rvol_grid'], self.pro_grid), 
                         'rhop_grid': (['rvol_grid'], self.rhop_grid)
                         },
                        coords={'time': self.time_out, 
                                'rvol_grid': self.rvol_grid,
                                'charge_states': np.arange(nz.shape[1])
                                })

        return particle_conserv.check_particle_conserv(self.Raxis_cm, ds = ds, plot=plot, axs=axs)


    def centrifugal_asym(self, omega, Zeff, plot=False):
        """Estimate impurity poloidal asymmetry effects from centrifugal forces. See notes the 
        :py:func:`~aurora.synth_diags.centrifugal_asym` function docstring for details.

        In this function, we use the average Z of the impurity species in the Aurora simulation result, using only
        the last time slice to calculate fractional abundances. The CF lambda factor

        Parameters
        -----------------
        omega : array (nt,nr) or (nr,) [ rad/s ] 
             Toroidal rotation on Aurora temporal time_grid and radial rhop_grid (or, equivalently, rvol_grid) grids.
        Zeff : array (nt,nr), (nr,) or float
             Effective plasma charge on Aurora temporal time_grid and radial rhop_grid (or, equivalently, rvol_grid) grids.
             Alternatively, users may give Zeff as a float (taken constant over time and space).
        plot : bool
            If True, plot asymmetry factor :math:`\lambda` vs. radius

        Returns
        ------------
        CF_lambda : array (nr,)
            Asymmetry factor, defined as :math:`\lambda` in the :py:func:`~aurora.synth_diags.centrifugal_asym` function
            docstring.
        """
        fz = self.res[0][...,-1] / np.sum(self.res[0][...,-1],axis=1)[:,None]
        Z_ave_vec = np.mean(self.Z_imp * fz * np.arange(self.Z_imp+1)[None,:],axis=1)
        
        self.CF_lambda = synth_diags.centrifugal_asym(self.rhop_grid, self.Rlfs, omega, Zeff, 
                                                      self.A_imp, Z_ave_vec, 
                                                      self.Te, self.Ti, main_ion_A=self.main_ion_A, 
                                                      plot=plot, nz=self.res[0][...,-1], geqdsk=self.geqdsk).mean(0)

        return self.CF_lambda
