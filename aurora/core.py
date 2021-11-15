'''This module includes the core class to set up simulations with :py:mod:`aurora`. The :py:class:`~aurora.core.aurora_sim` takes as input a namelist dictionary and a g-file dictionary (and possibly other optional argument) and allows creation of grids, interpolation of atomic rates and other steps before running the forward model.
'''
# MIT License
#
# Copyright (c) 2021 Francesco Sciortino
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os,sys
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.constants import e as q_electron, m_p
import pickle as pkl
from copy import deepcopy
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from . import interp
from . import atomic
from . import grids_utils
from . import source_utils
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
          
    '''
    def __init__(self, namelist, geqdsk=None):

        if namelist is None:
            # option useful for calls like omfit_classes.OMFITaurora(filename)
            # A call like omfit_classes.OMFITaurora('test', namelist, geqdsk=geqdsk) is also possible
            # to initialize the class as a dictionary.
            return

        # make sure that any changes in namelist will not propagate back to the calling function
        self.namelist = deepcopy(namelist)
        self.kin_profs = self.namelist['kin_profs']
        self.imp = namelist['imp']

        # import here to avoid issues when building docs or package
        from omfit_classes.utils_math import atomic_element

        # get nuclear charge Z and atomic mass number A
        out = atomic_element(symbol=self.imp)
        spec = list(out.keys())[0]
        self.Z_imp = int(out[spec]['Z'])
        self.A_imp = int(out[spec]['A'])
        
        self.reload_namelist()
        
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

        # allow for ion superstaging
        self.superstages = self.namelist.get('superstages',[])

        # set up radial and temporal grids
        self.setup_grids()

        # set up kinetic profiles and atomic rates
        self.setup_kin_profs_depts()
        

    def reload_namelist(self, namelist=None):
        '''(Re-)load namelist to update scalar variables.
        '''
        if namelist is not None:
            self.namelist = namelist
        
        # Extract other inputs from namelist:
        self.mixing_radius = self.namelist['saw_model']['rmix']
        self.decay_length_boundary = self.namelist['SOL_decay']
        self.wall_recycling = self.namelist['wall_recycling']
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
        if len(self._ne) > 1: # all have the same shape now
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
        self.S_rates, self.R_rates = self.get_time_dept_atomic_rates(
            superstages = self.namelist.get('superstages',[]))

        S0 = self.S_rates[:,0,:]
        # get radial profile of source function
        if len(save_time) == 1:  # if time averaged profiles were used
            S0 = S0[:, [0]]  # 0th charge state (neutral)

        if self.namelist['source_type'] == 'arbitrary_2d_source':
            # interpolate explicit source values on time and rhop grids of simulation
            # NB: explicit_source_vals should be in units of particles/s/cm^3 <-- ionization rate
            srho = self.namelist['explicit_source_rhop']
            stime = self.namelist['explicit_source_time']
            source = np.array(self.namelist['explicit_source_vals']).T
            
            spl = RectBivariateSpline(srho,stime,source,kx=1,ky=1)
            #extrapolate by the nearest values
            self.source_rad_prof = spl(np.clip(self.rhop_grid, min(srho),  max(srho)), 
                                np.clip(self.time_grid, min(stime), max(stime)))
            # Change units to particles/cm^3
            self.src_core = self.source_rad_prof/S0
        else:
            # get time history and radial profiles separately
            source_time_history = source_utils.get_source_time_history(
                self.namelist, self.Raxis_cm, self.time_grid)  # units of particles/s/cm
            
            # get radial profile of source function for each time step
            # dimensionless, normalized such that pnorm=1
            # i.e. source_time_history*source_rad_prof = # particles/cm^3
            self.source_rad_prof = source_utils.get_radial_source(
                self.namelist,
                self.rvol_grid, self.pro_grid,
                S0,   # 0th charge state (neutral)
                self._Ti)   
            
            # construct source from separable radial and time dependences
            self.src_core = self.source_rad_prof*source_time_history[None,:]
        
        self.src_core = np.asfortranarray(self.src_core)

        # if wall_recycling>=0, return flows from the divertor are enabled   
        if self.wall_recycling>=0 and\
           'source_div_time' in self.namelist and 'source_div_vals' in self.namelist:
    
            # interpolate divertor source time history
            self.src_div = interp1d(
                self.namelist['source_div_time'], self.namelist['source_div_vals'])(self.time_grid)
        else:
            # no source into the divertor
            self.src_div = np.zeros_like(self.time_grid)
        
        # total number of injected ions, used for a check of particle conservation
        self.total_source = np.pi*np.sum(self.src_core*S0*(self.rvol_grid/self.pro_grid)[:,None],0)  # sum over radius
        self.total_source += self.src_div   # units of particles/s/cm

        if self.wall_recycling >= 0: # recycling activated
            
            # if recycling radial profile is given, interpolate it on radial grid
            if 'rcl_prof_vals' in self.namelist and 'rcl_prof_rhop' in self.namelist:

                # Check that at least part of the recycling prof is within Aurora radial grid
                if np.min(self.namelist['rcl_prof_rhop'])<np.max(self.rhop_grid):
                    raise ValueError('Input recycling radial grid is too far out!')

                rcl_rad_prof = interp1d(
                    self.namelist['rcl_prof_rhop'], self.namelist['rcl_prof_vals'],
                    fill_value='extrapolate')(
                        self.rhop_grid
                    )
                self.rcl_rad_prof = np.broadcast_to(rcl_rad_prof, (rcl_rad_prof.shape[0], len(self.time_grid)))

            else:
                # set recycling prof to exp decay from wall
                # use all time steps, specified neutral stage energy
                nml_rcl_prof = {key: self.namelist[key] for key in
                                ['imp_source_energy_eV', 'rvol_lcfs',
                                 'source_cm_out_lcfs', 'imp',
                                 'prompt_redep_flag', 'Baxis', 'main_ion_A']}
                nml_rcl_prof['source_width_in'] = 0
                nml_rcl_prof['source_width_out'] = 0

                # NB: we assume here that the 0th time is a good representation of how recycling is radially distributed
                self.rcl_rad_prof = source_utils.get_radial_source(
                    nml_rcl_prof, # namelist specifically to obtain exp decay from wall
                    self.rvol_grid, self.pro_grid,
                    S0,
                    self._Ti)

        else:
            # dummy profile -- recycling is turned off
            self.rcl_rad_prof = np.zeros((len(self.rhop_grid), len(self.time_grid)))
        

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


    def get_time_dept_atomic_rates(self,  superstages=[]):
        '''Obtain time-dependent ionization and recombination rates for a simulation run.
        If kinetic profiles are given as time-independent, atomic rates for each time slice
        will be set to be the same.

        Parameters
        ----------
        superstages : list or 1D array
            Indices of charge states that should be kept as superstages.
            The default is to have this as an empty list, in which case all charge states are kept.

        Returns
        -------
        S_rates : array (space, nZ(-super), time)
            Effective ionization rates. If superstages were indicated, these are the rates of superstages.
        R_rates : array (space, nZ(-super), time)
            Effective recombination rates. If superstages were indicated, these are the rates of superstages.
        '''
        
        # get electron impact ionization and radiative recombination rates in units of [s^-1]
        _, S, R, cx = atomic.get_cs_balance_terms(
                self.atom_data, ne_cm3=self._ne, Te_eV = self._Te, Ti_eV=self._Ti,
                include_cx=self.namelist['cxr_flag'])
                    
        if self.namelist['cxr_flag']:
            # Get an effective recombination rate by summing radiative & CX recombination rates
            R += cx*(self._n0/self._ne)[:,None] 
        
        if self.namelist['nbi_cxr_flag']:
            # include charge exchange between NBI neutrals and impurities
            self.nbi_cxr = interp1d(self.namelist['nbi_cxr']['rhop'], self.namelist['nbi_cxr']['vals'], axis=0,
                                    bounds_error=False, fill_value=0.0)(self.rhop_grid)

            R += self.nbi_cxr.T[None,:,:]
 
        if len(superstages):
            self.superstages, R, S, self.fz_upstage = \
                atomic.superstage_rates(R, S, superstages,save_time=self.save_time)
             
        # S and R for the Z+1 stage must be zero for the forward model.
        # Use Fortran-ordered arrays for speed in forward modeling (both Fortran and Julia)
        S_rates = np.zeros((S.shape[2], S.shape[1] + 1, self.time_grid.size), order='F')
        S_rates[:, :-1] = S.T

        R_rates = np.zeros((R.shape[2], R.shape[1] + 1, self.time_grid.size), order='F')
        R_rates[:, :-1] = R.T

        return S_rates, R_rates
    

    def get_par_loss_rate(self, trust_SOL_Ti=False):
        '''Calculate the parallel loss frequency on the radial and temporal grids [1/s].

        Parameters
        ----------
        trust_SOL_Ti : bool
            If True, the input Ti is trusted also in the SOL to calculate a parallel loss rate. 
            Often, Ti measurements in the SOL are unrealiable, so this parameter is set to False by default.

        Returns
        -------
        dv : array (space,time)
            Parallel loss rates in :math:`s^{-1}` units. 
            Values are zero in the core region and non-zero in the SOL. 
        
        '''
        # import here to avoid issues when building docs or package
        from omfit_classes.utils_math import atomic_element

        # background mass number (=2 for D)
        self.main_element = self.namelist['main_element']
        out = atomic_element(symbol=self.namelist['main_element'])
        spec = list(out.keys())[0]
        self.main_ion_A = self.namelist['main_ion_A'] = int(out[spec]['A'])
        self.main_ion_Z = self.namelist['main_ion_Z'] = int(out[spec]['Z'])

        # factor for v = machnumber * sqrt((3T_i+T_e)k/m)
        vpf = self.namelist['SOL_mach']*np.sqrt(q_electron/m_p/self.main_ion_A)  
        # v[m/s]=vpf*sqrt(T[ev])

        # number of points inside of LCFS
        ids = self.rvol_grid.searchsorted(self.namelist['rvol_lcfs'],side='left')
        idl = self.rvol_grid.searchsorted(self.namelist['rvol_lcfs']+self.namelist['lim_sep'],side='left')

        # Calculate parallel loss frequency using different connection lengths in the SOL and in the limiter shadow
        dv = np.zeros_like(self._Te.T) # space x time

        # Ti may not be reliable in SOL, replace it by Te
        Ti = self._Ti if trust_SOL_Ti else self._Te

        # open SOL
        dv[ids:idl] = vpf*np.sqrt(3.*Ti.T[ids:idl] + self._Te.T[ids:idl])/self.namelist['clen_divertor']

        # limiter shadow
        dv[idl:] = vpf*np.sqrt(3.*Ti.T[idl:] + self._Te.T[idl:])/self.namelist['clen_limiter']

        dv,_ = np.broadcast_arrays(dv,self.time_grid[None])

        return np.asfortranarray(dv)

    def superstage_DV(self, D_z, V_z, times_DV=None, opt=1):
        '''Reduce the dimensionality of D and V time-dependent profiles for the case in which superstaging is applied.

        Three options are currently available: 

        #. opt=1 gives a simple selection of D_z and V_z fields corresponding to each superstage index.
        
        #. opt=2 averages D_z and V_z over the charge states that are part of each superstage.

        #. opt=3 weights D_z and V_z corresponding to each superstage by the fractional abundances at ionization
        equilibrium. This is mostly untested -- use with care!

        Parameters
        ---------
        D_z: array, shape of (space,time,nZ)
            Diffusion coefficients, in units of :math:`cm^2/s`.
        V_z: array, shape of (space,time,nZ)
            Convection coefficients, in units of :math:`cm/s`.

        Returns
        -------
        Dzf: array, shape of (space,time,nZ-superstages)
            Diffusion coefficients of superstages, in units of :math:`cm^2/s`.
        Vzf: array, shape of (space,time,nZ-superstages)
            Convection coefficients of superstages, in units of :math:`cm/s`.
        
        '''
        # simple selection of elements corresponding to superstage
        Dzf = D_z[:,:,self.superstages]
        Vzf = V_z[:,:,self.superstages]
    
        if opt==1:
            # see selection above 
            pass

        elif opt==2:
            # average D,V over superstage

            superstages = np.r_[self.superstages, self.Z_imp+1]

            for i in range(len(self.superstages)):
                if superstages[i]+1 != superstages[i+1]:
                    Dzf[:,:,i] = D_z[:,:,superstages[i]: superstages[i+1]].mean(2)
                    Vzf[:,:,i] = V_z[:,:,superstages[i]: superstages[i+1]].mean(2)

        elif opt==3:
            # weighted average of D and V 
            superstages = np.r_[self.superstages, self.Z_imp+1]
           
            # calculate fractional abundances inside of each superstage
            for i in range(len(superstages)-1):
                if  superstages[i]+1 < superstages[i+1]:
                    # need to interpolate fz_upstage on time base of D and V -- do this only once
                    if not hasattr(self, 'fz_upstage_DV') or self.fz_upstage_DV.shape[2]==len(times_DV):
                        self.fz_upstage_DV = interp1d(self.time_grid, self.fz_upstage, axis=2)(times_DV)

                    ind = slice(superstages[i],superstages[i+1])
                    Dzf[:,:,i] = np.sum(D_z[:,:,ind]*self.fz_upstage_DV[:,ind].transpose(0,2,1), 2)
                    Vzf[:,:,i] = np.sum(V_z[:,:,ind]*self.fz_upstage_DV[:,ind].transpose(0,2,1), 2)

        else:
            raise ValueError('Unrecognized option for D and V superstaging!')
        
        return Dzf, Vzf


    def run_aurora(self, D_z, V_z, times_DV=None, nz_init=None, 
                   unstage=True, alg_opt=1, evolneut=False, use_julia=False, plot=False):
        '''Run a simulation using the provided diffusion and convection profiles as a function of space, time 
        and potentially also ionization state. Users can give an initial state of each ion charge state as an input. 

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
        unstage : bool, optional
            If superstages are indicated in the namelist, this parameter sets whether the output 
            should be "unstaged" by multiplying by the appropriate fractional abundances of all 
            charge states at ionization equilibrium. 
            Note that this unstaging process cannot account for transport and is therefore
            only an approximation, to be used carefully.
        alg_opt : int, optional
            If `alg_opt=1`, use the finite-volume algorithm proposed by Linder et al. NF 2020. 
            If `alg_opt=0`, use the older finite-differences algorithm in the 2018 version of STRAHL.
        evolneut : bool, optional
            If True, evolve neutral impurities based on their D,V coefficients. Default is False, in
            which case neutrals are only taken as a source and those that are not ionized immediately after
            injection are neglected.
            NB: It is recommended to only use this with explicit 2D sources, otherwise
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
        D_z, V_z = np.asarray(D_z), np.asarray(V_z)
        
        # D_z and V_z must have the same shape
        assert np.shape(D_z) == np.shape(V_z)
        
        if (times_DV is None) and (D_z.ndim>1 or V_z.ndim>1):
            raise ValueError('D_z and V_z given as time dependent, but times were not specified!')
 
        if times_DV is None or np.size(times_DV) == 0:
            times_DV = [1.] # dummy, no time dependence

        num_cs = int(self.Z_imp+1)

        # D and V were given for all stages -- define D and V for superstages
        if len(self.superstages):
            num_cs = len(self.superstages)
            if D_z.ndim==3 and D_z.shape[2]==self.Z_imp+1:
                D_z, V_z = self.superstage_DV(D_z, V_z, times_DV, opt=1)
         
        if not evolneut:
            # prevent recombination back to neutral state to maintain good particle conservation 
            self.R_rates[:,0] = 0

        if nz_init is None:
            # default: start in a state with no impurity ions
            nz_init = np.zeros((len(self.rvol_grid),num_cs))

        if D_z.ndim < 3:
            # set all charge states to have the same transport
            # num_cs = Z+1 - include elements for neutrals
            D_z =  np.tile(D_z.T, (num_cs, 1, 1)).T # create fortran contiguous arrays
            D_z[:,:,0] = 0.0
        
        if V_z.ndim < 3:
            # set all charge states to have the same transport
            V_z =  np.tile(V_z.T, (num_cs, 1, 1)).T# create fortran contiguous arrays
            V_z[:,:,0] = 0.0
 
        nt = len(self.time_out)

        # NOTE: for both Fortran and Julia, use f_configuous arrays for speed
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
                                     self.src_core,# source profile in radius and time
                                     self.rcl_rad_prof, # recycling radial profile
                                     self.S_rates, # ioniz_rate,
                                     self.R_rates, # recomb_rate,
                                     self.rvol_grid, self.pro_grid, self.qpr_grid,
                                     self.mixing_radius, self.decay_length_boundary,
                                     self.time_grid, self.saw_on,
                                     self.save_time, self.sawtooth_erfc_width, # dsaw width  [cm]
                                     self.wall_recycling,
                                     self.tau_div_SOL_ms * 1e-3,  # [s]
                                     self.tau_pump_ms * 1e-3,     # [s]
                                     self.tau_rcl_ret_ms * 1e-3,  # [s] 
                                     self.rvol_lcfs, self.bound_sep, self.lim_sep, self.prox_param,
                                     nz_init, alg_opt, evolneut, self.src_div)
        else:
            # import here to avoid import when building documentation or package (negligible slow down)
            from ._aurora import run as fortran_run

            self.res = fortran_run(nt,  # number of times at which simulation outputs results
                                   times_DV,
                                   D_z, V_z, # cm^2/s & cm/s    #(ir,nt_trans,nion)
                                   self.par_loss_rate,  # time dependent
                                   self.src_core, # source profile in radius and time
                                   self.rcl_rad_prof, # recycling radial profile
                                   self.S_rates, # ioniz_rate
                                   self.R_rates, # recomb_rate
                                   self.rvol_grid, self.pro_grid, self.qpr_grid,
                                   self.mixing_radius, self.decay_length_boundary,
                                   self.time_grid, self.saw_on,
                                   self.save_time, self.sawtooth_erfc_width, # dsaw width [cm]
                                   self.wall_recycling,
                                   self.tau_div_SOL_ms * 1e-3,  # [s]
                                   self.tau_pump_ms * 1e-3,     # [s]
                                   self.tau_rcl_ret_ms * 1e-3,  # [s]  
                                   self.rvol_lcfs, self.bound_sep, self.lim_sep, self.prox_param,
                                   rn_t0 = nz_init,  # if omitted, internally set to 0's
                                   alg_opt = alg_opt,
                                   evolneut = evolneut,
                                   src_div = self.src_div)
            
        if plot:

            # plot charge state density distributions over radius and time
            plot_tools.slider_plot(self.rvol_grid, self.time_out, self.res[0].transpose(1,0,2),
                                   xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'$n_z$ [$cm^{-3}$]',
                                   labels=[str(i) for i in np.arange(0,self.res[0].shape[1])],
                                   plot_sum=True, x_line=self.rvol_lcfs)
            
            # check particle conservation by summing over simulation reservoirs
            _ = self.check_conservation(plot=True)
        


        if len(self.superstages) and unstage:
            # "unstage" superstages to recover estimates for density of all charge states
            nz_unstaged = np.zeros(( len(self.rvol_grid), self.Z_imp+1, nt))
            
            superstages = np.r_[self.superstages, self.Z_imp+1]
           
            # calculate fractional abundances inside of each superstage
            for i in range(len(superstages)-1):
                if  superstages[i]+1 < superstages[i+1]:
                    # fill skipped stages from ionization equilibrium
                    ind = slice(superstages[i],superstages[i+1])
                    nz_unstaged[:,ind] = self.res[0][:,[i]]*self.fz_upstage[:,ind]
                else:
                    nz_unstaged[:,superstages[i]] = self.res[0][:,i]

            self.res = nz_unstaged, *self.res[1:]
        
        
        # nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = self.res
        return self.res


    def run_aurora_steady(self, D_z, V_z, nz_init=None, unstage=False,
                          alg_opt=1, evolneut=False, use_julia=False,
                          tolerance=0.01, max_sim_time = 100, dt=1e-4, dt_increase=1.05, 
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
        unstage : bool, optional
            If a list of superstages are provided in the namelist, this parameter sets whether the 
            output should be "unstaged". See docs for :py:meth:`~aurora.core.run_aurora` for details.
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
            Number of time steps (>2) before convergence is checked.
        plot : bool
            If True, plot time evolution of charge state density profiles to show convergence.
        '''

        if n_steps<2:
            raise ValueError('n_steps must be greater than 2!')
        
        if  self.ne.shape[0] > 1:
            raise ValueError('This method is designed to operate with time-independent background profiles!')

        if  D_z.ndim > 2 or V_z.ndim > 2:
            raise ValueError('This method is designed to operate with time-independent D and V profiles!')

        #set constant timesource
        self.namelist['source_type'] = 'const'
        self.namelist['source_rate'] = 1.
 
        # build timing dictionary
        self.namelist['timing'] = {'dt_start': [dt,dt],
                                   'dt_increase':[dt_increase, 1.],
                                   'steps_per_cycle': [1, 1],
                                   'times' : [0. , max_sim_time]}
        
        #prepare radial and temporal grid
        self.setup_grids()

        # update kinetic profile dependencies to get everything to the right shape
        self.setup_kin_profs_depts()
        
        times_DV = None
        if D_z.ndim==2:
            # make sure that transport coefficients were given as a function of space and nZ, not time!
            assert D_z.shape[0]==self.rhop_grid and D_z.shape[1]==self.Z_imp+1
            assert V_z.shape[0]==self.rhop_grid and V_z.shape[1]==self.Z_imp+1
            
            D_z = D_z[:,None] #(ir,nt_trans,nion)
            V_z = V_z[:,None] 

        sim_steps = 0
        
        time_grid = self.time_grid.copy()
        time_out = self.time_out.copy()
        save_time = self.save_time.copy()
        par_loss_rate = self.par_loss_rate.copy()
        src_core = self.src_core.copy()
        S_rates = self.S_rates.copy()
        R_rates = self.R_rates.copy()
        saw_on = self.saw_on.copy()
        nz_all = None if nz_init is None else nz_init

        while sim_steps < len(time_grid):
            
            self.time_grid = self.time_out = time_grid[sim_steps:sim_steps+n_steps]
            self.save_time = save_time[sim_steps:sim_steps+n_steps]
            self.par_loss_rate = par_loss_rate[:,sim_steps:sim_steps+n_steps]
            self.src_core = src_core[:,sim_steps:sim_steps+n_steps]
            self.S_rates = S_rates[:,:,sim_steps:sim_steps+n_steps]
            self.R_rates = R_rates[:,:,sim_steps:sim_steps+n_steps]
            self.saw_on = saw_on[sim_steps:sim_steps+n_steps]

            sim_steps+= n_steps

            # get charge state densities from latest time step
            if nz_all is None:
                nz_init = None
            else:
                nz_init = nz_all[:,:,-1] if nz_all.ndim==3 else nz_all

            nz_new = self.run_aurora(D_z, V_z, times_DV, nz_init=nz_init,
                                     unstage=unstage, alg_opt=alg_opt,
                                     evolneut=evolneut, use_julia=use_julia, plot=False)[0]

            if nz_all is None:
                nz_all = np.dstack((np.zeros_like(nz_new[:,:,[0]]), nz_new))
                nz_init = np.zeros_like(nz_new[:,:,0])
            else:
                nz_all = np.dstack((nz_all, nz_new))

            # check if normalized profiles have converged
            if np.linalg.norm(nz_new[:,:,-1]-nz_init)/(np.linalg.norm(nz_init)+1e-99) < tolerance: 
                break
            
        # store final time grids
        self.time_grid = time_grid[:sim_steps]
        self.time_out  = time_grid[:sim_steps] # identical because steps_per_cycle is fixed to 1
        self.save_time = save_time[:sim_steps]
        
        if plot:
            # plot charge state distributions over radius and time
            plot_tools.slider_plot(self.rhop_grid, self.time_grid, nz_all.transpose(1,0,2),
                                   xlabel=r'$\rho_p$', ylabel='time [s]', zlabel=r'$n_z$ [$cm^{-3}$]',
                                   labels=[str(i) for i in np.arange(0,nz_all.shape[1])],
                                   plot_sum=True)
        
        if sim_steps >= len(time_grid):
            raise ValueError(f'Could not reach convergence before {max_sim_time:.3f}s of simulated time!')
        
        # compute effective particle confinement time from latest few time steps
        circ = 2*np.pi*self.Raxis_cm # cm
        zvol = circ * np.pi * self.rvol_grid / self.pro_grid

        wh = self.rhop_grid <= 1
        var_volint = np.nansum(nz_new[wh,:,-2]*zvol[wh,None])

        # compute effective particle confinement time
        source_time_history = grids_utils.vol_int(
            self.src_core.T, self.rvol_grid, self.pro_grid, self.Raxis_cm)
        self.tau_imp = var_volint / source_time_history[-2] # avoid last time point because source may be 0 there
        
        return nz_new[:,:,-1]
    
        
    def calc_Zeff(self):
        '''Compute Zeff from each charge state density, using the result of an AURORA simulation.
        The total Zeff change over time and space due to the simulated impurity can be simply obtained by summing 
        over charge states.

        Results are stored as an attribute of the simulation object instance. 
        '''
        # This method requires that a simulation has already been run:
        assert hasattr(self,'res')

        # extract charge state densities from the simulation result
        nz = self.res[0]

        # this method requires all charge states to be made available
        try:
            assert nz.shape[1] == self.Z_imp+1
        except AssertionError:
            raise ValueError('calc_Zeff method requires all charge state densities to be availble! Unstage superstages.')

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
        ----------
        plot : bool, optional
            If True, plot time histories in each particle reservoir and display quality of particle conservation.
        axs : 2-tuple or array
            Array-like structure containing two matplotlib.Axes instances: the first one 
            for the separate particle time variation in each reservoir, the second for 
            the total particle-conservation check. This can be used to plot results 
            from several aurora runs on the same axes. 
        
        Returns
        -------
        out : dict
            Dictionary containing density of particles in each reservoir.
        axs : matplotlib.Axes instances, only returned if plot=True
            Array-like structure containing two matplotlib.Axes instances, (ax1,ax2).
            See optional input argument.
        '''
        nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = self.res
        nz = nz.transpose(2,1,0)   # time,nZ,space

        # factor to account for cylindrical geometry:
        circ = 2*np.pi*self.Raxis_cm # cm

        # collect all the relevant quantities for particle conservation
        out = {}
        
        # particles that entered as "source":
        out['source'] = grids_utils.vol_int(
            self.src_core.T, self.rvol_grid, self.pro_grid, self.Raxis_cm) * circ

        # integrated source over timee
        out['integ_source'] =  cumtrapz(out['source'], self.time_out, initial=0)
    
        # calculate total impurity density (summed over charge states)
        total_impurity_density = np.nansum(nz, axis=1) # time, space

        # Compute total number of particles for particle conservation checks:
        all_particles = grids_utils.vol_int(total_impurity_density, self.rvol_grid, self.pro_grid, self.Raxis_cm)
        
        out['total'] = all_particles + (N_wall + N_div + N_pump + N_ret)*circ 
        out['plasma_particles'] = grids_utils.vol_int(
            total_impurity_density, self.rvol_grid, self.pro_grid, self.Raxis_cm)
        out['particles_at_wall'] = N_wall * circ
        out['particles_retained_at_wall'] = N_ret * circ
        out['particles_in_divertor'] = N_div * circ
        out['particles_in_pump'] = N_pump * circ
        out['parallel_loss'] = N_dsu * circ
        out['edge_loss'] = N_tsu * circ
        out['parallel_loss_to_limiter'] = N_dsul * circ
        out['recycling_from_divertor'] = rcld_rate * circ
        out['recycling_from_wall'] = rclw_rate * circ
        if hasattr(self, 'rad'): # radiation has already been compputed
            out['impurity_radiation'] = grids_utils.vol_int(
                self.rad['tot'], self.rvol_grid, self.pro_grid, self.Raxis_cm)

        if plot:
            # -------------------------------------------------
            # plot time histories for each particle reservoirs:
            if axs is None:
                fig, ax1 = plt.subplots(nrows=4, ncols=3, sharex=True, figsize=(15,10))
            else:
                ax1 = axs[0]

            ax1[0,0].plot(self.time_out, out['source'], label='Influx ($s^{-1}$)')
            ax1[0,1].plot(self.time_out, out['particles_in_divertor'], label='Particles in divertor')
            ax1[0,2].plot(self.time_out, out['particles_in_pump'], label='Particles in pump')
            
            ax1[1,0].plot(self.time_out, out['parallel_loss'], label='Parallel Loss')
            ax1[1,1].plot(self.time_out, out['parallel_loss_to_limiter'], label='Parallel Loss to Limiter')
            ax1[1,2].plot(self.time_out, out['edge_loss'], label='Edge Loss')
                  
            ax1[2,0].plot(self.time_out, out['particles_at_wall'], label='Particles stuck at wall')
            ax1[2,1].plot(self.time_out, out['particles_retained_at_wall'], label='Particles retained at wall')
            ax1[2,2].plot(self.time_out, out['recycling_from_wall'], label='Wall rec. rate')
            
            ax1[3,0].plot(self.time_out, out['recycling_from_divertor'], label='Divertor rec. rate')
            ax1[3,1].plot(self.time_out, out['total'], label='Core impurity particles')
            for aa in ax1.flatten()[:11]:
                aa.legend(loc='best').set_draggable(True)
                
            if 'impurity_radiation' in out:
                ax1[3,2].plot(self.time_out, out['impurity_radiation'], label='Core radiation (W)')
                ax1[3,2].legend(loc='best').set_draggable(True)
                
            for ii in [0,1,2]:
                ax1[3,ii].set_xlabel('Time (s)')
            ax1[3,0].set_xlim(self.time_out[[0,-1]])

            # ----------------------------------------------------------------
            # now plot all particle reservoirs to check particle conservation:
            if axs is None:
                fig, ax2 = plt.subplots()
            else:
                ax2 = axs[1]
                
            ax2.set_xlabel('time [s]')

            ax2.plot(self.time_out, all_particles, label='Particles in Plasma')
            ax2.plot(self.time_out, out['particles_at_wall'], label='Particles stuck at wall')
            ax2.plot(self.time_out, out['particles_in_divertor'], label='Particles in Divertor')
            ax2.plot(self.time_out, out['particles_in_pump'], label='Particles in Pump')
            ax2.plot(self.time_out, out['particles_retained_at_wall'],label='Particles retained at wall')
            ax2.plot(self.time_out, out['total'], label='Total')
            ax2.plot(self.time_out, out['integ_source'], label='Integrated source')

            if abs((out['total'][-1]-out['integ_source'][-1])/out['integ_source'][-1])> .1:
                print('Warning: significant error in particle conservation!')

            ax2.set_ylim(0,None)
            ax2.legend(loc='best')
            plt.tight_layout()

        if plot:
            return out, (ax1,ax2)
        else:
            return out


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
        # this method requires all charge states to be made available
        try:
            assert self.res[0].shape[1] == self.Z_imp+1
        except AssertionError:
            raise ValueError('centrifugal_asym method requires all charge state densities to be availble! Unstage superstages.')

        fz = self.res[0][...,-1] / np.sum(self.res[0][...,-1],axis=1)[:,None]
        Z_ave_vec = np.sum(fz * np.arange(self.Z_imp+1)[None,:],axis=1)
        
        self.CF_lambda = synth_diags.centrifugal_asym(self.rhop_grid, self.Rlfs, omega, Zeff, 
                                                      self.A_imp, Z_ave_vec, 
                                                      self.Te, self.Ti, main_ion_A=self.main_ion_A, 
                                                      plot=plot, nz=self.res[0][...,-1], geqdsk=self.geqdsk).mean(0)

        return self.CF_lambda
