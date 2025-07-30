"""This module includes the core class to set up simulations with :py:mod:`aurora`. The :py:class:`~aurora.core.aurora_sim` takes as input a namelist dictionary and a g-file dictionary (and possibly other optional argument) and allows creation of grids, interpolation of atomic rates and other steps before running the forward model.
"""
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

import os, sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import e as q_electron, m_p
import pickle as pkl
from copy import deepcopy
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
from . import interp
from . import atomic
from . import grids_utils
from . import source_utils
from . import plot_tools
from . import synth_diags
from . import adas_files
from . import surface
from . import radiation
 

class aurora_sim:
    """Setup the input dictionary for an Aurora ion transport simulation from the given namelist.

    Parameters
    ----------
    namelist : dict
        Dictionary containing aurora inputs. See default_nml.py for some defaults,
        which users should modify for their runs.
    geqdsk : dict, optional
        EFIT gfile as returned after postprocessing by the :py:mod:`omfit_classes.omfit_eqdsk`
        package (OMFITgeqdsk class). If left to None (default), the minor and major radius must be
        indicated in the namelist in order to create a radial grid.

    """

    def __init__(self, namelist, geqdsk=None):

        if namelist is None:
            # option useful for calls like omfit_classes.OMFITaurora(filename)
            # A call like omfit_classes.OMFITaurora('test', namelist, geqdsk=geqdsk) is also possible
            # to initialize the class as a dictionary.
            return

        # make sure that any changes in namelist will not propagate back to the calling function
        self.namelist = deepcopy(namelist)
        self.geqdsk = geqdsk  # if None, minor (rvol_lcfs) and major radius (Raxis_cm) must be in namelist
       
        self.imp = namelist["imp"]

        # import here to avoid issues when building docs or package
        from omfit_classes.utils_math import atomic_element

        # get nuclear charge Z and atomic mass number A
        out = atomic_element(symbol=self.imp)
        spec = list(out.keys())[0]
        self.Z_imp = int(out[spec]["Z"])
        self.A_imp = int(out[spec]["A"])

        self.reload_namelist()
 
        self.full_PWI_flag = False

        if "Raxis_cm" in self.namelist:
            self.Raxis_cm = self.namelist["Raxis_cm"]  # cm
        elif self.geqdsk is not None and "RMAXIS" in self.geqdsk:
            self.Raxis_cm = self.geqdsk["RMAXIS"] * 100.0  # cm

        if self.geqdsk is not None and "BCENTR" in self.geqdsk:
            self.namelist["Baxis"] = self.geqdsk["BCENTR"]
        if (
            "prompt_redep_flag" in self.namelist and self.namelist["prompt_redep_flag"]
        ) and not hasattr(self, "Baxis"):
            # need magnetic field to model prompt redeposition
            raise ValueError(
                "Missing magnetic field on axis! Please define this in the namelist"
            )

        # specify which atomic data files should be used -- use defaults unless user specified in namelist

        atom_files = {}
        names = ["acd", "scd"]
        if self.namelist.get("cxr_flag", False):
            names += ["ccd"]
        if self.namelist.get("metastable_flag", False):
            names += ["qcd", "xcd"]
            
        for name in names:
            if name in self.namelist:
                atom_files[name] = self.namelist[name] 
            elif imp in  adas_files.adas_files_dict():
                atom_files[name] = self.namelist[name][self.imp].get(name, None)


        # now load ionization and recombination rates
        self.atom_data = atomic.get_atom_data(self.imp, files=atom_files)

        # allow for ion superstaging
        self.superstages = self.namelist.get("superstages", [])

        # set up radial and temporal grids
        self.setup_grids()
        
        #interpolate kinetics profiles of the profiles are in namelist
        if "kin_profs" in self.namelist:
            self.kin_profs = self.namelist["kin_profs"]
            # set up kinetic profiles and atomic rates
            self.setup_kin_profs_depts()
            
        # full PWI model not used - dummy values for all related input variables to fortran routine
        self.setup_dummy_pwi_vars() 

    def reload_namelist(self, namelist=None):
        """(Re-)load namelist to update scalar variables."""
        if namelist is not None:
            self.namelist = namelist
            
        # Extract one by one all the inputs from namelist
        # as attributes of asim, keeping the same name  
        #NOTE T.O. I'm not a fan of this MATLAB approach.. 

        for parameter in self.namelist:
            if parameter == 'kin_profs':
                continue
            elif isinstance(self.namelist[parameter], dict):
                for sub_parameter in self.namelist[parameter]:
                     if sub_parameter != 'nbi_cxr':
                        setattr(self, sub_parameter, self.namelist[parameter][sub_parameter])
            else:
                setattr(self, parameter, self.namelist[parameter])
       

        # consistency checks for divertor parameters
        if not 0.0 <= self.div_neut_screen <= 1.0:
            raise ValueError("div_neut_screen must be between 0.0 and 1.0!") 
        if not 0.0 <= self.div_recomb_ratio <= 1.0:
            raise ValueError("div_recomb_ratio must be between 0.0 and 1.0!") 
         
        self.num_background_species = len(self.background_species)
        # if phys_volumes and pump_chamber flags are set to False, pumping is done
        # directly from the divertor chamber and is defined by the time tau_pump_ms
        if not self.phys_volumes and not self.pump_chamber:
            self.vol_div = 0.0  # adimensional divertor chamber
            self.vol_pump = 0.0  # adimensional pump chamber
            self.L_divpump = 0.0  # no conductance
            self.L_leak = 0.0  # no leakage
            self.S_pump = 0.0
        # if pump_chamber flag is set to False but phys_volumes flag is set to True,
        # pumping is done directly from the divertor chamber and is defined by the
        # pumping speed S_pump
        if self.phys_volumes and not self.pump_chamber:
            self.vol_pump = 0.0  # adimensional pump chamber
            self.L_divpump = 0.0  # no conductance
            self.L_leak = 0.0  # no leakage
            self.tau_pump_ms = 0.0
        # if phys_volumes and pump_chamber flags are set to True, pumping is done from
        # a pump chamber, the transport between divertor and pump chambers is defined
        # by the conductance L_divpump, and the pumping from the pump chamber is defined
        # by a pumping speed S_pump
        if self.phys_volumes and self.pump_chamber:
            self.tau_pump_ms = 0.0
        # consistency check:
        if not self.phys_volumes and self.pump_chamber:    
            raise ValueError("Assuming a secondary pump chamber requires defining the physical chamber volumes!")       
            
        # if recycling flag is set to False, avoid any divertor return flows
        # To include divertor return flows but no recycling, set wall_recycling=0
        if not self.namelist["recycling_flag"]:
            self.wall_recycling = -1.0  # no divertor return flows

    def save(self, filename):
        """Save state of `aurora_sim` object."""
        with open(filename, "wb") as f:
            pkl.dump(self, f)

    def load(self, filename):
        """Load `aurora_sim` object."""
        with open(filename, "rb") as f:
            obj = pkl.load(f)
        self.__dict__.update(obj.__dict__)

    def save_dict(self):
        return self.__dict__

    def load_dict(self, aurora_dict):
        self.__dict__.update(aurora_dict)

    def setup_grids(self):
        """Method to set up radial and temporal grids given namelist inputs."""
        if self.geqdsk is not None:
            # Get r_V to rho_pol mapping
            rho_pol, _rvol = grids_utils.get_rhopol_rvol_mapping(self.geqdsk)
            rvol_lcfs = np.round(np.interp(1.0, rho_pol, _rvol), 3)
            self.rvol_lcfs = self.namelist["rvol_lcfs"] = rvol_lcfs 

        elif "rvol_lcfs" in self.namelist:
            # separatrix location explicitly given by user
            self.rvol_lcfs = self.namelist["rvol_lcfs"]

        else:
            raise ValueError(
                "Could not identify rvol_lcfs. Either provide this in the namelist or provide a geqdsk equilibrium"
            )

        # create radial grid
        grid_params = grids_utils.create_radial_grid(self.namelist, plot=False)
        self.rvol_grid, self.pro_grid, self.qpr_grid, self.prox_param = grid_params

        if self.geqdsk is not None:
            # get rho_poloidal grid corresponding to aurora internal (rvol) grid
            self.rhop_grid = interp1d(_rvol, rho_pol, fill_value="extrapolate")(
                self.rvol_grid
            )
            self.rhop_grid[0] = 0.0  # enforce on axis
        else:
            # use rho_vol = rvol/rvol_lcfs
            self.rhop_grid = self.rvol_grid / self.rvol_lcfs

        # define time grid ('timing' must be in namelist)
        self.time_grid, self.save_time = grids_utils.create_time_grid(
            timing=self.namelist["timing"], plot=False)
        
        self.time_out = self.time_grid[self.save_time]

        # create array of 0's of length equal to self.time_grid, with 1's where sawteeth must be triggered
        self.saw_on = np.zeros_like(self.time_grid)
        input_saw_times = self.namelist["saw_model"]["times"]

        self.saw_times = np.array(input_saw_times)[input_saw_times < self.time_grid[-1]]
        if self.namelist["saw_model"]["saw_flag"] and len(self.saw_times) > 0:
            self.saw_on[self.time_grid.searchsorted(self.saw_times)] = 1
            
        # calculate core plasma volume (i.e. volume of region until rvol = rvol_lcfs) and total plasma volume
        ones = np.ones_like(self.rvol_grid)
        self.core_vol = grids_utils.vol_int(ones,self.rvol_grid,self.pro_grid,self.Raxis_cm,rvol_max=self.rvol_lcfs)
        self.plasma_vol = grids_utils.vol_int(ones,self.rvol_grid,self.pro_grid,self.Raxis_cm,rvol_max=None)

    def setup_kin_profs_depts(self):
        """Method to set up Aurora inputs related to the kinetic background from namelist inputs."""
        # get kinetic profiles on the radial and (internal) temporal grids
        self._ne, self._Te, self._Ti, self._n0 = self.get_aurora_kin_profs()

        # store also kinetic profiles on output time grid
        if len(self._ne) > 1:  # all have the same shape now
            save_time = self.save_time
        else:
            save_time = [0]

        self.ne = self._ne[save_time, :]
        self.Te = self._Te[save_time, :]
        self.Ti = self._Ti[save_time, :]
        self.n0 = self._n0[save_time, :]

        # Get time-dependent parallel loss rate
        self.par_loss_rate = self.get_par_loss_rate()

        metastables = self.namelist.get("metastable_flag", False)
        superstages = self.namelist.get("superstages", [])
        # Obtain atomic rates on the computational time and radial grids
        self.set_time_dept_atomic_rates(
            superstages=superstages, metastables=metastables
        )

        Sne0 = self.Sne_rates[:, 0, :]
        # get radial profile of source function
        if len(save_time) == 1:  # if time averaged profiles were used
            Sne0 = Sne0[:, [0]]  # 0th charge state (neutral)


        # get time history and radial profiles separately
        source_time_history = source_utils.get_source_time_history(
            self.namelist, self.Raxis_cm, self.time_grid
        )  # units of particles/s/cm for 1D source or particles/s/cm^3  for 2D source

            
        if self.namelist["source_type"] == "arbitrary_2d_source":
            # interpolate explicit source values on time and rhop grids of simulation
            # NB: explicit_source_vals should be in units of particles/s/cm^3 <-- ionization rate
            srho = self.namelist["explicit_source_rhop"]
            self.source_rad_prof  = interp1d(srho, source_time_history.T, 
                                            axis=0,bounds_error=False,
                                             fill_value=0)(self.rhop_grid)
            # Change units to particles/cm^3
            self.src_core = self.source_rad_prof / Sne0
        else:

            # get radial profile of source function for each time step
            # dimensionless, normalized such that pnorm=1
            # i.e. source_time_history*source_rad_prof = # particles/cm^3
            self.source_rad_prof = source_utils.get_radial_source(
                self.namelist,
                self.rvol_grid,
                self.pro_grid,
                Sne0,  # 0th charge state (neutral)
                self._Ti,
            )

            # construct source from separable radial and time dependences
            self.src_core = self.source_rad_prof * source_time_history[None, :]

        self.src_core = np.asfortranarray(self.src_core)

        # if wall_recycling>=0, return flows from the divertor are enabled
        if (
            self.wall_recycling >= 0
            and "source_div_time" in self.namelist
            and "source_div_vals" in self.namelist
        ):

            # interpolate divertor source time history
            self.src_div = interp1d(
                self.namelist["source_div_time"], self.namelist["source_div_vals"]
            )(self.time_grid)
        else:
            # no source into the divertor
            self.src_div = np.zeros_like(self.time_grid)

        # total number of injected ions, used for a check of particle conservation
        self.total_source = np.pi * np.sum(
            self.src_core * Sne0 * (self.rvol_grid / self.pro_grid)[:, None], 0
        )  # sum over radius
        self.total_source += self.src_div  # units of particles/s/cm
        # NB: src_core [1/cm^3] and src_div [1/cm/s] have different units!

        if self.wall_recycling >= 0:  # recycling activated

            # if recycling radial profile is given, interpolate it on radial grid
            if "rcl_prof_vals" in self.namelist and "rcl_prof_rhop" in self.namelist:

                # Check that at least part of the recycling prof is within Aurora radial grid
                if np.min(self.namelist["rcl_prof_rhop"]) < np.max(self.rhop_grid):
                    raise ValueError("Input recycling radial grid is too far out!")

                rcl_rad_prof = interp1d(
                    self.namelist["rcl_prof_rhop"],
                    self.namelist["rcl_prof_vals"],
                    fill_value="extrapolate",
                )(self.rhop_grid)

            else:
                # set recycling prof to exp decay from wall
                # use all time steps, specified neutral stage energy
                nml_keys = [
                        "imp_source_energy_eV",
                        "rvol_lcfs",
                        "source_cm_out_lcfs",
                        "imp",
                        "prompt_redep_flag",
                        "main_ion_A",
                    ]
                nml_rcl_prof = {k: self.namelist[k] for k in nml_keys}

                if "prompt_redep_flag" in self.namelist and self.namelist["prompt_redep_flag"]:
                    # only need Baxis for prompt redeposition model
                    nml_rcl_prof["Baxis"] = self.namelist["Baxis"]

                nml_rcl_prof["source_width_in"] = 0
                nml_rcl_prof["source_width_out"] = 0

                # NB: we assume here that the 0th time is a good representation of how recycling is radially distributed
                rcl_rad_prof = source_utils.get_radial_source(
                    nml_rcl_prof,  # namelist specifically to obtain exp decay from wall
                    self.rvol_grid,
                    self.pro_grid,
                    Sne0,
                    self._Ti,
                )

            self.rcl_rad_prof = np.broadcast_to(
                rcl_rad_prof, (rcl_rad_prof.shape[0], len(self.time_grid))
            )

        else:
            # dummy profile -- recycling is turned off
            self.rcl_rad_prof = np.zeros((len(self.rhop_grid), len(self.time_grid)))
        

        
    def interp_kin_prof(self, prof):
        """Interpolate the given kinetic profile on the radial and temporal grids [units of s].
        This function extrapolates in the SOL based on input options using the same methods as in STRAHL.
        """
        times = self.kin_profs[prof]["times"]

        r_lcfs = np.interp(1, self.rhop_grid, self.rvol_grid)

        # extrapolate profiles outside of LCFS by exponential decays
        r = interp1d(self.rhop_grid, self.rvol_grid, fill_value="extrapolate")(
            self.kin_profs[prof]["rhop"]
        )

        if self.kin_profs[prof]["fun"] == "interp":
            if "decay" not in self.kin_profs[prof]:
                # if decay length in the SOL was not given by the user, assume a decay length of 1cm
                # print(
                #    f"Namelist did not provide a {prof} decay length for the SOL. Setting it to 1cm."
                # )
                self.kin_profs[prof]["decay"] = np.ones(
                    len(self.kin_profs[prof]["vals"])
                )

            data = interp.interp_quad(
                r / r_lcfs,
                self.kin_profs[prof]["vals"],
                self.kin_profs[prof]["decay"],
                r_lcfs,
                self.rvol_grid,
            )
            data[data < 1.01] = 1

        elif self.kin_profs[prof]["fun"] == "interpa":
            data = interp.interpa_quad(
                r / r_lcfs, self.kin_profs[prof]["vals"], r_lcfs, self.rvol_grid
            )

        # linear interpolation in time
        if len(times) > 1:  # time-dept
            data = interp1d(times, data, axis=0)(
                np.clip(self.time_grid, *times[[0, -1]])
            )

        return data

    def get_aurora_kin_profs(self, min_T=1.01, min_ne=1e10):
        """Get kinetic profiles on radial and time grids."""
        # ensure 2-dimensional inputs:
        self.kin_profs["ne"]["vals"] = np.atleast_2d(self.kin_profs["ne"]["vals"])
        self.kin_profs["Te"]["vals"] = np.atleast_2d(self.kin_profs["Te"]["vals"])

        Te = self.interp_kin_prof("Te")
        ne = self.interp_kin_prof("ne")

        if "Ti" in self.kin_profs and "vals" in self.kin_profs["Ti"]:
            self.kin_profs["Ti"]["vals"] = np.atleast_2d(self.kin_profs["Ti"]["vals"])
            Ti = self.interp_kin_prof("Ti")
        else:
            Ti = Te

        # get neutral background neutral density
        if self.namelist.get("cxr_flag", False):
            n0 = self.interp_kin_prof("n0")
        else:
            n0 = None

        # set minima in temperature and density
        Te[Te < min_T] = min_T
        Ti[Ti < min_T] = min_T
        ne[ne < min_ne] = min_ne

        # make sure that Te,ne,Ti and n0 have the same shape at this stage
        ne, Te, Ti, n0 = np.broadcast_arrays(ne, Te, Ti, n0)

        return ne, Te, Ti, n0
    
    
    
    def setup_dummy_pwi_vars(self):
    
        # full PWI model not used - dummy values for all related input variables to fortran routine

 
        nt = len(self.time_grid)

        # the reflection profile = recycling profile
        self.rfl_rad_prof = self.rcl_rad_prof
       
        # sputtering profiles 
        self.spt_rad_prof = np.tile(self.rcl_rad_prof[:,None], (self.num_background_species+1,1))
  
        # effective wall surfaces, accounting for roughness
        self.surf_mainwall_eff = self.surf_mainwall*self.mainwall_roughness #cm^2
        self.surf_divwall_eff = self.surf_divwall*self.divwall_roughness #cm^2
        
        # keys for fortran routine
        self.Z_main_wall = 0
        self.Z_div_wall = 0
        

        # dummy values for reflection coefficients at each time step
        self.rn_main_wall = np.zeros(nt)
        self.rn_div_wall  = np.zeros(nt)
        
        # dummy values for reflected energies at each time step
        self.E_refl_main_wall = np.zeros(nt)
        self.E_refl_div_wall = np.zeros(nt)
        
        # dummy values for background fluxes onto the walls at each time step
        self.fluxes_main_wall_background = np.zeros((self.num_background_species,nt))
        self.fluxes_div_wall_background = np.zeros((self.num_background_species,nt))
        
        # dummy values for impurity sputtering coefficients at each time step
        self.y_main_wall = np.zeros((self.num_background_species+1,nt))
        self.y_div_wall = np.zeros((self.num_background_species+1,nt))
        
        # dummy values for impurity sputtered energies at each time step
        self.E_sput_main_wall = np.zeros((self.num_background_species+1,nt))
        self.E_sput_div_wall = np.zeros((self.num_background_species+1,nt))
        
        # dummy values for impurity implantation depths into the walls
        self.implantation_depth_main_wall = 0.0
        self.implantation_depth_div_wall = 0.0
        
        # dummy values for impurity saturation densities into the walls
        self.n_main_wall_sat = 0.0
        self.n_div_wall_sat = 0.0
        
            
            

    def set_time_dept_atomic_rates(self, superstages=[], metastables=False):
        """Obtain time-dependent ionization and recombination rates for a simulation run.
        If kinetic profiles are given as time-independent, atomic rates for each time slice
        will be set to be the same.

        Parameters
        ----------
        superstages : list or 1D array
            Indices of charge states that should be kept as superstages.
            The default is to have this as an empty list, in which case all charge states are kept.
        metastables : bool
            Load metastable resolved atomic data and rates. Default is False.

        Attributes
        ----------
        Sne_rates : array (space, nZ(-super), time)
            Effective ionization rates [s]. If superstages were indicated, these are the rates of superstages.
        Rne_rates : array (space, nZ(-super), time)
            Effective recombination rates [s]. If superstages were indicated, these are the rates of superstages.
        Qne_rates : array (space, nZ(-super), time)
            Cross-coupling coefficients, only computed if :param:`metastables`=True.
        Xne_rates : array (space, nZ(-super), time)
            Parent cross-coupling coefficients, only computed if :param:`metastables`=True.
        """

        # get electron impact ionization and radiative recombination rates in units of [s^-1]
        out = atomic.get_cs_balance_terms(
            self.atom_data,
            ne_cm3=self._ne,
            Te_eV=self._Te,
            Ti_eV=self._Ti,
            include_cx=self.namelist["cxr_flag"],
            metastables=metastables,
        )
        out.pop(0)
        Sne = out.pop(0)
        Rne = out.pop(0)

        # cache radiative & dielectronic recomb:
        self.alpha_RDR_rates = Rne
      
        if self.namelist["cxr_flag"]:
            # Get an effective recombination rate by summing radiative & CX recombination rates
            alpha_CX_rates = out.pop(0) * (self._n0 / self._ne)[:, None]
            #metastable resolved CCD file exist only for C and N, use unresolved file for others
            if metastables and alpha_CX_rates.shape[1] != Rne.shape[1]:
                self.alpha_CX_rates = np.zeros_like(Rne)
                for i, m in enumerate(self.atom_data["ccd"].meta_ind[1:]):
                    j = self.atom_data["acd"].meta_ind.index(m)
                    self.alpha_CX_rates[:,j] = alpha_CX_rates[:,i]
            else:
                self.alpha_CX_rates = alpha_CX_rates
            
            # inplace addition would change also self.alpha_RDR_rates
            Rne = Rne + self.alpha_CX_rates
            

        if self.namelist["nbi_cxr_flag"]:
            # include charge exchange between NBI neutrals and impurities
            self.nbi_cxr = interp1d(
                self.namelist["nbi_cxr"]["rhop"],
                np.atleast_3d(self.namelist["nbi_cxr"]["vals"]).T,
                bounds_error=False,
                fill_value=0.0,
            )(self.rhop_grid)
            
            if self.nbi_cxr.shape[2] > 1:
                #time-dependent, times are switch times of beams, values are mean in between switch times
                it = self.namelist["nbi_cxr"]["times"].searchsorted(self.time_grid)
                self.nbi_cxr = self.nbi_cxr[it]

            Rne = Rne + self.nbi_cxr

        if len(superstages):
            self.superstages, Rne, Sne, self.fz_upstage = atomic.superstage_rates(
                Rne, Sne, superstages, save_time=self.save_time
            )

        # Sne and Rne for the Z+1 stage must be zero for the forward model.
        # Use Fortran-ordered arrays for speed in forward modeling (both Fortran and Julia)
        self.Sne_rates = np.zeros((Sne.shape[2], Sne.shape[1] + 1, self.time_grid.size), order="F")
        self.Sne_rates[:, :-1] = Sne.T

        self.Rne_rates = np.zeros((Rne.shape[2], Rne.shape[1] + 1, self.time_grid.size), order="F")
        self.Rne_rates[:, :-1] = Rne.T

        if metastables:
            self.Qne_rates = out.pop(0).T
            self.Xne_rates = out.pop(0).T

    def get_par_loss_rate(self, trust_SOL_Ti=False):
        """Calculate the parallel loss frequency on the radial and temporal grids [1/s].

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

        """
        # import here to avoid issues when building docs or package
        from omfit_classes.utils_math import atomic_element

        # background mass number (=2 for D)
        self.main_element = self.namelist["main_element"]
        out = atomic_element(symbol=self.namelist["main_element"])
        spec = list(out.keys())[0]
        self.main_ion_A = self.namelist["main_ion_A"] = int(out[spec]["A"])
        self.main_ion_Z = self.namelist["main_ion_Z"] = int(out[spec]["Z"])

        # factor for v = machnumber * sqrt((3T_i+T_e)k/m)
        vpf = self.namelist["SOL_mach"] * np.sqrt(q_electron / m_p / self.main_ion_A)
        # v[m/s]=vpf*sqrt(T[ev])

        # number of points inside of LCFS
        ids = self.rvol_grid.searchsorted(self.namelist["rvol_lcfs"], side="left")
        idl = self.rvol_grid.searchsorted(
            self.namelist["rvol_lcfs"] + self.namelist["lim_sep"], side="left"
        )

        
        # Calculate parallel loss frequency using different connection lengths in the SOL and in the limiter shadow
        dv = np.zeros_like(self._Te.T)  # space x time

        # Ti may not be reliable in SOL, replace it by Te
        Ti = self._Ti if trust_SOL_Ti else self._Te

        # open SOL
        dv[ids:idl] = vpf * np.sqrt(3.0 * Ti.T[ids:idl] + self._Te.T[ids:idl])/ self.namelist["clen_divertor"]

        # limiter shadow
        dv[idl:] = vpf * np.sqrt(3.0 * Ti.T[idl:] + self._Te.T[idl:]) / self.namelist["clen_limiter"]
        
        dv, _ = np.broadcast_arrays(dv, self.time_grid[None])
        
   
        return np.asfortranarray(dv)


    def superstage_DV(self, D_z, V_z, times_DV=None, opt=1):
        """Reduce the dimensionality of D and V time-dependent profiles for the case in which superstaging is applied.

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

        """
        # simple selection of elements corresponding to superstage
        Dzf = D_z[:, :, self.superstages]
        Vzf = V_z[:, :, self.superstages]

        if opt == 1:
            # see selection above
            pass

        elif opt == 2:
            # average D,V over superstage

            superstages = np.r_[self.superstages, self.Z_imp + 1]

            for i in range(len(self.superstages)):
                if superstages[i] + 1 != superstages[i + 1]:
                    Dzf[:, :, i] = D_z[:, :, superstages[i] : superstages[i + 1]].mean(2)
                    Vzf[:, :, i] = V_z[:, :, superstages[i] : superstages[i + 1]].mean(2)

        elif opt == 3:
            # weighted average of D and V
            superstages = np.r_[self.superstages, self.Z_imp + 1]

            # calculate fractional abundances inside of each superstage
            for i in range(len(superstages) - 1):
                if superstages[i] + 1 < superstages[i + 1]:
                    # need to interpolate fz_upstage on time base of D and V -- do this only once
                    if not hasattr(self, "fz_upstage_DV") or self.fz_upstage_DV.shape[
                        2
                    ] == len(times_DV):
                        self.fz_upstage_DV = interp1d(
                            self.time_grid, self.fz_upstage, axis=2
                        )(times_DV)

                    ind = slice(superstages[i], superstages[i + 1])
                    Dzf[:, :, i] = np.sum(
                        D_z[:, :, ind] * self.fz_upstage_DV[:, ind].transpose(0, 2, 1),
                        2,
                    )
                    Vzf[:, :, i] = np.sum(
                        V_z[:, :, ind] * self.fz_upstage_DV[:, ind].transpose(0, 2, 1),
                        2,
                    )

        else:
            raise ValueError("Unrecognized option for D and V superstaging!")

        return Dzf, Vzf

    def run_aurora(
        self,
        D_z,
        V_z,
        times_DV=None,
        nz_init=None,
        ndiv_init=None,
        npump_init=None,
        nmainwall_init=None,
        ndivwall_init=None,
        unstage=True,
        alg_opt=1,
        evolneut=False,
        use_julia=False,
        plot=False,
        plot_radiation=False,
        plot_radial_coordinate = 'rho_vol',
        plot_PWI = False,
    ):
        """Run a simulation using the provided diffusion and convection profiles as a function of space, time
        and potentially also ionization state. Users can give an initial state of each ion charge state as an input.
        While the output impurity density in the plasma (in function of radial location, charge state and time)
        is given in [cm^-3], all the other outputs are given per unit of toroidal length, i.e. the absolute number of
        particles in reservoirs in [cm^-1] and fluxes from/to/between the reservoirs in [cm^-1 s^-1], so that
        effective numbers are achieved multiplying these by circ = 2 * np.pi * self.Raxis_cm.

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
            This may be given as a function of space only, (space,time) or (space,nZ,time),
            where nZ indicates the number of charge states. If given with 1 or 2 dimensions,
            it is assumed that all charge states should have the same diffusion coefficients.
            If given as 1D, it is further assumed that diffusion is time-independent.
            Note that it is assumed that radial profiles are already on the self.rvol_grid radial grid.
        V_z: array, shape of (space,time,nZ) or (space,time) or (space,)
            Convection coefficients, in units of :math:`cm/s`.
            This may be given as a function of space only, (space,time) or (space,nZ,time),
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
        ndiv_init: float
            Neutral impurity content in the divertor reservoir at the initial time of the simulation.
            If left to None, this is internally set to 0.
            If namelist["phys_volumes"] = False (default), this will be interpreted as absolute
            number of impurity neutrals in the divertor reservoir. If True, this will be interpreted
            as neutral impurity density (in cm^-3).
        npump_init: float
            Neutral impurity content in the pump reservoir at the initial time of the simulation,
            effective only if namelist["pump_chamber"] = True (otherwise the pump reservoir is not used).
            If left to None, this is internally set to 0.
            If namelist["phys_volumes"] = False (default), this will be interpreted as absolute
            number of impurity neutrals in the pump reservoir. If True, this will be interpreted
            as neutral impurity density (in cm^-3).
        nmainwall_init: float
            Implanted impurity content in the main wall reservoir at the initial time of the simulation.
            If left to None, this is internally set to 0.
            If namelist["phys_surfaces"] = False (default), this will be interpreted as absolute
            number of impurity neutrals retained at the main wall. If True, this will be interpreted
            as surface implantation density (in cm^-2).
        ndivwall_init: float
            Implanted impurity content in the divertor wall reservoir at the initial time of the simulation,
            effective only if namelist["div_recomb_ratio"] < 1.0 (otherwise the divertor wall is not used).    
            If left to None, this is internally set to 0.
            If namelist["phys_surfaces"] = False (default), this will be interpreted as absolute
            number of impurity neutrals retained at the divertor wall. If True, this will be interpreted
            as surface implantation density (in cm^-2).
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
        plot_radiation : bool, optional
            If True, plot line radiation for each charge state using a convenient slides over time and
            the total radiation time traces.
        plot_radial_coordinate : string, optional
            Radial coordinate shown in the plot. Options: 'rho_vol' (default) or 'rho_pol'

        Returns
        -------
        dict : 
            Dictionary containing simulation results. Depending on the chosen input options, different
            fields will be provided. Impurity charge state distributions in the plasma are always provided
            in the `nz` key, which gives an array of dimensions (nr,nZ,nt). See the notes below for details
            of other available fields.
            The following fields are available in the output dictionary, depending on input options:
    
            nz : array, (nr,nZ,nt)
                Always returned. Charge state densities over the space and time grids [:math:`cm^{-3}`]
                If a number of superstages are indicated in the input, only charge state densities for
                these are returned.
            N_mainwall : array (nt,)
                Number of particles at the main wall reservoir over time [:math:`cm^{-1}`]
            N_divwall : array (nt,)
                Number of particles at the divertor wall reservoir over time [:math:`cm^{-1}`].
                Not empty only if :param:`div_recomb_ratio` < 1.0.
            N_div : array (nt,)
                Number of particles in the divertor reservoir over time [:math:`cm^{-1}`]
            N_pump : array (nt,)
                Number of particles in the pump reservoir over time [:math:`cm^{-1}`].
                Not empty only if :param:`pump_chamber`=True, i.e. if a second reservoirs for neutral particles 
                before the pump is defined.
            N_out : array (nt,)
                Number of particles permanently removed through the pump over time [:math:`cm^{-1}`]
            N_mainret : array (nt,)
                 Number of particles temporarily held in the main wall reservoir [:math:`cm^{-1}`]
            N_divret : array (nt,)
                 Number of particles temporarily held in the divertor wall reservoir [:math:`cm^{-1}`].
                 Not empty only if :param:`div_recomb_ratio` < 1.0, so that ions are allowed to interact 
                 with a divertor wall.
            N_tsu : array (nt,)
                 Edge particle loss [:math:`cm^{-1} s^{-1}`]
            N_dsu : array (nt,)
                 Parallel particle loss [:math:`cm^{-1} s^{-1}`]
            N_dsul : array (nt,)
                 Parallel particle loss at the limiter [:math:`cm^{-1} s^{-1}`]
            rcld_rate : array (nt,)
                 Total recycled flux from the divertor wall reservoir [:math:`cm^{-1} s^{-1}`].
                 Not empty only if :param:`div_recomb_ratio` < 1.0, so that ions are allowed to interact 
                 with a divertor wall.
            rcld_refl_rate : array (nt,)
                 Reflected flux from the divertor wall [:math:`cm^{-1} s^{-1}`].
                 Not empty only if :param:`div_recomb_ratio` < 1.0 AND full PWI model used.
            rcld_recl_rate : array (nt,)
                 Promptly recycled flux from the divertor wall [:math:`cm^{-1} s^{-1}`].
                 Not empty only if :param:`div_recomb_ratio` < 1.0 AND full PWI model used.
            rcld_impl_rate : array (nt,)
                 Implanted flux into the divertor wall reservoir [:math:`cm^{-1} s^{-1}`]
                 Not empty only if :param:`div_recomb_ratio` < 1.0 AND full PWI model used.
            rcld_sput_rate : array (1+background_species,nt,)
                 Sputtered fluxes from the divertor wall reservoir, one for each sputtering species 
                 [:math:`cm^{-1} s^{-1}`]. Not empty only if :param:`div_recomb_ratio` < 1.0 AND
                 full PWI model used.
            rclb_rate : array (nt,)
                 Backflow from the divertor neutrals reservoir reaching the plasma [:math:`cm^{-1} s^{-1}`]
            rcls_rate : array (nt,)
                 Screened backflow from the divertor neutrals reservoir [:math:`cm^{-1} s^{-1}`].
                 Not empty only if :param:`div_neut_screen` > 0.0.
            rclp_rate : array (nt,)
                 Leakage from the pump neutrals reservoir [:math:`cm^{-1} s^{-1}`].
                 Not empty only if :param:`pump_chamber` = True.
            rclw_rate : array (nt,)
                 Total recycled flux from the main wall reservoir [:math:`cm^{-1} s^{-1}`]
            rclw_refl_rate : array (nt,)
                 Reflected flux from the main wall [:math:`cm^{-1} s^{-1}`].
                 Not empty only if full PWI model used.
            rclw_recl_rate : array (nt,)
                 Promptly recycled flux from the main wall [:math:`cm^{-1} s^{-1}`].
                 Not empty only if full PWI model used.
            rclw_impl_rate : array (nt,)
                 Implanted flux into the main wall reservoir [:math:`cm^{-1} s^{-1}`]
                 Not empty only if full PWI model used.
            rclw_sput_rate : array (1+background_species,nt,)
                 Sputtered fluxes from the main wall reservoir, one for each sputtering species 
                 [:math:`cm^{-1} s^{-1}`]. Not empty only if full PWI model used.
        """
        D_z, V_z = np.asarray(D_z), np.asarray(V_z)

        # D_z and V_z must have the same shape
        assert np.shape(D_z) == np.shape(V_z)

        if times_DV is None and D_z.ndim > 1 and D_z.shape[1] > 1:
            raise ValueError(
                "D_z and V_z given as time dependent, but times were not specified!"
            )

        if times_DV is None or np.size(times_DV) == 0:
            times_DV = [1.0]  # dummy, no time dependence

        num_cs = int(self.Z_imp + 1)

        # D and V were given for all stages -- define D and V for superstages
        if len(self.superstages):
            num_cs = len(self.superstages)
            if D_z.ndim == 3 and D_z.shape[2] == self.Z_imp + 1:
                D_z, V_z = self.superstage_DV(D_z, V_z, times_DV, opt=1)

        if not evolneut:
            # prevent recombination back to neutral state to maintain good particle conservation
            self.Rne_rates[:, 0] = 0

        if nz_init is None:
            # default: start in a state with no impurity ions
            nz_init = np.zeros((len(self.rvol_grid), num_cs))
        
        # factor to account for cylindrical geometry:
        circ = 2 * np.pi * self.Raxis_cm  # cm
            
        if ndiv_init is None:
            # default: start in a state with empty divertor neutrals reservoir
            ndiv_init = 0
        else:
            # start in a state with not empty divertor neutrals reservoir
            # convert to cm^-1 before passing it to fortran_run
            if self.namelist["phys_volumes"] == True:
                # the input ndiv_init is interpreted as a particle density in cm^-3
                ndiv_init = (ndiv_init * self.vol_div) / circ # ([cm^-3]*[cm^3])/[cm]
            else:
                # the input ndiv_init is interpreted as absolute number of particles
                ndiv_init = ndiv_init / circ        
                
        if npump_init is None or not self.namelist["pump_chamber"]:
            # default: start in a state with empty pump neutrals reservoir
            #   (or pump reservoir not used at all)
            npump_init = 0 
        else:
            # start in a state with not empty pump neutrals reservoir
            # convert to cm^-1 before passing it to fortran_run
            if self.namelist["phys_volumes"]:
                # the input npump_init is interpreted as a particle density in cm^-3
                npump_init = (npump_init * self.vol_pump) / circ # ([cm^-3]*[cm^3])/[cm]
            else:
                # the input npump_init is interpreted as absolute number of particles
                npump_init = npump_init / circ      
        
        if nmainwall_init is None:
            # default: start in a state with empty main wall reservoir
            nmainwall_init = 0
        else:
            # start in a state with not empty main wall reservoir
            # convert to cm^-1 before passing it to fortran_run
            if self.namelist["phys_surfaces"] == True:
                # the input nmainwall_init is interpreted as a particle density in cm^-2
                nmainwall_init = (nmainwall_init * self.surf_mainwall * self.mainwall_roughness) / circ # ([cm^-2]*[cm^2])/[cm]
            else:
                # the input nmainwall_init is interpreted as absolute number of particles
                nmainwall_init = nmainwall_init / circ     
        
        if ndivwall_init is None or self.namelist["div_recomb_ratio"] == 1.0:
            # default: start in a state with empty divertor wall reservoir
            #   (or divertor wall reservoir not used at all)
            ndivwall_init = 0  
        else:
            # start in a state with not empty divertor wall reservoir
            # convert to cm^-1 before passing it to fortran_run
            if self.namelist["phys_surfaces"] == True:
                # the input ndivwall_init is interpreted as a particle density in cm^-2
                ndivwall_init = (ndivwall_init * self.surf_divwall * self.divwall_roughness) / circ # ([cm^-2]*[cm^2])/[cm]
            else:
                # the input ndivwall_init is interpreted as absolute number of particles
                ndivwall_init = ndivwall_init / circ   

        if D_z.ndim < 3:
            # set all charge states to have the same transport
            # num_cs = Z+1 - include elements for neutrals
            D_z = np.tile(D_z.T, (num_cs, 1, 1)).T  # create fortran contiguous arrays
            D_z[:, :, 0] = 0.0

        if V_z.ndim < 3:
            # set all charge states to have the same transport
            V_z = np.tile(V_z.T, (num_cs, 1, 1)).T  # create fortran contiguous arrays
            V_z[:, :, 0] = 0.0

        nt = len(self.time_out)

        # NOTE: for both Fortran and Julia, use f_configuous arrays for speed
        if use_julia:
            
            if self.namelist["div_recomb_ratio"] < 1.0 or self.pump_chamber or self.div_neut_screen > 0.0:
                raise ValueError("Full recycling/pumping/PWI model not yet implemented in Julia!")
            
            # run Julia version of the code
            from julia.api import Julia

            jl = Julia(
                compiled_modules=False,
                sysimage=os.path.dirname(os.path.realpath(__file__))
                + "/../aurora.jl/sysimage.so",
            )
            from julia import aurora as aurora_jl

            _res = aurora_jl.run(
                nt,  # number of times at which simulation outputs results
                times_DV,
                D_z,
                V_z,  # cm^2/s & cm/s    #(ir,nt_trans,nion)
                self.par_loss_rate,  # time dependent
                self.src_core,  # source profile in radius and time
                self.rcl_rad_prof,  # recycling radial profile
                self.Sne_rates,  # ioniz_rate,
                self.Rne_rates,  # recomb_rate,
                self.rvol_grid,
                self.pro_grid,
                self.qpr_grid,
                self.mixing_radius,
                self.decay_length_boundary,  # cm
                self.time_grid,
                self.saw_on,
                self.save_time,
                self.crash_width,  # dsaw width  [cm]
                self.wall_recycling,
                self.tau_div_SOL_ms * 1e-3,  # [s]
                self.tau_pump_ms * 1e-3,  # [s]
                self.tau_rcl_ret_ms * 1e-3,  # [s]
                self.rvol_lcfs,
                self.bound_sep,
                self.lim_sep,
                self.prox_param,
                nz_init,
                alg_opt,
                evolneut,
                self.src_div,
            )
        else:


            # import here to avoid import when building documentation or package (negligible slow down)
            from ._aurora import run as fortran_run
            _res = fortran_run(
                nt,  # number of times at which simulation outputs results
                times_DV,
                D_z,
                V_z,  # cm^2/s & cm/s    #(ir,nt_trans,nion)
                self.par_loss_rate, # parallel loss rate values in the SOL on the time grid
                self.src_core, # radial source profile of externally injected neutrals in the plasma on the time grid
                self.rcl_rad_prof, # radial source profile of promptly recycled neutrals in the plasma on the time grid
                self.rfl_rad_prof, # radial source profile of reflected neutrals in the plasma on the time grid
                self.spt_rad_prof, # radial source profile of sputtered neutrals in the plasma on the time grid
                self.energetic_recycled_neutrals, # logic key for setting energetic reflected/sputtered neutrals
                self.Sne_rates, # ionization rates in the plasma
                self.Rne_rates, # recombination rates in the plasma
                self.Raxis_cm, # major radius at the magnetic axis [cm]
                self.rvol_grid, # radial grid values of rho_vol
                self.pro_grid,
                self.qpr_grid,
                self.mixing_radius,
                self.decay_length_boundary,
                self.time_grid, # time grid values
                self.saw_on, # logic key for sawteeth model
                self.save_time,
                self.crash_width,  # dsaw width [cm]
                self.wall_recycling,
                self.div_neut_screen, # screening coefficient
                self.div_recomb_ratio, # divertor recombination coefficient
                self.tau_div_SOL_ms * 1e-3,  # [s]
                self.tau_pump_ms * 1e-3,  # [s]
                self.tau_rcl_ret_ms * 1e-3,  # [s]
                self.S_pump, # pumping speed in the dimensional pumping model [cm^3/s]
                self.vol_div, # volume of the divertor neutrals reservoir [cm^3]
                self.L_divpump, # conductance between divertor and pump neutrals reservoirs [cm^3/s]
                self.vol_pump, # volume of the pump neutrals reservoirs[cm^3]
                self.L_leak, # leakage conductance from pump neutrals reservoir towards plasma [cm^3/s]
                self.surf_mainwall_eff, # effective main wall surface area [cm^2]
                self.surf_divwall_eff, # effective divertor wall surface area [cm^2]
                self.full_PWI_flag, # logic key for PWI model
                self.Z_main_wall, # atomic number of the main wall material
                self.Z_div_wall, # atomic number of the divertor wall material
                self.rn_main_wall, # reflection coefficients for the simulated impurity at the main wall on the time grid
                self.rn_div_wall, # reflection coefficients for the simulated impurity at the divertor wall on the time grid
                self.fluxes_main_wall_background, # fluxes for each background species onto the main wall on the time grid [s^-1]
                self.fluxes_div_wall_background, # fluxes for each background species onto the divertor wall on the time grid [s^-1]
                self.y_main_wall, # sputtering yields from simulated impurity + background species from the main wall on the time grid
                self.y_div_wall, # sputtering yields from simulated impurity + background species from the divertor wall on the time grid
                self.implantation_depth_main_wall, # considered impurity implantation depth in the main wall [A]
                self.implantation_depth_div_wall, # considered impurity implantation depth in the divertor wall [A]
                self.n_main_wall_sat, # considered saturation value of the impurity implantation density into the main wall [m^-2]
                self.n_div_wall_sat, # considered saturation value of the impurity implantation density into the divertor wall [m^-2]
                self.rvol_lcfs,
                self.bound_sep,
                self.lim_sep,
                self.prox_param,
                rn_t0=nz_init,  # if omitted, internally set to 0's
                ndiv_t0=ndiv_init,  # if omitted, internally set to 0       
                npump_t0=npump_init,  # if omitted, internally set to 0     
                nmainwall_t0=nmainwall_init,  # if omitted, internally set to 0     
                ndivwall_t0=ndivwall_init,  # if omitted, internally set to 0  
                alg_opt=alg_opt,
                evolneut=evolneut,
                src_div=self.src_div,
            )
             
        # add output fields in a dictionary
        self.res = {}
        
        if use_julia: # full recycling/pumping/PWI model not implemented yet in Julia --> self.res contains less elements
            
            (
                self.res['nz'],
                self.res['N_mainwall'],
                self.res['N_div'],
                self.res['N_self.res'],
                self.res['N_mainret'],
                self.res['N_tsu'],
                self.res['N_dsu'],
                self.res['N_dsul'],
                self.res['rclb_rate'],
                self.res['rclw_rate'],
            ) = _res
        
        else: # full recycling/pumping/PWI model fully implemented in Fortran
        
            (
                self.res['nz'],
                self.res['N_mainwall'],
                self.res['N_divwall'],
                self.res['N_div'],
                self.res['N_pump'],
                self.res['N_out'],
                self.res['N_mainret'],
                self.res['N_divret'],
                self.res['N_tsu'],
                self.res['N_dsu'],
                self.res['N_dsul'],
                self.res['rcld_rate'],
                self.res['rcld_refl_rate'],
                self.res['rcld_recl_rate'],
                self.res['rcld_impl_rate'],
                self.res['rcld_sput_rate'],
                self.res['rclb_rate'],
                self.res['rcls_rate'],
                self.res['rclp_rate'],
                self.res['rclw_rate'],
                self.res['rclw_refl_rate'],
                self.res['rclw_recl_rate'],
                self.res['rclw_impl_rate'],
                self.res['rclw_sput_rate']
            ) = _res
        
        if plot:
            
            if plot_radial_coordinate == 'rho_vol':
                x = self.rvol_grid
                xlabel = r"$r_V$ [cm]"
                x_line=self.rvol_lcfs
            elif plot_radial_coordinate == 'rho_pol':
                x = self.rhop_grid
                xlabel=r'$\rho_p$'
                x_line = 1
        
            # plot charge state density distributions over radius and time
            plot_tools.slider_plot(
                x,
                self.time_out,
                self.res['nz'].transpose(1, 0, 2),
                xlabel=xlabel,
                ylabel="time [s]",
                zlabel=f'$n_{{{self.imp}}}$ [$\mathrm{{cm}}$$^{{-3}}$]',
                plot_title = f'{self.imp} density profiles',
                labels=[str(i) for i in np.arange(0, self.res['nz'].shape[1])],
                plot_sum=True,
                x_line=x_line,
                zlim = True,
            )
        
            if plot_radiation:
                
                # compute line radiation
                self.rad = radiation.compute_rad(
                    self.imp,
                    self.res['nz'].transpose(2, 1, 0),
                    self.ne,
                    self.Te,
                    prad_flag=True,
                    thermal_cx_rad_flag=False,
                    spectral_brem_flag=False,
                    sxr_flag=False,
                )
                
                # plot line radiation distribution from each charge state over radius and time
                plot_tools.slider_plot(
                    x,
                    self.time_out,
                    self.rad["line_rad"].transpose(1, 2, 0),
                    xlabel=xlabel,
                    ylabel="time [s]",
                    zlabel="[$\mathrm{MW}/\mathrm{m}^3$]",
                    plot_title = f'{self.imp} line radiation',
                    labels=[str(i) for i in np.arange(0, self.res['nz'].shape[1])],
                    plot_sum=True,
                    x_line=x_line,
                    zlim = True,
                )
        
            # check particle conservation by summing over simulation reservoirs
            _ = self.reservoirs_time_traces(plot=True)


        # Plot PWI model
        if plot_PWI and self.full_PWI_flag:
            _ = self.PWI_time_traces(plot = True)



        if len(self.superstages) and unstage:
            # "unstage" superstages to recover estimates for density of all charge states
            nz_unstaged = np.zeros((len(self.rvol_grid), self.Z_imp + 1, nt))

            superstages = np.r_[self.superstages, self.Z_imp + 1]

            # calculate fractional abundances inside of each superstage
            for i in range(len(superstages) - 1):
                if superstages[i] + 1 < superstages[i + 1]:
                    # fill skipped stages from ionization equilibrium
                    ind = slice(superstages[i], superstages[i + 1])
                    nz_unstaged[:, ind] = self.res['nz'][:, [i]] * self.fz_upstage[:, ind]
                else:
                    nz_unstaged[:, superstages[i]] = self.res['nz'][:, i]
          
            self.res['nz'] = nz_unstaged
 

        return self.res

    def run_aurora_steady_analytic(self, D_z, V_z):
        """Evaluate the analytic steady state solution of the transport equation.
        Small differences in absolute densities from the full time-dependent, multi-reservoir Aurora solutions
        can be caused by recycling and divertor models, which are not included here.

        Parameters
        ----------
        D_z: array, shape of (space,nZ) or (space,)
            Diffusion coefficients, in units of :math:`cm^2/s`. This may be given as a function of space only or (space,nZ).
            No time dependence is allowed in this function. Here, nZ indicates the number of charge states.
            Note that it is assumed that radial profiles are already on the self.rvol_grid radial grid.
        V_z: array, shape of (space,nZ) or (space,)
            Convection coefficients, in units of :math:`cm/s`. This may be given as a function of space only or (space,nZ).
            No time dependence is allowed in this function. Here, nZ indicates the number of charge states.
        """
        if self.ne.shape[0] > 1 or self.Te.shape[0] > 1:
            raise ValueError(
                "This method is designed to operate with time-independent background profiles!"
            )
        if self.full_PWI_flag:
            raise Exception('It cannot be supported by PWI model!')
            
            
        if len(self.superstages) > 0:
            raise Exception("Superstages are not yet suported by analytical solver")

        if D_z.shape != V_z.shape:
            raise ValueError("Shape of D and V must be the same")

        if D_z.ndim > 2:
            raise ValueError(
                "This method is designed to operate with time-independent transport coefficients"
            )

        if D_z.ndim == 2:
            # make sure that transport coefficients were given as a function of space and nZ, not time!
            assert (
                D_z.shape[0] == len(self.rhop_grid) and D_z.shape[1] == self.Z_imp + 1
            )
            D_z = D_z[:, 1:]
            V_z = V_z[:, 1:]

        def between_grid(arr, axis=0):
            # calculate value in between grid points
            sl = (slice(None),) * axis
            return (arr[sl + (slice(1, None),)] + arr[sl + (slice(None, -1),)]) / 2.0

        # prepare inputs
        D_btw = between_grid(D_z, 0).T  # input D profile btw the spatial grids
        v_btw = between_grid(V_z, 0).T  # input v profile btw the spatial grids

        par_loss_rate = between_grid(
            self.par_loss_rate[:, 0] / 100
        )  # convert in the 1/s/cm units

        del_r = np.diff(self.rvol_grid)
        r_btw = between_grid(self.rvol_grid)
        self.rhop_btw = between_grid(self.rhop_grid)

        nr = len(r_btw)

        # diagonal of matrices for recombination, ionisation and parallel flux
        r_z = between_grid(self.Rne_rates.T[0, :-1], 1)
        s_z = between_grid(self.Sne_rates.T[0, :-1], 1)
        p_z = par_loss_rate * r_btw

        # in case that metastable resolved atomics data are used
        metastables = self.namelist.get("metastable_flag", False)
        if metastables:
            q_z = between_grid(self.Qne_rates.T[0], 1)
            x_z = between_grid(self.Xne_rates.T[0], 1)

            x_meta_ind = self.atom_data["xcd"].meta_ind
            q_meta_ind = self.atom_data["qcd"].meta_ind

        s_meta_ind = self.atom_data["scd"].meta_ind
        r_meta_ind = self.atom_data["acd"].meta_ind

        # number of metastables for each ion (ones if data are unresolved)
        Mmeta = self.atom_data["scd"].metastables
        # index of each metastable
        meta_ind = [(z, m + 1) for z, M in enumerate(Mmeta) for m in range(M)]
        n_state = len(meta_ind)

        # uvec
        exp_diag = np.exp(cumulative_trapezoid(v_btw / D_btw, r_btw, initial=0, axis=-1))
        exp_diag /= exp_diag[..., [-1]]

        exp_Dr = 1 / (exp_diag * D_btw * r_btw)  # diagonal of the matrix
        lam = self.decay_length_boundary
        edge_factor = 1 / (1 / lam + v_btw[..., [-1]] / D_btw[..., [-1]])  # units of m

        # calculate spatial integrals
        del_r_up = np.tri(nr + 1, nr, -1) * (del_r * r_btw)
        del_r_down = np.tri(nr, nr + 1).T * del_r

        def apply_integral(prof, i):
            # charge independent transport coefficients
            i = ... if D_btw.ndim == 1 else i - 1
            # the first cumulative trapezoid integral
            a = del_r_up * prof
            a = between_grid(a)
            a *= exp_Dr[i, :, None]
            # the second cumulative trapezoid integral
            a = np.dot(del_r_down, a)
            a = between_grid(a)
            # incorporate edge boundary condition with decay length lambda
            a += a[-1] / del_r[-1] * edge_factor[i]
            a *= exp_diag[i, :, None]
            return a

        R_z = {i: apply_integral(r, i[0]) for i, r in zip(r_meta_ind, r_z)}
        S_z = {i: apply_integral(s, i[0]) for i, s in zip(s_meta_ind, s_z)}

        if D_z.ndim == 2:
            P_z = [apply_integral(p_z, z) for z in range(self.Z_imp + 1)]
        else:
            P_z = [apply_integral(p_z, 0)] * (self.Z_imp + 1)

        if metastables:
            Q_z = {i: apply_integral(q, i[0]) for i, q in zip(q_meta_ind, q_z)}
            X_z = {i: apply_integral(x, i[0]) for i, x in zip(x_meta_ind, x_z)}

        # radial integral of the source used to calculate total impurity density profile
        nz_0 = np.zeros((n_state, nr))
        nimp0 = between_grid(self.src_core[:, 0])

        # metastable distributon of neutral influx is unknow
        meta_source = (1, 1)
        nz_0[meta_ind.index(meta_source)] = np.dot(S_z[meta_source + (1,)], nimp0)

        source_mtx = np.zeros([n_state, n_state, nr, nr])
        for i, (z, m) in enumerate(meta_ind):
            # paraell looses are on diagonal
            source_mtx[i, i] -= P_z[z]

            if z < self.Z_imp:  # for non-fully stripped species
                for g in range(1, Mmeta[z + 1] + 1):
                    source_mtx[i, i] -= S_z[(z + 1, m, g)]
                    j = meta_ind.index((z + 1, g))
                    source_mtx[i, j] += R_z[(z + 1, m, g)]

            if z > 0:  # for ionised species
                for g in range(1, Mmeta[z - 1] + 1):
                    source_mtx[i, i] -= R_z[(z, g, m)]
                    j = meta_ind.index((z - 1, g))
                    source_mtx[i, j] += S_z[(z, g, m)]

            if not metastables:
                continue

            # cross-coupling of the metastables
            for g in range(1, Mmeta[z] + 1):
                if g == m:
                    continue
                j = meta_ind.index((z, g))
                # cross-coupling coefficients important at lower Te
                source_mtx[i, j] += Q_z[(z + 1, m, g)]
                source_mtx[i, i] -= Q_z[(z + 1, g, m)]

                # Beringer 1989 is not including the parent cross coupling coefficients!
                # cross-coupling between parents via recombination to excited states
                # of the z-times ionised ion followed by re-ionisation to a different metastable
                # BUG when included, the summed ion metastable stage profiles disagree with unresolved profile
                # if g > m and z > 0:
                # source_mtx[i,j] += X_z[(z, g, m)] #BUG not yet sure about the order of m,g indexes
                # if g < m and z > 0:
                # source_mtx[i,i] -= X_z[(z, g, m)]

        # reshape to 2D matrix
        source_mtx = source_mtx.swapaxes(1, 2).reshape(nr * n_state, nr * n_state)

        if not metastables:
            # faster inversion if the matrix is block diagonal
            # convert source_mtx matrix to band diagonal form
            n_diags = 2 * nr
            source_mtx_diag = np.zeros((1 + 2 * n_diags, nr * n_state))
            for i, d in enumerate(range(n_diags, -n_diags - 1, -1)):
                source_mtx_diag[
                    i, np.maximum(0, d) : nr * n_state + np.minimum(0, d)
                ] = np.diagonal(source_mtx, d)

            # calculate eye( ) - source_mtx
            mtx_solv_diag = source_mtx_diag
            mtx_solv_diag *= -1
            mtx_solv_diag[n_diags] += 1

            # solve band matrix
            nz_steady = solve_banded(
                (n_diags, n_diags), mtx_solv_diag, nz_0.flatten()
            ).reshape(-1, nr)

        else:
            # standart numpy solver for a square matrix
            A = np.eye(nr * n_state) - source_mtx
            # slowest step
            nz_steady = np.linalg.solve(A, nz_0.flatten()).reshape(-1, nr)

        return meta_ind, nz_steady

    def run_aurora_steady(
        self,
        D_z,
        V_z,
        nz_init=None,
        unstage=False,
        alg_opt=1,
        evolneut=False,
        use_julia=False,
        tolerance=0.01,
        max_sim_time=100,
        dt=1e-4,
        dt_increase=1.05,
        n_steps=100,
        plot=False,
        plot_radial_coordinate = 'rho_vol',
    ):
        """Run an Aurora simulation until reaching steady state profiles. This method calls :py:meth:`~aurora.core.run_aurora`
        checking at every iteration whether profile shapes are still changing within a given fractional tolerance.
        Note that this method differs from :py:meth:`~aurora.core.run_aurora_steady_analytic` in that it runs time-dependent
        simulations until reaching steady-state, rather than using an analytic solution for steady-state profiles.

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
        plot_radial_coordinate : string, optional
            Radial coordinate shown in the plot. Options: 'rho_vol' (default) or 'rho_pol'
        """

        if n_steps < 2:
            raise ValueError("n_steps must be greater than 2!")

        if self.ne.shape[0] > 1:
            raise ValueError(
                "This method is designed to operate with time-independent background profiles!"
            )

        if D_z.ndim > 2 or V_z.ndim > 2:
            raise ValueError(
                "This method is designed to operate with time-independent D and V profiles!"
            )
            
        if self.full_PWI_flag:
            raise Exception('Not yet supported by PWI model!')

        # set constant timesource
        self.namelist["source_type"] = "const"
        # self.namelist["source_rate"] = 1.0

        # build timing dictionary
        self.namelist["timing"] = {
            "dt_start": [dt, dt],
            "dt_increase": [dt_increase, 1.0],
            "steps_per_cycle": [1, 1],
            "times": [0.0, max_sim_time],
        }

        # prepare radial and temporal grid
        self.setup_grids()

        # update kinetic profile dependencies to get everything to the right shape
        self.setup_kin_profs_depts()

        times_DV = None
        if D_z.ndim == 2:
            # make sure that transport coefficients were given as a function of space and nZ, not time!
            assert D_z.shape[0] == len(self.rhop_grid) and D_z.shape[1] == self.Z_imp + 1
            assert V_z.shape[0] == len(self.rhop_grid) and V_z.shape[1] == self.Z_imp + 1

            D_z = D_z[:, None]  # (ir,nt_trans,nion)
            V_z = V_z[:, None]

        sim_steps = 0

        time_grid = self.time_grid.copy()
        time_out = self.time_out.copy()
        save_time = self.save_time.copy()
        par_loss_rate = self.par_loss_rate.copy()
        rcl_rad_prof = self.rcl_rad_prof.copy()
        src_core = self.src_core.copy()
        Sne_rates = self.Sne_rates.copy()
        Rne_rates = self.Rne_rates.copy()
        saw_on = self.saw_on.copy()
        src_div = self.src_div.copy()
        nz_all = None if nz_init is None else nz_init

        while sim_steps < len(time_grid):
            self.time_grid = self.time_out = time_grid[sim_steps : sim_steps + n_steps]
            self.save_time = save_time[sim_steps : sim_steps + n_steps]
            self.par_loss_rate = par_loss_rate[:, sim_steps : sim_steps + n_steps]
            self.rcl_rad_prof = rcl_rad_prof[:, sim_steps : sim_steps + n_steps]
            self.src_core = src_core[:, sim_steps : sim_steps + n_steps]
            self.Sne_rates = Sne_rates[:, :, sim_steps : sim_steps + n_steps]
            self.Rne_rates = Rne_rates[:, :, sim_steps : sim_steps + n_steps]
            self.saw_on = saw_on[sim_steps : sim_steps + n_steps]
            self.src_div = src_div[sim_steps : sim_steps + n_steps]

            sim_steps += n_steps

            # get charge state densities from latest time step
            if nz_all is None:
                nz_init = None
            else:
                nz_init = nz_all[:, :, -1] if nz_all.ndim == 3 else nz_all
            
            #update dummy variable sizes of the PWI model
            self.setup_dummy_pwi_vars() 
            nz_new = self.run_aurora(
                D_z,
                V_z,
                times_DV,
                nz_init=nz_init,
                unstage=unstage,
                alg_opt=alg_opt,
                evolneut=evolneut,
                use_julia=use_julia,
                plot=False,
            )['nz']

            if nz_all is None:
                nz_all = np.dstack((np.zeros_like(nz_new[:, :, [0]]), nz_new))
                nz_init = np.zeros_like(nz_new[:, :, 0])
            else:
                nz_all = np.dstack((nz_all, nz_new))

            # check if normalized profiles have converged
            if (
                np.linalg.norm(nz_new[:, :, -1] - nz_init)
                / (np.linalg.norm(nz_init) + 1e-99)
                < tolerance
            ):
                break

        # store final time grids
        self.time_grid = time_grid[:sim_steps]
        # identical because steps_per_cycle is fixed to 1
        self.time_out  = time_grid[:sim_steps]
        self.save_time = save_time[:sim_steps]

        if plot:
            
            if plot_radial_coordinate == 'rho_vol':
                x = self.rvol_grid
                xlabel = r"$r_V$ [cm]"
                x_line=self.rvol_lcfs
            elif plot_radial_coordinate == 'rho_pol':
                x = self.rhop_grid
                xlabel=r'$\rho_p$'
                x_line = 1
            
            # plot charge state distributions over radius and time
            plot_tools.slider_plot(
                x,
                self.time_grid,
                nz_all.transpose(1, 0, 2),
                xlabel=xlabel,
                ylabel="time [s]",
                zlabel=f'$n_{{{self.imp}}}$ [cm$^{{-3}}$]',
                plot_title = f'{self.imp} density profiles',
                labels=[str(i) for i in np.arange(0, nz_all.shape[1])],
                plot_sum=True,
                x_line=x_line,
                zlim = True,
            )

        if sim_steps >= len(time_grid):
            raise ValueError(
                f"Could not reach convergence before {max_sim_time:.3f}s of simulated time!"
            )

        # compute effective particle confinement time from latest few time steps
        circ = 2 * np.pi * self.Raxis_cm  # cm
        zvol = circ * np.pi * self.rvol_grid / self.pro_grid

        wh = self.rhop_grid <= 1
        var_volint = np.nansum(nz_new[wh, :, -2] * zvol[wh, None])

        # compute effective particle confinement time
        source_time_history = grids_utils.vol_int(
            self.src_core.T,
            self.rvol_grid,
            self.pro_grid,
            self.Raxis_cm,
            rvol_max=self.rvol_lcfs,
        )
        
        # avoid last time point because source may be 0 there
        self.tau_imp = var_volint / source_time_history[-2]
        

        return nz_new[:, :, -1]

    def calc_Zeff(self):
        """Compute Zeff from each charge state density, using the result of an AURORA simulation.
        The total Zeff change over time and space due to the simulated impurity can be simply obtained by summing
        over charge states.

        Results are stored as an attribute of the simulation object instance.
        """
        # This method requires that a simulation has already been run:
        assert hasattr(self, "res")

        # extract charge state densities from the simulation result
        nz = self.res[0]

        # this method requires all charge states to be made available
        try:
            assert nz.shape[1] == self.Z_imp + 1
        except AssertionError:
            raise ValueError(
                "calc_Zeff method requires all charge state densities to be availble! Unstage superstages."
            )

        # Compute the variation of Zeff from these charge states
        Zmax = nz.shape[1] - 1
        Z = np.arange(Zmax + 1)
        self.delta_Zeff = nz * (Z * (Z - 1))[None, :, None]  # for each charge state
        self.delta_Zeff /= self.ne.T[:, None, :]

    def plot_resolutions(self):
        """Convenience function to show time and spatial resolution in Aurora simulation setup."""
        # display radial resolution
        _ = grids_utils.create_radial_grid(self.namelist, plot=True)

        # display time resolution
        _ = grids_utils.create_time_grid(timing=self.namelist["timing"], plot=True)
        
    def plot_kin_prof(self, var='ne', rad_coord = 'rvol'):
        """Convenience function to display the kinetic profiles used in the Aurora simulation.
        
        Parameters
        ----------
        var : str
            Variable to plot, one of ['ne','Te','Ti','n0']. Default is 'ne'.
        rad_coord : str
            Radial coordinate to use in plotting, one of ['rvol','rhop']. Default is 'rvol'.
        """
        _prof = getattr(self, var)
        prof = np.reshape(_prof, (1, _prof.shape[1], _prof.shape[0]))
        qlabels = {'ne': '$n_e$ [cm$^{{-3}}$]', 'Te': '$T_e$ [eV]', 'Ti': '$T_i$ [eV]', 'n0': '$n_n$ [cm$^{{-3}}$]'}
                   
        plot_tools.slider_plot(
            self.rvol_grid if rad_coord=='rvol' else self.rhop_grid,
            self.time_out,
            _prof,
            xlabel= r"$r_V$ [cm]" if rad_coord=='rvol' else r'$\rho_p$',
            ylabel="time [s]",
            zlabel=qlabels[var],
            x_line = self.rvol_lcfs if rad_coord=='rvol' else 1.0,
            zlim = True,
        )

    def reservoirs_time_traces(self, plot=True, ylim = True, axs=None, plot_resolutions=False):
        """Plot the particle content in the various reservoirs
        and check the particle conservation for an aurora simulation.

        Parameters
        ----------
        plot : bool, optional
            If True, plot time histories in each particle reservoir and display quality of particle conservation.
        axs : 2-tuple or array
            Array-like structure containing two matplotlib.Axes instances: the first one
            for the separate particle time variation in each reservoir and the main fluxes,
            and the second one for the total particle-conservation check.

        Returns
        -------
        reservoirs : dict
            Dictionary containing density of particles in each reservoir.
        axs : matplotlib.Axes instances, only returned if plot=True
            Array-like structure containing two matplotlib.Axes instances, (ax1,ax2).
            See optional input argument.
        """
        
        # get colors for plots
        colors = plot_tools.load_color_codes_reservoirs()
        blue,light_blue,green,light_green,grey,light_grey,red,light_red = colors

        # factor to account for cylindrical geometry:
        circ = 2 * np.pi * self.Raxis_cm  # cm



        # calculate total impurity density (summed over charge states)
        total_impurity_density = np.nansum(self.res['nz'], axis=1).T  # time, space

        # Compute total number of particles for particle conservation checks:
        all_particles = grids_utils.vol_int(
            total_impurity_density,
            self.rvol_grid,
            self.pro_grid,
            self.Raxis_cm,
            rvol_max=None,
        )
        
        # collect all the relevant quantities for particle conservation
        reservoirs = {}

        reservoirs["total"] = all_particles + (self.res['N_mainwall'] + self.res['N_divwall'] + self.res['N_div'] + self.res['N_pump'] + self.res['N_out'] + self.res['N_mainret'] + self.res['N_divret']) * circ

        # main fluxes
        reservoirs["source"] = self.total_source*circ
        reservoirs["plasma_source"] = self.total_source*circ
        reservoirs["wall_source"] = self.res['rclw_rate'] * circ
        reservoirs["divertor_source"] = self.res['rclb_rate'] * circ + self.res['rclp_rate'] * circ
        reservoirs["plasma_removal_rate"] = - self.res['N_dsu'] * circ - self.res['N_tsu'] * circ - self.res['N_dsul'] * circ       
        reservoirs["net_plasma_flow"] = reservoirs["plasma_source"] + reservoirs["wall_source"] + reservoirs["divertor_source"] + reservoirs["plasma_removal_rate"]

        # integrated source over time
        reservoirs["integ_source"] = cumulative_trapezoid(reservoirs["source"], self.time_out, initial=0) + reservoirs["total"][0] 
        
        # main plasma content
        if self.namelist["phys_volumes"]:    
            reservoirs["particle_density_in_plasma"] = all_particles/self.plasma_vol
        reservoirs["particles_in_plasma"] = all_particles
        
        # divertor and pump neutrals reservoirs
        if self.namelist["phys_volumes"]:
            reservoirs["particle_density_in_divertor"] = (self.res['N_div'] * circ)/self.vol_div
            if self.namelist["pump_chamber"]:
                reservoirs["particle_density_in_pump"] = (self.res['N_pump'] * circ)/self.vol_pump
        reservoirs["particles_in_divertor"] = self.res['N_div'] * circ
        reservoirs["particles_in_pump"] = self.res['N_pump'] * circ
        
        # fluxes towards main wall
        reservoirs["edge_loss"] = self.res['N_tsu'] * circ
        reservoirs["limiter_loss"] = self.res['N_dsul'] * circ
        reservoirs["total_flux_mainwall"] = reservoirs["edge_loss"] + reservoirs["limiter_loss"]
        
        # recycling rates from main wall
        reservoirs["mainwall_recycling"] = self.res['rclw_rate'] * circ
        
        # main wall reservoir
        if self.namelist["phys_surfaces"]:
            reservoirs["particle_density_stuck_at_main_wall"] = (self.res['N_mainwall'] * circ)/(self.surf_mainwall*self.mainwall_roughness)
            reservoirs["particle_density_retained_at_main_wall"] = (self.res['N_mainret'] * circ)/(self.surf_mainwall*self.mainwall_roughness)
        reservoirs["particles_stuck_at_main_wall"] = self.res['N_mainwall'] * circ
        reservoirs["particles_retained_at_main_wall"] = self.res['N_mainret'] * circ

        # flux towards divertor targets and backflow/leakage rates
        reservoirs["parallel_loss"] = self.res['N_dsu'] * circ
        reservoirs["divertor_backflow"] = self.res['rclb_rate'] * circ
        reservoirs["screened_divertor_backflow"] = self.res['rcls_rate'] * circ
        reservoirs["pump_leakage"] = self.res['rclp_rate'] * circ
        reservoirs["total_flux_divwall"] = (reservoirs["parallel_loss"]+reservoirs["screened_divertor_backflow"])*(1-self.div_recomb_ratio)
        
        # recycling rates from divertor wall
        reservoirs["divwall_recycling"] = self.res['rcld_rate'] * circ    
        
        # divertor wall reservoir
        if self.namelist["phys_surfaces"]:
            reservoirs["particle_density_stuck_at_div_wall"] = (self.res['N_divwall'] * circ)/(self.surf_divwall*self.divwall_roughness)
            reservoirs["particle_density_retained_at_div_wall"] = (self.res['N_divret'] * circ)/(self.surf_divwall*self.divwall_roughness)
        reservoirs["particles_stuck_at_div_wall"] = self.res['N_divwall'] * circ
        reservoirs["particles_retained_at_div_wall"] = self.res['N_divret'] * circ
        
        # particles pumped away
        reservoirs["particles_pumped"] = self.res['N_out'] * circ
        reservoirs["pumping_rate"] = np.r_[0, np.diff(reservoirs["particles_pumped"]) / np.diff(self.time_out)]
     
        if hasattr(self, "rad"):  # radiation has already been computed
            reservoirs["impurity_radiation"] = grids_utils.vol_int(
                self.rad["tot"], self.rvol_grid, self.pro_grid, self.Raxis_cm,
                rvol_max = self.rvol_lcfs
            )

        if plot:
            # -------------------------------------------------
            # plot time histories for each particle reservoirs:
            if axs is None:
                fig, ax1 = plt.subplots(nrows=4, ncols=3, sharex=True, figsize=(16, 12))
            else:
                ax1 = axs[0]
                
            fig.suptitle('Time traces',fontsize=18)

            ax1[0, 0].plot(self.time_out, reservoirs["plasma_source"], label="Ext. source", color = red, linestyle = 'dotted')
            ax1[0, 0].plot(self.time_out, reservoirs["wall_source"], label="Wall source", color = light_green)
            ax1[0, 0].plot(self.time_out, reservoirs["divertor_source"], label="Div. source", color = green)
            ax1[0, 0].plot(self.time_out, reservoirs["plasma_removal_rate"], label="Removal rate", color = red, linestyle = 'dashed')
            ax1[0, 0].plot(self.time_out, reservoirs["net_plasma_flow"], label="Net sum", color = 'black', linestyle = 'dashed')
            ax1[0, 0].set_title('Plasma particles balance', loc='right', fontsize = 11)
            ax1[0, 0].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax1[0, 0].legend(loc="best", fontsize = 9).set_draggable(True)

            if self.namelist["phys_volumes"]:
                ax1[0, 1].plot(self.time_out, reservoirs["particle_density_in_plasma"],
                               color = blue)
                if ylim:
                    ax1[0, 1].set_ylim(0,np.max(reservoirs["particle_density_in_plasma"])*1.15)
                ax1[0, 1].set_ylabel('[$\mathrm{cm}^{-3}$]')
            else:
                ax1[0, 1].plot(self.time_out, reservoirs["particles_in_plasma"],
                               color = blue)
                if ylim:
                    ax1[0, 1].set_ylim(0,np.max(reservoirs["particles_in_plasma"])*1.15)
                ax1[0, 1].set_ylabel('[#]')
            ax1[0, 1].set_title('Plasma', loc='right', fontsize = 11)

            if "impurity_radiation" in reservoirs:
                ax1[0, 2].plot(self.time_out, reservoirs["impurity_radiation"]/1e6, color = 'red')
                if ylim:
                    ax1[0, 2].set_ylim(0,np.max(reservoirs["impurity_radiation"]/1e6)*1.15)
                ax1[0, 2].set_ylabel('[$\mathrm{MW}$]')
                ax1[0, 2].set_title('Core radiation', loc='right', fontsize = 11) 

            ax1[1, 0].plot(self.time_out, reservoirs["total_flux_mainwall"], label="Tot. flux to main wall", color = blue)
            ax1[1, 0].plot(self.time_out, reservoirs["edge_loss"], label="Radial edge loss", color = light_blue, linestyle = 'dashed')
            ax1[1, 0].plot(self.time_out, reservoirs["limiter_loss"], label="Parallel limiter loss", color = light_blue, linestyle = 'dotted')
            if ylim:
                ax1[1, 0].set_ylim(0,np.max(reservoirs["total_flux_mainwall"])*1.15)
            ax1[1, 0].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax1[1, 0].set_title('Main wall fluxes', loc='right', fontsize = 11)
            ax1[1, 0].legend(loc="best", fontsize = 9).set_draggable(True)

            ax1[1, 1].plot(self.time_out, reservoirs["mainwall_recycling"], color = light_green)
            if ylim:
                ax1[1, 1].set_ylim(0,np.max(reservoirs["mainwall_recycling"])*1.15)
            ax1[1, 1].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax1[1, 1].set_title('Main wall recycling rate', loc='right', fontsize = 11)

            if self.namelist["phys_surfaces"]:
                ax1[1, 2].plot(self.time_out, reservoirs["particle_density_stuck_at_main_wall"], label="Particles stuck", color = light_grey, linestyle = 'dashed')
                ax1[1, 2].plot(self.time_out, reservoirs["particle_density_retained_at_main_wall"],
                    label="Particles retained", color = light_grey)
                ax1[1, 2].set_ylabel('[$\mathrm{cm}^{-2}$]')              
            else:
                ax1[1, 2].plot(self.time_out, reservoirs["particles_stuck_at_main_wall"], label="Particles stuck", color = light_grey, linestyle = 'dashed')
                ax1[1, 2].plot(self.time_out, reservoirs["particles_retained_at_main_wall"],
                    label="Particles retained", color = light_grey)
                ax1[1, 2].set_ylabel('[#]')  
            ax1[1, 2].set_title('Main wall reservoir', loc='right', fontsize = 11)
            ax1[1, 2].legend(loc="best", fontsize = 9).set_draggable(True)

            if self.div_recomb_ratio < 1.0:               
                ax1[2, 0].plot(self.time_out, reservoirs["parallel_loss"]+reservoirs["screened_divertor_backflow"],
                           label="Tot. parallel loss", color = light_blue) 
                ax1[2, 0].plot(self.time_out, (reservoirs["parallel_loss"]+reservoirs["screened_divertor_backflow"])*(1-self.div_recomb_ratio),
                           label="Tot. flux to div. wall", color = blue)
                ax1[2, 0].plot(self.time_out, (reservoirs["parallel_loss"]+reservoirs["screened_divertor_backflow"])*(self.div_recomb_ratio),
                               label="Recomb. flux to div. reservoir", color = green)
            elif self.div_recomb_ratio == 1.0:
                ax1[2, 0].plot(self.time_out, reservoirs["parallel_loss"]+reservoirs["screened_divertor_backflow"],
                           label="Parallel loss", color = blue) 
            if ylim:
                ax1[2, 0].set_ylim(0,np.max(reservoirs["parallel_loss"]+reservoirs["screened_divertor_backflow"])*1.15)
            ax1[2, 0].set_ylabel('[$\mathrm{s}^{-1}$]') 
            ax1[2, 0].set_title('Divertor fluxes', loc='right', fontsize = 11)
            ax1[2, 0].legend(loc="best", fontsize = 9).set_draggable(True)

            ax1[2, 1].plot(self.time_out, reservoirs["divwall_recycling"], color = green)
            if ylim:
                ax1[2, 1].set_ylim(0,np.max(reservoirs["divwall_recycling"])*1.15)
            ax1[2, 1].set_ylabel('[$\mathrm{s}^{-1}$]') 
            ax1[2, 1].set_title('Divertor wall recycling rate', loc='right', fontsize = 11)
            
            if self.namelist["phys_surfaces"]:
                ax1[2, 2].plot(self.time_out, reservoirs["particle_density_stuck_at_div_wall"], label="Particles stuck", color = grey, linestyle = 'dashed')
                ax1[2, 2].plot(self.time_out, reservoirs["particle_density_retained_at_div_wall"],
                    label="Particles retained", color = grey)
                ax1[2, 2].set_ylabel('[$\mathrm{cm}^{-2}$]')              
            else:
                ax1[2, 2].plot(self.time_out, reservoirs["particles_stuck_at_div_wall"], label="Particles stuck", color = grey, linestyle = 'dashed')
                ax1[2, 2].plot(self.time_out, reservoirs["particles_retained_at_div_wall"],
                    label="Particles retained", color = grey)
                ax1[2, 2].set_ylabel('[#]')   
            ax1[2, 2].set_title('Divertor wall reservoir', loc='right', fontsize = 11)
            ax1[2, 2].legend(loc="best", fontsize = 9).set_draggable(True)
            
            if self.div_neut_screen > 0.0:
                ax1[3, 0].plot(self.time_out, reservoirs["divertor_backflow"]+reservoirs["screened_divertor_backflow"],
                           label="Tot. backflow rate", color = green) 
                ax1[3, 0].plot(self.time_out, reservoirs["divertor_backflow"],
                           label="Backflow to core", color = blue) 
                ax1[3, 0].plot(self.time_out, reservoirs["screened_divertor_backflow"],
                           label="Screened backflow", color = light_blue) 
            elif self.div_neut_screen == 0.0:
                ax1[3, 0].plot(self.time_out, reservoirs["divertor_backflow"]+reservoirs["screened_divertor_backflow"],
                           label="Backflow rate", color = green)
            if ylim:
                ax1[3, 0].set_ylim(0,np.max(reservoirs["divertor_backflow"]+reservoirs["screened_divertor_backflow"])*1.15)
            ax1[3, 0].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax1[3, 0].set_title('Divertor backflow rates', loc='right', fontsize = 11)
            ax1[3, 0].legend(loc="best", fontsize = 9).set_draggable(True)

            ax1[3, 1].plot(self.time_out, reservoirs["pump_leakage"],
                           label="Leakage to core", color = light_green)
            if ylim and np.max(reservoirs["pump_leakage"])!=0:
                ax1[3, 1].set_ylim(0,np.max(reservoirs["pump_leakage"])*1.15)
            ax1[3, 1].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax1[3, 1].set_title('Pump leakage rates', loc='right', fontsize = 11)
            ax1[3, 1].legend(loc="best", fontsize = 9).set_draggable(True)
            
            if self.namelist["phys_volumes"]:
                ax1[3, 2].plot(self.time_out, reservoirs["particle_density_in_divertor"],
                           label="Div. reservoir", color = green)
                if self.namelist["pump_chamber"]:
                    ax1[3, 2].plot(self.time_out, reservoirs["particle_density_in_pump"],
                           label="Pump reservoir", color = light_green)
                if ylim:
                    ax1[3, 2].set_ylim(0,np.max(reservoirs["particle_density_in_divertor"])*1.15)
                ax1[3, 2].set_ylabel('[$\mathrm{cm}^{-3}$]')
            else:
                ax1[3, 2].plot(self.time_out, reservoirs["particles_in_divertor"],
                           label="Div. reservoir", color = green)
                if self.namelist["pump_chamber"]:
                    ax1[3, 2].plot(self.time_out, reservoirs["particles_in_pump"],
                           label="Pump reservoir", color = light_green)
                if ylim:
                    ax1[3, 2].set_ylim(0,np.max(reservoirs["particles_in_divertor"])*1.15)
                ax1[3, 2].set_ylabel('[#]')
            ax1[3, 2].set_title('Neutrals reservoirs', loc='right', fontsize = 11)
            ax1[3, 2].legend(loc="best", fontsize = 9).set_draggable(True)

            for ii in [0, 1, 2]:
                ax1[3, ii].set_xlabel('$\mathrm{time}$ [$\mathrm{s}$]')
            ax1[3, 0].set_xlim(self.time_out[[0, -1]])
            
            plt.tight_layout()

            # ----------------------------------------------------------------
            # now plot all particle reservoirs to check particle conservation:
            if axs is None:
                fig, ax2 = plt.subplots(figsize=(9, 5))
            else:
                ax2 = axs[1]
            
            fig.suptitle('Particle conservation',fontsize=14)

            ax2.set_xlabel('$\mathrm{time}$ [$\mathrm{s}$]')

            ax2.plot(self.time_out, all_particles, label="Particles in the plasma", color = blue)
            ax2.plot(self.time_out, reservoirs["particles_in_divertor"], label="Particles in the divertor chamber", color = green)
            if self.namelist["pump_chamber"]:
                ax2.plot(self.time_out, reservoirs["particles_in_pump"], label="Particles in the pump chamber", color = light_green)
            ax2.plot(self.time_out, reservoirs["particles_stuck_at_main_wall"]+reservoirs["particles_retained_at_main_wall"], label="Particles stored at the main wall", color = light_grey)
            ax2.plot(self.time_out, reservoirs["particles_stuck_at_div_wall"]+reservoirs["particles_retained_at_div_wall"], label="Particles stored at the divertor wall", color = grey)
            ax2.plot(self.time_out, reservoirs["particles_pumped"], label="Particles pumped away", color = red, linestyle = 'dashed')
            ax2.plot(self.time_out, reservoirs["integ_source"], label="Integrated source", color = red, linestyle = 'dotted')
            ax2.plot(self.time_out, reservoirs["total"], label="Total particles in the system", color = 'black')

            if (abs((reservoirs["total"][-1] - reservoirs["integ_source"][-1]) / reservoirs["integ_source"][-1]) > 0.1):
                print("Warning: significant error in particle conservation!")

            Ntot = reservoirs["integ_source"][-1]
            dN = np.trapz((reservoirs["total"] / Ntot - reservoirs["integ_source"] / Ntot) ** 2, self.time_out)
            dN /= np.trapz((reservoirs["integ_source"] / Ntot) ** 2, self.time_out)
            print('Particle conservation error %.1f%%' % (np.sqrt(dN) * 100))
        
            ax2.set_xlim(self.time_out[[0, -1]])
            ax2.set_ylim(0, None)
            ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            plt.tight_layout()

        if plot:
            return reservoirs, (ax1, ax2)
        else:

            return reservoirs


    def centrifugal_asym(self, omega, Zeff, plot=False):
         """Estimate impurity poloidal asymmetry effects from centrifugal forces. See notes the
         :py:func:`~aurora.synth_diags.centrifugal_asymmetry` function docstring for details.

         In this function, we use the average Z of the impurity species in the Aurora simulation result, using only
         the last time slice to calculate fractional abundances. The CF lambda factor

         Parameters
         -----------------
         omega : array (nt,nr) or (nr,) [ rad/s ]
              Toroidal rotation on Aurora temporal time_grid and radial rhop_grid (or, equivalently, rvol_grid) grids.
         Zeff : array (nt,nr), (nr,) or float
              Effective plasma charge on Aurora temporal time_grid and radial rhop_grid (or, equivalently, rvol_grid) grids.
              Alternatively, users may give Zeff as a float (taken constant over time and space).
              If impurity is not trace, Zeff should include also the modelled impurity
              Iz the Zeff have a large poloidal asymmetry, it nust be included in the calculation! Not done yet. 
         plot : bool
             If True, plot asymmetry factor :math:`\lambda` vs. radius

         Returns
         ------------
         CF_lambda : array (nr,)
             Asymmetry factor, defined as :math:`\lambda` in the :py:func:`~aurora.synth_diags.centrifugal_asymmetry` function
             docstring.
         """
         # this method requires all charge states to be made available
         nz = self.res['nz']
         try:
             assert nz.shape[1] == self.Z_imp + 1
         except AssertionError:
             raise ValueError(
                 "centrifugal_asym method requires all charge state densities to be availble! Unstage superstages."
             )
         
         #calculate only for the last timeslice
         nz = nz[..., -1]
         fz = nz / np.sum(nz, axis=1)[:, None]
         Z_ave_vec = np.sum(fz * np.arange(self.Z_imp + 1)[None, :], axis=1)
         _, self.Rlfs = grids_utils.get_HFS_LFS(self.geqdsk, rho_pol=self.rhop_grid)
         self.CF_lambda = synth_diags.centrifugal_asymmetry(
             self.rhop_grid,
             self.Rlfs,
             omega,
             Zeff,
             self.A_imp,
             Z_ave_vec,
             self.Te,
             self.Ti,
             main_ion_A=self.main_ion_A,
             plot=plot,
             nz=nz,
             geqdsk=self.geqdsk,
         ).mean(0)

         return self.CF_lambda
