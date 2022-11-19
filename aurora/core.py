"""This module includes the core class to set up simulations with :py:mod:`aurora`. The :py:class:`~aurora.core.aurora_sim` takes as input a namelist dictionary and a g-file dictionary and allows creation of grids, interpolation of atomic rates and other steps before running the forward model.
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
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.constants import e as q_electron, m_p
import pandas as pd
import pickle as pkl
from copy import deepcopy
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from . import interp
from . import atomic
from . import grids_utils
from . import source_utils
from . import transport_utils
from . import plot_tools
from . import synth_diags
from . import adas_files
from . import radiation


class aurora_sim:
    """Class to setup, run, and post-process 1.5D simulations of particle/impurity transport in
    magnetically-confined plasmas.

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

        print('hey')
        if namelist is None:
            # option useful for calls like omfit_classes.OMFITaurora(filename)
            # A call like omfit_classes.OMFITaurora('test', namelist, geqdsk=geqdsk) is also possible
            # to initialize the class as a dictionary.
            return

        # make sure that any changes in namelist will not propagate back to the calling function
        self.namelist = deepcopy(namelist)
        self.geqdsk = geqdsk # if None, minor (rvol_lcfs) and major radius (Raxis_cm) must be in namelist
        self.kin_profs = self.namelist["kin_profs"]
        self.imp = namelist["imp"]

        # import here to avoid issues when building docs or package
        from omfit_classes.utils_math import atomic_element

        # get nuclear charge Z and atomic mass number A
        out = atomic_element(symbol=self.imp)
        spec = list(out.keys())[0]
        self.Z_imp = int(out[spec]["Z"])
        self.A_imp = int(out[spec]["A"])

        self.reload_namelist()

        if 'Raxis_cm' in self.namelist:
            self.Raxis_cm = self.namelist['Raxis_cm'] # cm
        elif self.geqdsk is not None and 'RMAXIS' in self.geqdsk:
            self.Raxis_cm = self.geqdsk["RMAXIS"] * 100.0  # cm

        if self.geqdsk is not None and 'BCENTR' in self.geqdsk:
            self.namelist['Baxis'] = self.geqdsk["BCENTR"]
        if ('prompt_redep_flag' in self.namelist and self.namelist['prompt_redep_flag'])\
           and not hasattr(self, 'Baxis'):
            # need magnetic field to model prompt redeposition
            raise ValueError('Missing magnetic field on axis! Please define this in the namelist')

        # specify which atomic data files should be used -- use defaults unless user specified in namelist
        atom_files = {}
        atom_files["acd"] = self.namelist.get(
            "acd", adas_files.adas_files_dict()[self.imp]["acd"]
        )
        atom_files["scd"] = self.namelist.get(
            "scd", adas_files.adas_files_dict()[self.imp]["scd"]
        )
        if self.namelist["cxr_flag"]:
            atom_files["ccd"] = self.namelist.get(
                "ccd", adas_files.adas_files_dict()[self.imp]["ccd"]
            )

        # now load ionization and recombination rates
        self.atom_data = atomic.get_atom_data(self.imp, files=atom_files)

        # allow for ion superstaging
        self.superstages = self.namelist.get("superstages", [])

        # set up radial and temporal grids
        self.setup_grids()
        
        # set up kinetic profiles and atomic rates
        self.setup_kin_profs_depts()

    def reload_namelist(self, namelist=None):
        """(Re-)load namelist to update scalar variables.
        """
        if namelist is not None:
            self.namelist = namelist
            
        # Extract one by one all the inputs from namelist
        # as attributes of asim, keeping the same name  
        for parameter in self.namelist:
            if isinstance(self.namelist[parameter], dict):
                for sub_parameter in self.namelist[parameter]:
                     setattr(self, sub_parameter, self.namelist[parameter][sub_parameter])
            else:
                setattr(self, parameter, self.namelist[parameter])

        # consistency checks for divertor parameters
        if self.screening_eff < 0.0 or self.screening_eff > 1.0:
            raise ValueError("screening_eff must be between 0.0 and 1.0!") 
        if self.div_recomb_ratio < 0.0 or self.div_recomb_ratio > 1.0:
            raise ValueError("div_recomb_ratio must be between 0.0 and 1.0!") 
        
        # consistency checks for advanced PWI model
        if self.advanced_PWI_flag and not self.phys_surfaces:
            raise ValueError("Implementing the advanced PWI model requires defining the physical wall surface areas!") 
         
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

        # if recycling flag is set to False, avoid any divertor/pump return flows
        # To include divertor/pump return flows but no recycling, set wall_recycling=0
        if not self.recycling_flag:
            self.wall_recycling = -1.0  # no divertor/pump return flows
            
    def save(self, filename):
        """Save state of `aurora_sim` object.
        """
        with open(filename, "wb") as f:
            pkl.dump(self, f)

    def load(self, filename):
        """Load `aurora_sim` object.
        """
        with open(filename, "rb") as f:
            obj = pkl.load(f)
        self.__dict__.update(obj.__dict__)

    def save_dict(self):
        return self.__dict__

    def load_dict(self, aurora_dict):
        self.__dict__.update(aurora_dict)

    def setup_grids(self):
        """Method to set up radial and temporal grids given namelist inputs.
        """
        if self.geqdsk is not None:
            # Get r_V to rho_pol mapping
            rho_pol, _rvol = grids_utils.get_rhopol_rvol_mapping(self.geqdsk)
            rvol_lcfs = interp1d(rho_pol, _rvol)(1.0)
            self.rvol_lcfs = self.namelist["rvol_lcfs"] = np.round(
                rvol_lcfs, 3
            )  # set limit on accuracy
        
        elif 'rvol_lcfs' in self.namelist:
            # separatrix location explicitly given by user
            self.rvol_lcfs = self.namelist['rvol_lcfs']

        else:
            raise ValueError('Could not identify rvol_lcfs. Either provide this in the namelist or provide a geqdsk equilibrium')
            
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

        
        if self.namelist['ELM_model']['ELM_flag'] and self.namelist['ELM_model']['adapt_time_grid']:

            # possibly adapt the namelist 'timing' for ELMs, before creating the time grid
            self.namelist['timing'] = grids_utils.ELM_time_grid(self.namelist['timing'], self.namelist['ELM_model'])

        # define time grid ('timing' must be in namelist)
        self.time_grid, self.save_time = grids_utils.create_time_grid(
            timing=self.namelist["timing"], plot=False
        )
        self.time_out = self.time_grid[self.save_time]

        # create array of 0's of length equal to self.time_grid, with 1's where sawteeth must be triggered
        self.saw_on = np.zeros_like(self.time_grid)
        input_saw_times = self.namelist["saw_model"]["times"]
        self.saw_times = np.array(input_saw_times)[input_saw_times < self.time_grid[-1]]
        if self.namelist["saw_model"]["saw_flag"] and len(self.saw_times) > 0:
            self.saw_on[self.time_grid.searchsorted(self.saw_times)] = 1
            
        # calculate core plasma volume (i.e. volume of region until rvol = rvol_lcfs) and total plasma volume
        ones = np.ones((len(self.time_out),len(self.rvol_grid)))
        self.core_vol = grids_utils.vol_int(
            ones[:,:],self.rvol_grid,self.pro_grid,self.Raxis_cm,rvol_max=self.rvol_lcfs)[0]
        self.plasma_vol = grids_utils.vol_int(
            ones[:,:],self.rvol_grid,self.pro_grid,self.Raxis_cm,rvol_max=None)[0]
                 
    def setup_kin_profs_depts(self):
        """Method to set up Aurora inputs related to the kinetic background from namelist inputs.
        """
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

        # Obtain atomic rates on the computational time and radial grids
        self.Sne_rates, self.Rne_rates = self.get_time_dept_atomic_rates(
            superstages=self.namelist.get("superstages", [])
        )

        Sne0 = self.Sne_rates[:, 0, :]
        # get radial profile of source function
        if len(save_time) == 1:  # if time averaged profiles were used
            Sne0 = Sne0[:, [0]]  # 0th charge state (neutral)

        if self.namelist["source_type"] == "arbitrary_2d_source":
            # interpolate explicit source values on time and rhop grids of simulation
            # NB: explicit_source_vals should be in units of particles/s/cm^3 <-- ionization rate
            srho = self.namelist["explicit_source_rhop"]
            stime = self.namelist["explicit_source_time"]
            source = np.array(self.namelist["explicit_source_vals"]).T

            spl = RectBivariateSpline(srho, stime, source, kx=1, ky=1)
            # extrapolate by the nearest values
            self.source_rad_prof = spl(
                np.clip(self.rhop_grid, min(srho), max(srho)),
                np.clip(self.time_grid, min(stime), max(stime)),
            )
            # Change units to particles/cm^3
            self.src_core = self.source_rad_prof / Sne0
        else:
            # get time history and radial profiles separately
            source_time_history = source_utils.get_source_time_history(
                self.namelist, self.Raxis_cm, self.time_grid
            )  # units of particles/s/cm

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

            # recycling profile manually given by the user -- use it for all recycled impurities,
            #   regardless of the used PWI model
            
            # interpolate it on radial grid
            if "rcl_prof_vals" in self.namelist and "rcl_prof_rhop" in self.namelist:

                # Check that at least part of the recycling prof is within Aurora radial grid
                if np.min(self.namelist["rcl_prof_rhop"]) < np.max(self.rhop_grid):
                    raise ValueError("Input recycling radial grid is too far out!")

                rcl_rad_prof = interp1d(
                    self.namelist["rcl_prof_rhop"],
                    self.namelist["rcl_prof_vals"],
                    fill_value="extrapolate",
                )(self.rhop_grid)
                
                # dummy source profiles for reflected and sputtered neutrals
                #   passed to fortran routine but not used
                rfl_rad_prof = rcl_rad_prof
                spt_rad_prof = np.zeros((len(self.rvol_grid),len(self.background_species)+1))
                for i in range(0,len(self.background_species)+1):
                    spt_rad_prof[:,i] = rcl_rad_prof[:,0]
                    
                # set same profile for all times    
                self.rcl_rad_prof = np.broadcast_to(
                    rcl_rad_prof, (rcl_rad_prof.shape[0], len(self.time_grid))
                )

               self.rfl_rad_prof = np.broadcast_to(
                    rfl_rad_prof, (rfl_rad_prof.shape[0], len(self.time_grid))
                )
                self.spt_rad_prof = np.zeros(
                    (len(self.rvol_grid),len(self.background_species)+1,len(self.time_grid))
                )

                for i in range(0,len(self.background_species)+1):
                    for j in range(0,len(self.time_grid)):
                        self.spt_rad_prof[:,i,j] = spt_rad_prof[:,i]
                
            else:
                
                nml_rcl_prof = {
                    key: self.namelist[key]
                    for key in [
                            "imp_source_energy_eV",
                            "rvol_lcfs",
                            "source_cm_out_lcfs",
                            "imp",
                            "prompt_redep_flag",
                            "Baxis",
                            "main_ion_A",
                    ]
                }
                nml_rcl_prof["source_width_in"] = 0
                nml_rcl_prof["source_width_out"] = 0

                # set the energy of the recycled neutrals
                nml_rcl_prof["imp_source_energy_eV"] = self.namelist["imp_recycling_energy_eV"]

                # set start of the recycling source at the wall boundary
                nml_rcl_prof["source_cm_out_lcfs"] = self.namelist["bound_sep"]

                # NB: we assume here that the 0th time is a good representation of how recycling is radially distributed
                rcl_rad_prof = source_utils.get_radial_source(
                    nml_rcl_prof,  # namelist specifically to obtain exp decay from wall
                    self.rvol_grid,
                    self.pro_grid,
                    Sne0,
                    self._Ti,
                )

                # dummy source profiles for reflected and sputtered neutrals
                #   passed to fortran routine but not used
                rfl_rad_prof = rcl_rad_prof
                spt_rad_prof = np.zeros((len(self.rvol_grid),len(self.background_species)+1))
                for i in range(0,len(self.background_species)+1):
                    spt_rad_prof[:,i] = rcl_rad_prof[:,0]

                # set same profile for all times    
                self.rcl_rad_prof = np.broadcast_to(
                    rcl_rad_prof, (rcl_rad_prof.shape[0], len(self.time_grid))
                )
                self.rfl_rad_prof = np.broadcast_to(
                    rfl_rad_prof, (rfl_rad_prof.shape[0], len(self.time_grid))
                )
                self.spt_rad_prof = np.zeros((len(self.rvol_grid),len(self.background_species)+1,len(self.time_grid)))
                for i in range(0,len(self.background_species)+1):
                    for j in range(0,len(self.time_grid)):
                        self.spt_rad_prof[:,i,j] = spt_rad_prof[:,i]

        else:
            # dummy profile -- recycling is turned off
            self.rcl_rad_prof = np.zeros((len(self.rhop_grid), len(self.time_grid)))
            self.rfl_rad_prof = self.rcl_rad_prof
            self.spt_rad_prof = np.zeros((len(self.rvol_grid),len(self.background_species)+1,len(self.time_grid)))
            for i in range(0,len(self.background_species)+1):
                self.spt_rad_prof[:,i,:] = self.rcl_rad_prof


    def interp_kin_prof(self, prof):
        """ Interpolate the given kinetic profile on the radial and temporal grids [units of s].
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
                #print(
                #    f"Namelist did not provide a {prof} decay length for the SOL. Setting it to 1cm."
                #)
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
        """Get kinetic profiles on radial and time grids.
        """
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

        # get neutral background ion density
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

    def get_time_dept_atomic_rates(self, superstages=[]):
        """Obtain time-dependent ionization and recombination rates for a simulation run.
        If kinetic profiles are given as time-independent, atomic rates for each time slice
        will be set to be the same.

        Parameters
        ----------
        superstages : list or 1D array
            Indices of charge states that should be kept as superstages.
            The default is to have this as an empty list, in which case all charge states are kept.

        Returns
        -------
        Sne_rates : array (space, nZ(-super), time)
            Effective ionization rates [s]. If superstages were indicated, these are the rates of superstages.
        Rne_rates : array (space, nZ(-super), time)
            Effective recombination rates [s]. If superstages were indicated, these are the rates of superstages.
        """

        # get electron impact ionization and radiative recombination rates in units of [s^-1]
        _, Sne, Rne, cxne = atomic.get_cs_balance_terms(
            self.atom_data,
            ne_cm3=self._ne,
            Te_eV=self._Te,
            Ti_eV=self._Ti,
            include_cx=self.namelist["cxr_flag"],
        )

        # cache radiative & dielectronic recomb:
        self.alpha_RDR_rates = Rne
        
        if self.namelist["cxr_flag"]:
            # Get an effective recombination rate by summing radiative & CX recombination rates
            self.alpha_CX_rates = cxne * (self._n0 / self._ne)[:, None]
            Rne = Rne + self.alpha_CX_rates  #inplace addition would change also self.alpha_RDR_rates

        if self.namelist["nbi_cxr_flag"]:
            # include charge exchange between NBI neutrals and impurities
            self.nbi_cxr = interp1d(
                self.namelist["nbi_cxr"]["rhop"],
                self.namelist["nbi_cxr"]["vals"],
                axis=0,
                bounds_error=False,
                fill_value=0.0,
            )(self.rhop_grid)

            Rne += self.nbi_cxr.T[None, :, :]

        if len(superstages):
            self.superstages, Rne, Sne, self.fz_upstage = atomic.superstage_rates(
                Rne, Sne, superstages, save_time=self.save_time
            )

        # Sne and Rne for the Z+1 stage must be zero for the forward model.
        # Use Fortran-ordered arrays for speed in forward modeling (both Fortran and Julia)
        Sne_rates = np.zeros(
            (Sne.shape[2], Sne.shape[1] + 1, self.time_grid.size), order="F"
        )
        Sne_rates[:, :-1] = Sne.T

        Rne_rates = np.zeros(
            (Rne.shape[2], Rne.shape[1] + 1, self.time_grid.size), order="F"
        )
        Rne_rates[:, :-1] = Rne.T

        return Sne_rates, Rne_rates

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
        #   high collision frequency is assumed in the SOL -->
        #   impurities entrained into the main ion flow
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
        dv[ids:idl] = (
            vpf
            * np.sqrt(3.0 * Ti.T[ids:idl] + self._Te.T[ids:idl])
            / self.namelist["clen_divertor"]
        )

        # limiter shadow
        dv[idl:] = (
            vpf
            * np.sqrt(3.0 * Ti.T[idl:] + self._Te.T[idl:])
            / self.namelist["clen_limiter"]
        )

        dv, _ = np.broadcast_arrays(dv, self.time_grid[None])    
        
        
        # if a peak mach number during the ELM is desired, then
        #   rescale the rate accordingly in the open SOL
        
        if self.namelist['ELM_model']['ELM_flag'] and self.namelist["SOL_mach_ELM"] > self.namelist["SOL_mach"]:
            # time-dependent ratio wrt the intra-ELM value
            
            rescale_factor = transport_utils.ELM_cycle_SOL_mach(self.namelist["SOL_mach"],
                                                                self.namelist["SOL_mach_ELM"],
                                                                self.time_grid,
                                                                self.namelist["ELM_model"],
                                                                self.namelist['timing'])/self.namelist["SOL_mach"]
            dv_rescaled = np.zeros_like(dv)
            # rescale
            for i in range(0,ids):
                dv_rescaled[i,:] = dv[i,:]
            for i in range(ids,idl):
                dv_rescaled[i,:] = dv[i,:]*rescale_factor
            for i in range(idl,len(dv)):  
                dv_rescaled[i,:] = dv[i,:]
            
            return np.asfortranarray(dv_rescaled)
        
        else:
            
            return np.asfortranarray(dv)

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
        plot : bool
            If True, plot asymmetry factor :math:`\lambda` vs. radius

        Returns
        ------------
        CF_lambda : array (nr,)
            Asymmetry factor, defined as :math:`\lambda` in the :py:func:`~aurora.synth_diags.centrifugal_asymmetry` function
            docstring.
        """
        # this method requires all charge states to be made available
        try:
            assert self.res[0].shape[1] == self.Z_imp + 1
        except AssertionError:
            raise ValueError(
                "centrifugal_asym method requires all charge state densities to be availble! Unstage superstages."
            )

        fz = self.res[0][..., -1] / np.sum(self.res[0][..., -1], axis=1)[:, None]
        Z_ave_vec = np.sum(fz * np.arange(self.Z_imp + 1)[None, :], axis=1)

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
            nz=self.res[0][..., -1],
            geqdsk=self.geqdsk,
        )

        return self.CF_lambda
      

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
                    Dzf[:, :, i] = D_z[:, :, superstages[i] : superstages[i + 1]].mean(
                        2
                    )
                    Vzf[:, :, i] = V_z[:, :, superstages[i] : superstages[i + 1]].mean(
                        2
                    )

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
        init = {},
        unstage=True,
        alg_opt=1,
        evolneut=False,
        use_julia=False
    ):
        """Run a simulation using the provided diffusion and convection profiles as a function of space, time
        and potentially also ionization state. 
        Users can give an initial state of each ion charge state and reservoir content as an input.
        
        While the output impurity density in the plasma (function of radial location, charge state and time)
        is given in [:math:`cm^-3`], all the other outputs are given per unit of toroidal length, i.e. the absolute number of
        particles in reservoirs in [cm^-1] and fluxes from/to/between the reservoirs in [:math:`cm^-1 s^-1`], so that
        effective numbers are achieved multiplying these by :math:`circ = 2 * np.pi * self.Raxis_cm`.

        To visualize results, see the py:meth:`~aurora.core.plot_aurora_res` method.

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
        init : dict
            Dictionary describing the contents of particle reservoirs (main chamber plasma, divertor, wall,
            etc.). If not given, simulations begin with no pre-existing particles (default).
            See the notes below for details.
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
        use_julia : bool, optional
            If True, run the Julia pre-compiled version of the code. Run the julia makefile option to set
            this up. Default is False (still under development).

        Returns
        -------
        dict : 
            Dictionary containing simulation results. Depending on the chosen input options, different
            fields will be provided. Impurity charge state distributions in the plasma are always provided
            in the `nz` key, which gives an array of dimensions (nr,nZ,nt). See the notes below for details
            of other available fields.

        Notes: output dictionary contents
        ---------------------------------
        The following fields are available in the output dictionary, depending on input options:

        nz : array, (nr,nZ,nt)
            Always returned. Charge state densities over the space and time grids [:math:`cm^{-3}`]
            If a number of superstages are indicated in the input, only charge state densities for
            these are returned.
        N_mainwall : array (nt,)
            Number of particles at the main wall reservoir over time [:math:`cm^{-1}`]
        N_divwall : array (nt,)
            Number of particles at the divertor wall reservoir over time [:math:`cm^{-1}`].
            Only returned if :param:`div_recomb_ratio` < 1.0.
        N_div : array (nt,)
            Number of particles in the divertor reservoir over time [:math:`cm^{-1}`]
        N_pump : array (nt,)
            Number of particles in the pump reservoir over time [:math:`cm^{-1}`].
            Only returned if :param:`pump_chamber`=True, i.e. if a second reservoirs for neutral particles 
            before the pump is defined.
        N_out : array (nt,)
            Number of particles permanently removed through the pump over time [:math:`cm^{-1}`]
        N_mainret : array (nt,)
             Number of particles temporarily held in the main wall reservoir [:math:`cm^{-1}`]
        N_divret : array (nt,)
             Number of particles temporarily held in the divertor wall reservoir [:math:`cm^{-1}`].
             Only returned if :param:`div_recomb_ratio` < 1.0, so that ions are allowed to interact 
             with a divertor wall.
        N_tsu : array (nt,)
             Edge particle loss [:math:`cm^{-1} s^{-1}`]
        N_dsu : array (nt,)
             Parallel particle loss [:math:`cm^{-1} s^{-1}`]
        N_dsul : array (nt,)
             Parallel particle loss at the limiter [:math:`cm^{-1} s^{-1}`]
        rcld_rate : array (nt,)
             Total recycled flux from the divertor wall reservoir [:math:`cm^{-1} s^{-1}`].
             Only returned if :param:`div_recomb_ratio` < 1.0, so that ions are allowed to interact 
             with a divertor wall.
        rcld_refl_rate : array (nt,)
             Reflected flux from the divertor wall [:math:`cm^{-1} s^{-1}`].
             Only returned if :param:`div_recomb_ratio` < 1.0 AND :param:`advanced_PWI_flag` = True.
        rcld_recl_rate : array (nt,)
             Promptly recycled flux from the divertor wall [:math:`cm^{-1} s^{-1}`].
             Only returned if :param:`div_recomb_ratio` < 1.0 AND :param:`advanced_PWI_flag` = True.
        rcld_impl_rate : array (nt,)
             Implanted flux into the divertor wall reservoir [:math:`cm^{-1} s^{-1}`]
             Only returned if :param:`div_recomb_ratio` < 1.0 AND :param:`advanced_PWI_flag` = True.
        rcld_sput_rate : array (1+background_species,nt,)
             Sputtered fluxes from the divertor wall reservoir, one for each sputtering species 
             [:math:`cm^{-1} s^{-1}`]. Only returned if :param:`div_recomb_ratio` < 1.0 AND
             :param:`advanced_PWI_flag` = True.
        rclb_rate : array (nt,)
             Backflow from the divertor neutrals reservoir reaching the plasma [:math:`cm^{-1} s^{-1}`]
        rcls_rate : array (nt,)
             Screened backflow from the divertor neutrals reservoir [:math:`cm^{-1} s^{-1}`].
             Only returned if :param:`screening_eff` > 0.0.
        rclp_rate : array (nt,)
             Leakage from the pump neutrals reservoir [:math:`cm^{-1} s^{-1}`].
             Only returned if :param:`pump_chamber` = True.
        rclw_rate : array (nt,)
             Total recycled flux from the main wall reservoir [:math:`cm^{-1} s^{-1}`]
        rclw_refl_rate : array (nt,)
             Reflected flux from the main wall [:math:`cm^{-1} s^{-1}`].
             Only returned if :param:`advanced_PWI_flag` = True.
        rclw_recl_rate : array (nt,)
             Promptly recycled flux from the main wall [:math:`cm^{-1} s^{-1}`].
             Only returned if :param:`advanced_PWI_flag` = True.
        rclw_impl_rate : array (nt,)
             Implanted flux into the main wall reservoir [:math:`cm^{-1} s^{-1}`]
             Only returned if :param:`advanced_PWI_flag` = True.
        rclw_sput_rate : array (1+background_species,nt,)
             Sputtered fluxes from the main wall reservoir, one for each sputtering species 
             [:math:`cm^{-1} s^{-1}`]. Only returned if :param:`advanced_PWI_flag` = True.

        Notes: initial conditions
        -------------------------
        Simulations may be run starting from initial conditions that include particles in the main
        chamber plasma or in the edge reservoirs. If the :param:`init` dictionary is not provided 
        (or left as an empty dictionary), all reservoirs are initially empty. The following keys
        of the :param:`init` dictionary can be provided for specific initial conditions: 

        init['nz']: array, shape of (space, nZ)
            Impurity charge states at the initial time of the simulation. If left to None, this is
            internally set to an array of 0's.
        init['ndiv']: float
            Neutral impurity content in the divertor reservoir at the initial time of the simulation.
            If left to None, this is internally set to 0.
            If namelist["phys_volumes"] = False (default), this will be interpreted as absolute
            number of impurity neutrals in the divertor reservoir. If True, this will be interpreted
            as neutral impurity density (in cm^-3).
        init['npump']: float
            Neutral impurity content in the pump reservoir at the initial time of the simulation,
            effective only if namelist["pump_chamber"] = True (otherwise the pump reservoir is not used).
            If left to None, this is internally set to 0.
            If namelist["phys_volumes"] = False (default), this will be interpreted as absolute
            number of impurity neutrals in the pump reservoir. If True, this will be interpreted
            as neutral impurity density (in cm^-3).
        init['nmainwall']: float
            Implanted impurity content in the main wall reservoir at the initial time of the simulation.
            If left to None, this is internally set to 0.
            If namelist["phys_surfaces"] = False (default), this will be interpreted as absolute
            number of impurity neutrals retained at the main wall. If True, this will be interpreted
            as surface implantation density (in cm^-2).
        init['ndivwall']: float
            Implanted impurity content in the divertor wall reservoir at the initial time of the simulation,
            effective only if namelist["div_recomb_ratio"] < 1.0 (otherwise the divertor wall is not used).    
            If left to None, this is internally set to 0.
            If namelist["phys_surfaces"] = False (default), this will be interpreted as absolute
            number of impurity neutrals retained at the divertor wall. If True, this will be interpreted
            as surface implantation density (in cm^-2).
        """
        D_z, V_z = np.asarray(D_z), np.asarray(V_z)

        # D_z and V_z must have the same shape
        assert np.shape(D_z) == np.shape(V_z)

        if (times_DV is None) and (D_z.ndim > 1 or V_z.ndim > 1):
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

        if 'nz' not in init or init['nz'] is None:
            # default: start in a state with no impurity ions
            nz_init = np.zeros((len(self.rvol_grid), num_cs))
            
        # factor to account for cylindrical geometry:
        circ = 2 * np.pi * self.Raxis_cm  # cm
        
        if 'ndiv' not in init or init['ndiv'] is None:
            # default: start in a state with empty divertor neutrals reservoir
            ndiv_init = 0
        else:
            # start in a state with not empty divertor neutrals reservoir; convert to cm^-1
            if self.namelist["phys_volumes"] == True:
                # the input init['ndiv'] is interpreted as a particle density in cm^-3
                ndiv_init = (init['ndiv'] * self.vol_div) / circ # ([cm^-3]*[cm^3])/[cm]
            else:
                # the input ndiv_init is interpreted as absolute number of particles
                ndiv_init = init['ndiv'] / circ        
                
        if 'npump' not in init or init['npump'] is None or not self.namelist["pump_chamber"]:
            # default: start with empty pump neutrals reservoir (or pump reservoir not used at all)
            npump_init = 0 
        else:
            # start in a state with not empty pump neutrals reservoir
            # convert to cm^-1 before passing it to fortran_run
            if self.namelist["phys_volumes"]:
                # the input npump_init is interpreted as a particle density in cm^-3
                npump_init = (init['npump'] * self.vol_pump) / circ # ([cm^-3]*[cm^3])/[cm]
            else:
                # the input npump_init is interpreted as absolute number of particles
                npump_init = init['npump'] / circ      
        
        if 'nmainwall' not in init or init['nmainwall'] is None:
            # default: start in a state with empty main wall reservoir
            nmainwall_init = 0
        else:
            # start in a state with not empty main wall reservoir; convert to cm^-1
            if self.namelist["phys_surfaces"] == True:
                # the input init['nmainwall'] is interpreted as a particle density in cm^-2
                # ([cm^-2]*[cm^2])/[cm]
                nmainwall_init = (init['nmainwall'] * self.surf_mainwall * self.mainwall_roughness) / circ
            else:
                # the input init['nmainwall'] is interpreted as absolute number of particles
                nmainwall_init = init['nmainwall'] / circ     
        
        if 'ndivwall' not in init or init['ndivwall'] is None or self.namelist["div_recomb_ratio"] == 1.0:
            # default: start with empty divertor wall reservoir (or divertor wall reservoir not used at all)
            ndivwall_init = 0  
        else:
            # start with not empty divertor wall reservoir; convert to cm^-1
            if self.namelist["phys_surfaces"]:
                # the input ndivwall_init is interpreted as a particle density in cm^-2
                # ([cm^-2]*[cm^2])/[cm]
                ndivwall_init = (init['ndivwall'] * self.surf_divwall * self.divwall_roughness) / circ
            else:
                # the input ndivwall_init is interpreted as absolute number of particles
                ndivwall_init = init['ndivwall'] / circ         

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
            
            if self.div_recomb_ratio < 1.0 or self.pump_chamber or self.screening_eff > 0.0 or self.advanced_PWI_flag:
                raise ValueError("Advanced recycling/pumping/PWI model not yet implemented in Julia!")
 
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
                self.rmix,
                self.decay_length_boundary,
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
            # run Fortran version of the code
            # import here to avoid import when building documentation or package (negligible slow down)
            from ._aurora import run as fortran_run
            _res = fortran_run(
                nt,  # number of times at which simulation outputs results
                times_DV, # Times at which transport coeffs change [s]
                D_z, # diffusion coefficient on radial and time grids for each charge state [cm^2/s]
                V_z, # radial pinch veocity on radial and time grids for each charge state [cm^2/s]
                self.par_loss_rate, # parallel loss rate values in the SOL on the time grid
                self.src_core, # radial source profile of externally injected neutrals in the plasma on the time grid
                self.rcl_rad_prof, # radial source profile of promptly recycled neutrals in the plasma on the time grid
                self.rfl_rad_prof, # radial source profile of reflected neutrals in the plasma on the time grid
                self.spt_rad_prof, # radial source profile of sputtered neutrals in the plasma on the time grid
                self.energetic_recycled_neutrals, # logic key for setting energetic reflected/sputtered neutrals
                self.Sne_rates, # ionization rates in the plasma
                self.Rne_rates, # recombination rates in the plasma
                self.Raxis_cm, # major radius at the magnetic axis [cm]
                self.rvol_grid, # radial grid values of rvol
                self.pro_grid,
                self.qpr_grid,
                self.rmix,
                self.decay_length_boundary,
                self.time_grid, # time grid values
                self.saw_on, # logic key for sawteeth model
                self.save_time,
                self.crash_width,  # dsaw width [cm]
                self.wall_recycling, # recycling key
                self.screening_eff, # screening coefficient
                self.div_recomb_ratio, # divertor recombination coefficient
                self.tau_div_SOL_ms * 1e-3, # divertor retention time [s]
                self.tau_pump_ms * 1e-3, # pumping time in the adimensional pumping model [s]
                self.tau_rcl_ret_ms * 1e-3, # wall release time in the simple PWI model [s]
                self.S_pump, # pumping speed in the dimensional pumping model [cm^3/s]
                self.vol_div, # volume of the divertor neutrals reservoir [cm^3]
                self.L_divpump, # conductance between divertor and pump neutrals reservoirs [cm^3/s]
                self.vol_pump, # volume of the pump neutrals reservoirs[cm^3]
                self.L_leak, # leakage conductance from pump neutrals reservoir towards plasma [cm^3/s]
                self.surf_mainwall_eff, # effective main wall surface area [cm^2]
                self.surf_divwall_eff, # effective divertor wall surface area [cm^2]
                self.advanced_PWI_flag, # logic key for PWI model
                self.Z_main_wall, # atomic number of the main wall material
                self.Z_div_wall, # atomic number of the divertor wall material
                self.rn_main_wall, # refl. coeffs for the simulated impurity at the main wall on the time grid
                self.rn_div_wall, # refl. coeffs for the simulated impurity at the divertor wall on the time grid
                self.fluxes_main_wall_background, # fluxes for each bckg species onto main wall on the time grid [s^-1]
                self.fluxes_div_wall_background, # fluxes for each bckg species onto the divertor wall on the time grid [s^-1]
                self.y_main_wall, # sputtering yields from simulated impurity + bckg species from the main wall on the time grid
                self.y_div_wall, # sputtering yields from simulated impurity + bckg species from the divertor wall on the time grid
                self.implantation_depth_main_wall, # impurity implantation depth in the main wall [A]
                self.implantation_depth_div_wall, # impurity implantation depth in the divertor wall [A]
                self.n_main_wall_sat, # saturation value of the impurity implantation density into the main wall [m^-2]
                self.n_div_wall_sat, # saturation value of the impurity implantation density into the divertor wall [m^-2]
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
        if use_julia:
            # advanced recycling/pumping/PWI model not implemented yet in Julia
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
            ) = _res # less elements because of no PWI model implementation
        
        else:
            # advanced recycling/pumping/PWI model fully implemented in Fortran
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
            
    def plot_aurora_res(self,
                        plot_radiation=False,
                        plot_average = False,
                        interval = 0.01,
                        rad_coord = 'rvol',
                        plot_PWI = False):
        '''Plot the results of an Aurora simulation of particle transport, possibly calculating 
        
        plot_radiation : bool, optional
            If True, plot line radiation for each charge state using a convenient slides over time and
            the total radiation time traces.
        plot_average : bool, optional
            If True, plot density for each charge state averaged over ELM cycles using a convenient slide
            over time and check particle conservation in each particle reservoir averaged over ELM cycles.
        interval : float, optional
            Duration of time cycles to plot if plot_average is True, in s
        rad_coord : string, optional
            Radial coordinate shown in the plot. Options: 'rvol' (default) or 'rho_pol'
        plot_PWI : bool, optional
            If True, plot time traces related to the plasma-wall interaction model   
        '''
        
        if rad_coord == 'rvol':
            x = self.rvol_grid
            xlabel = r"$r_V$ [cm]"
            x_line=self.rvol_lcfs
        elif rad_coord == 'rho_pol':
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
            zlabel=f'$n_{{{self.imp}}}$ [cm$^{{-3}}$]',
            plot_title = f'{self.imp} density profiles',
            labels=[str(i) for i in np.arange(0, self.res[0].shape[1])],
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
                zlabel="[$MW/m^3$]",
                plot_title = f'{self.imp} line radiation',
                labels=[str(i) for i in np.arange(0, self.res['nz'].shape[1])],
                plot_sum=True,
                x_line=x_line,
                zlim = True,
            )

        # Plot reservoirs and particle conservation
        _ = self.reservoirs_time_traces(plot=True)

        if self.namelist['ELM_model']['ELM_flag'] and plot_average:

            time_average, data_average_profiles = plot_tools.time_average_profiles(self.namelist['timing'], self.time_out, self.res['nz'], interval = interval)

            # plot charge state density distributions over radius and time, averaged over cycles
            plot_tools.slider_plot(
                x,
                time_average
                data_average_profiles.transpose(1, 0, 2),
                xlabel=xlabel,
                ylabel="time [s]",
                zlabel=f'$n_{{{self.imp}}}$ [cm$^{{-3}}$]',
                plot_title = f'{self.imp} density profiles (averaged over ELM cycles)',
                labels=[str(i) for i in np.arange(0, self.res['nz'].shape[1])],
                plot_sum=True,
                x_line=x_line,
                zlim = True,
            )

            # Plot reservoirs and particle conservation, averaged over cycles
            _ = self.reservoirs_average_time_traces(interval = interval,plot = True)

        # Plot PWI model
        if plot_PWI and self.advanced_PWI_flag:
            _ = self.PWI_time_traces(interval = interval, plot = True)


    def run_aurora_steady(
        self,
        D_z,
        V_z,
        init = {},
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
        rad_coord = 'rvol',
    ):
        """Run an AURORA simulation until reaching steady state profiles. This method calls :py:meth:`~aurora.core.run_aurora`
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
        init: dict
            Dictionary describing the initial state from which the simulation should begin.
            See the description in :py:meth:`~aurora.core.run_aurora`.
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
        rad_coord : string, optional
            Radial coordinate shown in the plot. Options: 'rvol' (default) or 'rho_pol'
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

        # set constant timesource
        self.namelist["source_type"] = "const"
        self.namelist["source_rate"] = 1.0

        # build timing dictionary
        self.namelist["timing"] = {
            "dt_start": [dt, dt],
            "dt_increase": [dt_increase, 1.0],
            "steps_per_cycle": [1, 1],
            "times": [0.0, max_sim_time]
        }

        # prepare radial and temporal grid
        self.setup_grids()

        # update kinetic profile dependencies to get everything to the right shape
        self.setup_kin_profs_depts()

        times_DV = None
        if D_z.ndim == 2:
            # make sure that transport coefficients were given as a function of space and nZ, not time!
            assert D_z.shape[0] == self.rhop_grid and D_z.shape[1] == self.Z_imp + 1
            assert V_z.shape[0] == self.rhop_grid and V_z.shape[1] == self.Z_imp + 1

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
        nz_all = None if 'nz' not in init or else init['nz']

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

            nz_new = self.run_aurora(
                D_z,
                V_z,
                times_DV,
                init={'nz': nz_init},
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
        self.time_out = time_grid[
            :sim_steps
        ]  # identical because steps_per_cycle is fixed to 1
        self.save_time = save_time[:sim_steps]

        if plot:
            
            if rad_coord == 'rvol':
                x = self.rvol_grid
                xlabel = r"$r_V$ [cm]"
                x_line=self.rvol_lcfs
            elif rad_coord == 'rho_pol':
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
            self.src_core.T, self.rvol_grid, self.pro_grid, self.Raxis_cm,
            rvol_max = self.rvol_lcfs
        )
        self.tau_imp = (
            var_volint / source_time_history[-2]
        )  # avoid last time point because source may be 0 there

        return nz_new[:, :, -1]

    def calc_Zeff(self):
        """Compute Zeff from each charge state density, using the result of an AURORA simulation.
        The total Zeff change over time and space due to the simulated impurity can be simply obtained by summing
        over charge states.

        Results are stored as an attribute of the simulation object instance.
        """
        # This method requires that a simulation has already been run:
        assert hasattr(self, "res")

        # this method requires all charge states to be made available
        try:
            assert self.res['nz'].shape[1] == self.Z_imp + 1
        except AssertionError:
            raise ValueError(
                "calc_Zeff method requires all charge state densities to be availble! Unstage superstages."
            )

        # Compute the variation of Zeff from these charge states
        Zmax = self.res['nz'].shape[1] - 1
        Z = np.arange(Zmax + 1)
        self.delta_Zeff = self.res['nz'] * (Z * (Z - 1))[None, :, None]  # for each charge state
        self.delta_Zeff /= self.ne.T[:, None, :]
        
    def calc_taumain(self):
        '''Calculate the impurity confinement time.

        Returns
        -------
        tau_main : array????
            Impurity confinement time in units of :math:`s`.
        '''
        nz = self.res['nz'].transpose(2, 1, 0)  # time,nZ,space
        
        # factor to account for cylindrical geometry:
        circ = 2 * np.pi * self.Raxis_cm  # cm

        # calculate total impurity density (summed over charge states)
        total_impurity_density = np.nansum(nz, axis=1)  # time, space

        # compute total number of particles in the plasma
        N_main = grids_utils.vol_int(
            total_impurity_density, self.rvol_grid, self.pro_grid, self.Raxis_cm,
            rvol_max = None
        )
        
        # compute fluxes leaving the plasma
        edge_loss = self.res['N_tsu'] * circ
        limiter_loss = self.res['N_dsul'] * circ
        parallel_loss = self.res['N_dsu'] * circ
        flux_out = edge_loss + limiter_loss + parallel_loss
        
        # calculate the impurity confinement time in the plasma
        tau_main = np.divide(N_main, flux_out, out=np.zeros_like(N_main), where=flux_out!=0)
        
        return tau_main
    
    def calc_compression(self):
        '''Calculate the divertor impurity compression, i.e. the ratio of particles in the divertor
        compared to the core plasma.
        Note that calculating divertor compression requires the physical volumes of reservoirs, i.e.
        `phys_volumes` must be provided within the namelist.

        Returns
        -------
        n_core ???

        n_div  ???

        compression ???

        '''
        nz = self.res['nz'].transpose(2, 1, 0)  # time,nZ,space
        
        # factor to account for cylindrical geometry:
        circ = 2 * np.pi * self.Raxis_cm  # cm
        
        if not self.namelist["phys_volumes"]:    
            raise ValueError("Calculating divertor compression requires the physical volumes of reservoirs.") 
        
        n_core = grids_utils.vol_int(
            nz[:,-1,:],self.rvol_grid,self.pro_grid,self.Raxis_cm,rvol_max=self.rvol_lcfs)/self.core_vol
        n_div = (self.res['N_div']*circ)/self.vol_div
        
        compression = np.divide(n_div, n_core, out=np.zeros_like(n_div), where=n_core!=0)

        return n_core, n_div, compression

    def plot_resolutions(self, plot_radial_grid = True, plot_time_grid = True):
        """Convenience function to show time and spatial resolution in Aurora simulation setup.
        """
        # display radial resolution
        _ = grids_utils.create_radial_grid(self.namelist, plot=plot_radial_grid)

        # display time resolution
        _ = grids_utils.create_time_grid(timing=self.namelist["timing"], plot=plot_time_grid)
        
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

    def reservoirs_time_traces(self, plot=True, axs=None, plot_resolutions=False):
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

        nz = self.res['nz'].transpose(2, 1, 0)  # time,nZ,space

        # factor to account for cylindrical geometry:
        circ = 2 * np.pi * self.Raxis_cm  # cm

        # collect all the relevant quantities for particle conservation
        reservoirs = {}

        # calculate total impurity density (summed over charge states)
        total_impurity_density = np.nansum(nz, axis=1)  # time, space

        # Compute total number of particles for particle conservation checks:
        all_particles = grids_utils.vol_int(
            total_impurity_density, self.rvol_grid, self.pro_grid, self.Raxis_cm,
            rvol_max = None
        )
        reservoirs["total"] = all_particles + (self.res['N_mainwall'] + self.res['N_divwall'] + self.res['N_div'] + self.res['N_pump'] + self.res['N_out'] + self.res['N_mainret'] + self.res['N_divret']) * circ

        # main fluxes
        reservoirs["source"] = self.total_source*circ
        reservoirs["wall_source"] = rclw_rate * circ
        reservoirs["divertor_source"] = rclb_rate * circ + rclp_rate * circ
        reservoirs["plasma_removal_rate"] = - self.res['N_dsu'] * circ - self.res['N_tsu'] * circ - self.res['N_dsul'] * circ       
        reservoirs["net_plasma_flow"] = reservoirs["source"] + reservoirs["wall_source"] + reservoirs["divertor_source"] + reservoirs["plasma_removal_rate"]

        # integrated source over time
        reservoirs["integ_source"] = cumtrapz(reservoirs["source"], self.time_out, initial=0) + reservoirs["total"][0] 
        
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

            ax1[0, 0].plot(self.time_out, reservoirs["source"], label="Ext. source", color = red, linestyle = 'dotted')
            ax1[0, 0].plot(self.time_out, reservoirs["wall_source"], label="Wall source", color = light_green)
            ax1[0, 0].plot(self.time_out, reservoirs["divertor_source"], label="Div. source", color = green)
            ax1[0, 0].plot(self.time_out, reservoirs["plasma_removal_rate"], label="Removal rate", color = red, linestyle = 'dashed')
            ax1[0, 0].plot(self.time_out, reservoirs["net_plasma_flow"], label="Net sum", color = 'black', linestyle = 'dashed')
            ax1[0, 0].set_title('Plasma particles balance', loc='right', fontsize = 11)
            ax1[0, 0].set_ylabel('[$s^{-1}$]')
            ax1[0, 0].legend(loc="best", fontsize = 9).set_draggable(True)

            if self.namelist["phys_volumes"]:
                ax1[0, 1].plot(self.time_out, reservoirs["particle_density_in_plasma"],
                               color = blue)
                ax1[0, 1].set_ylabel('[$cm^{-3}$]')
            else:
                ax1[0, 1].plot(self.time_out, reservoirs["particles_in_plasma"],
                               color = blue)
                ax1[0, 1].set_ylabel('[#]')
            ax1[0, 1].set_title('Plasma', loc='right', fontsize = 11)

            if "impurity_radiation" in reservoirs:
                ax1[0, 2].plot(self.time_out, reservoirs["impurity_radiation"]/1e6, color = 'red')
                ax1[0, 2].set_ylabel('[MW]')
                ax1[0, 2].set_title('Core radiation', loc='right', fontsize = 11) 

            ax1[1, 0].plot(self.time_out, reservoirs["total_flux_mainwall"], label="Tot. flux to main wall", color = blue)
            ax1[1, 0].plot(self.time_out, reservoirs["edge_loss"], label="Radial edge loss", color = light_blue, linestyle = 'dashed')
            ax1[1, 0].plot(self.time_out, reservoirs["limiter_loss"], label="Parallel limiter loss", color = light_blue, linestyle = 'dotted')
            ax1[1, 0].set_ylabel('[$s^{-1}$]')
            ax1[1, 0].set_title('Main wall fluxes', loc='right', fontsize = 11)
            ax1[1, 0].legend(loc="best", fontsize = 9).set_draggable(True)

            ax1[1, 1].plot(self.time_out, reservoirs["mainwall_recycling"], color = light_green)
            ax1[1, 1].set_ylabel('[$s^{-1}$]')
            ax1[1, 1].set_title('Main wall recycling rate', loc='right', fontsize = 11)

            if self.namelist["phys_surfaces"]:
                if not self.advanced_PWI_flag:
                    ax1[1, 2].plot(self.time_out, reservoirs["particle_density_stuck_at_main_wall"], label="Particles stuck", color = light_grey, linestyle = 'dashed')
                ax1[1, 2].plot(self.time_out, reservoirs["particle_density_retained_at_main_wall"],
                    label="Particles retained", color = light_grey)
                ax1[1, 2].set_ylabel('[$cm^{-2}$]')              
            else:
                if not self.advanced_PWI_flag:
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
            ax1[2, 0].set_ylabel('[$s^{-1}$]') 
            ax1[2, 0].set_title('Divertor fluxes', loc='right', fontsize = 11)
            ax1[2, 0].legend(loc="best", fontsize = 9).set_draggable(True)

            ax1[2, 1].plot(self.time_out, reservoirs["divwall_recycling"], color = green)
            ax1[2, 1].set_ylabel('[$s^{-1}$]') 
            ax1[2, 1].set_title('Divertor wall recycling rate', loc='right', fontsize = 11)
            
            if self.namelist["phys_surfaces"]:
                if not self.advanced_PWI_flag:
                    ax1[2, 2].plot(self.time_out, reservoirs["particle_density_stuck_at_div_wall"], label="Particles stuck", color = grey, linestyle = 'dashed')
                ax1[2, 2].plot(self.time_out, reservoirs["particle_density_retained_at_div_wall"],
                    label="Particles retained", color = grey)
                ax1[2, 2].set_ylabel('[$cm^{-2}$]')              
            else:
                if not self.advanced_PWI_flag:
                    ax1[2, 2].plot(self.time_out, reservoirs["particles_stuck_at_div_wall"], label="Particles stuck", color = grey, linestyle = 'dashed')
                ax1[2, 2].plot(self.time_out, reservoirs["particles_retained_at_div_wall"],
                    label="Particles retained", color = grey)
                ax1[2, 2].set_ylabel('[#]')   
            ax1[2, 2].set_title('Divertor wall reservoir', loc='right', fontsize = 11)
            ax1[2, 2].legend(loc="best", fontsize = 9).set_draggable(True)
            
            if self.screening_eff > 0.0:
                ax1[3, 0].plot(self.time_out, reservoirs["divertor_backflow"]+reservoirs["screened_divertor_backflow"],
                           label="Tot. backflow rate", color = green) 
                ax1[3, 0].plot(self.time_out, reservoirs["divertor_backflow"],
                           label="Backflow to core", color = blue) 
                ax1[3, 0].plot(self.time_out, reservoirs["screened_divertor_backflow"],
                           label="Screened backflow", color = light_blue) 
            elif self.screening_eff == 0.0:
                ax1[3, 0].plot(self.time_out, reservoirs["divertor_backflow"]+reservoirs["screened_divertor_backflow"],
                           label="Backflow rate", color = green)
            ax1[3, 0].set_ylabel('[$s^{-1}$]')
            ax1[3, 0].set_title('Divertor backflow rates', loc='right', fontsize = 11)
            ax1[3, 0].legend(loc="best", fontsize = 9).set_draggable(True)

            ax1[3, 1].plot(self.time_out, reservoirs["pump_leakage"],
                           label="Leakage to core", color = light_green)
            ax1[3, 1].set_ylabel('[$s^{-1}$]')
            ax1[3, 1].set_title('Pump leakage rates', loc='right', fontsize = 11)
            ax1[3, 1].legend(loc="best", fontsize = 9).set_draggable(True)
            
            if self.namelist["phys_volumes"]:
                ax1[3, 2].plot(self.time_out, reservoirs["particle_density_in_divertor"],
                           label="Div. reservoir", color = green)
                if self.namelist["pump_chamber"]:
                    ax1[3, 2].plot(self.time_out, reservoirs["particle_density_in_pump"],
                           label="Pump reservoir", color = light_green)
                ax1[3, 2].set_ylabel('[$cm^{-3}$]')
            else:
                ax1[3, 2].plot(self.time_out, reservoirs["particles_in_divertor"],
                           label="Div. reservoir", color = green)
                if self.namelist["pump_chamber"]:
                    ax1[3, 2].plot(self.time_out, reservoirs["particles_in_pump"],
                           label="Pump reservoir", color = light_green)
                ax1[3, 2].set_ylabel('[#]')
            ax1[3, 2].set_title('Neutrals reservoirs', loc='right', fontsize = 11)
            ax1[3, 2].legend(loc="best", fontsize = 9).set_draggable(True)

            for ii in [0, 1, 2]:
                ax1[3, ii].set_xlabel("Time [s]")
            ax1[3, 0].set_xlim(self.time_out[[0, -1]])
            
            plt.tight_layout()

            # ----------------------------------------------------------------
            # now plot all particle reservoirs to check particle conservation:
            if axs is None:
                fig, ax2 = plt.subplots(figsize=(12,7))
            else:
                ax2 = axs[1]
            
            fig.suptitle('Particle conservation',fontsize=14)

            ax2.set_xlabel("time [s]")

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
