"""Extension of the core class :py:class:`~aurora.core.aurora_sim` including a more advanced model for Plasma Wall Interface (PWI) physics.
Module provided by Antonello Zito.
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
import pickle as pkl
from copy import deepcopy
from scipy.integrate import cumtrapz
from scipy.linalg import solve_banded
import pandas as pd
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
from . import core

class aurora_sim_pwi(core.aurora_sim):
    """Class to setup, run, and post-process 1.5D simulations of particle/impurity transport in
    magnetically-confined plasmas with the inclusion of an advanced plasma-wall interaction (PWI) model.
    This class inherits the :py:class:`~aurora.core.aurora_sim` class and adds additional methods
    related to PWI modelling.
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
        
        self.namelist = deepcopy(namelist)
        self.geqdsk = geqdsk
        
        self.reload_namelist()
        
        if "Raxis_cm" in self.namelist:
            self.Raxis_cm = self.namelist["Raxis_cm"]  # cm
        elif self.geqdsk is not None and "RMAXIS" in self.geqdsk:
            self.Raxis_cm = self.geqdsk["RMAXIS"] * 100.0  # cm
        
        # set up radial and temporal grids
        self.setup_grids()
        
        # set up parameters for plasma-wall interaction model
        self.setup_PWI_model()
        
        # inheritate all methods from main class core.py
        super(aurora_sim_pwi, self).__init__(namelist, geqdsk)
        
        # consistency checks
        if not self.phys_surfaces:
            raise ValueError("Implementing the advanced PWI model requires defining the physical wall surface areas!") 

        # # set up parameters for plasma-wall interaction model
        # self.setup_PWI_model()
      
        # set up kinetic profiles and atomic rates
        self.setup_kin_profs_depts()
        
    
    def reload_namelist(self, namelist=None):
        """(Re-)load namelist to update scalar variables."""
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
    
    
    def setup_PWI_model(self):
        """Method to set up Aurora inputs related to the advanced plasma-wall interaction model.
        """
        
        # effective wall surfaces, accounting for roughness
        self.surf_mainwall_eff = self.surf_mainwall*self.mainwall_roughness #cm^2
        self.surf_divwall_eff = self.surf_divwall*self.divwall_roughness #cm^2

        # Data for advanced PWI model for now available only for He as impurity and W as wall material
       
        if self.imp != 'He':
            raise ValueError(f"Advanced PWI model not available for impurity {self.imp}!")
        
        # keys for fortran routine
        if self.main_wall_material == 'W':
            self.Z_main_wall = 74
        else:
            raise ValueError(f"Advanced PWI model not available for main wall material {self.main_wall_material}!")
        if self.div_wall_material == 'W':
            self.Z_div_wall = 74
        else:
            raise ValueError(f"Advanced PWI model not available for divertor wall material {self.div_wall_material}!")
            
        # List of background species
        self.num_background_species = len(self.background_species)
        
        # calculate impact energies for simulated impurity and background species
        #   onto main and divertor walls at each time step
        self.energies_main_wall, self.energies_div_wall = self.get_impact_energy()
        self.E0_main_wall_imp = self.energies_main_wall[0,:] # Index 0 --> simulated species
        self.E0_div_wall_imp = self.energies_div_wall[0,:] # Index 0 --> simulated species
        self.E0_main_wall_background = self.energies_main_wall[1:,:] # Indices > 1 --> background species
        self.E0_div_wall_background = self.energies_div_wall[1:,:] # Indices > 1 --> background species
        
        # get angles of incidence
        angle_imp = surface.incidence_angle(self.imp)
        angle_background = np.zeros(len(self.background_species))
        for i in range(0,len(self.background_species)):
            angle_background[i] = surface.incidence_angle(self.background_species[i])
        
        # extract values for reflection coefficient of the simulated impurity at each time step
        self.rn_main_wall = surface.reflection_coeff_fit(self.imp,
                                                         self.main_wall_material,
                                                         angle_imp,
                                                         kind='rn',
                                                         energies = self.E0_main_wall_imp)["data"]
        self.rn_div_wall = surface.reflection_coeff_fit(self.imp,
                                                        self.div_wall_material,
                                                        angle_imp,
                                                        kind='rn',
                                                        energies = self.E0_div_wall_imp)["data"]
        
        # extract values for reflected energies of the simulated impurity at each time step
        self.E_refl_main_wall = surface.calc_reflected_energy(self.imp,
                                                              self.main_wall_material,
                                                              angle_imp,
                                                              energies = self.E0_main_wall_imp)["data"]
        self.E_refl_div_wall = surface.calc_reflected_energy(self.imp,
                                                             self.div_wall_material,
                                                             angle_imp,
                                                             energies = self.E0_div_wall_imp)["data"]
        
        # extract values for fluxes of background species onto main and divertor walls at each time step
        self.fluxes_main_wall_background, self.fluxes_div_wall_background = self.get_background_fluxes()
        
        # extract values for impurity sputtering coefficient for the wall retention of the
        #   simulated impurity at each time step
        self.y_main_wall = np.zeros((len(self.background_species)+1,len(self.time_out)))
        self.y_div_wall = np.zeros((len(self.background_species)+1,len(self.time_out)))
        self.y_main_wall[0,:] = surface.impurity_sputtering_coeff_fit(self.imp, # Index 0 --> simulated species
                                                                           self.imp, # simulated impurity itself as projectile
                                                                           self.main_wall_material,
                                                                           angle_imp,
                                                                           energies = self.E0_main_wall_imp)["normalized_data"]
        self.y_div_wall[0,:] = surface.impurity_sputtering_coeff_fit(self.imp, # Index 0 --> simulated species
                                                                          self.imp, # simulated impurity itself as projectile
                                                                          self.div_wall_material,
                                                                          angle_imp,
                                                                          energies = self.E0_div_wall_imp)["normalized_data"]
        for i in range(0,len(self.background_species)):
            self.y_main_wall[i+1,:] = surface.impurity_sputtering_coeff_fit(self.imp, # Indices > 1 --> background species
                                                                               self.background_species[i], # background species as projectiles
                                                                               self.main_wall_material,
                                                                               angle_background[i],
                                                                               energies = self.E0_main_wall_background[i,:])["normalized_data"]
            self.y_div_wall[i+1,:] = surface.impurity_sputtering_coeff_fit(self.imp, # Indices > 1 --> background species
                                                                              self.background_species[i], # background species as projectiles
                                                                              self.div_wall_material,
                                                                              angle_background[i],
                                                                              energies = self.E0_div_wall_background[i,:])["normalized_data"]
        
        # extract values for the impurity sputtered energy at each time step
        self.E_sput_main_wall = np.zeros((len(self.background_species)+1,len(self.time_out)))
        self.E_sput_div_wall = np.zeros((len(self.background_species)+1,len(self.time_out)))
        self.E_sput_main_wall[0,:] = surface.calc_imp_sputtered_energy(self.imp, # Index 0 --> simulated species
                                                                         self.imp, # simulated impurity itself as projectile
                                                                         self.main_wall_material,
                                                                         energies = self.E0_main_wall_imp)["data"]
        self.E_sput_div_wall[0,:] = surface.calc_imp_sputtered_energy(self.imp, # Index 0 --> simulated species
                                                                        self.imp, # simulated impurity itself as projectile
                                                                        self.div_wall_material,
                                                                        energies = self.E0_div_wall_imp)["data"]
        for i in range(0,len(self.background_species)):
            self.E_sput_main_wall[i+1,:] = surface.calc_imp_sputtered_energy(self.imp, # Indices > 1 --> background species
                                                                             self.background_species[i], # background species as projectiles
                                                                             self.main_wall_material,
                                                                             energies = self.E0_main_wall_background[i,:])["data"]
            self.E_sput_div_wall[i+1,:] = surface.calc_imp_sputtered_energy(self.imp, # Indices > 1 --> background species
                                                                            self.background_species[i], # background species as projectiles
                                                                            self.div_wall_material,
                                                                            energies = self.E0_div_wall_background[i,:])["data"]
        
        # extract the depth of the impurity implantation profile into main and divertor walls
        self.implantation_depth_main_wall = surface.implantation_depth_fit(self.imp,
                                                              self.main_wall_material,
                                                              angle_imp,
                                                              energies = self.characteristic_impact_energy_main_wall)["data"]
        self.implantation_depth_div_wall = surface.implantation_depth_fit(self.imp,
                                                             self.div_wall_material,
                                                             angle_imp,
                                                             energies = self.characteristic_impact_energy_div_wall)["data"]
        
        # extract saturation value of the density of the implanted impurity into main and divertor walls 
        self.n_main_wall_sat = self.namelist['advanced_PWI']['n_main_wall_sat'] # m^-2
        self.n_div_wall_sat = self.namelist['advanced_PWI']['n_div_wall_sat'] # m^-2   
        
        
    def get_background_fluxes(self):
        """Get the fluxes hitting main and divertor walls (for the background species), at all time
        steps, for a simulation run, in the framework of the advanced PWI model.

        Returns
        -------
        fluxes_main_wall : array (number of background species,time)
                                  Particle fluxes of the background species onto the
                                  main wall at each time of the simulation, in s^-1
        fluxes_div_wall :  array (number of background species,time)
                                  Particle fluxes of the background species onto the
                                  divertor wall at each time of the simulation, in s^-1
        
        """
        
        fluxes_main_wall = np.zeros((len(self.background_species),len(self.time_out)))
        fluxes_div_wall = np.zeros((len(self.background_species),len(self.time_out)))
        
        if self.background_mode == 'manual':
        # manually imposed fluxes of all the background species over the entire time grid
                    
            if (len(self.background_species) != len(self.background_main_wall_fluxes) or len(self.background_species) != len(self.background_div_wall_fluxes)):
                raise ValueError("Declared number of background species for advanced PWI model not consistent with declared number of background fluxes!")  
                
            # set wall fluxes of the background species to constant value over entire time grid
                
            for i in range(0,len(self.background_species)):
                fluxes_main_wall[i,:] = np.full(len(self.time_out),self.background_main_wall_fluxes[i])
                fluxes_div_wall[i,:] = np.full(len(self.time_out),self.background_div_wall_fluxes[i])
                

        elif self.background_mode == 'files':
        # fluxes of all the background species taken from aurora simulations performed
        #   in advance, one for each background species, on the same time grid of the
        #   current simulation
            
            if len(self.background_species) != len(self.background_files):
                raise ValueError("Declared number of background species for advanced PWI model not consistent with declared number of background simulation files!")       
            
            # Wall fluxes for the background species
            for i in range(0,len(self.background_species)):
                
                # Load background simulation data
                reservoirs_background = pd.read_pickle(self.background_files[i])
                
                if len(reservoirs_background['total_flux_mainwall']) != len(self.time_out):
                    raise ValueError("Time grids of current simulation and background simulations do not match!")       
                
                # Extract the fluxes
                fluxes_main_wall[i,:] = reservoirs_background['total_flux_mainwall']
                fluxes_div_wall[i,:] = reservoirs_background['total_flux_divwall']
                
        return fluxes_main_wall, fluxes_div_wall     
        
    def get_impact_energy(self):
        """Calculate the impact energy of ion projectile (both simulated and background species)
        hitting main and divertor wall, at all time steps, for a simulation run, in the framework
        of the advanced PWI model.

        Returns
        -------
        impact_energy_main_wall : array (1+number of background species,time)
                                  Impact energy of the ion projectiles of each species onto the
                                  main wall at each time of the simulation, in eV
        impact_energy_div_wall :  array (1+number of background species of species,time)
                                  Impact energy of the ion projectiles of each species onto the
                                  divertor wall at each time of the simulation, in eV
        
        """
        
        impact_energy_main_wall = np.zeros((len(self.background_species)+1,len(self.time_out)))
        impact_energy_div_wall = np.zeros((len(self.background_species)+1,len(self.time_out)))
                
        # Impact energy for the simulated species itself (index 0)
        # Single values
        E0_main_wall = surface.get_impact_energy(self.Te_lim,
                                                 self.imp,
                                                 mode = 'sheath',
                                                 Ti_over_Te = self.Ti_over_Te,
                                                 gammai = self.gammai)
        E0_div_wall = surface.get_impact_energy(self.Te_div,
                                                self.imp,
                                                mode = 'sheath',
                                                Ti_over_Te = self.Ti_over_Te,
                                                gammai = self.gammai)
        # Same values over entire time grid
        impact_energy_main_wall[0,:] = np.full(len(self.time_out),E0_main_wall)
        impact_energy_div_wall[0,:] = np.full(len(self.time_out),E0_div_wall)

        # Impact energies for the background species (indices 1+)
        for i in range(1,len(self.background_species)+1):
            # Single values
            E0_main_wall_temp = surface.get_impact_energy(self.Te_lim,
                                                          self.background_species[i-1],
                                                          mode = 'sheath',
                                                          Ti_over_Te = self.Ti_over_Te,
                                                          gammai = self.gammai)
            E0_div_wall_temp = surface.get_impact_energy(self.Te_div,
                                                         self.background_species[i-1],
                                                         mode = 'sheath',
                                                         Ti_over_Te = self.Ti_over_Te,
                                                         gammai = self.gammai)                
            # Same values over entire time grid 
            impact_energy_main_wall[i,:] = np.full(len(self.time_out),E0_main_wall_temp)
            impact_energy_div_wall[i,:] = np.full(len(self.time_out),E0_div_wall_temp)
        
        return impact_energy_main_wall, impact_energy_div_wall
        
    
    def setup_grids(self):
        """Method to set up radial and temporal grids given namelist inputs."""
        if self.geqdsk is not None:
            # Get r_V to rho_pol mapping
            rho_pol, _rvol = grids_utils.get_rhopol_rvol_mapping(self.geqdsk)
            rvol_lcfs = interp1d(rho_pol, _rvol)(1.0)
            self.rvol_lcfs = self.namelist["rvol_lcfs"] = np.round(
                rvol_lcfs, 3
            )  # set limit on accuracy

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
        self.core_vol = grids_utils.vol_int(ones[:,:],self.rvol_grid,self.pro_grid,self.Raxis_cm,rvol_max=self.rvol_lcfs)[0]
        self.plasma_vol = grids_utils.vol_int(ones[:,:],self.rvol_grid,self.pro_grid,self.Raxis_cm,rvol_max=None)[0]
        
        
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
        
        # explicit source into the divertor
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
            self.src_core * Sne0 * (self.rvol_grid / self.pro_grid)[:, None], 0)  # sum over radius
        self.total_source += self.src_div  # units of particles/s/cm
        # NB: src_core [1/cm^3] and src_div [1/cm/s] have different units!

        # recycling assumed to be activated
        
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
            self.spt_rad_prof = np.zeros((len(self.rvol_grid),len(self.background_species)+1,len(self.time_grid)))
            for i in range(0,len(self.background_species)+1):
                for j in range(0,len(self.time_grid)):
                    self.spt_rad_prof[:,i,j] = spt_rad_prof[:,i]

        else:
            
            # Advanced PWI model: set recycling profiles to exponentially decay from the wall
            #   with recycled neutrals having different energies depending on whether they
            #   were reflected, promptly recycled or sputtered from the wall

            # Source profile from promptly recycled particles from main wall:
            #   assumed to have the energy specified in imp_recycling_energy_eV
        
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

            # calculate the recycling profile
            rcl_rad_prof = source_utils.get_radial_source(
                nml_rcl_prof,  # namelist specifically to obtain exp decay from wall
                self.rvol_grid,
                self.pro_grid,
                Sne0,
                self._Ti,
            )
            
            # set same profile for all times    
            self.rcl_rad_prof = np.broadcast_to(
                rcl_rad_prof, (rcl_rad_prof.shape[0], len(self.time_grid))
            )
            
            # Source profile from reflected particles from main wall:
            #   use the reflected energy from TRIM data
            
            nml_rfl_prof = nml_rcl_prof
        
            # set energy value in the reflection profile namelist
            nml_rfl_prof["imp_source_energy_eV"] = self.E_refl_main_wall[0]
            
            # calculate the reflection profile
            rfl_rad_prof = source_utils.get_radial_source(
                nml_rfl_prof,  # namelist specifically to obtain exp decay from wall
                self.rvol_grid,
                self.pro_grid,
                Sne0,
                self._Ti,
            )
            
            # set same profile for all times    
            self.rfl_rad_prof = np.broadcast_to(
                rfl_rad_prof, (rfl_rad_prof.shape[0], len(self.time_grid))
            )
            
                
            # Source profile from sputtered particles from main wall:
            #   use the sputtered energy from TRIM data
            
            # calculate the sputtering profiles from simulated and background species
            spt_rad_prof = np.zeros((len(self.rvol_grid), len(self.background_species)+1))
            
            for j in range(0,len(self.background_species)+1):
            
                nml_spt_prof = nml_rcl_prof
        
                # set energy value in the sputtering profile namelist for the given projectile species
                nml_spt_prof["imp_source_energy_eV"] = self.E_sput_main_wall[j,0]
        
                spt_rad_prof[:,j] = source_utils.get_radial_source(
                    nml_spt_prof,  # namelist specifically to obtain exp decay from wall
                    self.rvol_grid,
                    self.pro_grid,
                    Sne0,
                    self._Ti,
                )[:,0]
                
                # set same profile for all times
                self.spt_rad_prof = np.zeros((len(self.rvol_grid),len(self.background_species)+1,len(self.time_grid)))
                for i in range(0,len(self.background_species)+1):
                    for j in range(0,len(self.time_grid)):
                        self.spt_rad_prof[:,i,j] = spt_rad_prof[:,i]
            
            
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
        radial_coordinate = 'rho_vol',
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
        radial_coordinate : string, optional
            Radial coordinate shown in the plot. Options: 'rho_vol' (default) or 'rho_pol'
        plot_PWI : bool, optional
            If True, plot time traces related to the plasma-wall interaction model

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
                 Not empty only if :param:`div_recomb_ratio` < 1.0 AND advanced PWI model used.
            rcld_recl_rate : array (nt,)
                 Promptly recycled flux from the divertor wall [:math:`cm^{-1} s^{-1}`].
                 Not empty only if :param:`div_recomb_ratio` < 1.0 AND advanced PWI model used.
            rcld_impl_rate : array (nt,)
                 Implanted flux into the divertor wall reservoir [:math:`cm^{-1} s^{-1}`]
                 Not empty only if :param:`div_recomb_ratio` < 1.0 AND advanced PWI model used.
            rcld_sput_rate : array (1+background_species,nt,)
                 Sputtered fluxes from the divertor wall reservoir, one for each sputtering species 
                 [:math:`cm^{-1} s^{-1}`]. Not empty only if :param:`div_recomb_ratio` < 1.0 AND
                 advanced PWI model used.
            rclb_rate : array (nt,)
                 Backflow from the divertor neutrals reservoir reaching the plasma [:math:`cm^{-1} s^{-1}`]
            rcls_rate : array (nt,)
                 Screened backflow from the divertor neutrals reservoir [:math:`cm^{-1} s^{-1}`].
                 Not empty only if :param:`screening_eff` > 0.0.
            rclp_rate : array (nt,)
                 Leakage from the pump neutrals reservoir [:math:`cm^{-1} s^{-1}`].
                 Not empty only if :param:`pump_chamber` = True.
            rclw_rate : array (nt,)
                 Total recycled flux from the main wall reservoir [:math:`cm^{-1} s^{-1}`]
            rclw_refl_rate : array (nt,)
                 Reflected flux from the main wall [:math:`cm^{-1} s^{-1}`].
                 Not empty only if advanced PWI model used.
            rclw_recl_rate : array (nt,)
                 Promptly recycled flux from the main wall [:math:`cm^{-1} s^{-1}`].
                 Not empty only if advanced PWI model used.
            rclw_impl_rate : array (nt,)
                 Implanted flux into the main wall reservoir [:math:`cm^{-1} s^{-1}`]
                 Not empty only if advanced PWI model used.
            rclw_sput_rate : array (1+background_species,nt,)
                 Sputtered fluxes from the main wall reservoir, one for each sputtering species 
                 [:math:`cm^{-1} s^{-1}`]. Not empty only if advanced PWI model used.
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
                
        if npump_init is None or self.namelist["pump_chamber"] == False:
            # default: start in a state with empty pump neutrals reservoir
            #   (or pump reservoir not used at all)
            npump_init = 0 
        else:
            # start in a state with not empty pump neutrals reservoir
            # convert to cm^-1 before passing it to fortran_run
            if self.namelist["phys_volumes"] == True:
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

        # import here to avoid import when building documentation or package (negligible slow down)
        from ._aurora import run as fortran_run
        
        # activate advanced PWI model in fortran main routines
        self.advanced_PWI_flag = True

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
            self.screening_eff, # screening coefficient
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
            self.advanced_PWI_flag, # logic key for PWI model
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
            
            if radial_coordinate == 'rho_vol':
                x = self.rvol_grid
                xlabel = r"$r_V$ [cm]"
                x_line=self.rvol_lcfs
            elif radial_coordinate == 'rho_pol':
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
        if plot_PWI:
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

            self.res = nz_unstaged, *self.res[1:]

        return self.res
    
    
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
        
        nz = self.res['nz']
        N_mainwall = self.res['N_mainwall']
        N_divwall = self.res['N_divwall']
        N_div = self.res['N_div']
        N_pump = self.res['N_pump']
        N_out = self.res['N_out']
        N_mainret = self.res['N_mainret']
        N_divret = self.res['N_divret']
        N_tsu = self.res['N_tsu']
        N_dsu = self.res['N_dsu']
        N_dsul = self.res['N_dsul']
        rcld_rate = self.res['rcld_rate']
        rcld_refl_rate = self.res['rcld_refl_rate']
        rcld_recl_rate = self.res['rcld_recl_rate']
        rcld_impl_rate = self.res['rcld_impl_rate']
        rcld_sput_rate = self.res['rcld_sput_rate']
        rclb_rate = self.res['rclb_rate']
        rcls_rate = self.res['rcls_rate']
        rclp_rate = self.res['rclp_rate']
        rclw_rate = self.res['rclw_rate']
        rclw_refl_rate = self.res['rclw_refl_rate']
        rclw_recl_rate = self.res['rclw_recl_rate']
        rclw_impl_rate = self.res['rclw_impl_rate']
        rclw_sput_rate = self.res['rclw_sput_rate']

        nz = nz.transpose(2, 1, 0)  # time,nZ,space

        # factor to account for cylindrical geometry:
        circ = 2 * np.pi * self.Raxis_cm  # cm

        # collect all the relevant quantities for particle conservation
        reservoirs = {}

        # calculate total impurity density (summed over charge states)
        total_impurity_density = np.nansum(nz, axis=1)  # time, space

        # Compute total number of particles for particle conservation checks:
        all_particles = grids_utils.vol_int(
            total_impurity_density,
            self.rvol_grid,
            self.pro_grid,
            self.Raxis_cm,
            rvol_max=None,
        )

        reservoirs["total"] = all_particles + (N_mainwall + N_divwall + N_div + N_pump + N_out + N_mainret + N_divret) * circ

        # main fluxes
        reservoirs["source"] = self.total_source*circ
        reservoirs["plasma_source"] = self.total_source*circ
        reservoirs["wall_source"] = rclw_rate * circ
        reservoirs["divertor_source"] = rclb_rate * circ + rclp_rate * circ
        reservoirs["plasma_removal_rate"] = - N_dsu * circ - N_tsu * circ - N_dsul * circ       
        reservoirs["net_plasma_flow"] = reservoirs["plasma_source"] + reservoirs["wall_source"] + reservoirs["divertor_source"] + reservoirs["plasma_removal_rate"]

        # integrated source over time
        reservoirs["integ_source"] = cumtrapz(reservoirs["source"], self.time_out, initial=0) + reservoirs["total"][0] 
        
        # main plasma content
        if self.namelist["phys_volumes"]:    
            reservoirs["particle_density_in_plasma"] = all_particles/self.plasma_vol
        reservoirs["particles_in_plasma"] = all_particles
        
        # divertor and pump neutrals reservoirs
        if self.namelist["phys_volumes"]:
            reservoirs["particle_density_in_divertor"] = (N_div * circ)/self.vol_div
            if self.namelist["pump_chamber"]:
                reservoirs["particle_density_in_pump"] = (N_pump * circ)/self.vol_pump
        reservoirs["particles_in_divertor"] = N_div * circ
        reservoirs["particles_in_pump"] = N_pump * circ
        
        # fluxes towards main wall
        reservoirs["edge_loss"] = N_tsu * circ
        reservoirs["limiter_loss"] = N_dsul * circ
        reservoirs["total_flux_mainwall"] = reservoirs["edge_loss"] + reservoirs["limiter_loss"]
        
        # recycling rates from main wall
        reservoirs["mainwall_recycling"] = rclw_rate * circ
        
        # main wall reservoir
        if self.namelist["phys_surfaces"]:
            reservoirs["particle_density_retained_at_main_wall"] = (N_mainret * circ)/(self.surf_mainwall*self.mainwall_roughness)
        reservoirs["particles_retained_at_main_wall"] = N_mainret * circ

        # flux towards divertor targets and backflow/leakage rates
        reservoirs["parallel_loss"] = N_dsu * circ
        reservoirs["divertor_backflow"] = rclb_rate * circ
        reservoirs["screened_divertor_backflow"] = rcls_rate * circ
        reservoirs["pump_leakage"] = rclp_rate * circ
        reservoirs["total_flux_divwall"] = (reservoirs["parallel_loss"]+reservoirs["screened_divertor_backflow"])*(1-self.div_recomb_ratio)
        
        # recycling rates from divertor wall
        reservoirs["divwall_recycling"] = rcld_rate * circ    
        
        # divertor wall reservoir
        if self.namelist["phys_surfaces"]:
            reservoirs["particle_density_retained_at_div_wall"] = (N_divret * circ)/(self.surf_divwall*self.divwall_roughness)
        reservoirs["particles_retained_at_div_wall"] = N_divret * circ
        
        # particles pumped away
        reservoirs["particles_pumped"] = N_out * circ
        reservoirs["pumping_rate"] = np.zeros(len(self.time_out))
        for i in range(1,len(self.time_out)):
            reservoirs["pumping_rate"][i] = (reservoirs["particles_pumped"][i]-reservoirs["particles_pumped"][i-1])/(self.time_out[i]-self.time_out[i-1])

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
                ax1[1, 2].plot(self.time_out, reservoirs["particle_density_retained_at_main_wall"],
                    label="Particles retained", color = light_grey)
                ax1[1, 2].set_ylabel('[$\mathrm{cm}^{-2}$]')              
            else:
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
                ax1[2, 2].plot(self.time_out, reservoirs["particle_density_retained_at_div_wall"],
                    label="Particles retained", color = grey)
                ax1[2, 2].set_ylabel('[$\mathrm{cm}^{-2}$]')              
            else:
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
        
        
    def PWI_time_traces(self, plot=True, ylim = True, axs=None):
        """Return and plot data regarding the plasma-wall interaction in the simulation.

        Parameters
        ----------
        plot : bool, optional
            If True, plot time traces of various quantities related to the plasma-wall interaction.
        axs : 2-tuple or array
            Array-like structure containing two matplotlib.Axes instances: the first one
            for the time traces referred to the main wall, the second one for the time traces
            referred to the divertor wall

        Returns
        -------
        PWI_traces : dict
            Dictionary containing various quantities related to the plasma-wall interaction.
        axs : matplotlib.Axes instances, only returned if plot=True
            Array-like structure containing two matplotlib.Axes instances, (ax1,ax2).
            See optional input argument.
        """
        
        # get colors for plots
        colors = plot_tools.load_color_codes_reservoirs()
        blue,light_blue,green,light_green,grey,light_grey,red,light_red = colors
        colors_PWI = plot_tools.load_color_codes_PWI()
        reds, blues, light_blues, greens = colors_PWI
        
        nz = self.res['nz']
        N_mainwall = self.res['N_mainwall']
        N_divwall = self.res['N_divwall']
        N_div = self.res['N_div']
        N_pump = self.res['N_pump']
        N_out = self.res['N_out']
        N_mainret = self.res['N_mainret']
        N_divret = self.res['N_divret']
        N_tsu = self.res['N_tsu']
        N_dsu = self.res['N_dsu']
        N_dsul = self.res['N_dsul']
        rcld_rate = self.res['rcld_rate']
        rcld_refl_rate = self.res['rcld_refl_rate']
        rcld_recl_rate = self.res['rcld_recl_rate']
        rcld_impl_rate = self.res['rcld_impl_rate']
        rcld_sput_rate = self.res['rcld_sput_rate']
        rclb_rate = self.res['rclb_rate']
        rcls_rate = self.res['rcls_rate']
        rclp_rate = self.res['rclp_rate']
        rclw_rate = self.res['rclw_rate']
        rclw_refl_rate = self.res['rclw_refl_rate']
        rclw_recl_rate = self.res['rclw_recl_rate']
        rclw_impl_rate = self.res['rclw_impl_rate']
        rclw_sput_rate = self.res['rclw_sput_rate']
        
        nz = nz.transpose(2, 1, 0)  # time,nZ,space

        # factor to account for cylindrical geometry:
        circ = 2 * np.pi * self.Raxis_cm  # cm
        
        # list of background species
        background_species = self.background_species
        
        # saturation densities in cm^-2
        n_main_wall_sat = self.n_main_wall_sat/1e4
        n_div_wall_sat = self.n_div_wall_sat/1e4

        # collect all the relevant quantities for the PWI model
        PWI_traces = {}
        
        # main wall
        
        # main wall impurity content
        PWI_traces["impurity_content_mainwall"] = ((N_mainret * circ)/(self.surf_mainwall*self.mainwall_roughness))/n_main_wall_sat
        # impurity flux towards main wall
        PWI_traces["impurity_flux_mainwall"] = N_tsu * circ + N_dsul * circ
        # impurity impact energy at main wall
        PWI_traces["impurity_impact_energy_mainwall"] = self.E0_main_wall_imp.T
        # impurity reflection coefficient at main wall
        PWI_traces["impurity_reflection_coeff_mainwall"] = self.rn_main_wall
        # impurity reflection energy at main wall
        PWI_traces["impurity_reflection_energy_mainwall"] = self.E_refl_main_wall
        # fluxes of background species
        PWI_traces["background_fluxes_mainwall"] = self.fluxes_main_wall_background.T
        # impact energies of background species at main wall
        PWI_traces["background_impact_energies_mainwall"] = self.E0_main_wall_background.T
        # impurity sputtering yields at main wall
        PWI_traces["impurity_sputtering_yields_mainwall"] = self.y_main_wall.T
        # impurity sputtering energies at main wall
        PWI_traces["impurity_sputtering_energies_mainwall"] = self.E_sput_main_wall.T
        # impurity reflection rate from main wall
        PWI_traces["impurity_reflection_rate_mainwall"] = rclw_refl_rate * circ
        # impurity prompt recycling rate from main wall
        PWI_traces["impurity_prompt_recycling_rate_mainwall"] = rclw_recl_rate * circ
        # impurity implantation rate into main wall
        PWI_traces["impurity_implantation_rate_mainwall"] = rclw_impl_rate * circ
        # impurity release rate from main wall
        PWI_traces["impurity_sputtering_rates_mainwall"] = np.zeros((len(self.time_out),len(background_species)+1))
        for i in range(0,len(background_species)+1):
            PWI_traces["impurity_sputtering_rates_mainwall"][:,i] = rclw_sput_rate[i,:] * circ
        PWI_traces["impurity_sputtering_rate_total_mainwall"] = PWI_traces["impurity_sputtering_rates_mainwall"].sum(axis=1)

        # divertor wall
        
        # divertor wall impurity content
        PWI_traces["impurity_content_divwall"] = ((N_divret * circ)/(self.surf_divwall*self.divwall_roughness))/n_div_wall_sat
        # impurity flux towards divertor wall
        PWI_traces["impurity_flux_divwall"] = (N_dsu * circ + rcls_rate * circ)*(1-self.div_recomb_ratio)
        # impurity impact energy at divertor wall
        PWI_traces["impurity_impact_energy_divwall"] = self.E0_div_wall_imp.T
        # impurity reflection coefficient at divertor wall
        PWI_traces["impurity_reflection_coeff_divwall"] = self.rn_div_wall
        # impurity reflection energy at divertor wall
        PWI_traces["impurity_reflection_energy_divwall"] = self.E_refl_div_wall
        # fluxes of background species
        PWI_traces["background_fluxes_divwall"] = self.fluxes_div_wall_background.T
        # impact energies of background species at divertor wall
        PWI_traces["background_impact_energies_divwall"] = self.E0_div_wall_background.T
        # impurity sputtering yields at divertor wall
        PWI_traces["impurity_sputtering_yields_divwall"] = self.y_div_wall.T
        # impurity sputtering energies at divertor wall
        PWI_traces["impurity_sputtering_energies_divwall"] = self.E_sput_div_wall.T 
        # impurity reflection rate from divertor wall
        PWI_traces["impurity_reflection_rate_divwall"] = rcld_refl_rate * circ
        # impurity prompt recycling rate from divertor wall
        PWI_traces["impurity_prompt_recycling_rate_divwall"] = rcld_recl_rate * circ
        # impurity implantation rate into divertor wall
        PWI_traces["impurity_implantation_rate_divwall"] = rcld_impl_rate * circ
        # impurity release rate from divertor wall
        PWI_traces["impurity_sputtering_rates_divwall"] = np.zeros((len(self.time_out),len(background_species)+1))
        for i in range(0,len(background_species)+1):
            PWI_traces["impurity_sputtering_rates_divwall"][:,i] = rcld_sput_rate[i,:] * circ
        PWI_traces["impurity_sputtering_rate_total_divwall"] = PWI_traces["impurity_sputtering_rates_divwall"].sum(axis=1)

        if plot:
            # -------------------------------------------------
            # plot time histories for the main wall:
            if axs is None:
                fig, ax1 = plt.subplots(nrows=3, ncols=5, sharex=True, figsize=(20, 12))
            else:
                ax1 = axs[0]
                
            fig.suptitle('Plasma - main wall interaction time traces',fontsize=18)

            ax1[0, 0].plot(self.time_out, PWI_traces["impurity_content_mainwall"], color = grey)
            if ylim:
                ax1[0, 0].set_ylim(0,1)
            ax1[0, 0].set_ylabel(f'$C_{{{self.imp}}}$/$C_{{{self.imp},sat}}$')
            ax1[0, 0].set_title(f"Retained {self.imp} content into wall", loc='right', fontsize = 11)

            ax1[0, 1].plot(self.time_out, PWI_traces["impurity_flux_mainwall"], color = light_blues[0])
            if ylim:
                ax1[0, 1].set_ylim(0,np.max(PWI_traces["impurity_flux_mainwall"])*1.15)
            ax1[0, 1].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax1[0, 1].set_title(f"{self.imp} flux towards wall", loc='right', fontsize = 11)

            ax1[0, 2].plot(self.time_out, PWI_traces["impurity_impact_energy_mainwall"], color = reds[0])
            if ylim:
                ax1[0, 2].set_ylim(0,np.max(PWI_traces["impurity_impact_energy_mainwall"])*1.15)
            ax1[0, 2].set_ylabel('[$\mathrm{eV}$]') 
            ax1[0, 2].set_title(f"Mean {self.imp} impact energy", loc='right', fontsize = 11)

            ax1[0, 3].plot(self.time_out, PWI_traces["impurity_reflection_coeff_mainwall"], color = blues[0])
            if ylim:
                ax1[0, 3].set_ylim(np.min(PWI_traces["impurity_reflection_coeff_mainwall"])/1.08,min(np.max(PWI_traces["impurity_reflection_coeff_mainwall"])*1.08,1))
            ax1[0, 3].set_ylabel('$R_N$')  
            ax1[0, 3].set_title(f"Mean {self.imp} reflection coeff. on {self.main_wall_material}", loc='right', fontsize = 11)
            
            ax1[0, 4].plot(self.time_out, PWI_traces["impurity_reflection_energy_mainwall"], color = reds[0])
            if ylim:
                ax1[0, 4].set_ylim(0,np.max(PWI_traces["impurity_reflection_energy_mainwall"])*1.15)
            ax1[0, 4].set_ylabel('[$\mathrm{eV}$]') 
            ax1[0, 4].set_title(f"Mean energy of reflected {self.imp}", loc='right', fontsize = 11)

            for i in range(0,len(background_species)):
                ax1[1, 1].plot(self.time_out, PWI_traces["background_fluxes_mainwall"][:,i], label=f"{background_species[i]} flux", color = light_blues[i+1])
            if ylim:
                ax1[1, 1].set_ylim(0,np.max(PWI_traces["background_fluxes_mainwall"])*1.15)
            ax1[1, 1].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax1[1, 1].set_title("Background fluxes towards wall", loc='right', fontsize = 11)
            ax1[1, 1].legend(loc="best", fontsize = 9).set_draggable(True)

            for i in range(0,len(background_species)):
                ax1[1, 2].plot(self.time_out, PWI_traces["background_impact_energies_mainwall"][:,i], label=f"{background_species[i]}", color = reds[i+1])
            if ylim:
                ax1[1, 2].set_ylim(0,np.max(PWI_traces["background_impact_energies_mainwall"])*1.15)
            ax1[1, 2].set_ylabel('[$\mathrm{eV}$]')
            ax1[1, 2].set_title("Mean background fluxes impact energy", loc='right', fontsize = 11)
            ax1[1, 2].legend(loc="best", fontsize = 9).set_draggable(True)

            ax1[1, 3].plot(self.time_out, PWI_traces["impurity_sputtering_yields_mainwall"][:,0], label=f"from {self.imp}", color = blues[0])
            for i in range(0,len(background_species)):
                ax1[1, 3].plot(self.time_out, PWI_traces["impurity_sputtering_yields_mainwall"][:,i+1], label=f"from {background_species[i]}", color = blues[i+1])
            if ylim:
                ax1[1, 3].set_ylim(0,np.max(PWI_traces["impurity_sputtering_yields_mainwall"])*1.15)
            ax1[1, 3].set_ylabel(f'$Y_{{{self.imp}}}$/$C_{{{self.imp},wall}}$')
            ax1[1, 3].set_title(f"Mean {self.imp} sputtering yields from {self.main_wall_material}", loc='right', fontsize = 11)
            ax1[1, 3].legend(loc="best", fontsize = 9).set_draggable(True)
            
            ax1[1, 4].plot(self.time_out, PWI_traces["impurity_sputtering_energies_mainwall"][:,0], label=f"from {self.imp}", color = reds[0])
            for i in range(0,len(background_species)):
                ax1[1, 4].plot(self.time_out, PWI_traces["impurity_sputtering_energies_mainwall"][:,i+1], label=f"from {background_species[i]}", color = reds[i+1])
            if ylim:
                ax1[1, 4].set_ylim(0,np.max(PWI_traces["impurity_sputtering_energies_mainwall"])*1.15)
            ax1[1, 4].set_ylabel('[$\mathrm{eV}$]')
            ax1[1, 4].set_title(f"Mean energy of sputtered {self.imp}", loc='right', fontsize = 11)
            ax1[1, 4].legend(loc="best", fontsize = 9).set_draggable(True)   
            
            ax1[2, 0].plot(self.time_out, PWI_traces["impurity_implantation_rate_mainwall"], label="Total absorption rate", color = greens[0])
            ax1[2, 0].plot(self.time_out, PWI_traces["impurity_sputtering_rate_total_mainwall"], label="Total release rate", color = blues[0])
            ax1[2, 0].plot(self.time_out, PWI_traces["impurity_implantation_rate_mainwall"]-PWI_traces["impurity_sputtering_rate_total_mainwall"], label="Balance", color = 'black')
            ax1[2, 0].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax1[2, 0].set_title(f"{self.imp} wall balance", loc='right', fontsize = 11)
            ax1[2, 0].legend(loc="best", fontsize = 9).set_draggable(True)
            
            ax1[2, 1].plot(self.time_out, PWI_traces["impurity_reflection_rate_mainwall"], color = blues[0])
            if ylim:
                ax1[2, 1].set_ylim(0,np.max(PWI_traces["impurity_reflection_rate_mainwall"])*1.15)
            ax1[2, 1].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax1[2, 1].set_title(f"{self.imp} reflection rate from wall", loc='right', fontsize = 11)

            ax1[2, 2].plot(self.time_out, PWI_traces["impurity_prompt_recycling_rate_mainwall"], color = blues[0])
            if ylim:
                ax1[2, 2].set_ylim(0,np.max(PWI_traces["impurity_prompt_recycling_rate_mainwall"])*1.15)
            ax1[2, 2].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax1[2, 2].set_title(f"{self.imp} prompt recycling rate from wall", loc='right', fontsize = 11)
            
            ax1[2, 3].plot(self.time_out, PWI_traces["impurity_implantation_rate_mainwall"], color = blues[0])
            if ylim:
                ax1[2, 3].set_ylim(0,np.max(PWI_traces["impurity_implantation_rate_mainwall"])*1.15)
            ax1[2, 3].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax1[2, 3].set_title(f"{self.imp} implantation rate into wall", loc='right', fontsize = 11)
            
            ax1[2, 4].plot(self.time_out, PWI_traces["impurity_sputtering_rates_mainwall"][:,0], label=f"from {self.imp}", color = blues[0])
            for i in range(0,len(background_species)):
                ax1[2, 4].plot(self.time_out, PWI_traces["impurity_sputtering_rates_mainwall"][:,i+1], label=f"from {background_species[i]}", color = blues[i+1])
            if ylim:
                ax1[2, 4].set_ylim(0,np.max(PWI_traces["impurity_sputtering_rates_mainwall"])*1.15)
            ax1[2, 4].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax1[2, 4].set_title(f"{self.imp} sputtering rates from wall", loc='right', fontsize = 11)
            ax1[2, 4].legend(loc="best", fontsize = 9).set_draggable(True)

            for ii in [0, 1, 2, 3, 4]:
                ax1[2, ii].set_xlabel('$\mathrm{time}$ [$\mathrm{s}$]')
            ax1[2, 0].set_xlim(self.time_out[[0, -1]])
            
            plt.tight_layout()
            
            # -------------------------------------------------
            # plot time histories for the divertor wall:
            if axs is None:
                fig, ax2 = plt.subplots(nrows=3, ncols=5, sharex=True, figsize=(20, 12))
            else:
                ax2 = axs[0]
                
            fig.suptitle('Plasma - divertor wall interaction time traces',fontsize=18)

            ax2[0, 0].plot(self.time_out, PWI_traces["impurity_content_divwall"], color = grey)
            if ylim:
                ax2[0, 0].set_ylim(0,1)
            ax2[0, 0].set_ylabel(f'$C_{{{self.imp}}}$/$C_{{{self.imp},sat}}$')
            ax2[0, 0].set_title(f"Retained {self.imp} content into wall", loc='right', fontsize = 11)

            ax2[0, 1].plot(self.time_out, PWI_traces["impurity_flux_divwall"], color = light_blues[0])
            if ylim:
                ax2[0, 1].set_ylim(0,np.max(PWI_traces["impurity_flux_divwall"])*1.15)
            ax2[0, 1].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax2[0, 1].set_title(f"{self.imp} flux towards wall", loc='right', fontsize = 11)

            ax2[0, 2].plot(self.time_out, PWI_traces["impurity_impact_energy_divwall"], color = reds[0])
            if ylim:
                ax2[0, 2].set_ylim(0,np.max(PWI_traces["impurity_impact_energy_divwall"])*1.15)
            ax2[0, 2].set_ylabel('[$\mathrm{eV}$]')
            ax2[0, 2].set_title(f"Mean {self.imp} impact energy", loc='right', fontsize = 11)

            ax2[0, 3].plot(self.time_out, PWI_traces["impurity_reflection_coeff_divwall"], color = blues[0])
            if ylim:
                ax2[0, 3].set_ylim(np.min(PWI_traces["impurity_reflection_coeff_divwall"])/1.08,min(np.max(PWI_traces["impurity_reflection_coeff_divwall"])*1.08,1))
            ax2[0, 3].set_ylabel('$R_N$')  
            ax2[0, 3].set_title(f"Mean {self.imp} reflection coeff. on {self.div_wall_material}", loc='right', fontsize = 11)
            
            ax2[0, 4].plot(self.time_out, PWI_traces["impurity_reflection_energy_divwall"], color = reds[0])
            if ylim:
                ax2[0, 4].set_ylim(0,np.max(PWI_traces["impurity_reflection_energy_divwall"])*1.15)
            ax2[0, 4].set_ylabel('[$\mathrm{eV}$]') 
            ax2[0, 4].set_title(f"Mean energy of reflected {self.imp}", loc='right', fontsize = 11)

            for i in range(0,len(background_species)):
                ax2[1, 1].plot(self.time_out, PWI_traces["background_fluxes_divwall"][:,i], label=f"{background_species[i]} flux", color = light_blues[i+1])
            if ylim:
                ax2[1, 1].set_ylim(0,np.max(PWI_traces["background_fluxes_divwall"])*1.15)
            ax2[1, 1].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax2[1, 1].set_title("Background fluxes towards wall", loc='right', fontsize = 11)
            ax2[1, 1].legend(loc="best", fontsize = 9).set_draggable(True)

            for i in range(0,len(background_species)):
                ax2[1, 2].plot(self.time_out, PWI_traces["background_impact_energies_divwall"][:,i], label=f"{background_species[i]}", color = reds[i+1])
            if ylim:
                ax2[1, 2].set_ylim(0,np.max(PWI_traces["background_impact_energies_divwall"])*1.15)
            ax2[1, 2].set_ylabel('[$\mathrm{eV}$]')
            ax2[1, 2].set_title("Mean background fluxes impact energy", loc='right', fontsize = 11)
            ax2[1, 2].legend(loc="best", fontsize = 9).set_draggable(True)

            ax2[1, 3].plot(self.time_out, PWI_traces["impurity_sputtering_yields_divwall"][:,0], label=f"from {self.imp}", color = blues[0])
            for i in range(0,len(background_species)):
                ax2[1, 3].plot(self.time_out, PWI_traces["impurity_sputtering_yields_divwall"][:,i+1], label=f"from {background_species[i]}", color = blues[i+1])
            if ylim:
                ax2[1, 3].set_ylim(0,np.max(PWI_traces["impurity_sputtering_yields_divwall"])*1.15)
            ax2[1, 3].set_ylabel(f'$Y_{{{self.imp}}}$/$C_{{{self.imp},wall}}$')
            ax2[1, 3].set_title(f"Mean {self.imp} sputtering yields from {self.div_wall_material}", loc='right', fontsize = 11)
            ax2[1, 3].legend(loc="best", fontsize = 9).set_draggable(True)

            ax2[1, 4].plot(self.time_out, PWI_traces["impurity_sputtering_energies_divwall"][:,0], label=f"from {self.imp}", color = reds[0])
            for i in range(0,len(background_species)):
                ax2[1, 4].plot(self.time_out, PWI_traces["impurity_sputtering_energies_divwall"][:,i+1], label=f"from {background_species[i]}", color = reds[i+1])
            if ylim:
                ax2[1, 4].set_ylim(0,np.max(PWI_traces["impurity_sputtering_energies_divwall"])*1.15)
            ax2[1, 4].set_ylabel('[$\mathrm{eV}$]')
            ax2[1, 4].set_title(f"Mean energy of sputtered {self.imp}", loc='right', fontsize = 11)
            ax2[1, 4].legend(loc="best", fontsize = 9).set_draggable(True)  

            ax2[2, 0].plot(self.time_out, PWI_traces["impurity_implantation_rate_divwall"], label="Total absorption rate", color = greens[0])
            ax2[2, 0].plot(self.time_out, PWI_traces["impurity_sputtering_rate_total_divwall"], label="Total release rate", color = blues[0])
            ax2[2, 0].plot(self.time_out, PWI_traces["impurity_implantation_rate_divwall"]-PWI_traces["impurity_sputtering_rate_total_divwall"], label="Balance", color = 'black')
            ax2[2, 0].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax2[2, 0].set_title(f"{self.imp} wall balance", loc='right', fontsize = 11)
            ax2[2, 0].legend(loc="best", fontsize = 9).set_draggable(True)

            ax2[2, 1].plot(self.time_out, PWI_traces["impurity_reflection_rate_divwall"], color = blues[0])
            if ylim:
                ax2[2, 1].set_ylim(0,np.max(PWI_traces["impurity_reflection_rate_divwall"])*1.15)
            ax2[2, 1].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax2[2, 1].set_title(f"{self.imp} reflection rate from wall", loc='right', fontsize = 11)

            ax2[2, 2].plot(self.time_out, PWI_traces["impurity_prompt_recycling_rate_divwall"], color = blues[0])
            if ylim:
                ax2[2, 2].set_ylim(0,np.max(PWI_traces["impurity_prompt_recycling_rate_divwall"])*1.15)
            ax2[2, 2].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax2[2, 2].set_title(f"{self.imp} prompt recycling rate from wall", loc='right', fontsize = 11)
            
            ax2[2, 3].plot(self.time_out, PWI_traces["impurity_implantation_rate_divwall"], color = blues[0])
            if ylim:
                ax2[2, 3].set_ylim(0,np.max(PWI_traces["impurity_implantation_rate_divwall"])*1.15)
            ax2[2, 3].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax2[2, 3].set_title(f"{self.imp} implantation rate into wall", loc='right', fontsize = 11)
            
            ax2[2, 4].plot(self.time_out, PWI_traces["impurity_sputtering_rates_divwall"][:,0], label=f"from {self.imp}", color = blues[0])
            for i in range(0,len(background_species)):
                ax2[2, 4].plot(self.time_out, PWI_traces["impurity_sputtering_rates_divwall"][:,i+1], label=f"from {background_species[i]}", color = blues[i+1])
            if ylim:
                ax2[2, 4].set_ylim(0,np.max(PWI_traces["impurity_sputtering_rates_divwall"])*1.15)
            ax2[2, 4].set_ylabel('[$\mathrm{s}^{-1}$]')
            ax2[2, 4].set_title(f"{self.imp} sputtering rates from wall", loc='right', fontsize = 11)
            ax2[2, 4].legend(loc="best", fontsize = 9).set_draggable(True)

            for ii in [0, 1, 2, 3, 4]:
                ax2[2, ii].set_xlabel('$\mathrm{time}$ [$\mathrm{s}$]')
            ax2[2, 0].set_xlim(self.time_out[[0, -1]])  
            
            plt.tight_layout()

        if plot:
            return PWI_traces, (ax1, ax2)
        else:
            return PWI_traces             
            