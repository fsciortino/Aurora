"""Extension of the core class for Aurora simulations in :py:class:`~aurora.core.aurora_sim` including a more advanced model for Plasma Wall Interface (PWI) physics.

Mostly based on work by Antonello Zito.
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

import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from . import source_utils
from . import transport_utils
from . import plot_tools
from . import surface

class aurora_sim_pwi(aurora_sim):
    """Class to setup, run, and post-process 1.5D simulations of particle/impurity transport in
    magnetically-confined plasmas with the inclusion of an advanced 0D edge model, including
    effects of recycling, plasma-wall interaction (PWI) and pumping.

    This class inherits the :py:class:`~aurora.core.aurora_sim` class and adds additional methods
    related to edge/global modeling.

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
        super(aurora_sim_pwi, self).__init__(namelist, geqdsk)

        # set up parameters for plasma-wall interaction model
        self.setup_PWI_model()

        self.setup_kin_profs_depts()

    def setup_kin_profs_depts(self):

        super(setup_kin_profs_depts, self).setup_kin_profs_depts()

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

        if self.namelist['ELM_model']['ELM_flag']:
            # ELMs activated --> Time-dependent reflection energy

            # The reflection energy will repeat itself over cycles as the ELM frequency
            #   --> calculate the reflection energy only for the first cycle then repeat it
            #   to save computation time
            ELM_frequency = self.namelist["ELM_model"]["ELM_frequency"]
            ELM_period = 1/ELM_frequency
            num_periods = int((self.namelist['timing']['times'][-1]-self.namelist['timing']['times'][0])/ELM_period)
            steps = int(len(self.time_out)/(num_periods))

            # reflection energies for the first ELM cycle
            E_refl_main_wall = self.E_refl_main_wall[0:steps]

            # calculate the reflection profiles for all times of the first ELM cycle
            rfl_rad_prof = np.zeros((len(self.rvol_grid), steps))

            for i in range(0,steps):

                nml_rfl_prof = copy.deepcopy(nml_rcl_prof)

                # set energy value in the reflection profile namelist
                nml_rfl_prof["imp_source_energy_eV"] = E_refl_main_wall[i]

                rfl_rad_prof[:,i] = source_utils.get_radial_source(
                    nml_rfl_prof,  # namelist specifically to obtain exp decay from wall
                    self.rvol_grid,
                    self.pro_grid,
                    Sne0,
                    self._Ti,
                )[:,0]

            # repeat the profiles throughout all the cycles in the time grid  
            self.rfl_rad_prof = np.tile(rfl_rad_prof,num_periods)

        else:
            # No ELMs --> Constant impact energy over the entire time grid

            nml_rfl_prof = copy.deepcopy(nml_rcl_prof)

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

        if self.namelist['ELM_model']['ELM_flag']:
        # ELMs activated --> Time-dependent sputtered energy

            # The sputtered energy will repeat itself over cycles as the ELM frequency
            #   --> calculate the sputtered energy only for the first cycle then repeat it
            #   to save computation time
            ELM_frequency = self.namelist["ELM_model"]["ELM_frequency"]
            ELM_period = 1/ELM_frequency
            num_periods = int((self.namelist['timing']['times'][-1]-self.namelist['timing']['times'][0])/ELM_period)
            steps = int(len(self.time_out)/(num_periods))

            # sputtered energies for the first ELM cycle from simulated and background species
            E_sput_main_wall = self.E_sput_main_wall[:,0:steps]

            # calculate the sputtering profiles from simulated and background species
            #   for all times of the first ELM cycle
            spt_rad_prof = np.zeros((len(self.rvol_grid), len(self.background_species)+1, steps))

            for i in range(0,steps):
                for j in range(0,len(self.background_species)+1):

                    nml_spt_prof = copy.deepcopy(nml_rcl_prof)

                    # set energy value in the sputtering profile namelist for the given projectile species
                    nml_spt_prof["imp_source_energy_eV"] = E_sput_main_wall[j,i]

                    spt_rad_prof[:,j,i] = source_utils.get_radial_source(
                        nml_spt_prof,  # namelist specifically to obtain exp decay from wall
                        self.rvol_grid,
                        self.pro_grid,
                        Sne0,
                        self._Ti,
                    )[:,0]

            # repeat the profiles throughout all the cycles in the time grid  
            self.spt_rad_prof = np.tile(spt_rad_prof,num_periods)

        else:
            # No ELMs --> Constant impact energy over the entire time grid

            # calculate the sputtering profiles from simulated and background species
            spt_rad_prof = np.zeros((len(self.rvol_grid), len(self.background_species)+1))

            for j in range(0,len(self.background_species)+1):

                nml_spt_prof = copy.deepcopy(nml_rcl_prof)

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


        
    def setup_PWI_model(self):
        """Method to set up Aurora inputs related to the advanced plasma-wall interaction model.
        """
        
        # effective wall surfaces, accounting for roughness
        self.surf_mainwall_eff = self.surf_mainwall*self.mainwall_roughness #cm^2
        self.surf_divwall_eff = self.surf_divwall*self.divwall_roughness #cm^2

        if self.advanced_PWI_flag: # advanced PWI model used: calculate parameters for fortran routine
        
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
            
        else:  # advanced PWI model not used: dummy input parameters for fortran routine
            
            # keys for fortran routine
            self.Z_main_wall = 0
            self.Z_div_wall = 0
            
            # List of background species
            self.num_background_species = len(self.background_species)
            
            # dummy values for reflection coefficients at each time step
            self.rn_main_wall = np.zeros(len(self.time_out))
            self.rn_div_wall = np.zeros(len(self.time_out))
            
            # dummy values for reflected energies at each time step
            self.E_refl_main_wall = np.zeros(len(self.time_out))
            self.E_refl_div_wall = np.zeros(len(self.time_out))
            
            # dummy values for background fluxes onto the walls at each time step
            self.fluxes_main_wall_background = np.zeros((len(self.background_species),len(self.time_out)))
            self.fluxes_div_wall_background = np.zeros((len(self.background_species),len(self.time_out)))
            
            # dummy values for impurity sputtering coefficients at each time step
            self.y_main_wall = np.zeros((len(self.background_species)+1,len(self.time_out)))
            self.y_div_wall = np.zeros((len(self.background_species)+1,len(self.time_out)))
            
            # dummy values for impurity sputtered energies at each time step
            self.E_sput_main_wall = np.zeros((len(self.background_species)+1,len(self.time_out)))
            self.E_sput_div_wall = np.zeros((len(self.background_species)+1,len(self.time_out)))
            
            # dummy values for impurity implantation depths into the walls
            self.implantation_depth_main_wall = 0.0
            self.implantation_depth_div_wall = 0.0
            
            # dummy values for impurity saturation densities into the walls
            self.n_main_wall_sat = 0.0
            self.n_div_wall_sat = 0.0
        


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
        #   on inter- and intra-ELM phases
                    
            if (len(self.background_species) != len(self.background_main_wall_fluxes) or len(self.background_species) != len(self.background_div_wall_fluxes)):
                raise ValueError("Declared number of background species for advanced PWI model not consistent with declared number of background fluxes!")  
                
            if not self.namelist['ELM_model']['ELM_flag']:
                # set wall fluxes of the background species to constant value over entire time grid
                
                for i in range(0,len(self.background_species)):
                    fluxes_main_wall[i,:] = np.full(len(self.time_out),self.background_main_wall_fluxes[i])
                    fluxes_div_wall[i,:] = np.full(len(self.time_out),self.background_div_wall_fluxes[i])
                
            else:
                # impose also a peak value during ELMs
                    
                    if (len(self.background_species) != len(self.background_main_wall_fluxes_ELM) or len(self.background_species) != len(self.background_div_wall_fluxes_ELM)):
                        raise ValueError("Declared number of background species for advanced PWI model not consistent with declared number of background ELM peak fluxes!")
                        
                    for i in range(0,len(self.background_species)):
                        
                        # general time-dependent shape following the ELMs onto the entire time grid
                        shape = transport_utils.ELM_cycle_shape(self.time_grid,self.namelist["ELM_model"],self.namelist['timing'])
                        max_shape = np.max(shape)
                        
                        # normalize the time-dependent shape so that its peaks value equates
                        #   the difference between flux_peak_intra_ELM and flux_inter_ELM
                        shape_norm_main = shape/(max_shape/(self.background_main_wall_fluxes_ELM[i] - self.background_main_wall_fluxes[i]))
                        shape_norm_div = shape/(max_shape/(self.background_div_wall_fluxes_ELM[i] - self.background_div_wall_fluxes[i]))

                        # now add this normalized shape to the inter-ELM value in order to achieve
                        #   background fluxes which flatten on their inter-ELM values during inter-ELM phases
                        #   but peak at wall_fluxes_ELM in the moment of maximum ELM-carried flux
                        flux_main_wall_inter_ELM = np.full(len(self.time_out),self.background_main_wall_fluxes[i])
                        fluxes_main_wall[i,:] = flux_main_wall_inter_ELM + shape_norm_main
                        flux_div_wall_inter_ELM = np.full(len(self.time_out),self.background_div_wall_fluxes[i])
                        fluxes_div_wall[i,:] = flux_div_wall_inter_ELM + shape_norm_div 

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
        
        if self.namelist['ELM_model']['ELM_flag']:
        # ELMs activated --> Time-dependent impact energy which peaks during the ELM crash
        #   whose time dependency is generated using the free streaming model
        
            # Impact energies for the simulated species itself (index 0)
            impact_energy_main_wall[0,:] = transport_utils.ELM_cycle_impact_energy_main_wall(self.imp,
                                                                                             self.time_grid,
                                                                                             self.namelist['advanced_PWI'],
                                                                                             self.namelist["ELM_model"],
                                                                                             self.namelist['timing'])
            impact_energy_div_wall[0,:] = transport_utils.ELM_cycle_impact_energy_div_wall(self.imp,
                                                                                             self.time_grid,
                                                                                             self.namelist['advanced_PWI'],
                                                                                             self.namelist["ELM_model"],
                                                                                             self.namelist['timing'])       
            
            # Impact energies for the background species (indices 1+)
            for i in range(1,len(self.background_species)+1):
                impact_energy_main_wall[i,:] = transport_utils.ELM_cycle_impact_energy_main_wall(self.background_species[i-1],
                                                                                                 self.time_grid,
                                                                                                 self.namelist['advanced_PWI'],
                                                                                                 self.namelist["ELM_model"],
                                                                                                 self.namelist['timing'])
                impact_energy_div_wall[i,:] = transport_utils.ELM_cycle_impact_energy_div_wall(self.background_species[i-1],
                                                                                                 self.time_grid,
                                                                                                 self.namelist['advanced_PWI'],
                                                                                                 self.namelist["ELM_model"],
                                                                                                 self.namelist['timing'])    
                
        else:
        # No ELMs --> Constant impact energy over the entire time grid
        
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



    def PWI_time_traces(self, interval = 0.01, plot=True, axs=None):
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

        nz = self.res['nz'].transpose(2, 1, 0)  # time,nZ,space

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
        PWI_traces["impurity_content_mainwall"] = ((self.res['N_mainret'] * circ)/(self.surf_mainwall*self.mainwall_roughness))/n_main_wall_sat
        time_average, PWI_traces["impurity_content_mainwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_content_mainwall"],interval)
        # impurity flux towards main wall
        PWI_traces["impurity_flux_mainwall"] = self.res['N_tsu'] * circ + self.res['N_dsul'] * circ
        _, PWI_traces["impurity_flux_mainwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_flux_mainwall"],interval)
        # impurity impact energy at main wall
        PWI_traces["impurity_impact_energy_mainwall"] = self.E0_main_wall_imp.T
        _, PWI_traces["impurity_impact_energy_mainwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_impact_energy_mainwall"],interval)
        # impurity reflection coefficient at main wall
        PWI_traces["impurity_reflection_coeff_mainwall"] = self.rn_main_wall
        _, PWI_traces["impurity_reflection_coeff_mainwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_reflection_coeff_mainwall"],interval)
        # impurity reflection energy at main wall
        PWI_traces["impurity_reflection_energy_mainwall"] = self.E_refl_main_wall
        _, PWI_traces["impurity_reflection_energy_mainwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_reflection_energy_mainwall"],interval)
        # fluxes of background species
        PWI_traces["background_fluxes_mainwall"] = self.fluxes_main_wall_background.T
        PWI_traces["background_fluxes_mainwall_av"] = np.zeros((len(time_average),len(background_species)))
        for i in range(0,len(background_species)):
            _, PWI_traces["background_fluxes_mainwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["background_fluxes_mainwall"][:,i],interval)
        # impact energies of background species at main wall
        PWI_traces["background_impact_energies_mainwall"] = self.E0_main_wall_background.T
        PWI_traces["background_impact_energies_mainwall_av"] = np.zeros((len(time_average),len(background_species)))
        for i in range(0,len(background_species)):
            _, PWI_traces["background_impact_energies_mainwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["background_impact_energies_mainwall"][:,i],interval)
        # impurity sputtering yields at main wall
        PWI_traces["impurity_sputtering_yields_mainwall"] = self.y_main_wall.T
        PWI_traces["impurity_sputtering_yields_mainwall_av"] = np.zeros((len(time_average),len(background_species)+1))
        for i in range(0,len(background_species)+1):
            _, PWI_traces["impurity_sputtering_yields_mainwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_sputtering_yields_mainwall"][:,i],interval)
        # impurity sputtering energies at main wall
        PWI_traces["impurity_sputtering_energies_mainwall"] = self.E_sput_main_wall.T
        PWI_traces["impurity_sputtering_energies_mainwall_av"] = np.zeros((len(time_average),len(background_species)+1))
        for i in range(0,len(background_species)+1):
            _, PWI_traces["impurity_sputtering_energies_mainwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_sputtering_energies_mainwall"][:,i],interval)
        # impurity reflection rate from main wall
        PWI_traces["impurity_reflection_rate_mainwall"] = self.res['rclw_refl_rate'] * circ
        _, PWI_traces["impurity_reflection_rate_mainwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_reflection_rate_mainwall"],interval) 
        # impurity prompt recycling rate from main wall
        PWI_traces["impurity_prompt_recycling_rate_mainwall"] = self.res['rclw_recl_rate'] * circ
        _, PWI_traces["impurity_prompt_recycling_rate_mainwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_prompt_recycling_rate_mainwall"],interval) 
        # impurity implantation rate into main wall
        PWI_traces["impurity_implantation_rate_mainwall"] = self.res['rclw_impl_rate'] * circ
        _, PWI_traces["impurity_implantation_rate_mainwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_implantation_rate_mainwall"],interval) 
        # impurity release rate from main wall
        PWI_traces["impurity_sputtering_rates_mainwall"] = np.zeros((len(self.time_out),len(background_species)+1))
        PWI_traces["impurity_sputtering_rates_mainwall_av"] = np.zeros((len(time_average),len(background_species)+1))
        for i in range(0,len(background_species)+1):
            PWI_traces["impurity_sputtering_rates_mainwall"][:,i] = self.res['rclw_sput_rate'][i,:] * circ
            _, PWI_traces["impurity_sputtering_rates_mainwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_sputtering_rates_mainwall"][:,i],interval) 
        PWI_traces["impurity_sputtering_rate_total_mainwall"] = PWI_traces["impurity_sputtering_rates_mainwall"].sum(axis=1)
        _, PWI_traces["impurity_sputtering_rate_total_mainwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_sputtering_rate_total_mainwall"],interval) 

        # divertor wall
        
        # divertor wall impurity content
        PWI_traces["impurity_content_divwall"] = ((self.res['N_divret'] * circ)/(self.surf_divwall*self.divwall_roughness))/n_div_wall_sat
        _, PWI_traces["impurity_content_divwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_content_divwall"],interval) 
        # impurity flux towards divertor wall
        PWI_traces["impurity_flux_divwall"] = (self.res['N_dsu'] * circ + self.res['rcls_rate'] * circ)*(1-self.div_recomb_ratio)
        _, PWI_traces["impurity_flux_divwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_flux_divwall"],interval) 
        # impurity impact energy at divertor wall
        PWI_traces["impurity_impact_energy_divwall"] = self.E0_div_wall_imp.T
        _, PWI_traces["impurity_impact_energy_divwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_impact_energy_divwall"],interval) 
        # impurity reflection coefficient at divertor wall
        PWI_traces["impurity_reflection_coeff_divwall"] = self.rn_div_wall
        _, PWI_traces["impurity_reflection_coeff_divwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_reflection_coeff_divwall"],interval)
        # impurity reflection energy at divertor wall
        PWI_traces["impurity_reflection_energy_divwall"] = self.E_refl_div_wall
        _, PWI_traces["impurity_reflection_energy_divwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_reflection_energy_divwall"],interval)
        # fluxes of background species
        PWI_traces["background_fluxes_divwall"] = self.fluxes_div_wall_background.T
        PWI_traces["background_fluxes_divwall_av"] = np.zeros((len(time_average),len(background_species)))
        for i in range(0,len(background_species)):
            _, PWI_traces["background_fluxes_divwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["background_fluxes_divwall"][:,i],interval)
        # impact energies of background species at divertor wall
        PWI_traces["background_impact_energies_divwall"] = self.E0_div_wall_background.T
        PWI_traces["background_impact_energies_divwall_av"] = np.zeros((len(time_average),len(background_species)))
        for i in range(0,len(background_species)):
            _, PWI_traces["background_impact_energies_divwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["background_impact_energies_divwall"][:,i],interval)
        # impurity sputtering yields at divertor wall
        PWI_traces["impurity_sputtering_yields_divwall"] = self.y_div_wall.T
        PWI_traces["impurity_sputtering_yields_divwall_av"] = np.zeros((len(time_average),len(background_species)+1))
        for i in range(0,len(background_species)+1):
            _, PWI_traces["impurity_sputtering_yields_divwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_sputtering_yields_divwall"][:,i],interval)
        # impurity sputtering energies at divertor wall
        PWI_traces["impurity_sputtering_energies_divwall"] = self.E_sput_div_wall.T
        PWI_traces["impurity_sputtering_energies_divwall_av"] = np.zeros((len(time_average),len(background_species)+1))
        for i in range(0,len(background_species)+1):
            _, PWI_traces["impurity_sputtering_energies_divwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_sputtering_energies_divwall"][:,i],interval)    
        # impurity reflection rate from divertor wall
        PWI_traces["impurity_reflection_rate_divwall"] = self.res['rcld_refl_rate'] * circ
        _, PWI_traces["impurity_reflection_rate_divwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_reflection_rate_divwall"],interval) 
        # impurity prompt recycling rate from divertor wall
        PWI_traces["impurity_prompt_recycling_rate_divwall"] = self.res['rcld_recl_rate'] * circ
        _, PWI_traces["impurity_prompt_recycling_rate_divwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_prompt_recycling_rate_divwall"],interval) 
        # impurity implantation rate into divertor wall
        PWI_traces["impurity_implantation_rate_divwall"] = self.res['rcld_impl_rate'] * circ
        _, PWI_traces["impurity_implantation_rate_divwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_implantation_rate_divwall"],interval) 
        # impurity release rate from divertor wall
        PWI_traces["impurity_sputtering_rates_divwall"] = np.zeros((len(self.time_out),len(background_species)+1))
        PWI_traces["impurity_sputtering_rates_divwall_av"] = np.zeros((len(time_average),len(background_species)+1))
        for i in range(0,len(background_species)+1):
            PWI_traces["impurity_sputtering_rates_divwall"][:,i] = self.res['rcld_sput_rate'][i,:] * circ
            _, PWI_traces["impurity_sputtering_rates_divwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_sputtering_rates_divwall"][:,i],interval) 
        PWI_traces["impurity_sputtering_rate_total_divwall"] = PWI_traces["impurity_sputtering_rates_divwall"].sum(axis=1)
        _, PWI_traces["impurity_sputtering_rate_total_divwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_sputtering_rate_total_divwall"],interval) 

        if plot:
            # -------------------------------------------------
            # plot time histories for the main wall:
            if axs is None:
                fig, ax1 = plt.subplots(nrows=3, ncols=5, sharex=True, figsize=(22, 12))
            else:
                ax1 = axs[0]
                
            fig.suptitle('Plasma - main wall interaction time traces',fontsize=18)

            ax1[0, 0].plot(self.time_out, PWI_traces["impurity_content_mainwall"], color = grey)
            ax1[0, 0].set_ylabel(f'$C_{{{self.imp}}}$/$C_{{{self.imp},sat}}$')
            ax1[0, 0].set_title(f"Retained {self.imp} content into wall", loc='right', fontsize = 11)

            ax1[0, 1].plot(self.time_out, PWI_traces["impurity_flux_mainwall"], color = light_blues[0])
            ax1[0, 1].set_ylabel('[$s^{-1}$]')
            ax1[0, 1].set_title(f"{self.imp} flux towards wall", loc='right', fontsize = 11)

            ax1[0, 2].plot(self.time_out, PWI_traces["impurity_impact_energy_mainwall"], color = reds[0])
            ax1[0, 2].set_ylabel('[eV]') 
            ax1[0, 2].set_title(f"Mean {self.imp} impact energy", loc='right', fontsize = 11)

            ax1[0, 3].plot(self.time_out, PWI_traces["impurity_reflection_coeff_mainwall"], color = blues[0])
            ax1[0, 3].set_ylabel('$R_N$')  
            ax1[0, 3].set_title(f"Mean {self.imp} reflection coeff. on {self.main_wall_material}", loc='right', fontsize = 11)
            
            ax1[0, 4].plot(self.time_out, PWI_traces["impurity_reflection_energy_mainwall"], color = reds[0])
            ax1[0, 4].set_ylabel('[eV]') 
            ax1[0, 4].set_title(f"Mean energy of reflected {self.imp}", loc='right', fontsize = 11)

            for i in range(0,len(background_species)):
                ax1[1, 1].plot(self.time_out, PWI_traces["background_fluxes_mainwall"][:,i], label=f"{background_species[i]} flux", color = light_blues[i+1])
            ax1[1, 1].set_ylabel('[$s^{-1}$]')
            ax1[1, 1].set_title("Background fluxes towards wall", loc='right', fontsize = 11)
            ax1[1, 1].legend(loc="best", fontsize = 9).set_draggable(True)

            for i in range(0,len(background_species)):
                ax1[1, 2].plot(self.time_out, PWI_traces["background_impact_energies_mainwall"][:,i], label=f"{background_species[i]}", color = reds[i+1])
            ax1[1, 2].set_ylabel('[eV]')
            ax1[1, 2].set_title("Mean background fluxes impact energy", loc='right', fontsize = 11)
            ax1[1, 2].legend(loc="best", fontsize = 9).set_draggable(True)

            ax1[1, 3].plot(self.time_out, PWI_traces["impurity_sputtering_yields_mainwall"][:,0], label=f"from {self.imp}", color = blues[0])
            # if self.namelist['ELM_model']['ELM_flag']:
            #     ax1[1, 3].plot(time_average, PWI_traces["impurity_sputtering_yields_mainwall_av"][:,0], linestyle = 'dashed', color = 'white')
            for i in range(0,len(background_species)):
                ax1[1, 3].plot(self.time_out, PWI_traces["impurity_sputtering_yields_mainwall"][:,i+1], label=f"from {background_species[i]}", color = blues[i+1])
                # if self.namelist['ELM_model']['ELM_flag']:
                #     ax1[1, 3].plot(time_average, PWI_traces["impurity_sputtering_yields_mainwall_av"][:,i+1], linestyle = 'dashed', color = 'white')
            ax1[1, 3].set_ylabel(f'$Y_{{{self.imp}}}$/$C_{{{self.imp},wall}}$')
            ax1[1, 3].set_title(f"Mean {self.imp} sputtering yields from {self.main_wall_material}", loc='right', fontsize = 11)
            ax1[1, 3].legend(loc="best", fontsize = 9).set_draggable(True)
            
            ax1[1, 4].plot(self.time_out, PWI_traces["impurity_sputtering_energies_mainwall"][:,0], label=f"from {self.imp}", color = reds[0])
            for i in range(0,len(background_species)):
                ax1[1, 4].plot(self.time_out, PWI_traces["impurity_sputtering_energies_mainwall"][:,i+1], label=f"from {background_species[i]}", color = reds[i+1])
            ax1[1, 4].set_ylabel('[eV]')
            ax1[1, 4].set_title(f"Mean energy of sputtered {self.imp}", loc='right', fontsize = 11)
            ax1[1, 4].legend(loc="best", fontsize = 9).set_draggable(True)   
            
            ax1[2, 0].plot(self.time_out, PWI_traces["impurity_reflection_rate_mainwall"], color = blues[0])
            ax1[2, 0].set_ylabel('[$s^{-1}$]')
            ax1[2, 0].set_title(f"{self.imp} reflection rate from wall", loc='right', fontsize = 11)

            ax1[2, 1].plot(self.time_out, PWI_traces["impurity_prompt_recycling_rate_mainwall"], color = blues[0])
            ax1[2, 1].set_ylabel('[$s^{-1}$]')
            ax1[2, 1].set_title(f"{self.imp} prompt recycling rate from wall", loc='right', fontsize = 11)
            
            ax1[2, 2].plot(self.time_out, PWI_traces["impurity_implantation_rate_mainwall"], color = blues[0])
            ax1[2, 2].set_ylabel('[$s^{-1}$]')
            ax1[2, 2].set_title(f"{self.imp} implantation rate into wall", loc='right', fontsize = 11)
            
            ax1[2, 3].plot(self.time_out, PWI_traces["impurity_sputtering_rates_mainwall"][:,0], label=f"from {self.imp}", color = blues[0])
            for i in range(0,len(background_species)):
                ax1[2, 3].plot(self.time_out, PWI_traces["impurity_sputtering_rates_mainwall"][:,i+1], label=f"from {background_species[i]}", color = blues[i+1])
            ax1[2, 3].set_ylabel('[$s^{-1}$]')
            ax1[2, 3].set_title(f"{self.imp} sputtering rates from wall", loc='right', fontsize = 11)
            ax1[2, 3].legend(loc="best", fontsize = 9).set_draggable(True)
            
            ax1[2, 4].plot(self.time_out, PWI_traces["impurity_implantation_rate_mainwall"], label="Total absorption rate", color = greens[0])
            ax1[2, 4].plot(self.time_out, PWI_traces["impurity_sputtering_rate_total_mainwall"], label="Total release rate", color = blues[0])
            ax1[2, 4].plot(self.time_out, PWI_traces["impurity_implantation_rate_mainwall"]-PWI_traces["impurity_sputtering_rate_total_mainwall"], label="Balance", color = 'black')
            ax1[2, 4].set_ylabel('[$s^{-1}$]')
            ax1[2, 4].set_title(f"{self.imp} wall balance", loc='right', fontsize = 11)
            ax1[2, 4].legend(loc="best", fontsize = 9).set_draggable(True)

            for ii in [0, 1, 2, 3, 4]:
                ax1[2, ii].set_xlabel("Time [s]")
            ax1[2, 0].set_xlim(self.time_out[[0, -1]])
            
            plt.tight_layout()
            
            # -------------------------------------------------
            # plot time histories for the divertor wall:
            if axs is None:
                fig, ax2 = plt.subplots(nrows=3, ncols=5, sharex=True, figsize=(22, 12))
            else:
                ax2 = axs[0]
                
            fig.suptitle('Plasma - divertor wall interaction time traces',fontsize=18)

            ax2[0, 0].plot(self.time_out, PWI_traces["impurity_content_divwall"], color = grey)
            ax2[0, 0].set_ylabel(f'$C_{{{self.imp}}}$/$C_{{{self.imp},sat}}$')
            ax2[0, 0].set_title(f"Retained {self.imp} content into wall", loc='right', fontsize = 11)

            ax2[0, 1].plot(self.time_out, PWI_traces["impurity_flux_divwall"], color = light_blues[0])
            ax2[0, 1].set_ylabel('[$s^{-1}$]')
            ax2[0, 1].set_title(f"{self.imp} flux towards wall", loc='right', fontsize = 11)

            ax2[0, 2].plot(self.time_out, PWI_traces["impurity_impact_energy_divwall"], color = reds[0])
            ax2[0, 2].set_ylabel('[eV]')
            ax2[0, 2].set_title(f"Mean {self.imp} impact energy", loc='right', fontsize = 11)

            ax2[0, 3].plot(self.time_out, PWI_traces["impurity_reflection_coeff_divwall"], color = blues[0])
            ax2[0, 3].set_ylabel('$R_N$')  
            ax2[0, 3].set_title(f"Mean {self.imp} reflection coeff. on {self.div_wall_material}", loc='right', fontsize = 11)
            
            ax2[0, 4].plot(self.time_out, PWI_traces["impurity_reflection_energy_divwall"], color = reds[0])
            ax2[0, 4].set_ylabel('[eV]') 
            ax2[0, 4].set_title(f"Mean energy of reflected {self.imp}", loc='right', fontsize = 11)

            for i in range(0,len(background_species)):
                ax2[1, 1].plot(self.time_out, PWI_traces["background_fluxes_divwall"][:,i], label=f"{background_species[i]} flux", color = light_blues[i+1])
            ax2[1, 1].set_ylabel('[$s^{-1}$]')
            ax2[1, 1].set_title("Background fluxes towards wall", loc='right', fontsize = 11)
            ax2[1, 1].legend(loc="best", fontsize = 9).set_draggable(True)

            for i in range(0,len(background_species)):
                ax2[1, 2].plot(self.time_out, PWI_traces["background_impact_energies_divwall"][:,i], label=f"{background_species[i]}", color = reds[i+1])
            ax2[1, 2].set_ylabel('[eV]')
            ax2[1, 2].set_title("Mean background fluxes impact energy", loc='right', fontsize = 11)
            ax2[1, 2].legend(loc="best", fontsize = 9).set_draggable(True)

            ax2[1, 3].plot(self.time_out, PWI_traces["impurity_sputtering_yields_divwall"][:,0], label=f"from {self.imp}", color = blues[0])
            for i in range(0,len(background_species)):
                ax2[1, 3].plot(self.time_out, PWI_traces["impurity_sputtering_yields_divwall"][:,i+1], label=f"from {background_species[i]}", color = blues[i+1])
            ax2[1, 3].set_ylabel(f'$Y_{{{self.imp}}}$/$C_{{{self.imp},wall}}$')
            ax2[1, 3].set_title(f"Mean {self.imp} sputtering yields from {self.div_wall_material}", loc='right', fontsize = 11)
            ax2[1, 3].legend(loc="best", fontsize = 9).set_draggable(True)

            ax2[1, 4].plot(self.time_out, PWI_traces["impurity_sputtering_energies_divwall"][:,0], label=f"from {self.imp}", color = reds[0])
            for i in range(0,len(background_species)):
                ax2[1, 4].plot(self.time_out, PWI_traces["impurity_sputtering_energies_divwall"][:,i+1], label=f"from {background_species[i]}", color = reds[i+1])
            ax2[1, 4].set_ylabel('[eV]')
            ax2[1, 4].set_title(f"Mean energy of sputtered {self.imp}", loc='right', fontsize = 11)
            ax2[1, 4].legend(loc="best", fontsize = 9).set_draggable(True)  

            ax2[2, 0].plot(self.time_out, PWI_traces["impurity_reflection_rate_divwall"], color = blues[0])
            ax2[2, 0].set_ylabel('[$s^{-1}$]')
            ax2[2, 0].set_title(f"{self.imp} reflection rate from wall", loc='right', fontsize = 11)

            ax2[2, 1].plot(self.time_out, PWI_traces["impurity_prompt_recycling_rate_divwall"], color = blues[0])
            ax2[2, 1].set_ylabel('[$s^{-1}$]')
            ax2[2, 1].set_title(f"{self.imp} prompt recycling rate from wall", loc='right', fontsize = 11)
            
            ax2[2, 2].plot(self.time_out, PWI_traces["impurity_implantation_rate_divwall"], color = blues[0])
            ax2[2, 2].set_ylabel('[$s^{-1}$]')
            ax2[2, 2].set_title(f"{self.imp} implantation rate into wall", loc='right', fontsize = 11)
            
            ax2[2, 3].plot(self.time_out, PWI_traces["impurity_sputtering_rates_divwall"][:,0], label=f"from {self.imp}", color = blues[0])
            for i in range(0,len(background_species)):
                ax2[2, 3].plot(self.time_out, PWI_traces["impurity_sputtering_rates_divwall"][:,i+1], label=f"from {background_species[i]}", color = blues[i+1])
            ax2[2, 3].set_ylabel('[$s^{-1}$]')
            ax2[2, 3].set_title(f"{self.imp} sputtering rates from wall", loc='right', fontsize = 11)
            ax2[2, 3].legend(loc="best", fontsize = 9).set_draggable(True)
            
            ax2[2, 4].plot(self.time_out, PWI_traces["impurity_implantation_rate_divwall"], label="Total absorption rate", color = greens[0])
            ax2[2, 4].plot(self.time_out, PWI_traces["impurity_sputtering_rate_total_divwall"], label="Total release rate", color = blues[0])
            ax2[2, 4].plot(self.time_out, PWI_traces["impurity_implantation_rate_divwall"]-PWI_traces["impurity_sputtering_rate_total_divwall"], label="Balance", color = 'black')
            ax2[2, 4].set_ylabel('[$s^{-1}$]')
            ax2[2, 4].set_title(f"{self.imp} wall balance", loc='right', fontsize = 11)
            ax2[2, 4].legend(loc="best", fontsize = 9).set_draggable(True)

            for ii in [0, 1, 2, 3, 4]:
                ax2[2, ii].set_xlabel("Time [s]")
            ax2[2, 0].set_xlim(self.time_out[[0, -1]])  
            
            plt.tight_layout()

        if plot:
            return PWI_traces, (ax1, ax2)
        else:
            return PWI_traces             
    

