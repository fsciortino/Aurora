"""Extension of the core class :py:class:`~aurora.core.aurora_sim` including the full model for Plasma Wall Interface (PWI) physics.
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

#TODO add paper references to this PWI model

import numpy as np
from scipy.constants import e as q_electron, m_p
import pandas as pd
import matplotlib.pyplot as plt
from . import source_utils
from . import transport_utils
from . import plot_tools
from . import surface
from . import radiation
from . import core

class aurora_sim_pwi(core.aurora_sim):
    """Class to setup, run, and post-process 1.5D simulations of particle/impurity transport in
    magnetically-confined plasmas with the inclusion of the full plasma-wall interaction (PWI) model.
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

        #PWI model assumes that the particle source is at the wall. 
        # set the energy of the recycled neutrals
        namelist["imp_source_energy_eV"] = namelist["imp_recycling_energy_eV"]
        
        # set start of the recycling source at the wall boundary
        namelist["source_cm_out_lcfs"] = namelist["bound_sep"]

        
        # inheritate all methods from main class core.py
        super(aurora_sim_pwi, self).__init__(namelist, geqdsk)
        
        # set up parameters for plasma-wall interaction model
        self.setup_PWI_model()
        
        # consistency checks
        if not self.phys_surfaces:
            raise ValueError("Implementing the full PWI model requires defining the physical wall surface areas!") 
      
        # recycling profiles for neutrals
        self.setup_rec_profs()
        
        # activate full PWI model in fortran main routines
        self.full_PWI_flag = True

    
    
    def setup_PWI_model(self):
        """Method to set up Aurora inputs related to the full plasma-wall interaction model.
        """
        
        # effective wall surfaces, accounting for roughness
        self.surf_mainwall_eff = self.surf_mainwall*self.mainwall_roughness #cm^2
        self.surf_divwall_eff = self.surf_divwall*self.divwall_roughness #cm^2

        # Data for full PWI model for now available only for He as impurity and W as wall material
       
        if self.imp != 'He':
            raise ValueError(f"Full PWI model not available for impurity {self.imp}!")
        
        # keys for fortran routine
        if self.main_wall_material == 'W':
            self.Z_main_wall = 74
        else:
            raise ValueError(f"Full PWI model not available for main wall material {self.main_wall_material}!")
        if self.div_wall_material == 'W':
            self.Z_div_wall = 74
        else:
            raise ValueError(f"Full PWI model not available for divertor wall material {self.div_wall_material}!")
            
        # List of background species
        self.background_species = list(self.background_species)
        self.num_background_species = len(self.background_species)
        self.all_species = [self.imp] + self.background_species
        
        # calculate impact energies for simulated impurity and background species
        #   onto main and divertor walls at each time step
        self.energies_main_wall, self.energies_div_wall = self.get_impact_energy()
        self.E0_main_wall_imp = self.energies_main_wall[0,:] # Index 0 --> simulated species
        self.E0_div_wall_imp = self.energies_div_wall[0,:] # Index 0 --> simulated species
        self.E0_main_wall_background = self.energies_main_wall[1:,:] # Indices > 1 --> background species
        self.E0_div_wall_background = self.energies_div_wall[1:,:] # Indices > 1 --> background species
        
        # get angles of incidence
        angle_imp = surface.incidence_angle(self.imp)

        
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
        
        
        # extract the depth of the impurity implantation profile into main and divertor walls
        self.implantation_depth_main_wall = surface.implantation_depth_fit(self.imp,
                                                              self.main_wall_material,
                                                              angle_imp,
                                                              energies = self.characteristic_impact_energy_main_wall)["data"]
        self.implantation_depth_div_wall = surface.implantation_depth_fit(self.imp,
                                                             self.div_wall_material,
                                                             angle_imp,
                                                             energies = self.characteristic_impact_energy_div_wall)["data"]
        
        
        
        # extract values for fluxes of background species onto main and divertor walls at each time step
        self.fluxes_main_wall_background, self.fluxes_div_wall_background = self.get_background_fluxes()
        
        # extract values for impurity sputtering coefficient for the wall retention of the
        #   simulated impurity at each time step
        self.y_main_wall = np.zeros((len(self.all_species),len(self.time_out)))
        self.y_div_wall  = np.zeros((len(self.all_species),len(self.time_out)))
        
        # extract values for the impurity sputtered energy at each time step
        self.E_sput_main_wall = np.zeros((len(self.all_species),len(self.time_out)))
        self.E_sput_div_wall  = np.zeros((len(self.all_species),len(self.time_out)))
        
        for i, s in enumerate(self.all_species):
            # get angles of incidence
            angle = surface.incidence_angle(s)
            
            self.y_main_wall[i] = surface.impurity_sputtering_coeff_fit(self.imp, 
                                                                s, # simulated impurity or background species as projectiles
                                                                self.main_wall_material,
                                                                angle,
                                                                energies = self.energies_main_wall[i])["normalized_data"]
    
            self.y_div_wall[i] = surface.impurity_sputtering_coeff_fit(self.imp,  
                                                                s, # simulated impurity or background species as projectiles
                                                                self.div_wall_material,
                                                                angle,
                                                                energies = self.energies_div_wall[i])["normalized_data"]
                
            self.E_sput_main_wall[i] = surface.calc_imp_sputtered_energy(self.imp,  
                                                                s, # simulated impurity or background species as projectiles
                                                                self.main_wall_material,
                                                                energies = self.energies_main_wall[i])["data"]
            self.E_sput_div_wall[i] = surface.calc_imp_sputtered_energy(self.imp,  
                                                                s, # simulated impurity or background species as projectiles
                                                                self.div_wall_material,
                                                                energies = self.energies_div_wall[i])["data"]
    

        
        # extract saturation value of the density of the implanted impurity into main and divertor walls 
        self.n_main_wall_sat = self.namelist['full_PWI']['n_main_wall_sat'] # m^-2
        self.n_div_wall_sat = self.namelist['full_PWI']['n_div_wall_sat'] # m^-2   
        
        
    def get_background_fluxes(self):
        """Get the fluxes hitting main and divertor walls (for the background species), at all time
        steps, for a simulation run, in the framework of the full PWI model.

        Returns
        -------
        fluxes_main_wall : array (number of background species,time)
                                  Particle fluxes of the background species onto the
                                  main wall at each time of the simulation, in s^-1
        fluxes_div_wall :  array (number of background species,time)
                                  Particle fluxes of the background species onto the
                                  divertor wall at each time of the simulation, in s^-1
        
        """
        
        fluxes_main_wall = np.zeros((self.num_background_species,len(self.time_out)))
        fluxes_div_wall  = np.zeros((self.num_background_species,len(self.time_out)))
        
        if self.background_mode == 'manual':
        # manually imposed fluxes of all the background species over the entire time grid
                    
            if not (self.num_background_species == len(self.background_main_wall_fluxes) == len(self.background_div_wall_fluxes)):
                raise ValueError("Declared number of background species for full PWI model not consistent with declared number of background fluxes!")  
         
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
            
            if self.num_background_species != len(self.background_files):
                raise ValueError("Declared number of background species for full PWI model not consistent with declared number of background simulation files!")       
            
            # Wall fluxes for the background species
            for i in range(0,self.num_background_species):
                
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
        of the full PWI model.

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
                                                                                             self.namelist['full_PWI'],
                                                                                             self.namelist["ELM_model"],
                                                                                             self.namelist['timing'])
            impact_energy_div_wall[0,:] = transport_utils.ELM_cycle_impact_energy_div_wall(self.imp,
                                                                                             self.time_grid,
                                                                                             self.namelist['full_PWI'],
                                                                                             self.namelist["ELM_model"],
                                                                                             self.namelist['timing'])       
            
            # Impact energies for the background species (indices 1+)
            for i in range(1,len(self.background_species)+1):
                impact_energy_main_wall[i,:] = transport_utils.ELM_cycle_impact_energy_main_wall(self.background_species[i-1],
                                                                                                 self.time_grid,
                                                                                                 self.namelist['full_PWI'],
                                                                                                 self.namelist["ELM_model"],
                                                                                                 self.namelist['timing'])
                impact_energy_div_wall[i,:] = transport_utils.ELM_cycle_impact_energy_div_wall(self.background_species[i-1],
                                                                                                 self.time_grid,
                                                                                                 self.namelist['full_PWI'],
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
        
        
    def setup_rec_profs(self):
        """Method to set up Aurora inputs related to the recycling profiles for neutrals released from the main wall.
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
        
        # Full PWI model: set recycling profiles to exponentially decay from the wall
        #   with recycled neutrals having different energies depending on whether they
        #   were reflected, promptly recycled or sputtered from the wall
              
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
                
                nml_rfl_prof = {
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
                nml_rfl_prof["source_width_in"] = 0
                nml_rfl_prof["source_width_out"] = 0
                
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
        
            nml_rfl_prof = {
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
            nml_rfl_prof["source_width_in"] = 0
            nml_rfl_prof["source_width_out"] = 0
            
            # set energy value in the reflection profile namelist
            nml_rfl_prof["imp_source_energy_eV"] = self.E_refl_main_wall[0]
            
            # set start of the recycling source at the wall boundary
            nml_rfl_prof["source_cm_out_lcfs"] = self.namelist["bound_sep"]
            
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
                
                    nml_spt_prof = {
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
                    nml_spt_prof["source_width_in"] = 0
                    nml_spt_prof["source_width_out"] = 0
                
                    # set energy value in the sputtering profile namelist for the given projectile species
                    nml_spt_prof["imp_source_energy_eV"] = E_sput_main_wall[j,i]
                    
                    # set start of the recycling source at the wall boundary
                    nml_spt_prof["source_cm_out_lcfs"] = self.namelist["bound_sep"]
            
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
            
                nml_spt_prof = {
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
                nml_spt_prof["source_width_in"] = 0
                nml_spt_prof["source_width_out"] = 0
                
                # set energy value in the reflection profile namelist
                nml_spt_prof["imp_source_energy_eV"] = self.E_sput_main_wall[j,0]
                
                # set start of the recycling source at the wall boundary
                nml_spt_prof["source_cm_out_lcfs"] = self.namelist["bound_sep"]
        
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
        plot_average = False,
        interval = 0.01,
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
        plot_average : bool, optional
            If True, plot density for each charge state averaged over ELM cycles using a convenient slide
            over time and check particle conservation in each particle reservoir averaged over ELM cycles.
        interval : float, optional
            Duration of time cycles to plot if plot_average is True, in s
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
        
        # activate full PWI model in fortran main routines
        self.full_PWI_flag = True

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
            
            if self.namelist['ELM_model']['ELM_flag'] and plot_average:
                
                time_average, data_average_profiles = plot_tools.time_average_profiles(self.namelist['timing'], self.time_out, self.res['nz'], interval = interval)
                
                # plot charge state density distributions over radius and time, averaged over cycles
                plot_tools.slider_plot(
                    x,
                    time_average,
                    data_average_profiles.transpose(1, 0, 2),
                    xlabel=xlabel,
                    ylabel="time [s]",
                    zlabel=f'$n_{{{self.imp}}}$ [$\mathrm{{cm}}$$^{{-3}}$]',
                    plot_title = f'{self.imp} density profiles (averaged over ELM cycles)',
                    labels=[str(i) for i in np.arange(0, self.res['nz'].shape[1])],
                    plot_sum=True,
                    x_line=x_line,
                    zlim = True,
                )
                
                print('Simulated ELM-averaged impurity density profiles slider plot prepared.')
                
                # Plot reservoirs and particle conservation, averaged over cycles
                _ = self.reservoirs_average_time_traces(interval = interval,plot = True)
                
                print('Simulated reservoirs ELM-averaged time traces plots prepared.')
            
        # Plot PWI model
        if plot_PWI:
            _ = self.PWI_time_traces(interval = interval,plot = True)

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
        

    def PWI_time_traces(self, interval = 0.01, plot=True, ylim = True, axs=None):
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
        # impurity flux towards main wall
        PWI_traces["impurity_flux_mainwall"] = self.res['N_tsu'] * circ + self.res['N_dsul'] * circ
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
        PWI_traces["impurity_reflection_rate_mainwall"] = self.res['rclw_refl_rate'] * circ
        # impurity prompt recycling rate from main wall
        PWI_traces["impurity_prompt_recycling_rate_mainwall"] = self.res['rclw_recl_rate'] * circ
        # impurity implantation rate into main wall
        PWI_traces["impurity_implantation_rate_mainwall"] = self.res['rclw_impl_rate'] * circ
        # impurity release rate from main wall
        PWI_traces["impurity_sputtering_rates_mainwall"] = np.zeros((len(self.time_out),len(background_species)+1))
        for i in range(0,len(background_species)+1):
            PWI_traces["impurity_sputtering_rates_mainwall"][:,i] = self.res['rclw_sput_rate'][i,:] * circ
        PWI_traces["impurity_sputtering_rate_total_mainwall"] = PWI_traces["impurity_sputtering_rates_mainwall"].sum(axis=1)

        # main wall, averaged
        
        # main wall impurity content
        time_average, PWI_traces["impurity_content_mainwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_content_mainwall"],interval)
        # impurity flux towards main wall
        _, PWI_traces["impurity_flux_mainwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_flux_mainwall"],interval)
        # impurity impact energy at main wall
        _, PWI_traces["impurity_impact_energy_mainwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_impact_energy_mainwall"],interval)
        # impurity reflection coefficient at main wall
        _, PWI_traces["impurity_reflection_coeff_mainwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_reflection_coeff_mainwall"],interval)
        # impurity reflection energy at main wall
        _, PWI_traces["impurity_reflection_energy_mainwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_reflection_energy_mainwall"],interval)
        # fluxes of background species
        PWI_traces["background_fluxes_mainwall_av"] = np.zeros((len(time_average),len(background_species)))
        for i in range(0,len(background_species)):
            _, PWI_traces["background_fluxes_mainwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["background_fluxes_mainwall"][:,i],interval)
        # impact energies of background species at main wall
        PWI_traces["background_impact_energies_mainwall_av"] = np.zeros((len(time_average),len(background_species)))
        for i in range(0,len(background_species)):
            _, PWI_traces["background_impact_energies_mainwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["background_impact_energies_mainwall"][:,i],interval)
        # impurity sputtering yields at main wall
        PWI_traces["impurity_sputtering_yields_mainwall_av"] = np.zeros((len(time_average),len(background_species)+1))
        for i in range(0,len(background_species)+1):
            _, PWI_traces["impurity_sputtering_yields_mainwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_sputtering_yields_mainwall"][:,i],interval)
        # impurity sputtering energies at main wall
        PWI_traces["impurity_sputtering_energies_mainwall_av"] = np.zeros((len(time_average),len(background_species)+1))
        for i in range(0,len(background_species)+1):
            _, PWI_traces["impurity_sputtering_energies_mainwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_sputtering_energies_mainwall"][:,i],interval)
        # impurity reflection rate from main wall
        _, PWI_traces["impurity_reflection_rate_mainwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_reflection_rate_mainwall"],interval) 
        # impurity prompt recycling rate from main wall
        _, PWI_traces["impurity_prompt_recycling_rate_mainwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_prompt_recycling_rate_mainwall"],interval) 
        # impurity implantation rate into main wall
        _, PWI_traces["impurity_implantation_rate_mainwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_implantation_rate_mainwall"],interval) 
        # impurity release rate from main wall
        PWI_traces["impurity_sputtering_rates_mainwall_av"] = np.zeros((len(time_average),len(background_species)+1))
        for i in range(0,len(background_species)+1):
            _, PWI_traces["impurity_sputtering_rates_mainwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_sputtering_rates_mainwall"][:,i],interval) 
        _, PWI_traces["impurity_sputtering_rate_total_mainwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_sputtering_rate_total_mainwall"],interval) 

        # divertor wall
        
        # divertor wall impurity content
        PWI_traces["impurity_content_divwall"] = ((self.res['N_divret'] * circ)/(self.surf_divwall*self.divwall_roughness))/n_div_wall_sat
        # impurity flux towards divertor wall
        PWI_traces["impurity_flux_divwall"] = (self.res['N_dsu'] * circ + self.res['rcls_rate'] * circ)*(1-self.div_recomb_ratio)
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
        PWI_traces["impurity_reflection_rate_divwall"] = self.res['rcld_refl_rate'] * circ
        # impurity prompt recycling rate from divertor wall
        PWI_traces["impurity_prompt_recycling_rate_divwall"] = self.res['rcld_recl_rate'] * circ
        # impurity implantation rate into divertor wall
        PWI_traces["impurity_implantation_rate_divwall"] = self.res['rcld_impl_rate'] * circ
        # impurity release rate from divertor wall
        PWI_traces["impurity_sputtering_rates_divwall"] = np.zeros((len(self.time_out),len(background_species)+1))
        for i in range(0,len(background_species)+1):
            PWI_traces["impurity_sputtering_rates_divwall"][:,i] = self.res['rcld_sput_rate'][i,:] * circ
        PWI_traces["impurity_sputtering_rate_total_divwall"] = PWI_traces["impurity_sputtering_rates_divwall"].sum(axis=1)
        
        # divertor wall, average
        
        # divertor wall impurity content
        _, PWI_traces["impurity_content_divwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_content_divwall"],interval) 
        # impurity flux towards divertor wall
        _, PWI_traces["impurity_flux_divwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_flux_divwall"],interval) 
        # impurity impact energy at divertor wall
        _, PWI_traces["impurity_impact_energy_divwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_impact_energy_divwall"],interval) 
        # impurity reflection coefficient at divertor wall
        _, PWI_traces["impurity_reflection_coeff_divwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_reflection_coeff_divwall"],interval)
        # impurity reflection energy at divertor wall
        _, PWI_traces["impurity_reflection_energy_divwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_reflection_energy_divwall"],interval)
        # fluxes of background species
        PWI_traces["background_fluxes_divwall_av"] = np.zeros((len(time_average),len(background_species)))
        for i in range(0,len(background_species)):
            _, PWI_traces["background_fluxes_divwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["background_fluxes_divwall"][:,i],interval)
        # impact energies of background species at divertor wall
        PWI_traces["background_impact_energies_divwall_av"] = np.zeros((len(time_average),len(background_species)))
        for i in range(0,len(background_species)):
            _, PWI_traces["background_impact_energies_divwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["background_impact_energies_divwall"][:,i],interval)
        # impurity sputtering yields at divertor wall
        PWI_traces["impurity_sputtering_yields_divwall_av"] = np.zeros((len(time_average),len(background_species)+1))
        for i in range(0,len(background_species)+1):
            _, PWI_traces["impurity_sputtering_yields_divwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_sputtering_yields_divwall"][:,i],interval)
        # impurity sputtering energies at divertor wall
        PWI_traces["impurity_sputtering_energies_divwall_av"] = np.zeros((len(time_average),len(background_species)+1))
        for i in range(0,len(background_species)+1):
            _, PWI_traces["impurity_sputtering_energies_divwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_sputtering_energies_divwall"][:,i],interval)    
        # impurity reflection rate from divertor wall
        _, PWI_traces["impurity_reflection_rate_divwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_reflection_rate_divwall"],interval) 
        # impurity prompt recycling rate from divertor wall
        _, PWI_traces["impurity_prompt_recycling_rate_divwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_prompt_recycling_rate_divwall"],interval) 
        # impurity implantation rate into divertor wall
        _, PWI_traces["impurity_implantation_rate_divwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_implantation_rate_divwall"],interval) 
        # impurity release rate from divertor wall
        PWI_traces["impurity_sputtering_rates_divwall_av"] = np.zeros((len(time_average),len(background_species)+1))
        for i in range(0,len(background_species)+1):
            _, PWI_traces["impurity_sputtering_rates_divwall_av"][:,i] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_sputtering_rates_divwall"][:,i],interval) 
        _, PWI_traces["impurity_sputtering_rate_total_divwall_av"] = plot_tools.time_average_reservoirs(self.namelist["timing"],self.time_out,PWI_traces["impurity_sputtering_rate_total_divwall"],interval) 

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
            
            if self.namelist['ELM_model']['ELM_flag']:
                ax1[1, 0].plot(time_average, PWI_traces["impurity_implantation_rate_mainwall_av"], label="Total absorption rate", color = greens[0])
                ax1[1, 0].plot(time_average, PWI_traces["impurity_sputtering_rate_total_mainwall_av"], label="Total release rate", color = blues[0])
                ax1[1, 0].plot(time_average, PWI_traces["impurity_implantation_rate_mainwall_av"]-PWI_traces["impurity_sputtering_rate_total_mainwall_av"], label="Balance", color = 'black')
                ax1[1, 0].set_ylabel('[$\mathrm{s}^{-1}$]')
                ax1[1, 0].set_title(f"{self.imp} wall balance (ELM-average)", loc='right', fontsize = 11)
                ax1[1, 0].legend(loc="best", fontsize = 9).set_draggable(True)

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
            
            if self.namelist['ELM_model']['ELM_flag']:
                ax2[1, 0].plot(time_average, PWI_traces["impurity_implantation_rate_divwall_av"], label="Total absorption rate", color = greens[0])
                ax2[1, 0].plot(time_average, PWI_traces["impurity_sputtering_rate_total_divwall_av"], label="Total release rate", color = blues[0])
                ax2[1, 0].plot(time_average, PWI_traces["impurity_implantation_rate_divwall_av"]-PWI_traces["impurity_sputtering_rate_total_divwall_av"], label="Balance", color = 'black')
                ax2[1, 0].set_ylabel('[$\mathrm{s}^{-1}$]')
                ax2[1, 0].set_title(f"{self.imp} wall balance (ELM-average)", loc='right', fontsize = 11)
                ax2[1, 0].legend(loc="best", fontsize = 9).set_draggable(True)

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
            
