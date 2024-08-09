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
                
            # set wall fluxes of the background species to constant value over entire time grid
            
            fluxes_main_wall[:] = np.asarray(self.background_main_wall_fluxes)[:,None]
            fluxes_div_wall[:]  = np.asarray(self.background_div_wall_fluxes)[:,None]

                

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
        
        impact_energy_main_wall = np.zeros((len(self.all_species),len(self.time_out)))
        impact_energy_div_wall = np.zeros((len(self.all_species),len(self.time_out)))
        
        # Impact energy for the simulated species itself (index 0) and for the background species (indices 1+)
        for i, s in enumerate(self.all_species):
             # Same values over entire time grid 
            impact_energy_main_wall[i] = surface.get_impact_energy(self.Te_lim,
                                                          s,
                                                          mode = 'sheath',
                                                          Ti_over_Te = self.Ti_over_Te,
                                                          gammai = self.gammai)
            impact_energy_div_wall[i] = surface.get_impact_energy(self.Te_div,
                                                         s,
                                                         mode = 'sheath',
                                                         Ti_over_Te = self.Ti_over_Te,
                                                         gammai = self.gammai)    
     
        
        return impact_energy_main_wall, impact_energy_div_wall
        
        
    def setup_rec_profs(self):
        """Method to set up Aurora inputs related to the recycling profiles for neutrals released from the main wall.
        """
 
        #impurity neutral ionization rate
        Sne0 = self.Sne_rates[:, 0, [0]]
    
        
        # Full PWI model: set recycling profiles to exponentially decay from the wall
        #   with recycled neutrals having different energies depending on whether they
        #   were reflected, promptly recycled or sputtered from the wall
              
        # Source profile from reflected particles from main wall:
        #   use the reflected energy from TRIM data
        
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
        
        # calculate the sputtering profiles from simulated and background species
        spt_rad_prof = np.zeros((len(self.rvol_grid), len(self.all_species)))
        
        #use almost the same namelist, change imp_source_energy_eV
        nml_spt_prof = nml_rfl_prof
        
        for j, s in enumerate(self.all_species):
            # set energy value in the reflection profile namelist
            nml_spt_prof["imp_source_energy_eV"] = self.E_sput_main_wall[j,0]
    
            spt_rad_prof[:,j] = source_utils.get_radial_source(
                nml_spt_prof,  # namelist specifically to obtain exp decay from wall
                self.rvol_grid,
                self.pro_grid,
                Sne0,
                self._Ti,
            )[:,0] #assume time independent?

            
        # set same profile for all times
        self.spt_rad_prof = np.tile(spt_rad_prof[:,:,None], (1,len(self.time_grid)))

        
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
            
