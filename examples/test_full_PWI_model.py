"""
Script to test the full plasma-wall interaction model for wall recycling.

It is recommended to run this in IPython.
"""
import numpy as np
import matplotlib.pyplot as plt
from omfit_classes import omfit_eqdsk
import sys
import os

# Make sure that package home is added to sys.path
sys.path.append("../")
import aurora

plt.ion()

# pass any argument via the command line to show plots
plot = len(sys.argv) > 1

# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()

# Use gfile and statefile in local directory:
examples_dir = os.path.dirname(os.path.abspath(__file__))
geqdsk = omfit_eqdsk.OMFITgeqdsk(examples_dir + "/example.gfile")

# save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid
# parameterization f=(f_center-f_edge)*(1-rhop**alpha1)**alpha2 + f_edge
kp = namelist["kin_profs"]
T_core = 5e3  # eV
T_edge = 100  # eV
T_alpha1 = 2.0
T_alpha2 = 1.5
n_core = 1e14  # cm^-3
n_edge = 0.4e14  # cm^-3
n_alpha1 = 2
n_alpha2 = 0.5

rhop = kp["Te"]["rhop"] = kp["ne"]["rhop"] = np.linspace(0, 1, 100)
kp["ne"]["vals"] = (n_core - n_edge) * (1 - rhop ** n_alpha1) ** n_alpha2 + n_edge
kp["Te"]["vals"] = (T_core - T_edge) * (1 - rhop ** T_alpha1) ** T_alpha2 + T_edge

# set impurity species and main ion species
# He for now is the only impurity species for which the full PWI is available!!!
namelist["imp"] = "He"
namelist["main_element"] = "D"

# set start and end time
namelist["timing"]["times"] = [0,1.0]

# set external source
namelist["source_type"] = "const"
namelist["source_rate"] = 2e20  # particles/s

# activate recycling
namelist['recycling_flag'] = True
    
# let's change some edge/divertor transport parameters in order to
# activate more features of the extended multi-reservoir particle balance    
    
namelist['div_neut_screen'] = 0.5
namelist['div_recomb_ratio'] = 0.2  # ms
namelist['tau_div_SOL_ms'] = 40.0  # ms

# let's also change some pumping parameters in order to
# activate more features of the extended multi-reservoir particle balance    

namelist['phys_volumes'] = True
namelist['vol_div'] = 0.4e6  # cm^3
namelist['pump_chamber'] = True
namelist['vol_pump'] = 1.0e6  # cm^3
namelist['L_divpump'] = 0.5e8 # cm^3/s 
namelist['L_leak'] = 1.0e6 # cm^3/s
namelist['S_pump'] = 1.0e8  # cm^3/s 

# finally let's set the parameters for the full plasma-wall interaction model

namelist['phys_surfaces'] = True
    # If True, the user can define physical surface areas for the main and divertor walls,
    #   e.g. corresponding to the ones of an actual device. In this case, the surface densities
    #   of particles retained/stuck at the walls are considered.
namelist['surf_mainwall'] = 1.0e4 # cm^2
    # Geometric surface area of the main wall, used if 'phys_surfaces' = True
namelist['surf_divwall'] = 1.0e3 # cm^2    
    # Geometric surface area of the divertor wall, used if 'phys_surfaces' = True
namelist['full_PWI']['main_wall_material'] = 'W'
    # Name of the bulk material of the main wall
    #   W for now is the only wall material for which the full PWI is available!!!
namelist['full_PWI']['div_wall_material'] = 'W'
    # Name of the bulk material of the divertor wall
    #   W for now is the only wall material for which the full PWI is available!!!
namelist['full_PWI']['n_main_wall_sat'] = 1e19 # m^-2
    # Saturation value of the implanted impurity surface density at the main wall
namelist['full_PWI']['n_div_wall_sat'] = 1e19 # m^-2
    # Saturation value of the implanted impurity surface density at the divertor wall
namelist['full_PWI']['characteristic_impact_energy_main_wall'] = 200 # eV
    # Characteristic impact energy of the simulated impurity onto the main wall over the entire
    #   device lifetime, used to estimate the implantation depth into the wall material
namelist['full_PWI']['characteristic_impact_energy_div_wall'] = 500 # eV
    # Characteristic impact energy of the simulated impurity onto the divertor wall over the entire
    #   device lifetime, used to estimate the implantation depth into the wall material
namelist['full_PWI']['background_mode'] = 'manual'
    # If 'mode' = 'manual', then the fluxes of all the background species (specified in 'species') on
    #   the time grid are manually specified in the lists 'main_wall_fluxes' and 'div_wall_fluxes'
    # If 'mode' = 'files', then the fluxes of all the background species (specified in 'species') are
    #   taken from aurora simulations performed in advance, one for each background species, using the same
    #   time grid of the current simulation, with corresponding simulated data contained in files specified
    #   in the list 'files'.
namelist['full_PWI']['background_species'] = ['D']
    # List of names of the simulated background species, whose fluxes towards main and divertor walls
    #   are employed, in the full PWI model, to determine the release of the currently simulated
    #   impurity from the wall through sputtering
namelist['full_PWI']['background_main_wall_fluxes'] = [1e22]
    # List of constant values of the fluxes of all the assumed background species, in s^-1, reaching the
    #   main wall in the framework of the full PWI model, used if 'mode' = 'manual'.
namelist['full_PWI']['background_div_wall_fluxes'] = [1e21]
    # List of constant values of the fluxes of all the assumed background species, in s^-1, reaching the
    #   divertor wall in the framework of the full PWI model, used if 'mode' = 'manual'.
namelist['full_PWI']['Te_div'] = 15.0 # eV
    # Electron temperature at the divertor target surface during inter-ELM phases
namelist['full_PWI']['Te_lim'] = 10.0 # eV
    # Electron temperature at the main wall surface during inter-ELM phases
namelist['full_PWI']['Ti_over_Te'] = 1.0
    # Ion/electron temperature ratio at the plasma-material interface
namelist['full_PWI']['gammai'] = 2.0
    # Ion sheath heat transmission coefficient
namelist['full_PWI']['energetic_recycled_neutrals'] = True
    # If True, then the reflected and sputtered neutrals from the main wall are emitted towards the plasma as
    #   energetic, with energy calculated from the TRIM tables, while the promptly recycled ones are still
    #   emitted as thermal.

# Now get aurora setup
# Note: instead of calling the core class, we need to call the pwi class to activate the full PWI model
asim = aurora.pwi.aurora_sim_pwi(namelist, geqdsk=geqdsk)

# set time-independent transport coefficients (flat D=1 m^2/s, V=-2 cm/s)
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -2e2 * np.ones(len(asim.rvol_grid))  # cm/s

# run Aurora forward model and plot results
out = asim.run_aurora(D_z, V_z, plot=plot, plot_PWI=plot, plot_radiation=plot)

