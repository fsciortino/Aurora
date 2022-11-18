"""
Script to test the advanced plasma-wall interaction model for wall recycling.

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
# He for now is the only impurity species for which the advanced PWI is available!!!
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
    
namelist['screening_eff'] = 0.5
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

# finally let's set the parameters for the advanced plasma-wall interaction model

namelist['phys_surfaces'] = True
    # If True, the user can define physical surface areas for the main and divertor walls,
    #   e.g. corresponding to the ones of an actual device. In this case, the surface densities
    #   of particles retained/stuck at the walls are considered.
namelist['surf_mainwall'] = 1.0e4 # cm^2
    # Geometric surface area of the main wall, used if 'phys_surfaces' = True
namelist['surf_divwall'] = 1.0e3 # cm^2    
    # Geometric surface area of the divertor wall, used if 'phys_surfaces' = True
namelist['advanced_PWI']['advanced_PWI_flag'] = True
    # If True, the advanced plasma-wall interaction model is used, in which wall recycling and
    #   retention are determined by realistic reflection and sputtering coefficient and wall saturation
    #   densities and impact energy/angles of projectile ions, and only a single dynamic reservoir
    #   is employed (for both main and divertor walls). This requires to specify the values of the
    #   fluxes of all the other species (both main plasma species and other possible impurities)
    #   which interact with the walls, on the same time grid used in the current simulation.
namelist['advanced_PWI']['main_wall_material'] = 'W'
    # Name of the bulk material of the main wall
    #   W for now is the only wall material for which the advanced PWI is available!!!
namelist['advanced_PWI']['div_wall_material'] = 'W'
    # Name of the bulk material of the divertor wall
    #   W for now is the only wall material for which the advanced PWI is available!!!
namelist['advanced_PWI']['n_main_wall_sat'] = 1e19 # m^-2
    # Saturation value of the implanted impurity surface density at the main wall
namelist['advanced_PWI']['n_div_wall_sat'] = 1e19 # m^-2
    # Saturation value of the implanted impurity surface density at the divertor wall
namelist['advanced_PWI']['characteristic_impact_energy_main_wall'] = 200 # eV
    # Characteristic impact energy of the simulated impurity onto the main wall over the entire
    #   device lifetime, used to estimate the implantation depth into the wall material
namelist['advanced_PWI']['characteristic_impact_energy_div_wall'] = 500 # eV
    # Characteristic impact energy of the simulated impurity onto the divertor wall over the entire
    #   device lifetime, used to estimate the implantation depth into the wall material
namelist['advanced_PWI']['background_mode'] = 'manual'
    # If 'mode' = 'manual', then the fluxes of all the background species (specified in 'species') on
    #   the time grid are manually specified in the lists 'main_wall_fluxes' and 'div_wall_fluxes'
    # If 'mode' = 'files', then the fluxes of all the background species (specified in 'species') are
    #   taken from aurora simulations performed in advance, one for each background species, using the same
    #   time grid of the current simulation, with corresponding simulated data contained in files specified
    #   in the list 'files'.
namelist['advanced_PWI']['background_species'] = ['D']
    # List of names of the simulated background species, whose fluxes towards main and divertor walls
    #   are employed, in the advanced PWI model, to determine the release of the currently simulated
    #   impurity from the wall through sputtering
namelist['advanced_PWI']['background_main_wall_fluxes'] = [1e22]
    # List of constant values of the fluxes of all the assumed background species, in s^-1, reaching the
    #   main wall in the framework of the advanced PWI model, used if 'mode' = 'manual'.
namelist['advanced_PWI']['background_div_wall_fluxes'] = [1e21]
    # List of constant values of the fluxes of all the assumed background species, in s^-1, reaching the
    #   divertor wall in the framework of the advanced PWI model, used if 'mode' = 'manual'.
namelist['advanced_PWI']['Te_div'] = 15.0 # eV
    # Electron temperature at the divertor target surface during inter-ELM phases
namelist['advanced_PWI']['Te_lim'] = 10.0 # eV
    # Electron temperature at the main wall surface during inter-ELM phases
namelist['advanced_PWI']['Ti_over_Te'] = 1.0
    # Ion/electron temperature ratio at the plasma-material interface
namelist['advanced_PWI']['gammai'] = 2.0
    # Ion sheath heat transmission coefficient
namelist['advanced_PWI']['energetic_recycled_neutrals'] = True
    # If True, then the reflected and sputtered neutrals from the main wall are emitted towards the plasma as
    #   energetic, with energy calculated from the TRIM tables, while the promptly recycled ones are still
    #   emitted as thermal.

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# set time-independent transport coefficients profiles at aribrary rho_pol locations

# arbitrary rho_pol locations:
rhop = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85,
        0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96,
        0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03,
        1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10]
        
# desired values of D_Z (in cm^2/s) corresponding to each radial location in rhop:
D = [2.00e4, 2.00e4, 2.00e4, 2.00e4, 2.00e4, 2.00e4, 2.00e4, 2.00e4, 2.00e4, 2.00e4,
     1.20e4, 1.00e4, 0.75e4, 0.75e4, 0.75e4, 0.75e4, 0.75e4,
     0.50e4, 0.50e4, 0.50e4, 0.50e4, 0.75e4, 1.00e4, 1.50e4, 
     2.00e4, 2.00e4, 4.00e4, 4.00e4, 4.00e4, 4.00e4, 4.00e4]  # cm^2/s  

# desired values of v_Z (in cm/s) corresponding to each radial location in rhop:
v = [-0.5e2, -0.5e2, -1e2, -3e2, -4e2, -3.5e2, -3.0e2, -1.0e2, -1.5e2, -2.5e2,
     -5e2, -5e2, -5e2, -5e2, -6e2, -6e2, -6e2,
     -8e2, -12e2, -15e2, -20e2, -15e2, -12e2, -10e2,
     -8e2, -6e2, -4e2, -2e2, -2e2, -2e2, -2e2]   # cm/s

# now create the transport arrays to be used as input for aurora and plot them
D_z = aurora.transport_utils.interp_coeffs(namelist, asim.rhop_grid, D, radial_dependency = True, rhop = rhop, method = 'Pchip_spline', plot = False, name = 'D')
v_z = aurora.transport_utils.interp_coeffs(namelist, asim.rhop_grid, v, radial_dependency = True, rhop = rhop, method = 'Pchip_spline', plot = False, name = 'v')

# run Aurora forward model and plot the results
#   (i.e. particle conservation and reservoirs plots)
#   including the plasma-wall interaction time traces
out = asim.run_aurora(D_z, v_z, plot=True, plot_PWI=True)

# extract densities and particle numbers in each simulation reservoir
nz, N_mainwall, N_divwall, N_div, N_pump, N_out, N_mainret, Ndivret, N_tsu, N_dsu, N_dsul, rcld_rate, rcld_refl_rate, rcld_recl_rate, rcld_impl_rate, rcld_sput_rate, rclb_rate, rcls_rate, rclp_rate, rclw_rate, rclw_refl_rate, rclw_recl_rate, rclw_impl_rate, rclw_sput_rate = out
