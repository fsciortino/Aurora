"""
Script to test the functionality of the time-dependent ELM transport model

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
namelist["imp"] = "He"
namelist["main_element"] = "D"

# set start and end time
namelist["timing"]["times"] = [0,1.0]

# set external source
namelist["source_type"] = "const"
namelist["source_rate"] = 2e20  # particles/s

# activate recycling
namelist["recycling_flag"] = True
namelist["wall_recycling"] = 0.5

# impose ELM parameters
namelist['ELM_model']['ELM_flag'] = True
    # Convenience flag for the ELM model
namelist['ELM_model']['ELM_time_windows'] = None
    # List of lists, which defines one or more time windows within the entire simulation times in which ELMs are desired
    # If None, then the ELMs take place for the entire duration of the simulation
namelist['ELM_model']['ELM_frequency'] = 100 # Hz
    # Frequency at which ELM cycles take place
namelist['ELM_model']['crash_duration'] = 0.5 # ms
    # Duration of the time windows, within an ELM, during which the transport coefficients
    #   (at each radial location) are ramped up linearly from their inter-ELM values to their intra-ELM values
namelist['ELM_model']['plateau_duration'] = 0.0 # ms
    # Duration of the time windows, within an ELM, during which the transport coefficients
    #   (at each radial location) stays constantly at their intra-ELM values
namelist['ELM_model']['recovery_duration'] = 0.1 # ms
    # Duration of the time windows, within an ELM, during which the transport coefficients
    #   (at each radial location) are ramped down linearly from their intra-ELM values to their inter-ELM values

# adapt the time grid for ELMs, in order to save computational time
namelist['ELM_model']['adapt_time_grid'] = True
    # Flag for adapting the time grid to the ELM characteristics, in order to get time steps which are
    #   small during ELMs and which start to gradually increase again in duration when the ELM is over
namelist['ELM_model']['dt_intra_ELM'] = 5e-5 # s
    # Constant dt values applied on the time grid during the intra-ELM phases if adapt_time_grid is True
namelist['ELM_model']['dt_increase_inter_ELM'] = 1.05 
    # dt multiplier applied on the time grid at every time steps in the inter-ELM phases if adapt_time_grid is True  
    
# Note: there is the possibility of also increasing the Mach number in the SOL during ELMs,
#   with the inputs SOL_mach (inter-ELM value) and SOL_mach_ELM (intra-ELM value) in the input namelist,
#   with a user-defined temporal shape which resembles the temporal ion parallel flux shape
#   onto the divertor (i.e. sharp peak, in which Mach abruptly increases from its inter-ELM value to its
#   intra-ELM value and then exponentially decays again to its inter-ELM value).
#   However, for now this only works when the time grid is build with a constant time step, e.g.
#   namelist['timing']['dt_increase'][0] = 1.05 and namelist['timing']['dt_start'][0] = 5e-5, and
#   the CPU-time-saving adaptation of the time grid for ELMs is not performed, i.e. 
#   namelist['ELM_model']['adapt_time_grid'] = False.

# Now get aurora setup plotting the resulting adapted time grid
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)
asim.plot_resolutions(plot_radial_grid = False, plot_time_grid = True)

# set transport coefficients profiles at aribrary rho_pol location
# for both inter-ELM and intra-ELM phases

# arbitrary rho_pol locations:
    
rhop = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85,
        0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96,
        0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03,
        1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10]

# desired values of D_Z (in cm^2/s) corresponding to each radial location in rhop:
    
D = [2.00e4, 2.00e4, 2.00e4, 2.00e4, 2.00e4, 2.00e4, 2.00e4, 2.00e4, 2.00e4, 2.00e4,
     1.20e4, 1.00e4, 0.75e4, 0.75e4, 0.75e4, 0.75e4, 0.75e4,
     0.50e4, 0.50e4, 0.50e4, 0.50e4, 0.75e4, 1.00e4, 1.50e4, 
     2.00e4, 2.00e4, 4.00e4, 4.00e4, 4.00e4, 4.00e4, 4.00e4]

D_ELM = [2.00e4, 2.00e4, 2.00e4, 2.00e4, 2.00e4, 2.00e4, 2.00e4, 2.00e4, 2.00e4, 2.00e4, 
         2.00e4, 1.50e4, 1.00e4, 1.50e4, 2.00e4, 3.00e4, 4.00e4,
         8.00e4, 16.00e4, 16.00e4, 12.00e4, 8.00e4, 6.00e4, 4.00e4, 
         4.00e4, 4.00e4, 4.00e4, 4.00e4, 4.00e4, 4.00e4, 4.00e4]

# desired values of v_Z (in cm/s) corresponding to each radial location in rhop:
    
v = [-0.5e2, -0.5e2, -1e2, -3e2, -4e2, -3.5e2, -3.0e2, -1.0e2, -1.5e2, -2.5e2,
     -5e2, -5e2, -5e2, -5e2, -6e2, -6e2, -6e2,
     -8e2, -12e2, -15e2, -20e2, -15e2, -12e2, -10e2,
     -8e2, -6e2, -4e2, -2e2, -2e2, -2e2, -2e2]

v_ELM = [-0.5e2, -0.5e2, -1e2, -3e2, -4e2, -3.5e2, -3.0e2, -1.0e2, -1.5e2, -2.5e2,
     -5e2, -5e2, -5e2, -5e2, -6e2, -6e2, -6e2,
     -8e2, -12e2, -15e2, -20e2, -15e2, -12e2, -10e2,
     -8e2, -6e2, -4e2, -2e2, -2e2, -2e2, -2e2]

# now create the time-dependent transport arrays (and the
# corresponding times) to be used as input for aurora
# and plot them
times_transport, D_z = aurora.transport_utils.ELM_model(namelist['timing'], namelist['ELM_model'], asim.rhop_grid, rhop, D, D_ELM, method = 'Pchip_spline', plot = True, name = 'D')
times_transport, v_z = aurora.transport_utils.ELM_model(namelist['timing'], namelist['ELM_model'], asim.rhop_grid, rhop, v, v_ELM, method = 'Pchip_spline', plot = True, name = 'v')

# run Aurora forward model, explicitly referring to the time array
#   corresponding to the input transport values, and and plot the results
#   (i.e. particle conservation and reservoirs plots)
# if plot_average is True, then the particle conservation and reservoirs plots
#   will be produced also with ELM-averaged values of the time traces, i.e.
#   showing their average values over the time interval 'interval' which
#   should be equal to the ELM period, i.e. 1/ELM_frequency
out = asim.run_aurora(D_z, v_z, times_transport, plot=True, plot_average = True, interval = 0.01)

# extract densities and particle numbers in each simulation reservoir
nz, N_mainwall, N_div, N_out, N_mainret, N_tsu, N_dsu, N_dsul, rclb_rate, rclw_rate = out
