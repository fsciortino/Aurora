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

# set impurity species and sources rate
imp = namelist["imp"] = "He"
namelist["source_type"] = "const"
namelist["source_rate"] = 2e20  # particles/s

# explicitly set some parameters for the time grid
namelist['timing']['times'] = [0.0, 2.0] # Start and end times of the simulation
namelist['timing']['dt_increase'][0] = 1.05 # dt multiplier at every time steps
namelist['timing']['dt_start'][0] = 5e-5 # dt values at the beginning of the simulation

# impose ELM parameters
namelist['ELM_model']['ELM_flag'] = True
namelist['ELM_model']['ELM_time_windows'] = None # ELMs for the entire duration of the simulation
namelist['ELM_model']['ELM_frequency'] = 100 # Hz
namelist['ELM_model']['crash_duration'] = 0.5 # ms
namelist['ELM_model']['plateau_duration'] = 0.0 # ms
namelist['ELM_model']['recovery_duration'] = 0.5 # ms

# adapt the time grid for ELMs
dt_intra_ELM = 5e-5 # constant dt values during the intra-ELM phases
dt_increase_inter_ELM = 1.05 # dt multiplier at every time steps in the inter-ELM phases
namelist['timing'] = aurora.transport_utils.ELM_time_grid(namelist['timing'], namelist['ELM_model'], dt_intra_ELM, dt_increase_inter_ELM)

# Now get aurora setup plotting the resulting adapted time grid
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)
asim.plot_resolutions(plot_radial_grid = False, plot_time_grid = True)

# set transport coefficients profiles at aribrary rho_pol location
# for both inter-ELM and intra-ELM phases

# arbitrary rho_pol locations:
    
rhop = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 
        0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96,
        0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03,
        1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10]

# desired values of D_Z (in cm^2/s) corresponding to each radial location in rhop:
    
D = [4.00e4, 4.00e4, 4.00e4, 4.00e4, 4.00e4, 4.00e4, 4.00e4, 4.00e4, 4.00e4, 
     4.00e4, 4.00e4, 4.00e4, 1.00e4, 0.80e4, 0.60e4, 0.40e4,
     0.20e4, 0.20e4, 0.20e4, 0.40e4, 0.60e4, 0.80e4, 1.20e4, 
     1.60e4, 2.00e4, 4.00e4, 8.00e4, 8.00e4, 8.00e4, 8.00e4]

D_ELM = [4.00e4, 4.00e4, 4.00e4, 4.00e4, 4.00e4, 4.00e4, 4.00e4, 4.00e4, 5.00e4, 
         6.00e4, 7.00e4, 8.00e4, 8.00e4, 8.00e4, 8.00e4, 8.00e4,
         8.00e4, 8.00e4, 8.00e4, 8.00e4, 8.00e4, 8.00e4, 8.00e4, 
         8.00e4, 8.00e4, 8.00e4, 8.00e4, 8.00e4, 8.00e4, 8.00e4]

# desired values of v_Z (in cm/s) corresponding to each radial location in rhop:
    
v = [-1.20e2, -1.20e2, -1.20e2, -1.20e2, -1.20e2, -1.20e2, -1.20e2, -1.50e2, -1.50e2, 
     -1.80e2, -2.00e2, -2.00e2, -2.00e2, -2.00e2, -2.00e2, -2.00e2,
     -2.00e2, -4.00e2, -6.00e2, -10.00e2, -10.00e2, -10.00e2, -6.00e2,
     -4.00e2, 0, 0, 0, 0, 0, 0]

v_ELM = v

# now create the time-dependent transport arrays (and the
# corresponding times) to be used as input for aurora
times_transport, D_z = aurora.transport_utils.ELM_model(namelist['timing'], namelist['ELM_model'], asim, rhop, D, D_ELM, method = 'Pchip_spline')
times_transport, v_z = aurora.transport_utils.ELM_model(namelist['timing'], namelist['ELM_model'], asim, rhop, v, v_ELM, method = 'Pchip_spline')

# run Aurora forward model, explicitly referring to the time array
# corresponding to the input transprot values, and and plot results
out = asim.run_aurora(D_z, v_z, times_transport, plot=True)

# extract densities and particle numbers in each simulation reservoir
nz, N_mainwall, N_div, N_out, N_mainret, N_tsu, N_dsu, N_dsul, rclb_rate, rclw_rate = out

# finally plot the reservoirs also averaged over ELM cycles, i.e. on
# intervals of 1/f_ELM = 1/100 = 0.01 s, for ELM-averaged results

time_average, data_average_profiles = aurora.plot_tools.time_average_profiles(namelist['timing'],asim.time_out,nz,interval=0.01)

aurora.slider_plot(asim.rhop_grid, time_average, data_average_profiles.transpose(1,0,2),
                xlabel=r'$\rho_p$', ylabel='time [s]', zlabel='$n_Z$ [cm$^{-3}$]',
                labels=[str(i) for i in np.arange(0,nz.shape[1])],
                plot_sum=
                True, x_line=1 ) # Impurity density in the plasma (average over cycles)

reservoirs_average = asim.plot_reservoirs_average(interval=0.01) # Particle conservation (average over cycles)
