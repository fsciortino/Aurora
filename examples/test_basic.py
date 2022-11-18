"""
Script to test functionality from namelist creation to run and postprocessing.

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
namelist["imp"] = "Ar"
namelist["main_element"] = "D"

# set start and end time
namelist["timing"]["times"] = [0,0.2]

# set external source
namelist["source_type"] = "const"
namelist["source_rate"] = 2e20  # particles/s

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
D_z = aurora.transport_utils.interp_coeffs(namelist, asim.rhop_grid, D, radial_dependency = True, rhop = rhop, method = 'Pchip_spline', plot = True, name = 'D')
v_z = aurora.transport_utils.interp_coeffs(namelist, asim.rhop_grid, v, radial_dependency = True, rhop = rhop, method = 'Pchip_spline', plot = True, name = 'v')

# run Aurora forward model and plot the results
#   (i.e. particle conservation and reservoirs plots)
#   including the line radiation
out = asim.run_aurora(D_z, v_z, plot=True, plot_radiation=True)

# extract densities and particle numbers in each simulation reservoir
nz, N_mainwall, N_div, N_out, N_mainret, N_tsu, N_dsu, N_dsul, rclb_rate, rclw_rate = out
