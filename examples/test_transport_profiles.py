"""
Script to test imposition of arbitrary radial profiles for transport coefficients

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
imp = namelist["imp"] = "Ar"
namelist["source_type"] = "const"
namelist["source_rate"] = 2e20  # particles/s

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# set time-independent transport coefficients profiles at aribrary rho_pol locations

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

# desired values of v_Z (in cm/s) corresponding to each radial location in rhop:
v = [-1.20e2, -1.20e2, -1.20e2, -1.20e2, -1.20e2, -1.20e2, -1.20e2, -1.50e2, -1.50e2, 
     -1.80e2, -2.00e2, -2.00e2, -2.00e2, -2.00e2, -2.00e2, -2.00e2,
     -2.00e2, -4.00e2, -6.00e2, -10.00e2, -10.00e2, -10.00e2, -6.00e2,
     -4.00e2, 0, 0, 0, 0, 0, 0]

# now create the transport arrays to be used as input for aurora and plot them
D_z = aurora.transport_utils.interp_coeffs(namelist, asim, D, radial_dependency = True, rhop = rhop, method = 'Pchip_spline')
v_z = aurora.transport_utils.interp_coeffs(namelist, asim, v, radial_dependency = True, rhop = rhop, method = 'Pchip_spline')

plt.figure()
plt.plot(asim.rhop_grid,D_z)
plt.xlabel(r'$\rho_p$')
plt.ylabel('$D_Z$ [$cm^{2}/s$]')
plt.show()

plt.figure()
plt.plot(asim.rhop_grid,v_z)
plt.xlabel(r'$\rho_p$')
plt.ylabel('$v_Z$ [$cm/s$]')
plt.show()

# run Aurora forward model and plot results
out = asim.run_aurora(D_z, v_z, plot=True)

# extract densities and particle numbers in each simulation reservoir
nz, N_wall, N_div, N_pump, N_out, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclp_rate, rclw_rate = out
