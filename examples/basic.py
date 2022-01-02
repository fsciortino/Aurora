'''
Script to test functionality from namelist creation to run and postprocessing.

It is recommended to run this in IPython.
'''

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from omfit_classes import omfit_eqdsk
import sys, os
from scipy.interpolate import interp1d

# Make sure that package home is added to sys.path
sys.path.append('../')
import aurora

try: # pass any argument via the command line to show plots
    plot = len(sys.argv)>1
except:
    plot = False

# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()

# Use gfile and statefile in local directory:
examples_dir = os.path.dirname(os.path.abspath(__file__))
geqdsk = omfit_eqdsk.OMFITgeqdsk(examples_dir+'/example.gfile')

# save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid
# parameterization f=(f_center-f_edge)*(1-rhop**alpha1)**alpha2 + f_edge
kp = namelist['kin_profs']
T_core = 5e3 # eV
T_edge = 100 # eV
T_alpha1 = 2.
T_alpha2 = 1.5
n_core = 1e14 # cm^-3
n_edge = 0.4e14 # cm^-3
n_alpha1 = 2
n_alpha2 = 0.5
                 
rhop = kp['Te']['rhop'] = kp['ne']['rhop'] = np.linspace(0, 1, 100)
kp['ne']['vals'] = (n_core - n_edge)*(1-rhop**n_alpha1)**n_alpha2 + n_edge
kp['Te']['vals'] = (T_core - T_edge)*(1-rhop**T_alpha1)**T_alpha2 + T_edge

# set impurity species and sources rate
imp = namelist['imp'] = 'Ar'
namelist['source_type'] = 'const'
namelist['source_rate'] = 2e20  # particles/s

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# set time-independent transport coefficients (flat D=1 m^2/s, V=-2 cm/s)
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -2e2 * np.ones(len(asim.rvol_grid)) # cm/s

# run Aurora forward model and plot results
out = asim.run_aurora(D_z, V_z, plot=plot)

# extract densities and particle numbers in each simulation reservoir
nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out

if plot:
    # plot charge state distributions over radius and time
    aurora.plot_tools.slider_plot(asim.rvol_grid, asim.time_out, nz.transpose(1,0,2),
                                  xlabel=r'$r_V$ [cm]', ylabel='time [s]',
                                  zlabel=r'$n_z$ [$cm^{-3}$]',
                                  labels=[str(i) for i in np.arange(0,nz.shape[1])],
                                  plot_sum=True, x_line=asim.rvol_lcfs)

# add radiation
asim.rad = aurora.compute_rad(imp, nz.transpose(2,1,0), asim.ne, asim.Te,
                              prad_flag=True, thermal_cx_rad_flag=False, 
                              spectral_brem_flag=False, sxr_flag=False)

if plot:
    # plot radiation profiles over radius and time
    aurora.slider_plot(asim.rvol_grid, asim.time_out, asim.rad['line_rad'].transpose(1,2,0),
                       xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'Line radiation [$MW/m^3$]',
                       labels=[str(i) for i in np.arange(0,nz.shape[1])],
                       plot_sum=True, x_line=asim.rvol_lcfs)
