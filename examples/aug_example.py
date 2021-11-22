'''
Script to test functionality for AUG.

It is recommended to run this in IPython.
'''

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import sys, os
from scipy.interpolate import interp1d
import aug_sfutils as sf
from omfit_classes.omfit_eqdsk import OMFITgeqdsk

# Make sure that package home is added to sys.path
sys.path.append('../')
import aurora

shot = 39649
time = 3.0

# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()

# get equilibrium for AUG through aug_sfutils and OMFITgeqdsk
geqdsk = OMFITgeqdsk('').from_aug_sfutils(
    shot=shot, time=time, eq_shotfile='EQI')

# get ne, Te from AUG IDA at specified time
kp = namelist['kin_profs']

ida = sf.SFREAD(shot, 'ida')
time_ida = ida.gettimebase('Te')
it_ida = np.argmin(np.abs(time_ida - time))
rhop_ida = ida.getareabase('Te')
Te_eV = ida.getobject('Te')
ne_m3 = ida.getobject('ne')

kp['Te']['rhop'] = kp['ne']['rhop'] = rhop_ida[:,it_ida]
kp['ne']['vals'] = ne_m3[:,it_ida] * 1e-6 # m^-3 --> cm^-3
kp['Te']['vals'] = Te_eV[:,it_ida] # eV

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
out = asim.run_aurora(D_z, V_z, plot=True)

# extract densities and particle numbers in each simulation reservoir
nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out

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

# plot radiation profiles over radius and time
aurora.slider_plot(asim.rvol_grid, asim.time_out, asim.rad['line_rad'].transpose(1,2,0),
                   xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'Line radiation [$MW/m^3$]',
                   labels=[str(i) for i in np.arange(0,nz.shape[1])],
                   plot_sum=True, x_line=asim.rvol_lcfs)

