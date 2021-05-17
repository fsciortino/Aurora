'''
Script to demonstrate use of superstages.

It is recommended to run this in IPython.
'''

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from omfit_classes import omfit_eqdsk, omfit_gapy
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
inputgacode = omfit_gapy.OMFITgacode(examples_dir+'/example.input.gacode')

# save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid
kp = namelist['kin_profs']
kp['Te']['rhop'] = kp['ne']['rhop'] = np.sqrt(inputgacode['polflux']/inputgacode['polflux'][-1])
kp['ne']['vals'] = inputgacode['ne']*1e13 # 1e19 m^-3 --> cm^-3
kp['Te']['vals'] = inputgacode['Te']*1e3  # keV --> eV

# set impurity species and sources rate
imp = namelist['imp'] = 'Ar'
namelist['source_type'] = 'const'
namelist['source_rate'] = 2e20  # particles/s

# choose superstages: always include 0 and 1!
namelist['superstages'] = [0,1,2,10,11,12,13,14,15,16,17,18]

#namelist['superstages'] = []

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# check radial grid (also internally run by aurora_sim):
#rvol_grid, pro_grid, qpr_grid, prox_param = aurora.create_radial_grid(namelist,plot=plot)

# check time grid (also internally run by aurora_sim):
#_ = aurora.create_time_grid(namelist['timing'], plot=plot)

# set time-independent transport coefficients (flat D=1 m^2/s, V=-2 cm/s)
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -2e2 * np.ones(len(asim.rvol_grid)) # cm/s

#D_z = np.tile(D_z, (len(namelist['superstages'])+1,1,1)).T
#V_z = np.tile(V_z, (len(namelist['superstages'])+1,1,1)).T

#D_z = np.tile(D_z, (asim.Z_imp+1,1,1)).T
#V_z = np.tile(V_z, (asim.Z_imp+1,1,1)).T

# run Aurora forward model and plot results
out = asim.run_aurora(D_z, V_z, times_DV=[1.0,], unstage=True, plot=plot)

# extract densities and particle numbers in each simulation reservoir
nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out


# plot charge state distributions over radius and time
aurora.plot_tools.slider_plot(asim.rvol_grid, asim.time_grid, nz.transpose(1,0,2),
                              xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'$n_z$ [$cm^{-3}$]',
                              labels=[str(i) for i in np.arange(0,nz.shape[1])],
                              plot_sum=True, x_line=asim.rvol_lcfs)


if nz.shape[1]==asim.Z_imp+1:
    # add radiation
    asim.rad = aurora.compute_rad(imp, nz.transpose(2,1,0), asim.ne, asim.Te,
                                  prad_flag=True, thermal_cx_rad_flag=False, 
                                  spectral_brem_flag=False, sxr_flag=False)
    
    # plot radiation profiles over radius and time
    aurora.slider_plot(asim.rvol_grid, asim.time_out, asim.rad['line_rad'].transpose(1,2,0),
                       xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'Line radiation [$MW/m^3$]',
                       labels=[str(i) for i in np.arange(0,nz.shape[1])],
                       plot_sum=True, x_line=asim.rvol_lcfs)
    

