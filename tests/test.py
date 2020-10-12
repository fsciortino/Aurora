'''
Script to test functionality from namelist creation to run and postprocessing.

It is recommended to run this in IPython.
'''

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import omfit_eqdsk
import pickle as pkl
import scipy,sys,os
import time

# Make sure that package home is added to sys.path
import sys
sys.path.append('../')
import aurora


namelist = aurora.default_nml.load_default_namelist()

# test for C-Mod:
namelist['device'] = 'CMOD'
namelist['shot'] = 1101014030
namelist['time'] = 1250 # ms


gfile_name=f'g{namelist["shot"]}.{str(namelist["time"]).zfill(5)}'


if os.path.exists(gfile_name):
    # fetch local g-file if available
    geqdsk = omfit_eqdsk.OMFITgeqdsk(gfile_name)
    print('Fetched local g-file')
else:
    # attempt to construct it via omfit_eqdsk if not available locally
    geqdsk = omfit_eqdsk.OMFITgeqdsk('').from_mdsplus(
        device=namelist['device'],shot=namelist['shot'],
        time=namelist['time'], SNAPfile='EFIT01',
        fail_if_out_of_range=False,time_diff_warning_threshold=20
    )
    # save g-file locally:
    geqdsk.save(raw=True)
    print('Saved g-file locally')


# example kinetic profiles
kin_profs = namelist['kin_profs']

with open('./test_kin_profs.pkl','rb') as f:
    ne_profs,Te_profs = pkl.load(f)

kin_profs['ne']['vals'] = ne_profs['ne']*1e14  # 10^20 m^-3 --> cm^-3
kin_profs['ne']['times'] = ne_profs['t']
rhop = kin_profs['ne']['rhop'] = ne_profs['rhop']
kin_profs['Te']['vals'] = Te_profs['Te']*1e3  # keV --> eV
kin_profs['Te']['times'] = Te_profs['t']
kin_profs['Te']['rhop'] = Te_profs['rhop']
kin_profs['Te']['decay'] = np.ones(len(Te_profs['Te']))*1.0

# set no sources of impurities
namelist['source_type'] = 'const'
namelist['Phi0'] = 1e24 #1.0

# Set up for 1 ion:
imp = namelist['imp'] = 'Ca' #'Ar' #'Ca'
namelist['Z_imp'] = 20 #18. #20.
namelist['imp_a'] = 40.078 #39.948  # 40.078

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# choose transport coefficients
D_eff = 1e4 #cm^2/s
v_eff = -2e2 #cm/s

# # set transport coefficients to the right format
D_z = np.ones((len(asim.rvol_grid),1)) * D_eff
V_z = np.ones((len(asim.rvol_grid),1)) * v_eff
times_DV = [1.0]  # dummy

# set initial charge state distributions to ionization equilibrium (no transport)
num=10
start = time.time()
for n in np.arange(num):
    out = asim.run_aurora(times_DV, D_z, V_z) #, nz_init=nz_init.T)
print('Average time per run: ', (time.time() - start)/num)
nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out
nz = nz.transpose(2,1,0)

# add radiation
asim.rad = aurora.radiation.compute_rad(imp, asim.rhop_grid, asim.time_out, nz, 
                                            asim.ne, asim.Te, prad_flag=True, thermal_cx_rad_flag=False, 
                                            spectral_brem_flag=False, sxr_flag=False, 
                                            main_ion_brem_flag=False)

# Calculate Delta-Z_eff
Zmax = nz.shape[1]-1
Z = np.arange(Zmax+1)
delta_Zeff = nz*(Z*(Z-1))[None,:,None]   # for each charge state
delta_Zeff/= asim.ne[:,None,:]


# ----------------------
# plot charge state distributions over radius and time
aurora.plot_tools.slider_plot(asim.rvol_grid, asim.time_out, nz.transpose(1,2,0), xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel='nz [A.U.]', labels=[str(i) for i in np.arange(0,nz.shape[1])], plot_sum=True, x_line=asim.rvol_lcfs)


# plot radiation profiles over radius and time
aurora.plot_tools.slider_plot(asim.rvol_grid, asim.time_out, asim.rad['impurity_radiation'].transpose(1,2,0)[:nz.shape[1],:,:], xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel='Total radiation [A.U.]', labels=[str(i) for i in np.arange(0,nz.shape[1])], plot_sum=True, x_line=asim.rvol_lcfs)

# plot Delta-Zeff profiles over radius and time
aurora.plot_tools.slider_plot(asim.rvol_grid, asim.time_out, delta_Zeff.transpose(1,2,0), xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'$\Delta$ $Z_{eff}$', labels=[str(i) for i in np.arange(0,nz.shape[1])], plot_sum=True,x_line=asim.rvol_lcfs)


##############################
# Check particle conservation
##############################
import xarray
ds = xarray.Dataset({'impurity_density': ([ 'time', 'charge_states','radius_grid'], nz),
                     'impurity_radiation':  ([ 'time', 'charge_states','radius_grid'], asim.rad['impurity_radiation'][:,:Zmax+1,:]),
                     'source_function': (['time'], asim.source_time_prof ),
                     'particles_in_divertor': (['time'], N_div), 
                     'particles_in_pump': (['time'], N_pump), 
                     'parallel_loss': (['time'], N_dsu), 
                     'parallel_loss_to_limiter': (['time'], N_dsul), 
                     'edge_loss': (['time'], N_tsu), 
                     'particles_at_wall': (['time'], N_wall), 
                     'particles_retained_at_wall': (['time'], N_ret), 
                     'recycling_from_wall':  (['time'], rclw_rate), 
                     'recycling_from_divertor':  (['time'], rcld_rate), 
                     'pro': (['radius_grid'], asim.pro), 
                     'rhop_grid': (['radius_grid'], asim.rhop_grid)
                     },
                    coords={'time': asim.time_out, 
                            'radius_grid': asim.rvol_grid,
                            'charge_states': np.arange(nz.shape[1])
                            })

_ = aurora.particle_conserv.plot_1d(ds = ds)

