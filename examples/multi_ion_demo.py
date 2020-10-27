'''
Script to demonstrate combinations of various ions.
It is recommended to run this in IPython
'''

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import omfit_eqdsk
import pickle as pkl
import scipy,sys,os
import time
from IPython import embed

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
kin_profs['Te']['decay'] = np.ones(len(Te_profs['Te']))

# set no sources of impurities
namelist['source_type'] = 'const'
namelist['Phi0'] = 1e24 #1.0

# Set up for 2 different ions:
imp = namelist['imp'] = 'Ca' 
asim_Ca = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# get charge state distributions from ionization equilibrium
atom_data = aurora.atomic.get_atom_data(imp,['acd','scd'])
ne_avg = np.mean(kin_profs['ne']['vals'],axis=0) # average over time
Te_avg = np.mean(kin_profs['Te']['vals'],axis=0)  # must be on the same radial basis as ne_avg

# get fractional abundances on ne (cm^-3) and Te (eV) grid
logTe, fz_Ca = aurora.atomic.get_frac_abundances(atom_data, ne_avg, Te_avg, rho=rhop)

imp = namelist['imp'] = 'Ar' 

asim_Ar = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# get charge state distributions from ionization equilibrium
atom_data = aurora.atomic.get_atom_data(imp,['acd','scd'])
ne_avg = np.mean(kin_profs['ne']['vals'],axis=0) # average over time
Te_avg = np.mean(kin_profs['Te']['vals'],axis=0)  # must be on the same radial basis as ne_avg

# get fractional abundances on ne (cm^-3) and Te (eV) grid
logTe, fz_Ar = aurora.atomic.get_frac_abundances(atom_data, ne_avg, Te_avg, rho=rhop)

############################################################

############################################################

# transform these fractional abundances to the r_V grid used by aurora
_rV = aurora.coords.rad_coord_transform(rhop, 'rhop','r_V', geqdsk)*1e2 # m --> cm (on kin profs grid)
cs_Ca = np.arange(asim_Ca.Z_imp+1)
cs_Ar = np.arange(asim_Ar.Z_imp+1)
nz_init_Ca = scipy.interpolate.interp2d(_rV,cs_Ca, fz_Ca.T)(asim_Ca.rvol_grid, cs_Ca)
nz_init_Ar = scipy.interpolate.interp2d(_rV,cs_Ar, fz_Ar.T)(asim_Ar.rvol_grid, cs_Ar)


# Take definition of peaking as q(psi_n=0.2)/<q>, where <> is a volume average
nominal_peaking=1.3
nominal_volavg = 1e12 # cm^-3

nz_tot_Ca = np.sum(nz_init_Ca,axis=0)
indLCFS = np.argmin(np.abs(asim_Ca.rhop_grid - 1.0))
nz_tot_volavg_Ca = aurora.coords.vol_average(nz_tot_Ca[:indLCFS], asim_Ca.rhop_grid[:indLCFS], geqdsk=geqdsk)[-1]
Psi_n = aurora.coords.rad_coord_transform(rhop, 'rhop','psin', geqdsk)
ind_psin02 = np.argmin(np.abs(Psi_n - 0.2))
peaking = nz_tot_Ca[ind_psin02]/nz_tot_volavg_Ca


########
# TODO: Apply peaking factor to nz_init_* and rescale to match approx source amplitude
########

# choose transport coefficients
D_eff = 1e4 #cm^2/s
v_eff = -2e2 #cm/s

# # set transport coefficients to the right format
D_z = np.ones(len(asim_Ca.rvol_grid)) * D_eff
V_z = np.ones(len(asim_Ca.rvol_grid)) * v_eff

# set initial charge state distributions to ionization equilibrium (no transport)
out = asim_Ca.run_aurora(D_z, V_z)
nz_Ca, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out
nz_Ca = nz_Ca.transpose(2,1,0)

asim_Ca.rad = aurora.radiation.compute_rad('Ca', asim_Ca.rhop_grid, asim_Ca.time_out, nz_Ca, 
                                             asim_Ca.ne, asim_Ca.Te, prad_flag=True, thermal_cx_rad_flag=False, 
                                             spectral_brem_flag=False, sxr_flag=False, 
                                             main_ion_brem_flag=False)

out = asim_Ar.run_aurora(D_z, V_z)
nz_Ar, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out
nz_Ar = nz_Ar.transpose(2,1,0)

asim_Ar.rad = aurora.radiation.compute_rad('Ar', asim_Ar.rhop_grid, asim_Ar.time_out, nz_Ar, 
                                            asim_Ar.ne, asim_Ar.Te, prad_flag=True, thermal_cx_rad_flag=False, 
                                            spectral_brem_flag=False, sxr_flag=False, 
                                            main_ion_brem_flag=False)


# ----------------------
# plot charge state distributions over radius and time
aurora.plot_tools.slider_plot(asim_Ar.rvol_grid, asim_Ar.time_out, nz_Ar.transpose(1,2,0), xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel='nz [A.U.]', labels=[fr'Ar$^{{{i}}}$$^+$' for i in np.arange(0,nz_Ar.shape[1])], plot_sum=True, x_line=asim_Ca.rvol_lcfs)

aurora.plot_tools.slider_plot(asim_Ca.rvol_grid, asim_Ca.time_out, nz_Ca.transpose(1,2,0), xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel='nz [A.U.]', labels=[fr'Ca$^{{{i}}}$$^+$' for i in np.arange(0,nz_Ca.shape[1])], plot_sum=True)
