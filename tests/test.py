'''
Script to test functionality from namelist creation to run and postprocessing.

It is recommended to run this in IPython.
Note that you might need to run %matplotlib qt in IPython in order to enable the animation to run. 
'''

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import omfit_eqdsk
import pickle as pkl
import scipy,sys,os
import time

# Allow test script to see package home (better ways?)
sys.path.insert(1, os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
import pylib


namelist = pylib.default_nml.load_default_namelist()

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

# Now get aurora setup dictionary
aurora_dict = pylib.utils.aurora_setup(namelist, geqdsk=geqdsk)

########## Atomic-only prediction (no transport) ###########
# get charge state distributions from ionization equilibrium
atom_data = pylib.atomic.get_all_atom_data(imp,['acd','scd'])
ne_avg = np.mean(kin_profs['ne']['vals'],axis=0) # average over time
Te_avg = np.mean(kin_profs['Te']['vals'],axis=0)  # must be on the same radial basis as ne_avg

# get_frac_abundances takes inputs in m^-3 and eV
logTe, fz = pylib.atomic.get_frac_abundances(atom_data, ne_avg*1e6, Te_avg, rho=rhop)
############################################################


# transform these fractional abundances to the r_V grid used by Aurora
_rV = pylib.coords.rad_coord_transform(rhop, 'rhop','r_V', geqdsk)*1e2 # m --> cm
cs = np.arange(aurora_dict['Z_imp']+1)
nz_init = scipy.interpolate.interp2d(_rV,cs, fz.T)(aurora_dict['radius_grid'], cs)


# Take definition of peaking as q(psi_n=0.2)/<q>, where <> is a volume average
nominal_peaking=1.3
nominal_volavg = 1e12 # cm^-3

nz_tot = np.sum(nz_init,axis=0)
indLCFS = np.argmin(np.abs(aurora_dict['rhop_grid'] - 1.0))
nz_tot_volavg = pylib.coords.vol_average(nz_tot[:indLCFS],aurora_dict['rhop_grid'][:indLCFS],
                                         geqdsk=geqdsk)[-1]
Psi_n = pylib.coords.rad_coord_transform(rhop, 'rhop','psin', geqdsk)
ind_psin02 = np.argmin(np.abs(Psi_n - 0.2))
peaking = nz_tot[ind_psin02]/nz_tot_volavg


# choose transport coefficients
D_eff = 1e4 #cm^2/s
v_eff = -2e2 #cm/s

# # set transport coefficients to the right format
D_z = np.ones((len(aurora_dict['radius_grid']),1)) * D_eff
V_z = np.ones((len(aurora_dict['radius_grid']),1)) * v_eff
times_DV = [1.0]  # dummy

# set initial charge state distributions to ionization equilibrium (no transport)
num=10
start = time.time()
for n in np.arange(num):
    out = pylib.utils.run_aurora(aurora_dict, times_DV, D_z, V_z) #, nz_init=nz_init.T)
print('Average time per run: ', (time.time() - start)/num)
nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out
nz = nz.transpose(2,1,0)

# convenient dictionary
res = {'nz': nz, 'time': aurora_dict['time_out'], 'rV': aurora_dict['radius_grid'], 
       'rhop': aurora_dict['rhop_grid'], 'ne':aurora_dict['ne'], 'Te':aurora_dict['Te']}

# radiation
res['rad'] = pylib.radiation.compute_rad(imp, res['rhop'], res['time'], res['nz'], 
                                            res['ne'],res['Te'], prad_flag=True, thermal_cx_rad_flag=False, 
                                            spectral_brem_flag=False, sxr_flag=False, 
                                            main_ion_brem_flag=False)

# Calculate Delta-Z_eff
Zmax = nz.shape[1]-1
Z = np.arange(Zmax+1)
delta_Zeff = res['nz']*(Z*(Z-1))[None,:,None]   # for each charge state
delta_Zeff/= res['ne'][:,None,:]


# ----------------------
# plot charge state distributions over radius and time
pylib.plot_tools.slider_plot(res['rV'], res['time'], nz.transpose(1,2,0), xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel='nz [A.U.]', labels=[str(i) for i in np.arange(0,nz.shape[1])], plot_sum=True, x_line=aurora_dict['rvol_lcfs'])


# plot radiation profiles over radius and time
pylib.plot_tools.slider_plot(res['rV'], res['time'], res['rad']['impurity_radiation'].transpose(1,2,0)[:nz.shape[1],:,:], xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel='Total radiation [A.U.]', labels=[str(i) for i in np.arange(0,nz.shape[1])], plot_sum=True, x_line=aurora_dict['rvol_lcfs'])

# plot Delta-Zeff profiles over radius and time
pylib.plot_tools.slider_plot(res['rV'], res['time'], delta_Zeff.transpose(1,2,0), xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'$\Delta$ $Z_{eff}$', labels=[str(i) for i in np.arange(0,nz.shape[1])], plot_sum=True,x_line=aurora_dict['rvol_lcfs'])


##############################
# Check particle conservation
##############################
import xarray
ds = xarray.Dataset({'impurity_density': ([ 'time', 'charge_states','radius_grid'], nz),
                     'impurity_radiation':  ([ 'time', 'charge_states','radius_grid'], res['rad']['impurity_radiation'][:,:Zmax+1,:]),
                     'source_function': (['time'], aurora_dict['source_function'] ),
                     'particles_in_divertor': (['time'], N_div), 
                     'particles_in_pump': (['time'], N_pump), 
                     'parallel_loss': (['time'], N_dsu), 
                     'parallel_loss_to_limiter': (['time'], N_dsul), 
                     'edge_loss': (['time'], N_tsu), 
                     'particles_at_wall': (['time'], N_wall), 
                     'particles_retained_at_wall': (['time'], N_ret), 
                     'recycling_from_wall':  (['time'], rclw_rate), 
                     'recycling_from_divertor':  (['time'], rcld_rate), 
                     'pro': (['radius_grid'], aurora_dict['pro']), 
                     'rhop_grid': (['radius_grid'], res['rhop'])
                     },
                    coords={'time': aurora_dict['time_out'], 
                            'radius_grid': res['rV'],
                            'charge_states': np.arange(nz.shape[1])
                            })

_ = pylib.particle_conserv.plot_1d(ds = ds)



pylib.animate.animate_aurora(res['rhop'], res['time'], nz.transpose(1,2,0), xlabel=r'$\rho_p$', ylabel='t={:.4f} [s]', zlabel=r'$n_z$ [A.U.]', labels=[str(i) for i in np.arange(0,nz.shape[1])], plot_sum=True, save_filename='test')
