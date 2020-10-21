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
from copy import deepcopy

# Make sure that package home is added to sys.path
import sys
sys.path.append('../')
import aurora

# number of repetitions to accurately time runs
num=1

###########
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

# set no sources of impurities
namelist['source_type'] = 'const'
namelist['Phi0'] = 1e24 #1.0

# Set up for 1 ion:
imp = namelist['imp'] = 'Ar' # 'W' #'Ca' #'Ar' #'Ca'

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# Choose radial resolution
namelist['dr_0']=0.1 #0.3
namelist['dr_1']=0.01   # 0.05
aurora.grids_utils.create_radial_grid(namelist,plot=True)

# Choose time resolution
namelist['timing']['dt_increase'] = np.array([1., 1.])
namelist['timing']['dt_start'] = np.array([0.0001, 0.001])
namelist['timing']['steps_per_cycle'] = np.array([1,1])
namelist['timing']['times'] = np.array([1.,2.])
aurora.grids_utils.create_time_grid(namelist['timing'], plot=True)


# choose transport coefficients
D_eff = 1e4 #cm^2/s
v_eff = 0.0 #-3e4 #-1e2 #-2e2 #cm/s

# # set transport coefficients to the right format
D_z = np.ones_like(asim.rvol_grid) * D_eff
V_z = asim.rhop_grid**10 * v_eff  # increasing towards edge

# plot transport coefficients
fig,ax = plt.subplots(2,1, sharex=True, figsize=(8,8))
ax[0].plot( asim.rhop_grid, D_z/1e4)
ax[1].plot( asim.rhop_grid, V_z/1e2)
ax[1].set_xlabel(r'$\rho_p$')
ax[0].set_ylabel(r'$D$ [$m^2/s$]')
ax[1].set_ylabel(r'$v$ [$m/s$]')
plt.subplots_adjust(wspace=0, hspace=0)


####### Original method #########
# # set transport coefficients to the right format
D_z = np.ones_like(asim.rvol_grid) * D_eff
V_z = asim.rhop_grid**4 * v_eff  # increasing towards edge

start = time.time()
for n in np.arange(num):
    out = asim.run_aurora(D_z, V_z)
print('Average time per run: ', (time.time() - start)/num)
nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out
nz = nz.transpose(2,1,0)   # time,nZ,space
print('np.mean(nz): ' ,np.mean(nz))


# ----------------------
# plot charge state distributions over radius and time
aurora.plot_tools.slider_plot(asim.rvol_grid, asim.time_out, nz.transpose(1,2,0),
                             xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel='nz [A.U.]',
                             labels=[str(i) for i in np.arange(0,nz.shape[1])],
                             plot_sum=True, x_line=asim.rvol_lcfs)

# Check particle conservation
axs = asim.check_conservation()




####### Linder method #########
start = time.time()
for n in np.arange(num):
    out_2 = asim.run_aurora(D_z, V_z, method='linder', evolneut=False) 
print('Average time per run: ', (time.time() - start)/num)
nz_2, N_wall_2, N_div_2, N_pump_2, N_ret_2, N_tsu_2, N_dsu_2, N_dsul_2, rcld_rate_2, rclw_rate_2 = out_2
nz_2 = nz_2.transpose(2,1,0)   # time,nZ,space
print('np.mean(nz_2): ' ,np.mean(nz_2))


# ----------------------
#plot charge state distributions over radius and time
aurora.plot_tools.slider_plot(asim.rvol_grid, asim.time_out, nz_2.transpose(1,2,0),
                             xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel='nz [A.U.]',
                             labels=[str(i) for i in np.arange(0,nz_2.shape[1])],
                             plot_sum=True, x_line=asim.rvol_lcfs)

# Check particle conservation
axs = asim.check_conservation()

######################################
#### Plot difference between the two algorithms with slider  #####
# aurora.plot_tools.slider_plot(asim.rvol_grid, asim.time_out,
#                              np.abs(nz.transpose(1,2,0) - nz_2.transpose(1,2,0)),
#                              xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel='$\Delta$ nz [A.U.]',
#                              labels=[str(i) for i in np.arange(0,nz_2.shape[1])],
#                              plot_sum=True, x_line=asim.rvol_lcfs)



######################################
# ####### Linder method evoling neutrals #########
start = time.time()
for n in np.arange(num):
    out_3 = asim.run_aurora(D_z, V_z, method='linder', evolneut=True) 
print('Average time per run: ', (time.time() - start)/num)
nz_3, N_wall_3, N_div_3, N_pump_3, N_ret_3, N_tsu_3, N_dsu_3, N_dsul_3, rcld_rate_3, rclw_rate_3 = out_3
nz_3 = nz_3.transpose(2,1,0)   # time,nZ,space
print('np.mean(nz_3): ' ,np.mean(nz_3))


# ----------------------
# plot charge state distributions over radius and time
aurora.plot_tools.slider_plot(asim.rvol_grid, asim.time_out, nz_3.transpose(1,2,0),
                             xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel='nz [A.U.]',
                             labels=[str(i) for i in np.arange(0,nz_3.shape[1])],
                             plot_sum=True, x_line=asim.rvol_lcfs)

# # Check particle conservation
axs = asim.check_conservation()


##############################################

### Compare all algorithmic choices at last time slice:
labels = [fr'{imp}$^{{{str(i)}}}$' for i in np.arange(0,nz_3.shape[1])]
colors = plt.cm.rainbow(np.linspace(0,1,nz.shape[1]))
fig = plt.figure()
fig.set_size_inches(10,7, forward=True)
a_plot = plt.subplot2grid((10,10),(0,0),rowspan = 10, colspan = 8, fig=fig) 
a_legend = plt.subplot2grid((10,10),(0,8),rowspan = 10, colspan = 2, fig=fig)
for ii,cc in zip(np.arange(nz.shape[1]),colors):
    a_plot.plot(asim.rhop_grid, nz[-1,ii,:].T, c=cc, ls='-')
    a_plot.plot(asim.rhop_grid, nz_2[-1,ii,:].T, c=cc, ls='--')

    #########
    factor = np.max(nz_2)/np.max(nz_3)   # why do we need this factor?
    print(f'Factor: {factor}')
    a_plot.plot(asim.rhop_grid, factor* nz_3[-1,ii,:].T, c=cc, ls=':')
    #a_plot.plot(asim.rhop_grid, nz_3[-1,ii,:].T, c=cc, ls=':')
    ########
    a_legend.plot([],[],c=cc,label=labels[ii],ls='-')
a_legend.plot([],[], c='k', ls='-',lw=2, label='Original')
a_legend.plot([],[], c='k', ls='--',lw=2, label='Linder w/o n')
a_legend.plot([],[], c='k', ls=':',lw=2, label='Linder w/ n')

a_legend.legend(loc='best').set_draggable(True)
a_plot.set_xlabel(r'$\rho_p$')
a_plot.set_ylabel(r'$n_z$ [$cm^{-3}$]')
a_legend.axis('off')
fig.suptitle('Algorithm comparison')



######## images/plots of total density ############
f = plt.figure()
a = f.add_subplot(1, 1, 1)
cmap = 'plasma'
pcm = a.pcolormesh(asim.rhop_grid, asim.time_out, nz.sum(axis=1), cmap=cmap, 
                   vmax=nz.sum(axis=1)[:, 0].max())
pcm.cmap.set_over('white')
f.colorbar(pcm, extend='max')
a.set_xlabel(r"$\rho_p$")
a.set_ylabel(r"$t$ [s]")
a.set_title("Tot impurity density (original)")
a.set_xlim([0.0,1.0])

f = plt.figure()
a = f.add_subplot(1, 1, 1)
cmap = 'plasma'
pcm = a.pcolormesh(asim.rhop_grid, asim.time_out, nz_2.sum(axis=1), cmap=cmap, 
                   vmax=nz_2.sum(axis=1)[:, 0].max())
pcm.cmap.set_over('white')
f.colorbar(pcm, extend='max')
a.set_xlabel(r"$\rho_p$")
a.set_ylabel(r"$t$ [s]")
a.set_title("Tot impurity density (Linder)")
a.set_xlim([0.0,1.0])
