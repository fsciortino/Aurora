'''
Script to test functionality from namelist creation to run and postprocessing.

It is recommended to run this in IPython.
'''

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from omfit_classes import omfit_eqdsk, omfit_gapy
import pickle as pkl
import scipy,sys,os
import time
from copy import deepcopy

# Make sure that package home is added to sys.path
sys.path.append('../')
import aurora

# number of repetitions to accurately time runs
num=10


# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()
kp = namelist['kin_profs']

# Use gfile and statefile in local directory:
examples_dir = os.path.dirname(os.path.abspath(__file__))
geqdsk = omfit_eqdsk.OMFITgeqdsk(examples_dir+'/example.gfile')
inputgacode = omfit_gapy.OMFITgacode(examples_dir+'/example.input.gacode')

# transform rho_phi (=sqrt toroidal flux) into rho_psi (=sqrt poloidal flux) and save kinetic profiles
kp['Te']['rhop'] = kp['ne']['rhop'] = np.sqrt(inputgacode['polflux']/inputgacode['polflux'][-1])
kp['ne']['vals'] = inputgacode['ne'][None,:]*1e13 # 1e19 m^-3 --> cm^-3
kp['Te']['vals'] = inputgacode['Te'][None,:]*1e3  # keV --> eV

# set impurity species and sources rate
imp = namelist['imp'] = 'C'
namelist['source_type'] = 'const'
namelist['source_rate'] = 1e21  # particles/s

# Change radial resolution from default:
# namelist['dr_0']=0.3
# namelist['dr_1']=0.05

# Change time resolution from default:
# namelist['timing']['dt_increase'] = np.array([1.01, 1.])
# namelist['timing']['dt_start'] = np.array([1e-5, 0.001])
# namelist['timing']['steps_per_cycle'] = np.array([1,1])
# namelist['timing']['times'] = np.array([0.,0.2])

# Now get aurora setup (let setup process create inputs needed to produce radial and temporal grids)
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# Visualize radial grid
rvol_grid, pro_grid, qpr_grid, prox_param = aurora.grids_utils.create_radial_grid(namelist,plot=True)

# Visualize time resolution
time_grid, save_grid = aurora.grids_utils.create_time_grid(namelist['timing'], plot=True)

# set time-independent transport coefficients (flat D=1 m^2/s, V=-2 cm/s)
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -2e2 * asim.rhop_grid**10 # cm/s, increasing towards the edge

# plot transport coefficients
fig,ax = plt.subplots(2,1, sharex=True, figsize=(8,8))
ax[0].plot( asim.rhop_grid, D_z/1e4)
ax[1].plot( asim.rhop_grid, V_z/1e2)
ax[1].set_xlabel(r'$\rho_p$')
ax[0].set_ylabel(r'$D$ [$m^2/s$]')
ax[1].set_ylabel(r'$v$ [$m/s$]')
plt.subplots_adjust(wspace=0, hspace=0)


####### Finite-differences method #########
start = time.time()
for n in np.arange(num):
    out = asim.run_aurora(D_z, V_z, alg_opt=0)
print('Average FD time per run: ', (time.time() - start)/num)
nz = out[0]   # extract only charge state densities in the plasma from output -- (time,nZ,space)

# plot charge state distributions over radius and time
aurora.plot_tools.slider_plot(asim.rvol_grid, asim.time_out, nz.transpose(1,0,2),
                              xlabel=r'$r_V$ [cm]', ylabel='time [s]',
                              zlabel=r'$n_z$ [$cm^{-3}$]',
                              labels=[str(i) for i in np.arange(0,nz.shape[1])],
                              plot_sum=True, x_line=asim.rvol_lcfs)

# Check particle conservation
out_fd, axs = asim.check_conservation(plot=True)




####### Finite volumes (Linder) method #########
start = time.time()
for n in np.arange(num):
    out_2 = asim.run_aurora(D_z, V_z, alg_opt=1, evolneut=False) 
print('Average FV time per run: ', (time.time() - start)/num)
nz_2 = out_2[0]

#plot charge state distributions over radius and time
aurora.plot_tools.slider_plot(asim.rvol_grid, asim.time_out, nz_2.transpose(1,0,2),
                              xlabel=r'$r_V$ [cm]', ylabel='time [s]',
                              zlabel=r'$n_z$ [$cm^{-3}$]',
                              labels=[str(i) for i in np.arange(0,nz_2.shape[1])],
                              plot_sum=True, x_line=asim.rvol_lcfs)

# Check particle conservation
out_fv, axs = asim.check_conservation(plot=True, axs=axs)


######### Finite volumes (Linder) evolving neutrals -- under development! #########
start = time.time()
for n in np.arange(num):
    out_3 = asim.run_aurora(D_z, V_z, alg_opt=1, evolneut=True) 
print('Average FVN time per run: ', (time.time() - start)/num)
nz_3 = out_3[0]

# plot charge state distributions over radius and time
aurora.plot_tools.slider_plot(asim.rvol_grid, asim.time_out, nz_3.transpose(1,0,2),
                              xlabel=r'$r_V$ [cm]', ylabel='time [s]',
                              zlabel=r'$n_z$ [$cm^{-3}$]',
                              labels=[str(i) for i in np.arange(0,nz_3.shape[1])],
                              plot_sum=True, x_line=asim.rvol_lcfs)

# Check particle conservation
out_fvn, axs = asim.check_conservation(plot=True, axs=axs)


###########################################################################
#
#           Compare all algorithmic choices at last time slice
#
############################################################################
labels = [fr'{imp}$^{{{str(i)}}}$' for i in np.arange(0,nz_3.shape[1])]
colors = plt.cm.rainbow(np.linspace(0,1,nz.shape[1]))
fig = plt.figure()
fig.set_size_inches(10,7, forward=True)
a_plot = plt.subplot2grid((10,10),(0,0),rowspan = 10, colspan = 8, fig=fig) 
a_legend = plt.subplot2grid((10,10),(0,8),rowspan = 10, colspan = 2, fig=fig)
for ii,cc in zip(np.arange(nz.shape[1]),colors):
    a_plot.plot(asim.rhop_grid, nz[:,ii,-1].T, c=cc, ls='-')
    a_plot.plot(asim.rhop_grid, nz_2[:,ii,-1].T, c=cc, ls='--')

    #########
    factor = np.max(nz_2)/np.max(nz_3)   # factor needed to match neutral evolution to basic FV case
    a_plot.plot(asim.rhop_grid, factor* nz_3[:,ii,-1].T, c=cc, ls=':')
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
fig,ax = plt.subplots()
pcm = ax.pcolormesh(asim.rhop_grid, asim.time_out, nz.sum(axis=1).T, cmap='plasma', 
                   vmax=nz.sum(axis=1)[0,:].max(), shading='auto')
fig.colorbar(pcm) 
ax.set_xlabel(r"$\rho_p$")
ax.set_ylabel(r"$t$ [s]")
ax.set_title("Tot impurity density (Finite differences)")
ax.set_xlim([0.0,1.0])

fig,ax = plt.subplots()
pcm = ax.pcolormesh(asim.rhop_grid, asim.time_out, nz_2.sum(axis=1).T, cmap='plasma', 
                   vmax=nz_2.sum(axis=1)[0,:].max(), shading='auto')
fig.colorbar(pcm, extend='max')
ax.set_xlabel(r"$\rho_p$")
ax.set_ylabel(r"$t$ [s]")
ax.set_title("Tot impurity density (Finite volumes)")
ax.set_xlim([0.0,1.0])
