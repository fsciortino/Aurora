'''
Script demonstrating use of Aurora with explicit radial profiles of impurity neutral sources, 
also allowing users to vary the ne,Te grids based on arbitrary heating, cooling or dilution 
processes. This may be useful, for example, for pellet ablation or massive gas injection studies. 

'''
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import omfit_eqdsk, omfit_gapy
import sys, copy
from scipy.interpolate import interp1d

# Make sure that package home is added to sys.path
import sys
sys.path.append('../')
import aurora

# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()

# Use gfile and statefile in local directory:
geqdsk = omfit_eqdsk.OMFITgeqdsk('example.gfile')
inputgacode = omfit_gapy.OMFITgacode('example.input.gacode')

# save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid
kp = namelist['kin_profs']
kp['Te']['rhop'] = kp['ne']['rhop'] = np.sqrt(inputgacode['polflux']/inputgacode['polflux'][-1])
kp['ne']['vals'] = inputgacode['ne']*1e13 # 1e19 m^-3 --> cm^-3
kp['Te']['vals'] = inputgacode['Te']*1e3  # keV --> eV

# set impurity species and sources rate
imp = namelist['imp'] = 'Ar'

# provide impurity neutral sources on explicit radial and time grids
namelist['explicit_source_time'] = np.linspace(0.,namelist['timing']['times'][-1],99)
namelist['explicit_source_rhop'] = np.linspace(0,1.3,101)
gaussian_rhop = 1e10 * np.exp(- (namelist['explicit_source_rhop']-0.5)**2/(2*0.1**2))
exp_time = np.exp(- namelist['explicit_source_time']/0.02)  # decay over 20ms time scale
namelist['explicit_source_vals'] = gaussian_rhop[None,:]*exp_time[:,None]

fig,ax = plt.subplots(num='Impurity neutral source')
ax.contourf(namelist['explicit_source_rhop'],
             namelist['explicit_source_time'],
             namelist['explicit_source_vals'])
ax.set_xlabel(r'$\rho_p$')
ax.set_ylabel('time [s]')

################## Simulation time steps and duration settings ##################
#
# Update background every n_rep iterations, each of dt [s] length
n_rep = 5
dt = 1e-5

# Total time to run [s] -- will be approximated by nearest multiplier of n_rep*dt
sim_time = 5e-3  # 20 ms
num_sims = int(sim_time/(n_rep*dt))
##################################################################################

# do only a few time steps per "run"
namelist['timing'] = {'dt_increase': np.array([1., 1.   ]),
                      'dt_start': np.array([1.e-05, 1.e-03]),
                      'steps_per_cycle': np.array([1, 1]),
                      'times': np.array([0. , n_rep*dt])}

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# set time-independent transport coefficients (flat D=1 m^2/s, V=-2 cm/s)
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -2e2 * np.ones(len(asim.rvol_grid)) # cm/s

# run Aurora forward model and plot results
out = asim.run_aurora(D_z, V_z, plot=False)
nz_all = out[0] # impurity charge state densities are the first element of "out"

# calculate dilution cooling
rad = aurora.compute_rad(
    imp, out[0].transpose(2,1,0), asim.ne, asim.Te, prad_flag=True)
tot_rad_dens = rad['tot'] # W/cm^3
line_rad_all = rad['line_rad'].T # W/cm^3
time_grid = copy.deepcopy(asim.time_grid)

# modify background temperature and density profiles based on tot_rad_dens
# TODO for user:
# namelist['Te']['vals'] =
# namelist['ne']['vals'] =

# update kinetic profile dependencies:
asim.setup_kin_profs_depts()


for i in np.arange(num_sims):
    # get charge state densities from latest time step
    nz_init = nz_all[:,:,-1]
    out = asim.run_aurora(D_z, V_z, nz_init=nz_init, plot=False)
    nz_all = np.dstack((nz_all, out[0]))

    rad = aurora.compute_rad(
        imp, out[0].transpose(2,1,0), asim.ne, asim.Te, prad_flag=True)
    tot_rad_dens = rad['tot'] # W/cm^3
    line_rad_all = np.dstack((line_rad_all, rad['line_rad'].T))
    time_grid = np.concatenate((time_grid, asim.time_grid))
    # modify background temperature and density profiles
    # TODO for user:
    # namelist['Te']['vals'] =
    # namelist['ne']['vals'] =

    # update kinetic profile dependencies:
    asim.setup_kin_profs_depts()
    

# plot charge state distributions over radius and time
aurora.plot_tools.slider_plot(asim.rvol_grid, time_grid, nz_all.transpose(1,0,2),
                              xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'$n_z$ [$cm^{-3}$]',
                              labels=[str(i) for i in np.arange(0,nz_all.shape[1])],
                              plot_sum=True, x_line=asim.rvol_lcfs)

# plot radiation profiles over radius and time
aurora.slider_plot(asim.rvol_grid, time_grid, line_rad_all.transpose(1,0,2),
                              xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel=r'Line radiation [$MW/m^3$]',
                              labels=[str(i) for i in np.arange(0,nz_all.shape[1])],
                              plot_sum=True, x_line=asim.rvol_lcfs)


