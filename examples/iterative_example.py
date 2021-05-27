'''
Script demonstrating use of Aurora with explicit radial profiles of impurity neutral sources, 
also allowing users to vary the ne,Te grids based on arbitrary heating, cooling or dilution 
processes. This may be useful, for example, for pellet ablation or massive gas injection studies. 

Run this in IPython, or uncomment plt.show(block=True) at the end. 

jmcclena and sciortino, Nov 2020
'''
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from omfit_classes import omfit_eqdsk, omfit_gapy
import sys, copy, os
from scipy.interpolate import interp1d

# Make sure that package home is added to sys.path
sys.path.append('../')
import aurora


def get_nall(rhop, rhop_grid, ne, nd, nz):
    # This function gives total ion+electron density
    nz_interp = np.zeros([len(rhop),len(nz[0,:])])
    for Z in range(len(nz[0,:])):
        nz_interp[:,Z] = np.interp(rhop, rhop_grid, nz[:,Z])

    nall = ne + nd
    Znz = np.zeros(len(rhop))
    for Z in range(len(nz_interp[0,:])):
        nall += nz_interp[:,Z]
        Znz += Z*nz_interp[:,Z]

    nall += nall + Znz

    return nall, Znz
    
def dilution_cooling(rhop, rhop_grid, ne_old, nd_old, T_old, nz_old, nz_new):
    # This function reduces temperature by the increase in density
    # Assumption of Ti=Te for simplicity

    nall_old, Znz_old  = get_nall(rhop, rhop_grid, ne_old, nd_old, nz_old)
    nall_new, Znz_new = get_nall(rhop, rhop_grid, ne_old, nd_old, nz_new)

    ne_new = ne_old - Znz_old + Znz_new
    T_new = T_old*nall_old/nall_new

    return ne_new, T_new

def radiation_cooling(rhop, rhop_grid, ne, nd, nz, T, Erad):
    # This function subracts radiation power from plasma energy
    nall, ZnZ = get_nall(rhop, rhop_grid, ne, nd, nz)
    Erad = np.interp(rhop, rhop_grid, Erad)
    T_new = T - 2*Erad/(3*nall) # 2/3 to account for stored energy = 3/2 integral p dV
    return T_new

# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()

# Use gfile and statefile in local directory:
examples_dir = os.path.dirname(os.path.abspath(__file__))
geqdsk = omfit_eqdsk.OMFITgeqdsk(examples_dir+'/example.gfile')
inputgacode = omfit_gapy.OMFITgacode(examples_dir+'/example.input.gacode')

# save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid
kp = namelist['kin_profs']
kp['Te']['rhop'] = rhop = kp['ne']['rhop'] = np.sqrt(inputgacode['polflux']/inputgacode['polflux'][-1])
kp['ne']['vals'] = ne_cm3 = inputgacode['ne']*1e13 # 1e19 m^-3 --> cm^-3
kp['Te']['vals'] = Te_eV = inputgacode['Te']*1e3  # keV --> eV
nd_cm3 = copy.deepcopy(ne_cm3)

################## Simulation time steps and duration settings ##################
#
# Update background every n_rep iterations, each of dt [s] length
n_rep = 5
dt = 1e-4

# Total time to run [s] -- will be approximated by nearest multiplier of n_rep*dt
sim_time = 5e-3
num_sims = int(sim_time/(n_rep*dt))
##################################################################################

# do only a few time steps per "run"
namelist['timing'] = {'dt_increase': np.array([1., 1.   ]),
                      'dt_start': np.array([dt, sim_time]),
                      'steps_per_cycle': np.array([1, 1]),
                      'times': np.array([0. , n_rep*dt])}


# set impurity species and sources rate
imp = namelist['imp'] = 'Ar'

'''
# provide impurity neutral sources on explicit radial and time grids
namelist['explicit_source_time'] = np.linspace(0.,namelist['timing']['times'][-1]*n_rep,99)
namelist['explicit_source_rhop'] = np.linspace(0,1.3,101)
gaussian_rhop = 1e9 * np.exp(- (namelist['explicit_source_rhop']-0.5)**2/(2*0.1**2))
exp_time = np.exp(- namelist['explicit_source_time']/0.02)  # decay over 20ms time scale
namelist['explicit_source_vals'] = gaussian_rhop[None,:]*exp_time[:,None]
'''

# provide explicit impurity neutral sources only as a function of time; radial distribution defined by source_width_in/out
namelist['explicit_source_time'] = np.linspace(0.,namelist['timing']['times'][-1]*n_rep,99)
namelist['explicit_source_vals'] = 1e10 * np.exp(- namelist['explicit_source_time']/0.02)  # decay over 20ms time scale
namelist['source_width_in'] = 1.0
namelist['source_width_out'] = 5.0
namelist['source_cm_out_lcfs'] = -18.0 # cm inside of LCFS


# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)


fig,ax = plt.subplots(num='Impurity neutral source')
ax.contourf(asim.rhop_grid,
            asim.time_grid,
            asim.source_rad_prof.T)
ax.set_xlabel(r'$\rho_p$')
ax.set_ylabel('time [s]')


# set time-independent transport coefficients (flat D=1 m^2/s, V=-2 cm/s)
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -2e2 * np.ones(len(asim.rvol_grid)) # cm/s

# run Aurora forward model and plot results
out = asim.run_aurora(D_z, V_z, plot=False)
nz_all = out[0] # impurity charge state densities are the first element of "out"
nz_init = nz_all[:,:,-1]

# calculate dilution cooling
rad = aurora.compute_rad(
    imp, out[0].transpose(2,1,0), asim.ne, asim.Te, prad_flag=True)
tot_rad_dens = rad['tot'] # W/cm^3
line_rad_all = rad['line_rad'].T # W/cm^3
time_grid = copy.deepcopy(asim.time_grid)

# modify background temperature and density profiles based on tot_rad_dens
rhop_grid = asim.rhop_grid
Erad = np.trapz(tot_rad_dens[-1*n_rep:,:],axis=0,dx=dt) * 1.6e-13
Te_eV = radiation_cooling(rhop, rhop_grid, ne_cm3, nd_cm3, nz_init, Te_eV, Erad)
ne_cm3, Te_eV = dilution_cooling(rhop, rhop_grid, ne_cm3, nd_cm3, Te_eV, nz_init*0., nz_init)
kp['Te']['vals'] = Te_eV
kp['ne']['vals'] = ne_cm3

# update kinetic profile dependencies:
asim.setup_kin_profs_depts()

Te_all = []
Te_all.append(Te_eV)
for i in np.arange(num_sims):
    # Update time array
    asim.namelist['timing']['times'] = np.array([(i+1)*n_rep*dt+dt, (i+2)*n_rep*dt])
    asim.setup_grids()
    
    # get charge state densities from latest time step
    nz_old = nz_all[:,:,-1*n_rep]
    nz_init = nz_all[:,:,-1]
    out = asim.run_aurora(D_z, V_z, nz_init=nz_init, plot=False)
    nz_all = np.dstack((nz_all, out[0]))

    rad = aurora.compute_rad(
        imp, out[0].transpose(2,1,0), asim.ne, asim.Te, prad_flag=True)
    tot_rad_dens = rad['tot'] # W/cm^3
    line_rad_all = np.dstack((line_rad_all, rad['line_rad'].T))
    time_grid = np.concatenate((time_grid, asim.time_grid))
    
    # modify background temperature and density profiles
    rhop_grid = asim.rhop_grid
    Erad = np.trapz(tot_rad_dens[-1*n_rep:,:], axis=0) * 1.6e-13
    Te_eV = radiation_cooling(rhop, rhop_grid, ne_cm3, nd_cm3, nz_init, Te_eV, Erad)
    ne_cm3, Te_eV = dilution_cooling(rhop, rhop_grid, ne_cm3, nd_cm3, Te_eV, nz_old, nz_init)
    kp['Te']['vals'] = Te_eV
    kp['ne']['vals'] = ne_cm3
    Te_all.append(Te_eV)
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


_Te_all = np.array(Te_all).T
Te_arr = np.reshape(_Te_all, (1,len(rhop),len(time_grid[::5])))
aurora.slider_plot(rhop, time_grid[::5], Te_arr, xlabel=r'$rho_p$', ylabel='time [s]', zlabel=r'Te [eV]')

#plt.show(block=True)

