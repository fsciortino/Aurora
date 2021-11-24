'''
Script to test use of KN1D, also integrating it within an Aurora impurity transport simulation.

It is recommended to run this in IPython.

sciortino, Jan 2021
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
# NB: this example.input.gacode file does not contain edge kinetic profiles and does not have a reasonable Te at the LCFS
# In this example, we will make up all kinetic data in the SOL based on exponential decay lengths.

# save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid
kp = namelist['kin_profs']
kp['Te']['rhop'] = kp['ne']['rhop'] = np.sqrt(inputgacode['polflux']/inputgacode['polflux'][-1])
kp['ne']['vals'] = inputgacode['ne']*1e13 # 1e19 m^-3 --> cm^-3
kp['Te']['vals'] = inputgacode['Te']*1e3  # keV --> eV

# set impurity species and sources rate
imp = namelist['imp'] = 'Ar' #'C'
namelist['source_type'] = 'const'
namelist['source_rate'] = 2e20  # particles/s

# setup Aurora simulation without neutrals and CXR
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# estimate connection lengths from the EFIT g-EQDSK
clen_divertor_m, clen_limiter_m = aurora.estimate_clen(geqdsk)

# Estimate radial separation of boundary to separatrix and limiter to separatrix.
# This can be done by hand if desired, or using the following Aurora function for a specific device/shot/time
# combination using the EFIT a-EQDSK
#bound_sep_cm, lim_sep_cm = aurora.grids_utils.estimate_boundary_distance(my_shot, my_device, time_ms.)
bound_sep_cm = 1.5
lim_sep_cm = 1.0
p_H2_mTorr=1.0 # mTorr
innermost_rmid_cm=2.0

kn1d_res = aurora.run_kn1d(kp['ne']['rhop'], kp['ne']['vals'], kp['Te']['vals'], kp['Te']['vals'],
                           geqdsk, p_H2_mTorr, clen_divertor_m*1e2, clen_limiter_m*1e2,
                           bound_sep_cm, lim_sep_cm, innermost_rmid_cm, plot_kin_profs=True)

# series of plots to visualize (processed) KN1D output
aurora.kn1d.plot_overview(kn1d_res)     # overview of inputs and outputs
aurora.kn1d.plot_exc_states(kn1d_res)   # excited states
aurora.kn1d.plot_emiss(kn1d_res)        # Ly-a and D-a emission profiles
aurora.kn1d.plot_transport(kn1d_res)    # gradient scale lengths and "effective transport"

# add atomic neutral density to inputs and turn on charge exchange recombination
kp['n0']['rhop'] = kp['ne']['rhop']

# interpolate and convert from m^-3 to cm^-3
n0 = interp1d(kn1d_res['kn1d_profs']['rhop'],kn1d_res['kn1d_profs']['n0']*1e-6)(kp['Te']['rhop'])
kp['n0']['vals'] = n0[None,:] # need to set to (time,radius) grid, here time-independent
namelist['cxr_flag'] = True

# Now get aurora setup
asim_cxr = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# set time-independent transport coefficients (flat D=1 m^2/s, V=-2 cm/s)
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -2e2 * np.ones(len(asim.rvol_grid)) # cm/s

# run Aurora forward model with and without CXR
out = asim.run_aurora(D_z, V_z, plot=plot)
out_cxr = asim_cxr.run_aurora(D_z, V_z, plot=plot)

nz = out[0]
nz_cxr = out_cxr[0]

#ls_cycle = aurora.get_ls_cycle()
c_cycle = aurora.get_color_cycle()

fig, ax = plt.subplots()
for cs in np.arange(nz.shape[1]):
    col = next(c_cycle)
    ax.plot(asim.rhop_grid, nz[:,cs,-1], c=col, ls='--')
    ax.plot(asim_cxr.rhop_grid, nz_cxr[:,cs,-1], c=col, ls='-', label=f'{imp}{cs}+')
ax.plot([],[], 'w-', label=' ')
ax.plot([],[],'k-', label='with CXR')
ax.plot([],[],'k--', label='without CXR')
ax.legend(loc='best').set_draggable(True)  
ax.set_xlabel(r'$\rho_p$')
ax.set_ylabel(r'$n_z$ [$cm^{-3}$]')
ax.set_xlim([0.85,np.max(asim.rhop_grid)])


# add radiation
asim.rad = aurora.compute_rad(imp, nz.transpose(2,1,0), asim.ne, asim.Te,
                              prad_flag=True, thermal_cx_rad_flag=False)
asim_cxr.rad = aurora.compute_rad(imp, nz_cxr.transpose(2,1,0), asim.ne, asim.Te,
                                  n0=asim_cxr.n0,Ti=asim_cxr.Ti, 
                                  prad_flag=True, thermal_cx_rad_flag=True)

c_cycle = aurora.get_color_cycle()
fig, ax = plt.subplots()
for cs in np.arange(nz.shape[1]-1):  # fully-stripped ions have no line radiation
    col = next(c_cycle)
    ax.plot(asim.rhop_grid, asim.rad['line_rad'][-1,cs,:], c=col, ls='--')
    ax.plot(asim_cxr.rhop_grid, asim_cxr.rad['line_rad'][-1,cs,:], c=col, ls='-', label=f'{imp}{cs}+')
ax.plot([],[], 'w-', label=' ')
ax.plot([],[],'k-', label='with CXR')
ax.plot([],[],'k--', label='without CXR')
ax.legend(loc='best').set_draggable(True)  
ax.set_xlabel(r'$\rho_p$')
ax.set_ylabel(r'$P_{rad}$ [$W/m^3$]')
ax.set_xlim([0.85,np.max(asim.rhop_grid)])


# Check total radiated power components
rhop = np.linspace(np.min(asim.rhop_grid),1.0, 303)
ne = interp1d(asim.rhop_grid, asim.ne[0,:], kind='cubic')(rhop)
Te = interp1d(asim.rhop_grid, asim.Te[0,:], kind='cubic')(rhop)
Ti = interp1d(asim.rhop_grid, asim.Ti[0,:], kind='cubic')(rhop)
n0 = interp1d(asim.rhop_grid, asim_cxr.n0[0,:], kind='cubic')(rhop)

# get flux surface volumes and coordinates
grhop = np.sqrt(geqdsk['fluxSurfaces']['geo']['psin'])
gvol = geqdsk['fluxSurfaces']['geo']['vol']

# interpolate on our grid
vol = interp1d(grhop, gvol)(rhop)

nz_interp = np.zeros((len(rhop),nz.shape[1]))
nz_cxr_interp = np.zeros((len(rhop),nz.shape[1]))

for cs in np.arange(nz.shape[1]):
    nz_interp[:,cs] = interp1d(asim.rhop_grid, nz[:,cs,-1])(rhop)
    nz_cxr_interp[:,cs] = interp1d(asim.rhop_grid, nz_cxr[:,cs,-1])(rhop)

# get total radiated power components
rad_model = aurora.radiation_model(imp, rhop, ne, Te, geqdsk, nz_cm3=nz_interp, plot=True)
rad_model_cxr = aurora.radiation_model(imp, rhop, ne, Te, geqdsk, nz_cm3=nz_cxr_interp,
                                       n0_cm3=n0, Ti_eV=Ti)

# print to screen 0-D prediction of ion total radiated power
print(f'Prad = {np.round(rad_model["Prad"]/1e3,3)} kW without neutrals')
print(f'Prad = {np.round(rad_model_cxr["Prad"]/1e3,3)} kW with neutrals')
