'''
Script to test quality of flux-surface-averaging (FSA) procedure by evaluating the normalized ionization rate, or equivalently by comparing the ionization rate of impurities with the parallel transport transit rate. 
This is inspired by Dux et al. NF 2020.

It is recommended to run this in IPython.
'''
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from omfit_classes import omfit_eqdsk, omfit_gapy
import scipy,sys,os
import time
from scipy.interpolate import interp1d
plt.style.use('/home/sciortino/SPARC/sparc_plots.mplstyle')
# Make sure that package home is added to sys.path
sys.path.append('../')
import aurora

# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()
kp = namelist['kin_profs']

# Use gfile and statefile in local directory:
examples_dir = os.path.dirname(os.path.abspath(__file__))
geqdsk = omfit_eqdsk.OMFITgeqdsk(examples_dir+'/example.gfile')
inputgacode = omfit_gapy.OMFITgacode(examples_dir+'/example.input.gacode')

# transform rho_phi (=sqrt toroidal flux) into rho_psi (=sqrt poloidal flux) and save kinetic profiles
rhop = kp['Te']['rhop'] = kp['ne']['rhop'] = np.sqrt(inputgacode['polflux']/inputgacode['polflux'][-1])
kp['ne']['vals'] = inputgacode['ne'][None,:]*1e13 # 1e19 m^-3 --> cm^-3
kp['Te']['vals'] = inputgacode['Te'][None,:]*1e3  # keV --> eV

# set impurity species and sources rate
imp = namelist['imp'] = 'F' #'Ar'
namelist['source_type'] = 'const'
namelist['source_rate'] = 1e24

# Setup aurora sim to efficiently setup atomic rates and profiles over radius
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# plot normalized ionization frequency for the last time point, only within LCFS:
rhop_in = asim.rhop_grid[asim.rhop_grid<1.0]
S_z = asim.S_rates[asim.rhop_grid<1.0,:,-1] # take last time point
q_prof = interp1d(geqdsk['AuxQuantities']['RHOp'], geqdsk['QPSI'])(rhop_in)
Rhfs,Rlfs = aurora.grids_utils.get_HFS_LFS(geqdsk, rho_pol=rhop_in)
R_prof = (Rhfs+Rlfs)/2.   # take as average of flux surface

Ti_prof = asim.Te[-1, asim.rhop_grid<1.0] # use Ti=Te, only last time point

# inverse aspect ratio profile
eps_prof = (Rlfs-geqdsk['RMAXIS'])/geqdsk['RMAXIS'] # use LFS radius for considerations on trapped particles
nu_ioniz_star = aurora.atomic.plot_norm_ion_freq( S_z, q_prof, R_prof, asim.A_imp, Ti_prof,
                                                      rhop=rhop_in, plot=True, eps_prof=eps_prof)


# get average over charge states using fractional abundances in ionization equilibrium (no transport)
ne_avg = np.mean(kp['ne']['vals'],axis=0) # average over time
Te_avg = np.mean(kp['Te']['vals'],axis=0) # assume on the same radial basis as ne_avg

# get fractional abundances on ne (cm^-3) and Te (eV) grid
atom_data = aurora.get_atom_data(imp,['acd','scd'])
_Te, fz = aurora.atomic.get_frac_abundances(atom_data, ne_avg, Te_avg, rho=rhop)

fz_profs = np.zeros_like(S_z)
for cs in np.arange(S_z.shape[1]):
    fz_profs[:,cs] = interp1d(rhop, fz[:,cs])(rhop_in)

nu_ioniz_star = aurora.atomic.plot_norm_ion_freq(S_z, q_prof, R_prof, asim.A_imp, Ti_prof,
                                                 nz_profs=fz_profs, rhop=rhop_in, plot=True,
                                                 eps_prof=eps_prof)
