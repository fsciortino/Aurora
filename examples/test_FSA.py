'''
Script to test quality of flux-surface-averaging (FSA) procedure by evaluating the normalized ionization rate, 
or equivalently by comparing the ionization rate of impurities with the parallel transport transit rate. 
This is inspired by Dux et al. NF 2020.

It is recommended to run this in IPython.
'''

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import omfit_eqdsk
import pickle as pkl
import scipy,sys,os
import time
from scipy.interpolate import interp1d
from omfit_commonclasses.utils_math import atomic_element

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

# set no sources of impurities
namelist['source_type'] = 'const'
namelist['Phi0'] = 1e24 #1.0

# Set up for 1 ion:
imp = namelist['imp'] = 'Ca' #'Ar' #'Ca'

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)


# plot normalized ionization frequency for the last time point, only within LCFS:
rhop_in = asim.rhop_grid[asim.rhop_grid<1.0]
S_z = asim.S_rates[asim.rhop_grid<1.0,:,-1] # take last time point
q_prof = interp1d(geqdsk['AuxQuantities']['RHOp'], geqdsk['QPSI'])(rhop_in)
Rhfs,Rlfs = aurora.grids_utils.get_HFS_LFS(geqdsk, rho_pol_arb=rhop_in)
R_prof = (Rhfs+Rlfs)/2.   # take as average of flux surface


# get impurity mass
out = atomic_element(symbol=imp)
spec = list(out.keys())[0]
imp_A = int(out[spec]['A'])

Ti_prof = asim.Te[-1, asim.rhop_grid<1.0] # use Ti=Te, only last time point

eps_prof = (Rlfs-geqdsk['RMAXIS'])/geqdsk['RMAXIS'] # use LFS radius for considerations on trapped particles
nu_ioniz_star = aurora.atomic.plot_norm_ion_freq( S_z, q_prof, R_prof, imp_A, Ti_prof,
                                                      rhop=rhop_in, plot=True, eps_prof=eps_prof)


# get average over charge states using fractional abundances in ionization equilibrium (no transport)

# average over time
ne_avg = np.mean(kin_profs['ne']['vals'],axis=0)
Te_avg = np.mean(kin_profs['Te']['vals'],axis=0)  # assume on the same radial basis as ne_avg

# get_frac_abundances takes inputs in m^-3 and eV
atom_data = aurora.get_atom_data(imp,['acd','scd'])
logTe, fz = aurora.atomic.get_frac_abundances(atom_data, ne_avg*1e6, Te_avg, rho=rhop)

fz_profs = np.zeros_like(S_z)
for cs in np.arange(S_z.shape[1]):
    fz_profs[:,cs] = interp1d(rhop, fz[:,cs])(rhop_in)

nu_ioniz_star = aurora.atomic.plot_norm_ion_freq( S_z, q_prof, R_prof, m_imp, Ti_prof,
                                                      nz_profs=fz_profs, rhop=rhop_in, plot=True,
                                                      eps_prof=eps_prof)
