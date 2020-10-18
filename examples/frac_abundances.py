'''
Script to obtain fractional abundances from aurora's reading of ADAS data.
Note that this is a calculation from atomic physics only, i.e. no transport is applied.

It is recommended to run this in IPython.
'''

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pickle as pkl

# Make sure that package home is added to sys.path
import sys
sys.path.append('../')
import aurora

# example kinetic profiles
with open('./test_kin_profs.pkl','rb') as f:
    ne_profs,Te_profs = pkl.load(f)

ne_vals = ne_profs['ne']*1e14  # 10^20 m^-3 --> cm^-3
rhop = ne_profs['rhop']
Te_vals = Te_profs['Te']*1e3  # keV --> eV

# average over time
ne_avg = np.mean(ne_vals,axis=0)
Te_avg = np.mean(Te_vals,axis=0)  # assume on the same radial basis as ne_avg


# get charge state distributions from ionization equilibrium for Ca
atom_data = aurora.atomic.get_all_atom_data('Ca',['acd','scd'])

# get_frac_abundances takes inputs in m^-3 and eV
logTe, fz = aurora.atomic.get_frac_abundances(atom_data, ne_avg*1e6, Te_avg, rho=rhop)
