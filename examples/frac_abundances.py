'''
Script to obtain fractional abundances from Aurora's reading of ADAS data.
Note that this is a calculation from atomic physics only, i.e. no transport is applied.

It is recommended to run this in IPython.
'''
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from omfit_classes import omfit_gapy
import scipy,sys,os
import time
from scipy.interpolate import interp1d
from matplotlib import cm

# Make sure that package home is added to sys.path
sys.path.append('../')
import aurora

try: # pass any argument via the command line to show plots
    plot = len(sys.argv)>1
except:
    plot = False

ion = 'Al'

# create some arbitrary kinetic profiles for this example
# parameterization f=(f_center-f_edge)*(1-rho**alpha1)**alpha2 + f_edge
T_core = 5e3 # eV
T_edge = 100 # eV
T_alpha1 = 2.
T_alpha2=1.5
n_core = 1e14 # cm^-3
n_edge = 0.4e14 # cm^-3
n_alpha1 = 2
n_alpha2 = 0.5

# now define kinetic profiles
rhop = np.linspace(0, 1, 100)
ne_cm3 = (n_core - n_edge)*(1-rhop**n_alpha1)**n_alpha2 + n_edge
Te_eV = (T_core - T_edge)*(1-rhop**T_alpha1)**T_alpha2 + T_edge

# get charge state distributions from ionization equilibrium for Ca
atom_data = aurora.atomic.get_atom_data(ion,['scd','acd', 'ccd'])

# get fractional abundances on ne (cm^-3) and Te (eV) grid
_Te, fz = aurora.atomic.get_frac_abundances(atom_data, ne_cm3, Te_eV, rho=rhop, plot=plot)

# include effect of CX with a given (here, arbitrary) density of neutrals
n0_by_ne = (1e-2*np.exp(rhop**5-1))**2  # arbitrary, exponentially decreasing from LCFS
_Te, fz = aurora.atomic.get_frac_abundances(atom_data, ne_cm3, Te_eV, n0_by_ne,
                                            rho=rhop, plot=plot, ax = plt.gca() if plot else None)

# compare to fractial abundances obtained with ne*tau=1e19 m^-3.s
# Use the get_atomic_relax_time function to use a finite ne*tau value
_Te, fz, rate_coeffs = aurora.get_atomic_relax_time(atom_data, ne_cm3, Te_eV, ne_tau=1e19, plot=False)

if plot:
    # overplot using cubic interpolation for smoother visualization:
    plt.gca().set_prop_cycle('color',cm.plasma(np.linspace(0,1,fz.shape[1])))
    x_fine = np.linspace(np.min(rhop), np.max(rhop),10000)
    fz_i = interp1d(rhop, fz, kind='cubic',axis=0)(x_fine)
    plt.gca().plot(x_fine, fz_i, ls='--')
    
# plot atomic relaxation time over Te (no transport) at a single value of ne
_Te, fz, rate_coeffs = aurora.get_atomic_relax_time(
    atom_data, np.ones(100)*1e14, np.logspace(1,4,100), ne_tau=np.inf, plot=plot)
