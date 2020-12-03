'''
Script to obtain fractional abundances from Aurora's reading of ADAS data.
Note that this is a calculation from atomic physics only, i.e. no transport is applied.

It is recommended to run this in IPython.
'''
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import omfit_gapy
import scipy,sys,os
import time
from scipy.interpolate import interp1d

# Make sure that package home is added to sys.path
sys.path.append('../')
import aurora

try: # pass any argument via the command line to show plots
    plot = len(sys.argv)>1
except:
    plot = False

# read in some kinetic profiles
examples_dir = os.path.dirname(os.path.abspath(__file__))
inputgacode = omfit_gapy.OMFITgacode(examples_dir+'/example.input.gacode')

# transform rho_phi (=sqrt toroidal flux) into rho_psi (=sqrt poloidal flux) and save kinetic profiles
rhop = np.sqrt(inputgacode['polflux']/inputgacode['polflux'][-1])
ne_vals = inputgacode['ne']*1e13 # 1e19 m^-3 --> cm^-3
Te_vals = inputgacode['Te']*1e3  # keV --> eV

# get charge state distributions from ionization equilibrium for Ca
atom_data = aurora.atomic.get_atom_data('Ca',['scd','acd'])

# get fractional abundances on ne (cm^-3) and Te (eV) grid
logTe, fz, rates = aurora.atomic.get_frac_abundances(
    atom_data, ne_vals, Te_vals, rho=rhop, plot=plot)

# compare to fractial abundances obtained with ne*tau=0.1e20 m^-3.s
logTe, fz, rates = aurora.atomic.get_frac_abundances(
    atom_data, ne_vals, Te_vals, rho=rhop, ne_tau=0.1e19,
    plot=plot, ax=plt.gca() if plot else None, ls='--')
