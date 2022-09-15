"""
Script to test simple functionality of radiation models.

It is recommended to run this in IPython.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.ion()
from omfit_classes import omfit_eqdsk
import sys, os
from scipy.interpolate import interp1d

# Make sure that package home is added to sys.path
sys.path.append("../")
import aurora

try:  # pass any argument via the command line to show plots
    plot = len(sys.argv) > 1
except:
    plot = False

# Use gfile and statefile in local directory:
examples_dir = os.path.dirname(os.path.abspath(__file__))
geqdsk = omfit_eqdsk.OMFITgeqdsk(examples_dir + "/example.gfile")

# save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid
# parameterization f=(f_center-f_edge)*(1-rhop**alpha1)**alpha2 + f_edge
T_core = 5e3  # eV
T_edge = 100  # eV
T_alpha1 = 2.0
T_alpha2 = 1.5
n_core = 1e14  # cm^-3
n_edge = 0.4e14  # cm^-3
n_alpha1 = 2
n_alpha2 = 0.5

rhop = np.linspace(0, 1, 100)
ne_cm3 = (n_core - n_edge) * (1 - rhop ** n_alpha1) ** n_alpha2 + n_edge
Te_eV = (T_core - T_edge) * (1 - rhop ** T_alpha1) ** T_alpha2 + T_edge

# choice of ion and fractional abundance wrt electron density:
ion = "Ar"
frac = 0.005

# look at radiation profiles
res = aurora.radiation_model(ion, rhop, ne_cm3, Te_eV, geqdsk, frac=frac, plot=True)
