"""
Script to test functionality of steady-state run with AURORA.

It is recommended to run this in IPython.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.ion()
from omfit_classes import omfit_eqdsk
import sys, os
from scipy.interpolate import interp1d
import copy

# Make sure that package home is added to sys.path
sys.path.append("../")
import aurora

# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()

# Use gfile and statefile in local directory:
examples_dir = os.path.dirname(os.path.abspath(__file__))
geqdsk = omfit_eqdsk.OMFITgeqdsk(examples_dir + "/example.gfile")

# save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid
# parameterization f=(f_center-f_edge)*(1-rhop**alpha1)**alpha2 + f_edge
kp = namelist["kin_profs"]
T_core = 5e3  # eV
T_edge = 100  # eV
T_alpha1 = 2.0
T_alpha2 = 1.5
n_core = 1e14  # cm^-3
n_edge = 0.4e14  # cm^-3
n_alpha1 = 2
n_alpha2 = 0.5

rhop = kp["Te"]["rhop"] = kp["ne"]["rhop"] = np.linspace(0, 1, 100)
ne = kp["ne"]["vals"] = (n_core - n_edge) * (1 - rhop ** n_alpha1) ** n_alpha2 + n_edge
Te = kp["Te"]["vals"] = (T_core - T_edge) * (1 - rhop ** T_alpha1) ** T_alpha2 + T_edge

# set impurity species and sources rate to 0
imp = namelist["imp"] = "Ar"
namelist["source_type"] = "const"
namelist["source_rate"] = 1e31  # particles/s

# get charge state distributions from ionization equilibrium
atom_data = aurora.atomic.get_atom_data(imp, ["scd", "acd"])

# get fractional abundances on ne (cm^-3) and Te (eV) grid
_Te, fz = aurora.atomic.get_frac_abundances(atom_data, ne, Te, rho=rhop, plot=False)

# initial guess for steady state Ar charge state densities
nz_init = 1e-25 * ne[:, None] * fz

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

nz_init = interp1d(rhop, nz_init, bounds_error=False, fill_value=0.0, axis=0)(
    asim.rhop_grid
)

# set time-independent transport coefficients
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -10e2 * asim.rhop_grid ** 5  # cm/s

n_steps = 10
max_sim_time = 500e-3

nz_norm_steady = asim.run_aurora_steady(
    D_z,
    V_z,
    nz_init=None,  # nz_init,
    tolerance=0.01,
    max_sim_time=max_sim_time,
    dt=1e-4,
    dt_increase=1.05,
    n_steps=n_steps,
    plot=True,
)

print("tau_imp:", asim.tau_imp)
