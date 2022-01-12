"""
Script to test quality of flux-surface-averaging (FSA) procedure by evaluating the normalized ionization rate, or equivalently by comparing the ionization rate of impurities with the parallel transport transit rate. 
This is inspired by Dux et al. NF 2020.

It is recommended to run this in IPython.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
from omfit_classes import omfit_eqdsk
import scipy, sys, os
import time
from scipy.interpolate import interp1d

# Make sure that package home is added to sys.path
sys.path.append("../")
import aurora

# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()
kp = namelist["kin_profs"]

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
ne_cm3 = kp["ne"]["vals"] = (n_core - n_edge) * (
    1 - rhop ** n_alpha1
) ** n_alpha2 + n_edge
Te_eV = kp["Te"]["vals"] = (T_core - T_edge) * (
    1 - rhop ** T_alpha1
) ** T_alpha2 + T_edge

# set impurity species and sources rate
imp = namelist["imp"] = "F"  #'Ar'
namelist["source_type"] = "const"
namelist["source_rate"] = 1e24

# Setup aurora sim to efficiently setup atomic rates and profiles over radius
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# plot normalized ionization frequency for the last time point, only within LCFS:
rhop_in = asim.rhop_grid[asim.rhop_grid < 1.0]
Sne_z = asim.Sne_rates[asim.rhop_grid < 1.0, :, -1]  # take last time point
q_prof = interp1d(geqdsk["AuxQuantities"]["RHOp"], geqdsk["QPSI"])(rhop_in)
Rhfs, Rlfs = aurora.grids_utils.get_HFS_LFS(geqdsk, rho_pol=rhop_in)
R_prof = (Rhfs + Rlfs) / 2.0  # take as average of flux surface

Ti_prof = asim.Te[-1, asim.rhop_grid < 1.0]  # use Ti=Te, only last time point

# inverse aspect ratio profile
eps_prof = (Rlfs - geqdsk["RMAXIS"]) / geqdsk[
    "RMAXIS"
]  # use LFS radius for considerations on trapped particles
nu_ioniz_star = aurora.atomic.plot_norm_ion_freq(
    Sne_z,
    q_prof,
    R_prof,
    asim.A_imp,
    Ti_prof,
    rhop=rhop_in,
    plot=True,
    eps_prof=eps_prof,
)

# get fractional abundances on ne (cm^-3) and Te (eV) grid
atom_data = aurora.get_atom_data(imp, ["acd", "scd"])
_Te, fz = aurora.atomic.get_frac_abundances(atom_data, ne_cm3, Te_eV, rho=rhop)

fz_profs = np.zeros_like(Sne_z)
for cs in np.arange(Sne_z.shape[1]):
    fz_profs[:, cs] = interp1d(rhop, fz[:, cs])(rhop_in)

nu_ioniz_star = aurora.atomic.plot_norm_ion_freq(
    Sne_z,
    q_prof,
    R_prof,
    asim.A_imp,
    Ti_prof,
    nz_profs=fz_profs,
    rhop=rhop_in,
    plot=True,
    eps_prof=eps_prof,
)
