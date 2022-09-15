"""
Script to demonstrate the creation of an animation of an Aurora run, purely for visualization purposes. 

It is recommended to run this in basic Python3 (not IPython or notebooks).
Note that you might need to run %matplotlib qt in IPython in order to enable the animation to run. 
"""

import numpy as np
import matplotlib.pyplot as plt

plt.ion()
from omfit_classes import omfit_eqdsk
import pickle as pkl
import scipy, sys, os
import time

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
kp["ne"]["vals"] = (n_core - n_edge) * (1 - rhop ** n_alpha1) ** n_alpha2 + n_edge
kp["Te"]["vals"] = (T_core - T_edge) * (1 - rhop ** T_alpha1) ** T_alpha2 + T_edge

# set impurity species and sources rate
imp = namelist["imp"] = "Ar"
namelist["source_type"] = "const"
namelist["source_rate"] = 2e20  # particles/s

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# set time-independent transport coefficients (flat D=1 m^2/s, V=-2 cm/s)
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -2e2 * np.ones(len(asim.rvol_grid))  # cm/s

# run Aurora forward model
out = asim.run_aurora(D_z, V_z)

# extract densities of each charge state:
nz = out[0]

# now create animation
aurora.animate.animate_aurora(
    asim.rhop_grid,
    asim.time_out,
    nz.transpose(1, 0, 2),
    xlabel=r"$\rho_p$",
    ylabel="t={:.4f} [s]",
    zlabel=r"$n_z$ [$cm^{-3}$]",
    labels=[str(i) for i in np.arange(0, nz.shape[1])],
    plot_sum=True,
    save_filename="aurora_anim",
)
