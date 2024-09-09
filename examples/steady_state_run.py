"""
Script to test functionality of steady-state run with AURORA.

It is recommended to run this in IPython.
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time
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
ne = kp["ne"]["vals"] = (n_core - n_edge) * (1 - rhop**n_alpha1) ** n_alpha2 + n_edge
Te = kp["Te"]["vals"] = (T_core - T_edge) * (1 - rhop**T_alpha1) ** T_alpha2 + T_edge


namelist["rvol_lcfs"] = 70
namelist["Raxis_cm"] = 170
namelist["lim_sep"] = 5.6
namelist["clen_divertor"] = 25
namelist["clen_limiter"] = 0.5
namelist["bound_sep"] = 8
namelist["source_cm_out_lcfs"] = 10
namelist["recycling_switch"] = 0
namelist["dr_0"] = 1
namelist["dr_1"] = 0.1
namelist["K"] = 10


# set impurity species and sources rate to 0
imp = namelist["imp"] = "Ne"
namelist["source_type"] = "const"
namelist["source_rate"] = 1e18  # particles/s
namelist["SOL_mach"] = 0.1

# use just a single time
namelist["timing"]["times"] = [0, 1e-6]

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist)

# set time-independent transport coefficients
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -10e2 * asim.rhop_grid**5  # cm/s

# Z dependent D and V
# D_z = np.tile(D_z, (asim.Z_imp+1,1)).T
# V_z = np.tile(V_z, (asim.Z_imp+1,1)).T


t = time()
meta_ind, nz_norm_steady = asim.run_aurora_steady_analytic( D_z, V_z) 
#,  rvol,  nz[0], ploss, Sne, Rne, lam = 1.0)
print("Analytical steady solution calculated in : %.3fs" % (time() - t))

n_steps = 10
max_sim_time = 1000
t = time()
nz_norm_steady2 = asim.run_aurora_steady(
    D_z,
    V_z,
    nz_init=None,
    tolerance=0.01,
    max_sim_time=max_sim_time,
    dt=1e-4,
    dt_increase=1.05,
    n_steps=n_steps,
    plot=False,
)
print(
    "Steady solution from convergence of time-dependent solver calculated in : %.3fs"
    % (time() - t)
)

plt.plot(asim.rhop_btw, nz_norm_steady.T)
plt.gca().set_prop_cycle(None)
plt.plot(asim.rhop_grid, nz_norm_steady2, "--")

plt.plot(asim.rhop_btw, nz_norm_steady.sum(0), "k-", lw=2, label="Analytical solution")
plt.plot(
    asim.rhop_grid, nz_norm_steady2.sum(1), "k--", lw=2, label="Full iterative solution"
)
plt.legend()
plt.xlabel("rhop")
plt.ylabel("impurity density [$cm^{-3}$]")
plt.ioff()
plt.show()

# TODO the two solutions have a different Z=0 profile - why??
