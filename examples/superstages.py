"""
Script to demonstrate use of superstages.

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
kp["ne"]["vals"] = (n_core - n_edge) * (1 - rhop ** n_alpha1) ** n_alpha2 + n_edge
kp["Te"]["vals"] = (T_core - T_edge) * (1 - rhop ** T_alpha1) ** T_alpha2 + T_edge

# set impurity species and sources rate
imp = namelist["imp"] = "Ar"
namelist["source_type"] = "const"
namelist["source_rate"] = 2e20  # particles/s

# selection of superstages for Ar
superstages = [0, 14, 15, 16, 17, 18]

########
# first run WITHOUT superstages
namelist["superstages"] = []

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# set time-independent transport coefficients (flat D=1 m^2/s, V=0 cm/s)

D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = 0.0 * np.ones(len(asim.rvol_grid))  # cm/s

# run Aurora forward model and plot results
out = asim.run_aurora(D_z, V_z, times_DV=[1.0,], unstage=True, plot=plot)

# extract densities and particle numbers in each simulation reservoir
nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out


# plot charge state distributions over radius and time
aurora.plot_tools.slider_plot(
    asim.rhop_grid,
    asim.time_grid,
    nz.transpose(1, 0, 2),
    xlabel=r"$\rho_p$",
    ylabel="time [s]",
    zlabel=r"$n_z$ [$cm^{-3}$]",
    labels=[str(i) for i in np.arange(0, nz.shape[1])],
    plot_sum=True,
)  # , x_line=asim.rvol_lcfs)


########
# now choose superstages: always include 0 and 1!
namelist["superstages"] = superstages

# set up aurora again, this time with superstages
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# run Aurora forward model and plot results
out = asim.run_aurora(D_z, V_z, times_DV=[1.0,], unstage=True, plot=plot)

# extract densities and particle numbers in each simulation reservoir
nzs, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out

# plot charge state distributions over radius and time
aurora.plot_tools.slider_plot(
    asim.rvol_grid,
    asim.time_grid,
    nzs.transpose(1, 0, 2),
    xlabel=r"$\rho_p$",
    ylabel="time [s]",
    zlabel=r"$n_z$ [$cm^{-3}$]",
    labels=[str(i) for i in np.arange(0, nzs.shape[1])],
    plot_sum=True,
)  # , x_line=asim.rvol_lcfs)


# compare at last slice
ls_cycle = aurora.get_ls_cycle()

fig = plt.figure()
fig.set_size_inches(12, 7, forward=True)
a_plot = plt.subplot2grid((10, 10), (0, 0), rowspan=10, colspan=8, fig=fig)
a_legend = plt.subplot2grid((10, 10), (0, 8), rowspan=10, colspan=2, fig=fig)
a_legend.axis("off")

for cs in np.arange(nz.shape[1]):
    ls = next(ls_cycle)
    a_plot.plot(asim.rhop_grid, nz[:, cs, -1], ls, lw=1.0)
    a_plot.plot(asim.rhop_grid, nzs[:, cs, -1], ls, lw=2.0)
    a_legend.plot([], [], ls, label=imp + f"$^{{{cs}+}}$")

a_plot.set_xlabel(r"$\rho_p$")
a_plot.set_ylabel(r"$n_z$ [A.U.]")

a_legend.legend(loc="best", ncol=1).set_draggable(True)
