"""
Script to test FACIT functionality integrated with Aurora simulations. 

It is recommended to run this in IPython.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
import aug_sfutils as sf
from omfit_classes import omfit_eqdsk
import sys
import os
import time

# Make sure that package home is added to sys.path
sys.path.append("../")
import aurora

# user choices for testing:
rotation_model = 0  # recommended: 0 for light impurities, 2 for heavier than Ar
D_an = 1e4  # anomalous D, cm^2/s
V_an = -1e2  # anomalous V, cm/s
# ------------------

# pass any argument via the command line to show plots
plot = len(sys.argv) > 1

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
T_alpha2 = 1.0
n_core = 1e14  # cm^-3
n_edge = 0.02e14  # cm^-3
n_alpha1 = 2.0
n_alpha2 = 1.5

rhop_kp = kp["Te"]["rhop"] = kp["ne"]["rhop"] = np.linspace(0, 1, 100)
kp["ne"]["vals"] = (n_core - n_edge) * (1 - rhop_kp**n_alpha1) ** n_alpha2 + n_edge
kp["Te"]["vals"] = (T_core - T_edge) * (1 - rhop_kp**T_alpha1) ** T_alpha2 + T_edge

# set impurity species and sources rate
imp = namelist["imp"] = "C"  # "W"
namelist["source_type"] = "const"
namelist["source_rate"] = 1e16  # particles/s

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

times_DV = np.array([0])
nz_init = np.zeros((asim.rvol_grid.size, asim.Z_imp + 1))

# initialize transport coefficients
D_z = np.zeros((asim.rvol_grid.size, times_DV.size, asim.Z_imp + 1))  # space, time, nZ
V_z = np.zeros(D_z.shape)

# set time-independent anomalous transport coefficients
Dz_an = np.zeros(D_z.shape)  # space, time, nZ
Vz_an = np.zeros(D_z.shape)

# set anomalous transport coefficients
Dz_an[:] = D_an
Vz_an[:] = V_an

# -------------------
# prepare FACIT input
rr = asim.rvol_grid / 100  # in m
idxsep = np.argmin(
    np.abs(1.0 - asim.rhop_grid)
)  # index of radial position of separatrix
amin = rr[idxsep]  # minor radius in m
roa = rr[: idxsep + 1] / amin  # normalized radial coordinate

B0 = np.abs(geqdsk["BCENTR"])  # magnetic field on axis
R0 = geqdsk["fluxSurfaces"]["R0"]  # major radius

qmag = np.interp(roa, geqdsk["RHOVN"], geqdsk["QPSI"])[: idxsep + 1]  # safety factor

rhop = asim.rhop_grid[: idxsep + 1]

# profiles
Ni = (
    np.interp(roa, rhop_kp, kp["ne"]["vals"]) * 1e6
)  # in m**3 instead of cm**3 in FACIT
TeovTi = 1.0  # electron to ion temperature ratio
Ti = np.interp(roa, rhop_kp, kp["Te"]["vals"]) / TeovTi

# gradNi = np.gradient(Ni, roa*amin)
# gradTi = np.gradient(Ti, roa*amin)

# analytical gradients
drhopdx = np.gradient(rhop, amin * roa)
gradTi = (
    (T_core - T_edge)
    * T_alpha2
    * (1 - rhop**T_alpha1) ** (T_alpha2 - 1)
    * (-T_alpha1 * rhop ** (T_alpha1 - 1))
    * drhopdx
)
gradNi = (
    (n_core - n_edge)
    * n_alpha2
    * (1 - rhop**n_alpha1) ** (n_alpha2 - 1)
    * (-n_alpha1 * rhop ** (n_alpha1 - 1))
    * drhopdx
)
gradTi[-1] = gradTi[-2]
gradNi *= 1e6  # density in m**3 instead of cm**3 in FACIT
gradNi[-1] = gradNi[-2]
Zeff = 1.5 * np.ones(roa.size)  # typical AUG value

if rotation_model == 0:

    Machi = np.zeros(
        roa.size
    )  # no rotation (not that it matters with rotation_model=0)
    RV = None
    ZV = None

elif rotation_model == 2:

    Mi_core = 0.35
    Mi_edge = 0.05

    Machi = (Mi_core - Mi_edge) * (1 - rhop**2) + Mi_edge

    nth = 51
    theta = np.linspace(0, 2 * np.pi, nth)

    RV, ZV = aurora.rhoTheta2RZ(geqdsk, rhop, theta, coord_in="rhop", n_line=201)
    RV, ZV = RV.T, ZV.T

else:
    raise ValueError("Other options of rotation_model are not enabled in this example!")

# ----------
# call FACIT

# uncomment to begin simulation from a pre-existing profile
# c_imp = 1e-7 # trace concentration
# for k in range(nz_init.shape[1]):
#    nz_init[:idxsep+1,k] = c_imp*Ni*1e-6 # in 1/cm**3

starttime = time.time()
for j, tj in enumerate(times_DV):

    for i, zi in enumerate(range(asim.Z_imp + 1)):

        if zi != 0:
            Nz = nz_init[: idxsep + 1, i] * 1e6  # in 1/m**3
            gradNz = np.gradient(Nz, roa * amin)

            fct = aurora.FACIT(
                roa,
                zi,
                asim.A_imp,
                asim.main_ion_Z,
                asim.main_ion_A,
                Ti,
                Ni,
                Nz,
                Machi,
                Zeff,
                gradTi,
                gradNi,
                gradNz,
                amin / R0,
                B0,
                R0,
                qmag,
                rotation_model=rotation_model,
                Te_Ti=TeovTi,
                RV=RV,
                ZV=ZV,
            )

            D_z[: idxsep + 1, j, i] = fct.Dz * 100**2  # convert to cm**2/s
            V_z[: idxsep + 1, j, i] = fct.Vconv * 100  # convert to cm/s

time_exec = time.time() - starttime
print("FACIT exec time [s]: ", time_exec)

# add anomalous transport
D_z += Dz_an
V_z += Vz_an

if plot:  # display neoclassical transport coefficients
    plt.figure()
    for i, zi in enumerate(range(asim.Z_imp + 1)):
        plt.plot(asim.rhop_grid, (D_z - Dz_an)[:, 0, i] / 100**2)
    plt.xlabel(r"$\rho_p$")
    plt.ylabel(r"D [m$^2$/s]")
    plt.figure()
    for i, zi in enumerate(range(asim.Z_imp + 1)):
        plt.plot(asim.rhop_grid, (V_z - Vz_an)[:, 0, i] / 100)
    plt.xlabel(r"$\rho_p$")
    plt.ylabel(r"v [m/s]")

# run Aurora forward model and plot results
out = asim.run_aurora(D_z, V_z, times_DV=times_DV, nz_init=nz_init, plot=plot)

# extract densities and particle numbers in each simulation reservoir
nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out

if plot:
    # plot charge state distributions over radius and time
    aurora.plot_tools.slider_plot(
        asim.rvol_grid,
        asim.time_out,
        nz.transpose(1, 0, 2),
        xlabel=r"$r_V$ [cm]",
        ylabel="time [s]",
        zlabel=r"$n_z$ [$cm^{-3}$]",
        labels=[str(i) for i in np.arange(0, nz.shape[1])],
        plot_sum=True,
        x_line=asim.rvol_lcfs,
    )
