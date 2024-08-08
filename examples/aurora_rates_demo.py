"""This scrit shows how one can use Aurora to load ADAS rates and process them for transport modeling and/or spectroscopic analysis in simple ways. In particular, it shows
(a) how to load individual ADF11 files (ionization and recombination rates);
(b) how to interpolate rates in 2D, either in (time,radius) or (R,Z) coordinates;
(c) how to read/process photon emissivity coefficients (PECs);
(d) how to get total radiated power (line and continuum radius) on a 2D grid for many atomic species.
"""
import aurora
import os
import matplotlib.pyplot as plt

plt.ion() 
import numpy as np

ion = "Ar"  # chosen species
rates_label = (
    "scd"  # chosen ADF11 file to load (SCD: ionization; ACD: recombination; CCD: CX)
)

# first, find name of a default file for the chosen species and process
filename = aurora.adas_files_dict()[ion][
    rates_label
]  # default SCD file for chosen species

# this file can be provided by the user, or fetched/downloaded by Aurora using the following:
filepath = aurora.get_adas_file_loc(filename)

# now, load the file:
atomdat = aurora.adas_file(filepath)

# visualize content of specific ADAS file
atomdat.plot()

# now get multiple data files
atom_data = aurora.get_atom_data(
    ion, files=["acd", "scd", "ccd"]  # recom  # ioniz  # CX
)

# atom_data[rates_label][0] <-- density grid
# atom_data[rates_label][1] <-- temperature grid
# atom_data[rates_label][2] <-- data for each charge state

# ------------------------------------------------------- #
# interpolate on arbitrary ne and Te grid on the 2D plane
# load an example gEQDSK from the Aurora examples directory
from omfit_classes.omfit_eqdsk import OMFITgeqdsk

geqdsk = OMFITgeqdsk(os.path.expanduser("~") + os.sep + "Aurora/examples/example.gfile")

# To plot the equilibrium:
# geqdsk.plot()

R = geqdsk["AuxQuantities"]["R"]
Z = geqdsk["AuxQuantities"]["Z"]
rhop = geqdsk["AuxQuantities"]["RHOpRZ"]

# make up some arbitrary (example) density and temperature profiles
ne_cm3_2d = np.maximum(0,1 - rhop**2)**.5 * 1e13  # cm^-3
Te_eV_2d = np.maximum(0,1- rhop**2)**.5 * 3e3  # eV

# plt.figure()
# cntr = plt.contourf(R,Z,ne)
# plt.colorbar(cntr)

# Routines like :py:fun:`~aurora.atomic.interp_atom_prof` can work with 2 input dimensions for
# ne and Te arrays, which can equally be (time,radius) or (R,Z)
dat_2d = aurora.interp_atom_prof(
    atom_data[rates_label],
    np.log10(ne_cm3_2d),
    np.log10(Te_eV_2d),  # given as log10 of values
    log_val=False,  # False --> give real units
    x_multiply=True,
)  # True --> gives units of s^-1


fig, axs = plt.subplots(1, 3, figsize=(13, 5))
cntr0 = axs[0].contourf(R, Z, ne_cm3_2d)
cbar0 = plt.colorbar(cntr0, ax=axs[0])
cbar0.set_label(r"$n_e$ [cm${^-3}$]")
cntr1 = axs[1].contourf(R, Z, Te_eV_2d)
cbar1 = plt.colorbar(cntr1, ax=axs[1])
cbar1.set_label(r"$T_e$ [eV]")
cntr2 = axs[2].contourf(R, Z, dat_2d[:, 0, :])
cbar2 = plt.colorbar(cntr2, ax=axs[2])
cbar2.set_label(rates_label.upper() + ", Z=0")
plt.tight_layout()

# ----------------------------
# Now for PECs, e.g. Ly-a
filename = "pec96#h_pju#h0.dat"

# fetch file automatically, locally, from AURORA_ADAS_DIR, or directly from the web:
path = aurora.get_adas_file_loc(filename, filetype="adf15")

# load all transitions provided in the chosen ADF15 file:
trs = aurora.read_adf15(path)

# select the excitation-driven component of the Lyman-alpha transition:
tr = trs[(trs["lambda [A]"] == 1215.2) & (trs["type"] == "excit")]

# now plot the rates:
aurora.plot_pec(tr)

# ---------------------------
# Now, let's say we want to compute the radiated power by an impurity
# For simplicity, let's construct some charge state densities using fractional
# abundances at ionization equilibrium. Choose ion concentration (wrt to electrons):
frac = 1e-3

_Te, fz = aurora.atomic.get_frac_abundances(atom_data, ne_cm3_2d, Te_eV_2d, plot=False)

# create some constant-fraction impurity charge state density spatial profiles
nz_cm3 = frac * ne_cm3_2d[:, None, :] * fz.transpose(0, 2, 1)  # (R,nZ,Z)

# calculate radiation from chosen ion for these conditions
rad = aurora.compute_rad(ion, nz_cm3, ne_cm3_2d, Te_eV_2d, prad_flag=True)  # R,nZ,Z

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
cntr = axs[0].contourf(R, Z, nz_cm3[:, 16, :])
cbar = plt.colorbar(cntr, ax=axs[0])
cbar.set_label(rf"$n_{{{ion}17+}}$")
cntr = axs[1].contourf(R, Z, rad["line_rad"][:, 16, :])
cbar = plt.colorbar(cntr, ax=axs[1])
cbar.set_label(rf"$P_{{line,{ion}17+}}$ [MW/m$^3$]")
 
