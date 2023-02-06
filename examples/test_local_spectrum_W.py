"""Demo for calculation of a local spectrum for an ADAS ADF15 file.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.ion()
import aurora


Te_eV = np.geomspace(100, 5000)  # eV
ne_cm3 = 1e14  # cm^-3

# Ion temperature for Doppler broadenning
# unrealistic value is used to mimic intrumental broadening of the used spectrometer
Ti_eV = 4e6

atom_data = aurora.get_atom_data("W", ["scd", "acd"])

# always include charge exchange, although n0_cm3 may be 0
logTe, fz = aurora.get_frac_abundances(atom_data, ne_cm3, Te_eV, plot=False)


file_template = "pec40#w_ic#w%d.dat"

Z = range(1, 48)

f, ax = plt.subplots(1)

lam_min = 120  # A
lam_max = 140  # A

lam_common = np.linspace(lam_min, lam_max, 1000)

total_emiss = np.zeros((len(lam_common), len(Te_eV)))


for z in Z:

    # load W PEC file
    filepath = aurora.get_adas_file_loc(file_template % z, filetype="adf15")

    trs = aurora.read_adf15(filepath)
    # now pass transitions pandas DataFrame with reduced number of lines:

    trs = trs.loc[trs["lambda [A]"] < lam_max]
    trs = trs.loc[trs["lambda [A]"] > lam_min]

    for i, te in enumerate(Te_eV):
        out = aurora.get_local_spectrum(
            trs,
            ne_cm3,
            te,
            Ti_eV=Ti_eV,
            ion_exc_rec_dens=[fz[i, z - 1], fz[i, z], fz[i, z + 1]],
            plot_spec_tot=False,
        )
        lam, ioniz, excit, recom, drsat, chexc, _ = out

        if excit is None:
            continue

        total_emiss[:, i] += np.interp(lam_common, lam, excit)


lemiss = np.log(total_emiss.T)
plt.contourf(
    lam_common / 10,
    Te_eV / 1000,
    lemiss,
    np.linspace(6, lemiss.max(), 30),
    cmap="CMRmap_r",
)
cb = plt.colorbar(ticks=range(6, 11))
plt.xlabel("$\lambda$ [nm]")
plt.ylabel("$T_e$ [keV]")
cb.set_label("log$_{10}$(Emissivity)")
plt.title("W spectrum")
plt.ioff()
plt.show()
