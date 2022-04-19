'''Demo for calculation of a local spectrum for an ADAS ADF15 file.
'''

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import aurora

# load neutral H ADF15 file
filename = 'pec96#h_pju#h0.dat'
filepath = aurora.get_adas_file_loc(filename, filetype='adf15')  
trs = aurora.read_adf15(filepath)

Te_eV = 80. # eV
ne_cm3 = 1e14 # cm^-3

atom_data = aurora.get_atom_data('H',['scd','acd'])

# always include charge exchange, although n0_cm3 may be 0
logTe, fz = aurora.get_frac_abundances(
    atom_data, np.array([ne_cm3,]), np.array([Te_eV,]), plot=False)

# now add spectra
out = aurora.get_local_spectrum(filepath, 'H', ne_cm3, Te_eV,
                                # H0+ cannot recombine, so first element is 0
                                ion_exc_rec_dens=[0.0, fz[0,0], fz[0,1]])

# The plot above covers a large wavelength range. Reduce it to focus on near-UV
trs = aurora.read_adf15(filepath)
trs = trs.loc[trs['lambda [A]']<2000]

# now pass transitions pandas DataFrame with reduced number of lines:
out = aurora.get_local_spectrum(trs, 'H', ne_cm3, Te_eV,
                                ion_exc_rec_dens=[0.0, fz[0,0], fz[0,1]])

