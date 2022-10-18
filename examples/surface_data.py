"""
Script to plot relevant data for the plasma-surface interaction models
(reflection coefficients, bulk/impurity sputtering yields)
Note that for some data files, the fit are not properly produced yet
The user is encouraged to manually check the quality of the data,
stored in /Aurora/aurora/trim_data/, and read the README files
"""

import matplotlib.pyplot as plt
import sys
import os

# Make sure that package home is added to sys.path
sys.path.append("../")
import aurora

# Select impurity and wall materials
imp = 'He'
wall = 'W'

# Extract the TRIM-generated data for the particle and energy
#   reflection coefficients for the selected impurity hitting
#   the surface, in function of the impact energy, specifying
#   the incidence angle of the projectile onto the surface
angle = 65 # degrees
refl_data_rn = aurora.surface.get_reflection_data(imp, wall, angle, 'rn') # particle reflection coeff.
energies_rn = refl_data_rn["energies"]
data_rn = refl_data_rn["data"]
refl_data_re = aurora.surface.get_reflection_data(imp, wall, angle, 're') # energy reflection coeff.
energies_re = refl_data_re["energies"]
data_re = refl_data_re["data"]

# Extract the Eckstein fits for both the coefficients at the
#   same incidence angle of the projectile onto the surface
refl_data_rn_fit = aurora.surface.reflection_coeff_fit(imp, wall, angle, 'rn') # particle reflection coeff.
energies_rn_fit = refl_data_rn_fit["energies"]
data_rn_fit = refl_data_rn_fit["data"]
refl_data_re_fit = aurora.surface.reflection_coeff_fit(imp, wall, angle, 're') # energy reflection coeff.
energies_re_fit = refl_data_re_fit["energies"]
data_re_fit = refl_data_re_fit["data"]

# Plot together the TRIM-generated data and the Eckstein fit
#   for particle and energy reflection coefficients
fig, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), squeeze=False)
ax1[0,0].scatter(energies_rn, data_rn, c='r', label="TRIM data")
ax1[0,0].plot(energies_rn_fit, data_rn_fit, c='b', label="Eckstein fit")
ax1[0,0].set_ylabel(f'$R_N$ ({imp}-->{wall}, angle={angle}°)')
ax1[0,1].scatter(energies_re, data_re, c='r', label="TRIM data")
ax1[0,1].plot(energies_re_fit, data_re_fit, c='b', label="Eckstein fit")
ax1[0,1].set_ylabel(f'$R_E$ ({imp}-->{wall}, angle={angle}°)')
for aa in ax1.flatten()[:2]:
     aa.legend(loc="best").set_draggable(True)
for ii in [0,1]:
    ax1[0, ii].set_xlabel("$E_0$ [eV]")
ax1[0,0].set_xlim([0,1000])
ax1[0,1].set_xlim([0,1000])

# Extract the TRIM-generated data for the bulk sputtering yield and
#   the energy sputtering yield for the selected impurity hitting
#   the surface, in function of the impact energy, specifying
#   the incidence angle of the projectile onto the surface
angle = 65 # degrees
bulk_sputter_data_y = aurora.surface.get_bulk_sputtering_data(imp, wall, angle, 'y') # sputtering yield
energies_y = bulk_sputter_data_y["energies"]
data_y = bulk_sputter_data_y["data"]
bulk_sputter_data_ye = aurora.surface.get_bulk_sputtering_data(imp, wall, angle, 'ye') # energy sputtering yield
energies_ye = bulk_sputter_data_ye["energies"]
data_ye = bulk_sputter_data_ye["data"]

# Extract the Bohdansky fit for the sputtering yield at the
#   same incidence angle of the projectile onto the surface
bulk_sputter_data_y_fit = aurora.surface.bulk_sputtering_coeff_fit(imp, wall, angle, 'y') # sputtering yield
energies_y_fit = bulk_sputter_data_y_fit["energies"]
data_y_fit = bulk_sputter_data_y_fit["data"]

# Plot together the TRIM-generated data and the Bohdansky fit
#   for the sputtering yields
fig, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), squeeze=False)
ax2[0,0].scatter(energies_y, data_y, c='r', label="TRIM data")
ax2[0,0].plot(energies_y_fit, data_y_fit, c='b', label="Bohdansky fit")
ax2[0,0].set_ylabel(f'$Y$ ({imp}-->{wall}, angle={angle}°)')
ax2[0,0].set_xscale('log')
ax2[0,0].set_yscale('log')
ax2[0,1].scatter(energies_ye, data_ye, c='r', label="TRIM data")
ax2[0,1].set_ylabel(f'$Y_E$ ({imp}-->{wall}, angle={angle}°)')
ax2[0,1].set_xscale('log')
ax2[0,1].set_yscale('log')
for aa in ax2.flatten()[:2]:
     aa.legend(loc="best").set_draggable(True)
for ii in [0,1]:
    ax2[0, ii].set_xlabel("$E_0$ [eV]")
ax2[0,0].set_xlim([10,10000])
ax2[0,1].set_xlim([10,10000])
ax2[0,0].set_ylim([0.00001,10])
ax2[0,1].set_ylim([0.00001,10])

# Extract the TRIM-generated data for the impurity sputtering yield
#   for the selected impurity implanted in the surface, hit by
#   different projectiles, in function of the impact energy, specifying
#   the incidence angle of the projectile onto the surface
#   Note: such data are for now available only for He implanted in W
angle = 65 # degrees
# D as projectile
imp_sputter_data_1 = aurora.surface.get_impurity_sputtering_data(imp, 'D', wall, angle)
energies_1 = imp_sputter_data_1["energies"]
concentrations_1 = imp_sputter_data_1["impurity_concentrations"]
data_1 = imp_sputter_data_1["data"]
# N as projectile
imp_sputter_data_2 = aurora.surface.get_impurity_sputtering_data(imp, 'N', wall, angle)
energies_2 = imp_sputter_data_2["energies"]
concentrations_2 = imp_sputter_data_2["impurity_concentrations"]
data_2 = imp_sputter_data_2["data"]

# Extract the Bohdansky fit for the sputtering yield (normalized)
#   to the impurity concentration into the bulk material) at the
#   same incidence angle of the projectile onto the surface
# D as projectile
imp_sputter_data_fit_1 = aurora.surface.impurity_sputtering_coeff_fit(imp, 'D', wall, angle)
energies_1_fit = imp_sputter_data_fit_1["energies"]
data_1_fit = imp_sputter_data_fit_1["normalized_data"]
# N as projectile
imp_sputter_data_fit_2 = aurora.surface.impurity_sputtering_coeff_fit(imp, 'N', wall, angle)
energies_2_fit = imp_sputter_data_fit_2["energies"]
data_2_fit = imp_sputter_data_fit_2["normalized_data"]

# Plot together the TRIM-generated data and the Bohdansky fit
#   for the sputtering yields
fig, ax3 = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), squeeze=False)
for i in range(1,len(concentrations_1)):
    ax3[0,0].scatter(energies_1, data_1[:,i]/concentrations_1[i], label=f"TRIM data, $f_{{{imp}}}$ = {concentrations_1[i]}")
ax3[0,0].plot(energies_1_fit, data_1_fit, c='k', label="Bohdansky fit")
ax3[0,0].set_ylabel(f'$Y_{{He}}/f_{{He}}$ (D-->{wall}, angle={angle}°)')
for i in range(1,len(concentrations_2)):
    ax3[0,1].scatter(energies_2, data_2[:,i]/concentrations_2[i], label=f"TRIM data, $f_{{{imp}}}$ = {concentrations_1[i]}")
ax3[0,1].plot(energies_2_fit, data_2_fit, c='k', label="Bohdansky fit")
ax3[0,1].set_ylabel(f'$Y_{{He}}/f_{{He}}$ (N-->{wall}, angle={angle}°)')
for aa in ax3.flatten()[:2]:
     aa.legend(loc="best").set_draggable(True)
for ii in [0,1]:
    ax3[0, ii].set_xlabel("$E_0$ [eV]")
ax3[0,0].set_xlim([0,800])
ax3[0,1].set_xlim([0,800])
ax3[0,0].set_ylim([0,4])
ax3[0,1].set_ylim([0,4])