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

# Select two different projectiles and wall material
projectile_1 = 'He'
projectile_2 = 'Ar'
target = 'W'

# Select angle of incidence
angle = 55 # degrees

# Extract the TRIM-generated data for the bulk sputtering yield and
#   the energy sputtering yield for the selected impurity hitting
#   the surface, in function of the impact energy, specifying
#   the incidence angle of the projectile onto the surface
# D as projectile
energies_y_1 = aurora.surface.get_bulk_sputtering_data(projectile_1, target, angle, 'y')["energies"] # energy values of the sputtering yields
data_y_1 = aurora.surface.get_bulk_sputtering_data(projectile_1, target, angle, 'y')["data"] # sputtering yields
energies_ye_1 = aurora.surface.get_bulk_sputtering_data(projectile_1, target, angle, 'ye')["energies"] # energy values of the energy sputtering yields
data_ye_1 = aurora.surface.get_bulk_sputtering_data(projectile_1, target, angle, 'ye')["data"] # energy sputtering yields
# Ar as projectile
energies_y_2 = aurora.surface.get_bulk_sputtering_data(projectile_2, target, angle, 'y')["energies"] # energy values of the sputtering yields
data_y_2 = aurora.surface.get_bulk_sputtering_data(projectile_2, target, angle, 'y')["data"] # sputtering yields
energies_ye_2 = aurora.surface.get_bulk_sputtering_data(projectile_2, target, angle, 'ye')["energies"] # energy values of the energy sputtering yields
data_ye_2 = aurora.surface.get_bulk_sputtering_data(projectile_2, target, angle, 'ye')["data"] # energy sputtering yields

# Extract the Bohdansky fit for the sputtering yield at the
#   same incidence angle of the projectile onto the surface
# D as projectile
energies_y_fit_1 = aurora.surface.bulk_sputtering_coeff_fit(projectile_1, target, angle, 'y')["energies"] # energy values of the sputtering yields
data_y_fit_1 = aurora.surface.bulk_sputtering_coeff_fit(projectile_1, target, angle, 'y')["data"] # sputtering yields
# Ar as projectile
energies_y_fit_2 = aurora.surface.bulk_sputtering_coeff_fit(projectile_2, target, angle, 'y')["energies"] # energy values of the sputtering yields
data_y_fit_2 = aurora.surface.bulk_sputtering_coeff_fit(projectile_2, target, angle, 'y')["data"] # sputtering yields

# Plot together the TRIM-generated data and the Bohdansky fit
#   for the sputtering yields
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), squeeze=False)
ax[0,0].scatter(energies_y_1, data_y_1, c='r', label="TRIM data")
ax[0,0].plot(energies_y_fit_1, data_y_fit_1, c='b', label="Bohdansky fit")
ax[0,0].set_ylabel(f'$Y$ ({projectile_1}-->{target}, angle={angle}°)')
ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')
ax[0,1].scatter(energies_ye_1, data_ye_1, c='r', label="TRIM data")
ax[0,1].set_ylabel(f'$Y_E$ ({projectile_1}-->{target}, angle={angle}°)')
ax[0,1].set_xscale('log')
ax[0,1].set_yscale('log')
ax[1,0].scatter(energies_y_2, data_y_2, c='r', label="TRIM data")
ax[1,0].plot(energies_y_fit_2, data_y_fit_2, c='b', label="Bohdansky fit")
ax[1,0].set_ylabel(f'$Y$ ({projectile_2}-->{target}, angle={angle}°)')
ax[1,0].set_xscale('log')
ax[1,0].set_yscale('log')
ax[1,1].scatter(energies_ye_2, data_ye_2, c='r', label="TRIM data")
ax[1,1].set_ylabel(f'$Y_E$ ({projectile_2}-->{target}, angle={angle}°)')
ax[1,1].set_xscale('log')
ax[1,1].set_yscale('log')
for aa in ax.flatten()[:4]:
     aa.legend(loc="best").set_draggable(True)
for ii in [0,1]:
    ax[0, ii].set_xlabel(f"$E_{{0,{projectile_1}}}$ [eV]")
    ax[0, ii].set_xlim([10,10000])
    ax[0, ii].set_ylim([0.00001,10])
    ax[1, ii].set_xlabel(f"$E_{{0,{projectile_2}}}$ [eV]")
    ax[1, ii].set_xlim([10,10000])
    ax[1, ii].set_ylim([0.00001,10])