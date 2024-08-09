import matplotlib.pyplot as plt
import sys
import os

# Make sure that package home is added to sys.path
sys.path.append("../")
import aurora

# Select impurity, two different projectiles and wall material
imp = 'He'
projectile_1 = 'He'
projectile_2 = 'N'
target = 'W'

# Select angle of incidence
angle = 65 # degrees

# Extract the TRIM-generated data for the impurity sputtering yield
#   for the selected impurity implanted in the surface, hit by
#   different projectiles, in function of the impact energy, specifying
#   the incidence angle of the projectile onto the surface
#   Note: such data are for now available only for He implanted in W

# D as projectile
energies_1 = aurora.surface.get_impurity_sputtering_data(imp, projectile_1, target, angle)["energies"]
concentrations_1 = aurora.surface.get_impurity_sputtering_data(imp, projectile_1, target, angle)["impurity_concentrations"]
data_1 = aurora.surface.get_impurity_sputtering_data(imp, projectile_1, target, angle)["data"]
# N as projectile
energies_2 = aurora.surface.get_impurity_sputtering_data(imp, projectile_2, target, angle)["energies"]
concentrations_2 = aurora.surface.get_impurity_sputtering_data(imp, projectile_2, target, angle)["impurity_concentrations"]
data_2 = aurora.surface.get_impurity_sputtering_data(imp, projectile_2, target, angle)["data"]

# Extract the Bohdansky fit for the sputtering yield (normalized)
#   to the impurity concentration into the bulk material) at the
#   same incidence angle of the projectile onto the surface
# D as projectile
energies_1_fit = aurora.surface.impurity_sputtering_coeff_fit(imp, projectile_1, target, angle)["energies"]
data_1_fit = aurora.surface.impurity_sputtering_coeff_fit(imp, projectile_1, target, angle)["normalized_data"]
# N as projectile
energies_2_fit = aurora.surface.impurity_sputtering_coeff_fit(imp, projectile_2, target, angle)["energies"]
data_2_fit = aurora.surface.impurity_sputtering_coeff_fit(imp, projectile_2, target, angle)["normalized_data"]

# Extract the mean sputtered energy
# D as projectile
impact_energies_1 = aurora.surface.calc_imp_sputtered_energy(imp, projectile_1, target)["energies"]
sputtered_energies_1_fit = aurora.surface.calc_imp_sputtered_energy(imp, projectile_1, target)["data"]
# N as projectile
impact_energies_2 = aurora.surface.calc_imp_sputtered_energy(imp, projectile_2, target)["energies"]
sputtered_energies_2_fit = aurora.surface.calc_imp_sputtered_energy(imp, projectile_2, target)["data"]

# Make the plots
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), squeeze=False)
for i in range(1,len(concentrations_1)):
    ax[0,0].scatter(energies_1, data_1[:,i]/concentrations_1[i], label=f"TRIM data, $f_{{{imp}}}$ = {concentrations_1[i]}")
ax[0,0].plot(energies_1_fit, data_1_fit, c='k', label="Bohdansky fit")
ax[0,0].set_ylabel(f'$Y_{{He}}/f_{{He}}$ ({projectile_1}-->{target}, angle={angle}°)')
for i in range(1,len(concentrations_2)):
    ax[0,1].scatter(energies_2, data_2[:,i]/concentrations_2[i], label=f"TRIM data, $f_{{{imp}}}$ = {concentrations_1[i]}")
ax[0,1].plot(energies_2_fit, data_2_fit, c='k', label="Bohdansky fit")
ax[0,1].set_ylabel(f'$Y_{{He}}/f_{{He}}$ ({projectile_2}-->{target}, angle={angle}°)')
ax[1,0].plot(impact_energies_1, sputtered_energies_1_fit, c='k', label="TRIM data, linear fit")
ax[1,0].set_ylabel(f'<$E_{{sput,He}}$> ({projectile_1}-->{target}, angle={angle}°)')
ax[1,1].plot(impact_energies_2, sputtered_energies_2_fit, c='k', label="TRIM data, linear fit")
ax[1,1].set_ylabel(f'<$E_{{sput,He}}$> ({projectile_2}-->{target}, angle={angle}°)')
for ii in [0,1]:
    ax[ii, 0].set_xlabel(f"$E_{{0,{projectile_1}}}$ [eV]")
    ax[0, ii].set_xlim([0,5000])
    ax[0, ii].set_ylim([0,5])
    ax[0, ii].legend(loc="best").set_draggable(True)
    ax[ii, 1].set_xlabel(f"$E_{{0,{projectile_2}}}$ [eV]")
    ax[1, ii].set_xlim([0,5000])
    ax[1, ii].set_ylim([0,150])
    ax[1, ii].legend(loc="best").set_draggable(True)