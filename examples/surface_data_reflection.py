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
angle = 65 # degrees

# Extract the TRIM-generated data for the particle and energy
#   reflection coefficients for the selected impurity hitting
#   the surface, in function of the impact energy, specifying
#   the incidence angle of the projectile onto the surface
# He as projectile
energies_rn_1 = aurora.surface.get_reflection_data(projectile_1, target, angle, 'rn')["energies"] # energy values of the particle refl. coeff.
data_rn_1 = aurora.surface.get_reflection_data(projectile_1, target, angle, 'rn')["data"] # particle reflection coeff.
energies_re_1 = aurora.surface.get_reflection_data(projectile_1, target, angle, 're')["energies"] # energy values of the energy refl. coeff.
data_re_1 = aurora.surface.get_reflection_data(projectile_1, target, angle, 're')["data"] # energy refl. coeff.
# N as projectile
energies_rn_2 = aurora.surface.get_reflection_data(projectile_2, target, angle, 'rn')["energies"] # energy values of the particle refl. coeff
data_rn_2 = aurora.surface.get_reflection_data(projectile_2, target, angle, 'rn')["data"] # particle refl. coeff.
energies_re_2 = aurora.surface.get_reflection_data(projectile_2, target, angle, 're')["energies"] # energy values of the energy refl. coeff.
data_re_2 = aurora.surface.get_reflection_data(projectile_2, target, angle, 're')["data"] # energy refl. coeff.

# Extract the Eckstein fits for both the coefficients at the
#   same incidence angle of the projectile onto the surface
# He as projectile
energies_rn_fit_1 = aurora.surface.reflection_coeff_fit(projectile_1, target, angle, 'rn')["energies"] # energy values of the particle refl. coeff.
data_rn_fit_1 = aurora.surface.reflection_coeff_fit(projectile_1, target, angle, 'rn')["data"] # particle refl. coeff.
energies_re_fit_1 = aurora.surface.reflection_coeff_fit(projectile_1, target, angle, 're')["energies"] # energy values of the energy refl. coeff.
data_re_fit_1 = aurora.surface.reflection_coeff_fit(projectile_1, target, angle, 're')["data"] # energy refl. coeff.
# N as projectile
energies_rn_fit_2 = aurora.surface.reflection_coeff_fit(projectile_2, target, angle, 'rn')["energies"] # energy values of the particle refl. coeff.
data_rn_fit_2 = aurora.surface.reflection_coeff_fit(projectile_2, target, angle, 'rn')["data"] # particle refl. coeff.
energies_re_fit_2 = aurora.surface.reflection_coeff_fit(projectile_2, target, angle, 're')["energies"] # energy values of the energy refl. coeff.
data_re_fit_2 = aurora.surface.reflection_coeff_fit(projectile_2, target, angle, 're')["data"] # energy refl. coeff.

# Calculate the mean reflection energy
# He as projectile
impact_energies_1 = aurora.surface.calc_reflected_energy(projectile_1, target, angle)["energies"]
refl_energies_1 = aurora.surface.calc_reflected_energy(projectile_1, target, angle)["data"]
# N as projectile
impact_energies_2 = aurora.surface.calc_reflected_energy(projectile_2, target, angle)["energies"]
refl_energies_2 = aurora.surface.calc_reflected_energy(projectile_2, target, angle)["data"]

# Plot together the TRIM-generated data and the Eckstein fit
#   for particle and energy reflection coefficients,
#   and the mean reflected energy
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 10), squeeze=False)
ax[0,0].scatter(energies_rn_1, data_rn_1, c='r', label="TRIM data")
ax[0,0].plot(energies_rn_fit_1, data_rn_fit_1, c='b', label="Eckstein fit")
ax[0,0].set_ylabel(f'$R_N$ ({projectile_1}-->{target}, angle={angle}°)')
ax[0,1].scatter(energies_re_1, data_re_1, c='r', label="TRIM data")
ax[0,1].plot(energies_re_fit_1, data_re_fit_1, c='b', label="Eckstein fit")
ax[0,1].set_ylabel(f'$R_E$ ({projectile_1}-->{target}, angle={angle}°)')
ax[0,2].plot(impact_energies_1, refl_energies_1, c='b')
ax[0,2].set_ylabel(f'<E>$_{{refl}}$ ({projectile_1}-->{target}, angle={angle}°)')
ax[1,0].scatter(energies_rn_2, data_rn_2, c='r', label="TRIM data")
ax[1,0].plot(energies_rn_fit_2, data_rn_fit_2, c='b', label="Eckstein fit")
ax[1,0].set_ylabel(f'$R_N$ ({projectile_2}-->{target}, angle={angle}°)')
ax[1,1].scatter(energies_re_2, data_re_2, c='r', label="TRIM data")
ax[1,1].plot(energies_re_fit_2, data_re_fit_2, c='b', label="Eckstein fit")
ax[1,1].set_ylabel(f'$R_E$ ({projectile_2}-->{target}, angle={angle}°)')
ax[1,2].plot(impact_energies_2, refl_energies_2, c='b')
ax[1,2].set_ylabel(f'<E>$_{{refl}}$ ({projectile_2}-->{target}, angle={angle}°)')
for aa in ax.flatten()[:2]:
    aa.legend(loc="best").set_draggable(True)
for aa in ax.flatten()[3:5]:
    aa.legend(loc="best").set_draggable(True)
for ii in [0,1,2]:
    ax[0, ii].set_xlabel(f"$E_{{0,{projectile_1}}}$ [eV]")
    ax[0, ii].set_xlim([0,1000])
for ii in [0,1,2]:
    ax[1, ii].set_xlabel(f"$E_{{0,{projectile_2}}}$ [eV]")
    ax[1, ii].set_xlim([0,1000])
for ii in [0,1]:
    ax[0, ii].set_ylim([0.3,1])
    ax[1, ii].set_ylim([0.3,1])
for ii in [2]:
    ax[0, ii].set_ylim([0,1000])
    ax[1, ii].set_ylim([0,1000])