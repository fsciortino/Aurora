'''Cooling curves for multiple ions from Aurora. 

Here we exclude charge exchange for simplicity, but it may easily be added by specifying background D neutral densities.
Note that the cooling factors shown here only use ionization equilibrium, i.e. no transport effects.

sciortino, 2021
'''

import aurora
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


# scan Te and fix a value of ne
Te_eV = np.logspace(np.log10(100), np.log10(1e5), 1000)
ne_cm3 = 5e13 * np.ones_like(Te_eV)

imp = 'C'

# basic cooling curve, considering ionization equilibrium between charge states
line_rad_tot, cont_rad_tot = aurora.get_cooling_factors(
    imp, ne_cm3, Te_eV, plot=True, ion_resolved=False)


# Total cooling coefficients are the same regardless of whether we apply superstaging or not!
line_rad_tot, cont_rad_tot = aurora.get_cooling_factors(
    imp, ne_cm3, Te_eV, plot=True, ion_resolved=False, superstages=[0,3,6])


# plot contributions from each charge state at ionization equilibrium
line_rad_tot, cont_rad_tot = aurora.get_cooling_factors(
    imp, ne_cm3, Te_eV, plot=True, ion_resolved=True)


###### Overplot total cooling coefficients for multiple ions
ions_list = ['He','C','Ar','W']

fig = plt.figure()
a_plot = plt.subplot2grid((10,10),(0,0),rowspan = 10, colspan = 8, fig=fig) 
a_legend = plt.subplot2grid((10,10),(0,8),rowspan = 10, colspan = 8, fig=fig) 

a_legend.axis('off')

ls_cycle = aurora.get_ls_cycle()
for imp in ions_list:

    ls = next(ls_cycle)

    # read atomic data, interpolate and plot cooling factors
    line_rad_tot, cont_rad_tot = aurora.get_cooling_factors(imp, ne_cm3, Te_eV,
                                                            ion_resolved=False, plot=False)

    # total radiation (includes hard X-ray, visible, UV, etc.)
    a_plot.loglog(Te_eV/1e3, cont_rad_tot+line_rad_tot, ls)
    a_legend.plot([],[], ls, label=f'{imp}')
    
    
a_legend.legend(loc='best').set_draggable(True)
a_plot.grid('on', which='both')
a_plot.set_xlabel('T$_e$ [keV]')
a_plot.set_ylabel('$L_z$ [$W$ $m^3$]')
plt.tight_layout()

  
