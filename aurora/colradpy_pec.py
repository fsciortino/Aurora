import matplotlib.pyplot as plt
import numpy as np
from colradpy import colradpy
plt.ion()
from matplotlib import cm

Te_grid = np.array([100])
ne_grid = np.array([1.e13]) #,1.e14,1.e15])


filepath = '/home/sciortino/adf04_files/ca/ca_adf04_adas/'

files = {'ca8': 'mglike_lfm14#ca8.dat',
         'ca9': 'nalike_lgy09#ca9.dat',
         'ca10': 'nelike_lgy09#ca10.dat',
         'ca11': 'flike_mcw06#ca11.dat',
         'ca14': 'clike_jm19#ca14.dat',
         'ca15': 'blike_lgy12#ca15.dat',
         'ca16': 'belike_lfm14#ca16.dat',
         'ca17': 'lilike_lgy10#ca17.dat',
         'ca18': 'helike_adw05#ca18.dat'}

colors = cm.rainbow(np.linspace(0, 1, len(files)))
fig, ax = plt.subplots()

res = {}
for ii,cs in enumerate(files.keys()):
    res[cs] = colradpy(filepath+files[cs],[0],Te_grid,ne_grid,use_recombination=False,use_recombination_three_body=False)
    
    res[cs].make_ioniz_from_reduced_ionizrates()
    res[cs].suppliment_with_ecip()
    res[cs].make_electron_excitation_rates()
    res[cs].populate_cr_matrix()
    res[cs].solve_quasi_static()

    # plot lines
    ax.vlines(res[cs].data['processed']['wave_vac'],np.zeros_like(res[cs].data['processed']['wave_vac']),
              res[cs].data['processed']['pecs'][:,0,0,0],label=cs, color=colors[ii])
    

ax.set_xlim(0,200)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('PEC (ph cm$^3$ s$^{-1}$)')
ax.legend().set_draggable(True)


###############
# Te_grid2 = np.geomspace(10,1e3,100)
# ne_grid2 = np.array([1.e13])

# figs, axs = plt.subplots()
# res2 = {}
# for ii,cs in enumerate(files.keys()):
#     res2[cs] = colradpy(filepath+files[cs],[0],Te_grid2,ne_grid2,use_recombination=False,use_recombination_three_body=False)
    
#     res2[cs].make_ioniz_from_reduced_ionizrates()
#     res2[cs].suppliment_with_ecip()
#     res2[cs].make_electron_excitation_rates()
#     res2[cs].populate_cr_matrix()
#     res2[cs].solve_quasi_static()

#     # plot lines
#     ax.plot(Te_grid,res2[cs].data['processed']['pecs'][:,0,0,:].T, label=cs, color=colors[ii])
    
