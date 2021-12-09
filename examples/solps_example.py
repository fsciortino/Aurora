'''
Script to demonstrate capabilities to load and post-process SOLPS results.
Note that SOLPS output is not distributed with Aurora; so, in order to run this script,
either you are able to access the default AUG SOLPS MDS+ server, or you need
to appropriately modify the script to point to your own SOLPS results.
'''
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from omfit_classes import omfit_eqdsk
import sys, os
import aurora

# if one wants to load a SOLPS case from MDS+ (defaults for AUG):
so = aurora.solps_case(solps_id = 141349)

# alternatively, one may want to load SOLPS results from files on disk:
so2 = aurora.solps_case(b2fstate_path = '/afs/ipp/home/s/sciof/SOLPS/141349/b2fstate',
                        b2fgmtry_path = '/afs/ipp/home/s/sciof/SOLPS/141349/b2fgmtry')

# plot some important fields
fig,axs = plt.subplots(1,2, figsize=(10,6),sharex=True) 
ax = axs.flatten()
so.plot2d_b2(so.data('ne'), ax=ax[0], scale='log', label=r'$n_e$ [$m^{-3}$]')
so.plot2d_b2(so.data('te'), ax=ax[1], scale='linear', label=r'$T_e$ [eV]')


# if EIRENE data files (e.g. fort.44, .46, etc.) are available, one can plot EIRENE results
# on the original EIRENE grid. SOLPS results also include EIRENE outputs on B2 grid:
fig,axs = plt.subplots(1,2, figsize=(10,6),sharex=True) 
so.plot2d_eirene(so.fort46['pdena'][:,0]*1e6, scale='log', label=r'$n_n$ [$m^{-3}$]', ax=axs[0])
so.plot2d_b2(so.fort44['dab2'][:,:,0].T, label=r'$n_n$ [$m^{-3}$]', ax=axs[1])
plt.tight_layout()

