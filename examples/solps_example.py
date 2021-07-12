'''
Script to test functionality from namelist creation to run and postprocessing.

It is recommended to run this in IPython.
'''

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from omfit_classes import omfit_eqdsk
import sys, os
import aurora


# info to find and load SOLPS-ITER case
path = '/home/sciortino/ITER/iter_solps_jl'
solps_run = 'orig_D1.95e23_Ne2.00e20.done.ids'
gfilepath = '/home/sciortino/ITER/gfile_iter'
        
# load SOLPS results
so = aurora.solps_case(path, gfilepath, solps_run=solps_run,form='full')


# plot some important fields
fig,axs = plt.subplots(1,4, figsize=(20,6),sharex=True) 
ax = axs.flatten()
so.plot2d_b2(so.quants['ne'], ax=ax[0], scale='log', label=so.labels['ne'])
so.plot2d_b2(so.quants['Te'], ax=ax[1], scale='linear', label=so.labels['Te'])
so.plot2d_b2(so.quants['nn'], ax=ax[2], scale='log', label=so.labels['nn'])
so.plot2d_b2(so.quants['Tn'], ax=ax[3], scale='linear', label=so.labels['Tn'])
for axx in ax:
    # overplot 2D flux surfaces and vacuum contour
    so.geqdsk.plot(only2D=True, ax=axx)
    axx.grid(False)
plt.tight_layout()


# comparison of neutrals on B2 (fort.44) and EIRENE (fort.46) grids
fig,axs = plt.subplots(1,2, figsize=(10,6),sharex=True) 
so.plot2d_eirene(so.fort46['pdena'][:,0]*1e6, scale='log', label=so.labels['nn'], ax=axs[0])
so.plot2d_b2(so.fort44['dab2'][:,:,0].T, label=so.labels['nn'], ax=axs[1])
plt.tight_layout()

