'''
Script to demonstrate the creation of an animation of an Aurora run, purely for visualization purposes. 

It is recommended to run this in basic Python3 (not IPython or notebooks).
Note that you might need to run %matplotlib qt in IPython in order to enable the animation to run. 
'''

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from omfit_classes import omfit_eqdsk, omfit_gapy
import pickle as pkl
import scipy,sys,os
import time

# Make sure that package home is added to sys.path
sys.path.append('../')
import aurora


# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()
kp = namelist['kin_profs']

# Use gfile and statefile in local directory:
examples_dir = os.path.dirname(os.path.abspath(__file__))
geqdsk = omfit_eqdsk.OMFITgeqdsk(examples_dir+'/example.gfile')
inputgacode = omfit_gapy.OMFITgacode(examples_dir+'/example.input.gacode')

# transform rho_phi (=sqrt toroidal flux) into rho_psi (=sqrt poloidal flux) and save kinetic profiles
kp['Te']['rhop'] = kp['ne']['rhop'] = np.sqrt(inputgacode['polflux']/inputgacode['polflux'][-1])
kp['ne']['vals'] = inputgacode['ne'][None,:]*1e13 # 1e19 m^-3 --> cm^-3
kp['Te']['vals'] = inputgacode['Te'][None,:]*1e3  # keV --> eV

# set impurity species and sources rate
imp = namelist['imp'] = 'Ar'
namelist['source_type'] = 'const'
namelist['source_rate'] = 2e20 # particles/s

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# set time-independent transport coefficients (flat D=1 m^2/s, V=-2 cm/s)
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -2e2 * np.ones(len(asim.rvol_grid)) # cm/s

# run Aurora forward model
out = asim.run_aurora(D_z, V_z)

# extract densities of each charge state:
nz = out[0]

# now create animation
aurora.animate.animate_aurora(asim.rhop_grid, asim.time_out, nz.transpose(1,0,2),
                              xlabel=r'$\rho_p$', ylabel='t={:.4f} [s]', zlabel=r'$n_z$ [$cm^{-3}$]',
                              labels=[str(i) for i in np.arange(0,nz.shape[1])],
                              plot_sum=True, save_filename='aurora_anim')
