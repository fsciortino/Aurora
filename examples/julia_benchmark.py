'''
Script to benchmark basic Julia version against Fortran one. 
'''

import numpy as np
import matplotlib.pyplot as plt
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
namelist['source_rate'] = 1e24

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# set time-independent transport coefficients (flat D=1 m^2/s, V=-2 cm/s)
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -2e2 * np.ones(len(asim.rvol_grid)) # cm/s

# do several runs in series to more accurately time runs
num=10
start = time.time()
for i in range(num):
    pyout = out = asim.run_aurora(D_z, V_z) 
print("Fortran: ", (time.time() - start)/num, " seconds on average")

# First call includes precompilation, not a good timing example. Time second set of runs!
start = time.time()
juout = asim.run_aurora(D_z, V_z, use_julia=True)
print("Julia time for first call (compiling): ", time.time() - start, " second")

start = time.time()
for i in range(num):
    juout = asim.run_aurora(D_z, V_z, use_julia=True)
print("Julia: ", (time.time() - start)/num, " seconds on average")

# check that results are the same between Fortran and Julia:
all_good = True
for i in range(len(juout)):
    if not np.allclose(pyout[i], juout[i]):
        print("Result incongruency")
        all_good = False
        break
if all_good:
    print("Fortran and Julia results are equivalent!")
