'''
Script to benchmark basic Julia version against Fortran one. 
'''

import numpy as np
import matplotlib.pyplot as plt
import omfit_eqdsk
import pickle as pkl
import scipy,sys,os
import time

# Allow test script to see package home
sys.path.insert(1, os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))

from pylib import default_nml
from pylib import atomic
from pylib import utils
from pylib import coords
from pylib import plot_tools

namelist = default_nml.load_default_namelist()
namelist['device'] = 'CMOD'
namelist['shot'] = 1101014030
namelist['time'] = 1250 # ms

gfile_name=f'g{namelist["shot"]}.{str(namelist["time"]).zfill(5)}'

if os.path.exists(gfile_name):
    # fetch local g-file if available
    geqdsk = omfit_eqdsk.OMFITgeqdsk(gfile_name)
    print('Fetched local g-file')
else:
    # attempt to construct it via omfit_eqdsk if not available locally
    geqdsk = omfit_eqdsk.OMFITgeqdsk('').from_mdsplus(
        device=namelist['device'],shot=namelist['shot'],
        time=namelist['time'], SNAPfile='EFIT01',
        fail_if_out_of_range=False,time_diff_warning_threshold=20
    )
    # save g-file locally:
    geqdsk.save(raw=True)
    print('Saved g-file locally')


# example kinetic profiles
kin_profs = namelist['kin_profs']

with open('./test_kin_profs.pkl','rb') as f:
    ne_profs,Te_profs = pkl.load(f)

kin_profs['ne']['vals'] = ne_profs['ne']*1e14  # 10^20 m^-3 --> cm^-3
kin_profs['ne']['times'] = ne_profs['t']
rhop = kin_profs['ne']['rhop'] = ne_profs['rhop']
kin_profs['Te']['vals'] = Te_profs['Te']*1e3  # keV --> eV
kin_profs['Te']['times'] = Te_profs['t']
kin_profs['Te']['rhop'] = Te_profs['rhop']
kin_profs['Te']['decay'] = np.ones(len(Te_profs['Te']))*1.0

# set no sources of impurities
namelist['source_type'] = 'const'
namelist['Phi0'] = 1e24 #1.0
imp = namelist['imp'] = 'Ca' #'Ar' #'Ca'
namelist['Z_imp'] = 20 #18. #20.
namelist['imp_a'] = 40.078 #39.948  # 40.078

# Now get aurora setup dictionary
aurora_dict = utils.aurora_setup(namelist, geqdsk=geqdsk)

# choose transport coefficients
D_eff = 1e4 #cm^2/s
v_eff = 0.0

# # set transport coefficients to the right format
D_z = np.ones((len(aurora_dict['radius_grid']),1)) * D_eff
V_z = np.ones((len(aurora_dict['radius_grid']),1)) * v_eff
times_DV = [1.0]  # dummy

num=10

start = time.time()
for i in range(num):
    pyout = utils.run_aurora(aurora_dict, times_DV, D_z, V_z) #, nz_init=nz_init.T)
print("Fortran: ", (time.time() - start)/num, " seconds on average")

# First call includes precompilation, not a good timing example. Time second run
start = time.time()
juout = utils.run_julia(aurora_dict, times_DV, D_z, V_z)
print("Julia time for first call (compiling): ", time.time() - start, " second")

start = time.time()
for i in range(num):
    juout = utils.run_julia(aurora_dict, times_DV, D_z, V_z)
print("Julia: ", (time.time() - start)/num, " seconds on average")


all_good = True
for i in range(len(juout)):
    if not np.allclose(pyout[i], juout[i]):
        print("Result incongruency")
        all_good = False
        break
if all_good:
    print("Results equivalent")
