'''
Script to test functionality of Aurora from namelist creation to run and postprocessing.

It is recommended to run this in IPython
'''

import numpy as np
import matplotlib.pyplot as plt
import omfit_eqdsk
import pickle as pkl
import scipy,sys,os
import time

# Allow test script to see package home (better ways?)
sys.path.insert(1, os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))

from pylib import default_nml
from pylib import atomic
from pylib import utils
from pylib import coords
from pylib import plot_tools

namelist = default_nml.load_default_namelist()

# test for C-Mod:
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

###
# Set up for 2 different ions:
imp = namelist['imp'] = 'Ca' #'Ar' #'Ca'
namelist['Z_imp'] = 20 #18. #20.
namelist['imp_a'] = 40.078 #39.948  # 40.078

# Now get aurora setup dictionary
aurora_dict = utils.aurora_setup(namelist, geqdsk=geqdsk)
###

##########
# get charge state distributions from ionization equilibrium
atom_data = atomic.get_all_atom_data(imp,['acd','scd'])
ne_avg = np.mean(kin_profs['ne']['vals'],axis=0) # average over time
Te_avg = np.mean(kin_profs['Te']['vals'],axis=0)  # must be on the same radial basis as ne_avg

# get_frac_abundances takes inputs in m^-3 and eV
logTe, fz = atomic.get_frac_abundances(atom_data, ne_avg*1e6, Te_avg, rho=rhop)
############


# transform these fractional abundances to the r_V grid used by aurora
_rV = coords.rad_coord_transform(rhop, 'rhop','r_V', geqdsk)*1e2 # m --> cm
cs = np.arange(aurora_dict['Z_imp']+1)
nz_init = scipy.interpolate.interp2d(_rV,cs, fz.T)(aurora_dict['radius_grid'], cs)


# Take definition of peaking as q(psi_n=0.2)/<q>, where <> is a volume average
nominal_peaking=1.3
nominal_volavg = 1e12 # cm^-3

nz_tot = np.sum(nz_init,axis=0)
nz_tot_volavg = np.sum(2.*np.pi*np.diff(aurora_dict['radius_grid'], prepend=0.0)*nz_tot)
Psi_n = coords.rad_coord_transform(rhop, 'rhop','psin', geqdsk)
ind_psin02 = np.argmin(np.abs(Psi_n - 0.2))
peaking = nz_tot[ind_psin02]/nz_tot_volavg


# choose transport coefficients
D_eff = 1e4 #cm^2/s
v_eff = 0.0

# # set transport coefficients to the right format
D_z = np.ones((len(aurora_dict['radius_grid']),1)) * D_eff
V_z = np.ones((len(aurora_dict['radius_grid']),1)) * v_eff
times_DV = [1.0]  # dummy

# set initial charge state distributions to ionization equilibrium (no transport)
start = time.time()
pyout = utils.run_aurora(aurora_dict, times_DV, D_z, V_z) #, nz_init=nz_init.T)
print("Fortan: ", time.time() - start, " seconds")

# First call includes precompilation, not a good timing example. Time second run
juout = utils.run_julia(aurora_dict, times_DV, D_z, V_z)

start = time.time()
juout = utils.run_julia(aurora_dict, times_DV, D_z, V_z)
print("Julia: ", time.time() - start, " seconds")

all_good = True
for i in range(len(juout)):
    if not np.allclose(pyout[i], juout[i]):
        print("Result incongruency")
        all_good = False
        break
if all_good:
    print("Results equivalent")
