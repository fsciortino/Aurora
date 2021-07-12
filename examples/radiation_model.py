'''
Script to test simple functionality of radiation models.

It is recommended to run this in IPython.
'''

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from omfit_classes import omfit_eqdsk, omfit_gapy
import sys, os
from scipy.interpolate import interp1d

# Make sure that package home is added to sys.path
sys.path.append('../')
import aurora

try: # pass any argument via the command line to show plots
    plot = len(sys.argv)>1
except:
    plot = False

# Use gfile and statefile in local directory:
examples_dir = os.path.dirname(os.path.abspath(__file__))
geqdsk = omfit_eqdsk.OMFITgeqdsk(examples_dir+'/example.gfile')
inputgacode = omfit_gapy.OMFITgacode(examples_dir+'/example.input.gacode')

# save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid

rhop = np.sqrt(inputgacode['polflux']/inputgacode['polflux'][-1])
ne_cm3 = inputgacode['ne']*1e13 # 1e19 m^-3 --> cm^-3
Te_eV = inputgacode['Te']*1e3  # keV --> eV


# choice of ion and fractional abundance wrt electron density:
ion = 'Ar'
frac = 0.005

# look at radiation profiles
res = aurora.radiation_model(ion,rhop,ne_cm3,Te_eV, geqdsk,
                             frac=frac, plot=True)
  
