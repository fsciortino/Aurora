import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from omfit_classes.omfit_eqdsk import OMFITgeqdsk
import sys

# Make sure that package home is added to sys.path
sys.path.append("../")
import aurora



shot = 38996
time = 3.0
geqdsk = OMFITgeqdsk("").from_aug_sfutils(shot=shot, time=time, eq_shotfile="EQI")

rho = np.linspace(0,1, 99)
theta = np.linspace(0, 2*np.pi, 222)
R,Z = aurora.rhoTheta2RZ(geqdsk, rho, theta, coord_in='rhop', n_line=201)



