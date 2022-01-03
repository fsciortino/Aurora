""" Example script to demonstrate/test how Plasma OPeration CONtour (POPCON) plots can be created.
"""

import sys
import numpy as np

# Make sure that package home is added to sys.path
sys.path.append('../')
import aurora

# Initialize POPCON with default parameters
popcon = aurora.POPCON()

# Adjust input parameters that won't be scanned
popcon.M   = 50
popcon.a        = 1.15      # minor radius (m)
popcon.kappa    = 2         # elongation
popcon.H        = 1.15      # H Factor

popcon.imcharg  = 54
popcon.Ti_min   = 2
popcon.f_LH     = 0.2

popcon.volavgcurr = True
popcon.fixed_quantity = "Psol"

popcon.plot_Psol  = not (popcon.fixed_quantity == "Psol")
popcon.plot_impf  = not (popcon.fixed_quantity == "impfrac")
popcon.plot_f_LH  = not (popcon.fixed_quantity == "f_LH")

# Adjust input parameters that will be scanned and run openpopcon
for Ip in [10]:
    for R in [3.7]:
        popcon.Ip = Ip
        popcon.R  = R

        popcon.make_popcons()

        popcon.plot_contours(save_filename='/tmp/popcon_example_Ip{Ip}_R{R}.pdf'.format(Ip, R))
        
