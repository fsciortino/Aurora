import os, copy
import numpy as np
import matplotlib.pylab as plt
plt.ion()
import sys

# Make sure that package home is added to sys.path
sys.path.append("../")
import aurora

rdb = aurora.amdata.reactions_database()


RR_alpha = [
    "AMJUEL,12,2_1_5a",  # H(3)/H 
    "AMJUEL,12,2_1_8a",  # H(3)/H+ 
    "AMJUEL,12,2_2_5a",  # H(3)/H2
    "AMJUEL,12,2_2_14a", # H(3)/H2+
    "AMJUEL,12,2_0c",    # H2+/H2
    "AMJUEL,12,7_2a",    # H(3)/H-
    "AMJUEL,11,7_0a",    # H-/H2
    "AMJUEL,12,2_2_15a", # H(3)/Hll3+    # not used
    "AMJUEL,11,4_0a"     # H3+/H2/H2+/ne  # not used
]

# Lyman series spontaneous emission coeffs for n=2 to 1, 3 to 1, ... 16 to 1
A_lyman = [4.699e8, 5.575e7, 1.278e7, 4.125e6, 1.644e6, 7.568e5, 3.869e5, 2.143e5,
           1.263e5, 7.834e4, 5.066e4, 3.393e4, 2.341e4, 1.657e4, 1.200e4]

# Balmer series spontaneous emission coeffs for n=3 to 2, 4 to 2, ... 17 to 2
A_balmer = [4.41e7, 8.42e6, 2.53e6, 9.732e5, 4.389e5, 2.215e5, 1.216e5, 7.122e4,
            4.397e4, 2.83e4, 18288.8, 12249.1, 8451.26, 5981.95, 4332.13]

fig, ax = plt.subplots(2, 1, sharex=True)
ls_list = ['-','--','-.',':']

nt = 200
lines = ['alpha','beta','gamma','delta']
rates = np.zeros((len(RR_alpha), nt, len(lines)))

for ii,choice in enumerate(lines):

    ls = ls_list[ii]
    
    # allow one to read various Balmer-series lines by changing final reaction letter
    sub = {"alpha": "a", "beta": "c", "gamma": "d", "delta": "e"}
    RR = copy.deepcopy(RR_alpha)
    for i in [0, 1, 2, 3, 5, 7]:  # only cross sections specific to Halpha
        RR[i] = "AMJUEL" + RR_alpha[i].split("AMJUEL")[1].replace("a", sub[choice])

    # select Einstein Aki coefficients
    aa = {'alpha':0, 'beta':1,'gamma':2,'delta':3}
    a0 = A_balmer[aa[choice]]
    
    te = np.linspace(0.2, 10.0, nt)
    ne = np.ones(nt) * 1e19
    ni = np.ones(nt) * 1e19
    nh = np.ones(nt) * 1e19
    nh2 = np.ones(nt) * 1e19
    
    for j, R in enumerate(RR):
        rdb.select_reaction(R)
        rates[j, :, ii] = rdb.reaction(ne, te)

    c1 = a0 * rates[0, :, ii] * nh
    c2 = a0 * rates[1, :, ii] * ni
    c3 = a0 * rates[2, :, ii] * nh2
    c4 = a0 * rates[3, :, ii] * rates[4, :, ii] * nh2
    c5 = a0 * rates[5, :, ii] * rates[6, :, ii] * nh2
    ct = c1 + c2 + c3 + c4 + c5 # np.ones_like(c1)

    ax[0].plot(te, ct, c='k', ls=ls)
    ax[1].plot(te, c1 / ct, c='b', ls=ls)
    ax[1].plot(te, c2 / ct, c='c', ls=ls)
    ax[1].plot(te, c3 / ct, c='g', ls=ls)
    ax[1].plot(te, c4 / ct, c='r', ls=ls)
    ax[1].plot(te, c5 / ct, c='m', ls=ls)

ax[0].plot([],[], c='k', ls='-', label='alpha')
ax[0].plot([],[], c='k', ls='--', label='beta')
ax[0].plot([],[], c='k', ls='-.', label='gamma')
ax[0].plot([],[], c='k', ls=':', label='delta')
ax[0].legend(loc='best').set_draggable(True)
ax[0].set_yscale("log")
ax[0].set_ylabel("$\epsilon_{tot}$ [ph/m$^{-3}$]")

ax[1].plot([],[], c='b', label="H")
ax[1].plot([],[], c='c', label="H+")
ax[1].plot([],[], c='g', label="H2")
ax[1].plot([],[], c='r', label="H2+")
ax[1].plot([],[], c='m', label="H2-")
ax[1].legend(loc='best').set_draggable(True)
ax[1].set_yscale("log")
ax[1].set_ylim((1e-3, 1e0))
ax[1].set_xlabel("$T_e$ [eV]")
ax[1].set_ylabel("$\epsilon_i/\epsilon_{tot}$")

