"""
Script to test functionality from namelist creation to run and postprocessing.

It is recommended to run this in IPython.
"""
import numpy as np
import matplotlib.pyplot as plt
import aug_sfutils as sf
from omfit_classes import omfit_eqdsk
import sys
import os
#from scipy.interpolate import interp1d, CubicSpline

# Make sure that package home is added to sys.path
sys.path.append("../")
import aurora

sys.path.append('/afs/ipp/home/d/dfajardo/facit/python/')
from facit4Aurora import FACIT

import time

plt.ion()

# pass any argument via the command line to show plots
#plot = len(sys.argv) > 1
plot = True

# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()

# Use gfile and statefile in local directory:
#examples_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = '/afs/ipp/home/d/dfajardo/Aurora/examples'
geqdsk = omfit_eqdsk.OMFITgeqdsk(examples_dir + "/example.gfile")

plt.figure()
for key, value in geqdsk['fluxSurfaces']['flux'].items():
    plt.plot(value['R'], value['Z'])
    
plt.axis('equal')

# save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid
# parameterization f=(f_center-f_edge)*(1-rhop**alpha1)**alpha2 + f_edge
kp = namelist["kin_profs"]
T_core = 5e3  # eV
T_edge = 100  # eV
T_alpha1 = 2.0
T_alpha2 = 1.0
n_core = 1e14  # cm^-3
n_edge = 0.02e14  # cm^-3
n_alpha1 = 2.0
n_alpha2 = 1.5

rhop = kp["Te"]["rhop"] = kp["ne"]["rhop"] = np.linspace(0, 1, 100)
kp["ne"]["vals"] = (n_core - n_edge) * (1 - rhop ** n_alpha1) ** n_alpha2 + n_edge
kp["Te"]["vals"] = (T_core - T_edge) * (1 - rhop ** T_alpha1) ** T_alpha2 + T_edge


# set impurity species and sources rate
imp = namelist["imp"] = "W"
namelist["source_type"] = "const"
namelist["source_rate"] = 1e16  # particles/s

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# transport coefficients

use_facit = True

if not use_facit:

    # set time-independent transport coefficients (flat D=1 m^2/s, V=-1 m/s)
    D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
    V_z = -1e2 * np.ones(len(asim.rvol_grid))  # cm/s

    times_DV = None
    nz_init  = None

else:

    times_DV = np.array([0])
    nz_init  = np.zeros((asim.rvol_grid.size, asim.Z_imp+1))
    
    
    # initialize transport coefficients
    D_z = np.zeros((asim.rvol_grid.size, times_DV.size, asim.Z_imp+1)) # space, time, nZ
    V_z = np.zeros(D_z.shape)


    # set time-independent anomalous transport coefficients
    # flat D=1 m^2/s, V=-1 m/s in core, 0 in pedestal, half of core in SOL (?)

    Dz_an = np.zeros(D_z.shape) # space, time, nZ
    Vz_an = np.zeros(D_z.shape)

    rped   = 0.90
    idxped = np.argmin(np.abs(rped - asim.rhop_grid))
    idxsep = np.argmin(np.abs(1.0 - asim.rhop_grid))

    Dz_an[:idxped,:,:] = 1e4  # cm^2/s
    Vz_an[:idxped,:,:] = -1e2 # cm/s

    Dz_an[idxsep:,:,:] = 1e4/2  # cm^2
    Vz_an[idxsep:,:,:] = -1e2/2 # cm/s

    # collisional (neoclassical + classical) transport coefficients with FACIT
    
    # prepare FACIT input
    
    rotation_model = 2 # calculate collisional D, V without rotation
                       # recommended for "light" impurities (Be, B, C, O, N,...)
                       # Ar has small but non-negligible rotation effects, so set to 2
                       # for Ar and heavier impurities, particularly W

    rr      = asim.rvol_grid/100 # in m
    amin    = rr[idxsep] # minor radius in m
    rova    = rr[:idxsep+1]/amin # normalized radial coordinate

    B0 = np.abs(geqdsk['BCENTR']) # magnetic field on axis
    R0 = geqdsk['fluxSurfaces']['R0'] # major radius
    
    qmag = np.interp(rova, geqdsk['RHOVN'], geqdsk['QPSI'])[:idxsep+1] # safety factor
    
    
    rhop1 = np.linspace(0,1,rova.size)

    # profiles

    Ni = np.interp(rova, rhop, kp['ne']['vals'])*1e6 # in m**3 instead of cm**3 in FACIT

    TeovTi = 1.0 # electron to ion temperature ratio
    Ti = np.interp(rova, rhop, kp['Te']['vals'])/TeovTi

    #gradNi = np.gradient(Ni, rova*amin)
    #gradTi = np.gradient(Ti, rova*amin)
    
    # analytical gradients
    drhopdx = np.gradient(rhop1, amin*rova)
    gradTi     = (T_core - T_edge)*T_alpha2*(1 - rhop1**T_alpha1)**(T_alpha2-1)*\
                 (-T_alpha1*rhop1**(T_alpha1-1))*drhopdx
    gradNi     = (n_core - n_edge)*n_alpha2*(1 - rhop1**n_alpha1)**(n_alpha2-1)*\
                 (-n_alpha1*rhop1**(n_alpha1-1))*drhopdx
    
    gradTi[-1] = gradTi[-2]
    
    gradNi *= 1e6 # density in m**3 instead of cm**3 in FACIT
    gradNi[-1] = gradNi[-2]

    Zeff   = 1.5*np.ones(rova.size) # typical AUG value

    if rotation_model == 0:

        Machi  = np.zeros(rova.size)    # no rotation (not that it matters with rotation_model=0)
        RV = None
        JV = None

    elif rotation_model == 2:

        Mi_core = 0.35
        Mi_edge = 0.05

        Machi = (Mi_core-Mi_edge)*(1 - rhop1**2) + Mi_edge
        
        nth  = 51
        mapping = None #geqdsk['RHOVN']
        geqdsk['fluxSurfaces'].findSurfaces(np.linspace(0,1,rr[:idxsep+1].size), map = mapping)
        geqdsk['fluxSurfaces'].resample(npts=nth)
        
        Z0 = geqdsk['fluxSurfaces']['Z0'] 
        

        R1    = np.zeros((geqdsk['fluxSurfaces']['flux'].keys()[-1]+1, nth))
        Z1    = np.zeros(R1.shape)
        theta = np.linspace(0,2*np.pi,nth)

        # new radial positions

        RV = np.zeros((rova.size,nth))
        ZV = np.zeros((rova.size,nth))
        

        for key, value in geqdsk['fluxSurfaces']['flux'].items():

            thg = np.arctan2(value['Z']-Z0, value['R']-R0)
            thg[thg<0] += 2*np.pi
            
            idxsort = np.argsort(thg)

            Rtemp = value['R'][idxsort]
            Ztemp = value['Z'][idxsort]

            RV[key] = Rtemp
            ZV[key] = Ztemp

        # Jacobian of coordinate system

        dRdr  = np.gradient(RV, rova*amin, axis = 0)
        dRdth = np.gradient(RV, theta, axis = 1)
        dZdr  = np.gradient(ZV, rova*amin, axis = 0)
        dZdth = np.gradient(ZV, theta, axis = 1)

        grr   = dRdr**2 + dZdr**2
        grth  = dRdr*dRdth + dZdr*dZdth
        gthth = dRdth**2 + dZdth**2

        JV = (RV/(amin*np.maximum(rova[:,None],0.001)*B0/qmag[:,None]))*np.sqrt(grr*gthth - grth**2)

    

    # call FACIT

    c_imp = 1e-7 # trace concentration

    for k in range(nz_init.shape[1]):
        nz_init[:idxsep+1,k] = c_imp*Ni*1e-6 # in 1/cm**3
        

    starttime = time.time()
    
    for j, tj in enumerate(times_DV):

        for i, zi in enumerate(range(asim.Z_imp + 1)):

            if zi != 0:
                Nz     = nz_init[:idxsep+1,i]*1e6 # in 1/m**3
                gradNz = np.gradient(Nz, rova*amin)
            
            
                fct = FACIT(rova,\
                            zi, asim.A_imp,\
                            asim.main_ion_Z, asim.main_ion_A,\
                            Ti, Ni, Nz, Machi, Zeff, \
                            gradTi, gradNi, gradNz, \
                            amin/R0, B0, R0, qmag, \
                            rotation_model = rotation_model, Te_Ti = TeovTi,\
                            RV = RV, JV = JV)
            
                D_z[:idxsep+1,j,i] = fct.Dz*100**2 # converto to cm**2/s     
                V_z[:idxsep+1,j,i] = fct.Vconv*100 # convert to cm/s

            else:
                D_z[:idxsep+1,j,i] = 0.0 #1.0*100**2     
                V_z[:idxsep+1,j,i] = 0.0

            # transport coefficients after separatrix in SOL

            D_z[idxsep+1:,j,i] = 0.0 # D_z[idxsep,j,i] #?
            V_z[idxsep+1:,j,i] = 0.0 # V_z[idxsep,j,i] #?


    time_exec = time.time()-starttime

    print('FACIT exec time [s]: ', time_exec)

    # add anomalous transport

    D_z += Dz_an
    V_z += Vz_an

plt.figure()
for i, zi in enumerate(range(asim.Z_imp + 1)):
    #plt.plot(asim.rhop_grid, (D_z-Dz_an)[:,0,i]/100**2)
    plt.plot(asim.rhop_grid, (V_z-Vz_an)[:,0,i]/100)


# run Aurora forward model and plot results
out = asim.run_aurora(D_z, V_z, times_DV = times_DV, nz_init = nz_init, plot=plot)

# extract densities and particle numbers in each simulation reservoir
nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out

if plot:
    # plot charge state distributions over radius and time
    aurora.plot_tools.slider_plot(
        asim.rvol_grid,
        asim.time_out,
        nz.transpose(1, 0, 2),
        xlabel=r"$r_V$ [cm]",
        ylabel="time [s]",
        zlabel=r"$n_z$ [$cm^{-3}$]",
        labels=[str(i) for i in np.arange(0, nz.shape[1])],
        plot_sum=True,
        x_line=asim.rvol_lcfs,
    )

# add radiation
asim.rad = aurora.compute_rad(
    imp,
    nz.transpose(2, 1, 0),
    asim.ne,
    asim.Te,
    prad_flag=True,
    thermal_cx_rad_flag=False,
    spectral_brem_flag=False,
    sxr_flag=False,
)

if plot:
    # plot radiation profiles over radius and time
    aurora.slider_plot(
        asim.rvol_grid,
        asim.time_out,
        asim.rad["line_rad"].transpose(1, 2, 0),
        xlabel=r"$r_V$ [cm]",
        ylabel="time [s]",
        zlabel=r"Line radiation [$MW/m^3$]",
        labels=[str(i) for i in np.arange(0, nz.shape[1])],
        plot_sum=True,
        x_line=asim.rvol_lcfs,
    )