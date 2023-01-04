"""
Script to test functionality of steady-state run with AURORA.

It is recommended to run this in IPython.
"""



import numpy as np
import matplotlib.pyplot as plt
from time import time

#from omfit_classes import omfit_eqdsk
import sys, os
from scipy.interpolate import interp1d
import copy
from matplotlib import cm


# Make sure that package home is added to sys.path
sys.path.append("../")
import aurora

 
# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()

# Use gfile and statefile in local directory:
examples_dir = os.path.dirname(os.path.abspath(__file__))

 
# save kinetic profiles on a rhop (sqrt of norm. pol. flux) grid
# parameterization f=(f_center-f_edge)*(1-rhop**alpha1)**alpha2 + f_edge

kp = namelist["kin_profs"]
T_core = 5e3  # eV
T_edge = 100  # eV
T_alpha1 = 2.0
T_alpha2 = 1.5
n_core = 1e14  # cm^-3
n_edge = 0.4e14  # cm^-3
n_alpha1 = 2
n_alpha2 = 0.5

rhop = kp["Te"]["rhop"] = kp["ne"]["rhop"] = np.linspace(0, 1.1, 100)
ne = kp["ne"]["vals"] = (n_core - n_edge) * np.maximum(0,1 - rhop ** n_alpha1) ** n_alpha2 + n_edge
Te = kp["Te"]["vals"] = (T_core - T_edge) * np.maximum(0,1 - rhop ** T_alpha1) ** T_alpha2 + T_edge
#add exponantial decay of Te in SOL
Te[rhop > 1] *= np.exp((1-rhop[rhop > 1])/0.03)
 
#assume 1e-6 neutral concentration
kp["n0"] = copy.copy(kp["ne"])
n0 = kp["n0"]["vals"] = kp["ne"]["vals"]/1e6
namelist['cxr_flag'] = True

 
 
namelist['rvol_lcfs'] = 70
namelist['Raxis_cm'] = 170
namelist['lim_sep'] = 5.6
namelist['clen_divertor'] = 25
namelist['clen_limiter'] = 0.5
namelist['bound_sep'] = 8
namelist['source_cm_out_lcfs'] = 10
namelist['recycling_switch'] = 0
namelist['dr_0'] = 1
namelist['dr_1'] = 0.1
namelist['K'] = 10

#load  meta resolved atomic data files 
namelist['acd'] = 'acd96r_c.dat'
namelist['scd'] = 'scd96r_c.dat'
namelist['ccd'] = 'ccd96r_c.dat'
namelist['qcd'] = 'qcd96r_c.dat'
namelist['xcd'] = 'xcd96r_c.dat'
namelist['metastable_flag'] = True



 
  
# set impurity species and sources rate to 0
namelist["imp"] = 'C' 
namelist["source_type"] = "const"
namelist["source_rate"] = 1e18  # particles/s
namelist["SOL_mach"] = 0.1

#use just a single time 
namelist["timing"]['times'] = [0,1e-6]
 
# Now get aurora setup
asim = aurora.core.aurora_sim(namelist)
 
# set time-independent transport coefficients
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -10e2 * asim.rhop_grid ** 5  # cm/s

#Z dependent D and V
D_z = np.tile(D_z, (asim.Z_imp+1,1)).T
V_z = np.tile(V_z, (asim.Z_imp+1,1)).T
 

t = time()
meta_ind, nz_norm_steady = asim.run_aurora_steady_analytic( D_z, V_z)
print('Analytical solution with metastables calculated in : %.3fs'%(time()-t))


namelist['acd'] = 'acd96_c.dat'
namelist['scd'] = 'scd96_c.dat'
namelist['ccd'] = 'ccd96_c.dat'
namelist['metastable_flag'] = False
asim = aurora.core.aurora_sim(namelist )


t = time()
meta_ind2, nz_norm_steady2 = asim.run_aurora_steady_analytic( D_z, V_z)
print('Analytical solution without metastables calculated in : %.3fs'%(time()-t))
 
 
colormap = cm.rainbow
colors = cm.rainbow(np.linspace(0, 1, asim.Z_imp+1))
        
for i, c in enumerate(colors):
    total = np.zeros(nz_norm_steady.shape[1])
    for ind, (z, j) in enumerate(meta_ind):
        if z != i: continue
        plt.plot(asim.rhop_btw,nz_norm_steady[ind], lw=.5,c=c)
        k = np.argmax(nz_norm_steady[ind])
        plt.text(asim.rhop_btw[k],nz_norm_steady[ind, k], str((z, j)), verticalalignment ='top')
        total += nz_norm_steady[ind]
        
    plt.plot(asim.rhop_btw,total, lw=2,c=c, label='Z=%d'%i)
    plt.plot(asim.rhop_btw,nz_norm_steady2[i], '--',lw=2,c=c)


plt.plot([],[],'k-',label='unresolved')
plt.plot([],[],'k--',label='resolved')
plt.title('Density profiles of carbon ions')
plt.xlabel('rhop')
plt.ylabel('Carbon density [cm$^{-3}$]')
plt.legend()
plt.xlim(.9, 1.1)
plt.ylim(0, 4e8)
plt.grid(True)
 
resolved_files = ['pec96#c_pjr#c0.dat', 
            'pec96#c_vsr#c1.dat',
            'pec96#c_vsr#c2.dat',
            'pec96#c_pjr#c3.dat',
            'pec96#c_pjr#c4.dat',
            'pec96#c_bnd#c5.dat',]  


#C5 is not including CX component, use the one from Ralph?
#calculate n=2 state density for hydrogen , use CX crossections 

observed_lines = [[], [6581.5, 5143.3], [4650.1, 5697.5], [5806.5], [], [5292.7]]

 
n0 = np.interp(asim.rhop_btw, rhop, n0)
ne = np.interp(asim.rhop_btw, rhop, ne)
Te = np.interp(asim.rhop_btw, rhop, Te)
 

line_emiss_dict_resolved = {}
for filename, lines in zip(resolved_files, observed_lines):
    if len(lines) == 0:
        continue
    
    # fetch file automatically, locally, from AURORA_ADAS_DIR, or directly from the web:
    path = aurora.get_adas_file_loc(filename, filetype='adf15')

    # load all transitions provided in the chosen ADF15 file:
    trs = aurora.read_adf15(path)
 
    Z = trs.attrs['Z']
 
    for lam in lines:
        #get line emissivity in  ph/s
        line_emiss = aurora.radiation.get_photon_emissivity(trs, lam, ne, Te, nz_norm_steady,n0, meta_ind)
        line_emiss_dict_resolved[(Z, lam)] = np.sum([emiss for emiss in line_emiss.values()],0)
        

           
            
 
unresolved_files = ['pec96#c_pju#c0.dat',
                    'pec96#c_vsu#c1.dat',
                    'pec96#c_vsu#c2.dat',
                    'pec96#c_pju#c3.dat',
                    'pec96#c_pju#c4.dat',
                    'pec96#c_bnd#c5.dat',]


line_emiss_dict_unresolved = {}
for filename, lines in zip(unresolved_files, observed_lines):
    if len(lines) == 0:
        continue
    
    # fetch file automatically, locally, from AURORA_ADAS_DIR, or directly from the web:
    path = aurora.get_adas_file_loc(filename, filetype='adf15')

    # load all transitions provided in the chosen ADF15 file:
    trs = aurora.read_adf15(path)
 
    Z = trs.attrs['Z']
     
    for lam in lines:
        ##get line emissivity in  ph/s
        line_emiss = aurora.radiation.get_photon_emissivity(trs, lam, ne, Te, nz_norm_steady2,n0, meta_ind2)
        line_emiss_dict_unresolved[(Z, lam)] = np.sum([emiss for emiss in line_emiss.values()],0)
 
##CX lines, TODO included CX cross-sections!
line_emiss_dict_resolved[(4, 4660)] = nz_norm_steady[meta_ind.index((4, 1))]
line_emiss_dict_resolved[(6, 5290)] = nz_norm_steady[meta_ind.index((6, 1))]

 
colormap = cm.rainbow
colors = cm.rainbow(np.linspace(0, 1, asim.Z_imp+1))

plt.figure()
plt.title('Normalised intensity profiles of main carbon lines')
z_ = 0
for (z,lam), emiss in  line_emiss_dict_resolved.items():
    plt.plot(asim.rhop_btw , emiss/emiss.max(), c=colors[z], label=f'z={z},$\lambda$={round(lam/10)}')
    i = np.argmax(emiss)
    if z_ == z:
        s = -0.05
    else:
        z_  = z
        s = 0
    plt.text(asim.rhop_btw[i] ,1.02+s, str((z,round(lam/10))))
    plt.plot(asim.rhop_btw[i] ,1, 'o', c=colors[z])

for (z,lam), emiss in  line_emiss_dict_unresolved.items():
    plt.plot(asim.rhop_btw , emiss/emiss.max(), '--', c=colors[z])


plt.plot([],[],'k--',label='unresolved')
plt.plot([],[],'k-',label='resolved')

plt.legend()
plt.xlim(.9, 1.1)
plt.ylim(0, 1.06)
plt.xlabel('rhop')
plt.ylabel('Normalised line emissivity [-]')



    
plt.ioff()    
plt.show()
           
 
