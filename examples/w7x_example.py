'''Example of Aurora simulation of core impurity transport for a W7-X case.

Note that this example assumes that the user can access a number of private routines. Please email francesco.sciortino@ipp.mpg.de and thilo.romba@ipp.mpg.de if these are of interest.
'''

import os, sys
import numpy as np
from osa import Client

# Make sure that package home is added to sys.path
sys.path.append("../")
import aurora

# choose W7-X shot number and get VMEC ID of reference equilibrium
shot = '20180920.042'
#vmecID, _ = geometry_routines.get_reference_equilibrium(shot)
vmecID = 'w7x_ref_348'  # fixed for this example

vmec = Client('http://esb.ipp-hgw.mpg.de:8280/services/vmec_v5?wsdl')
aminor = max(vmec.service.getReffProfile(vmecID))

# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()

# now adapt for W7-X run
namelist['K'] = 10
namelist['dr_0'] = 0.3
namelist['dr_1'] = 0.05
namelist['rvol_lcfs'] = aminor * 100.0 #cm
namelist['bound_sep'] = 3.0
namelist["source_cm_out_lcfs"] = 0.0
namelist["lim_sep"] = 2.0
namelist["clen_divertor"] = 1000.0 
namelist["clen_lim"] = 1.0
namelist["shot"] = shot    
namelist["time"] = None

# magnetic axis at phi=0
pnt3d = vmec.service.getMagneticAxis(vmecID, 0.0)
namelist['Raxis_cm'] = float(pnt3d.x1[0]) * 100.0 # cm
B3d = vmec.service.magneticField(vmecID, pnt3d)
namelist['Baxis'] = np.sqrt(B3d.x1[0]**2+B3d.x2[0]**2+B3d.x3[0]**2)

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

rhop = kp["Te"]["rhop"] = kp["ne"]["rhop"] = np.linspace(0, 1, 100)
kp["ne"]["vals"] = (n_core - n_edge) * (1 - rhop ** n_alpha1) ** n_alpha2 + n_edge
kp["Te"]["vals"] = (T_core - T_edge) * (1 - rhop ** T_alpha1) ** T_alpha2 + T_edge

# set impurity species and sources rate
imp = namelist["imp"] = "Ar"
namelist["source_type"] = "const"
namelist["source_rate"] = 2e20  # particles/s

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk = {})

# set time-independent transport coefficients (flat D=1 m^2/s, V=-2 cm/s)
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -2e2 * np.ones(len(asim.rvol_grid))  # cm/s

# run Aurora forward model and plot results
out = asim.run_aurora(D_z, V_z, plot=True)

# extract densities and particle numbers in each simulation reservoir
nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out

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
