"""
Script to test the addition of new reservoirs wrt to the default ones
in the framework of the new advanced recycling model.

It is recommended to run this in IPython.
"""
import numpy as np
import matplotlib.pyplot as plt
from omfit_classes import omfit_eqdsk
import sys
import os

# Make sure that package home is added to sys.path
sys.path.append("../")
import aurora

plt.ion()

# pass any argument via the command line to show plots
plot = len(sys.argv) > 1

# read in default Aurora namelist
namelist = aurora.default_nml.load_default_namelist()

# Use gfile and statefile in local directory:
examples_dir = os.path.dirname(os.path.abspath(__file__))
geqdsk = omfit_eqdsk.OMFITgeqdsk(examples_dir + "/example.gfile")

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

# set the options for the recycling model
namelist['recycling_flag'] = True
    # if False, then all flows from the walls and from the neutral reservoirs back towards the plasma are prevented
    # if True, then particles from the walls and from the neutral reservoirs can return back to core plasma
namelist['wall_recycling'] = 0.8
    # if <0, then all particles arriving at the main/divertor walls remain stuck, and there is no backflow from the divertor/pump reservoirs
    # if >=0, then a fraction 1-'wall_recycling' of the particles arriving at the main/divertor walls remain stuck,
    #   while the rest is re-emitted with time constant 'tau_rcl_ret_ms',
    #   and there can be backflow from the divertor reservoir with time constant 'tau_div_SOL_ms'
    #   and from the pump reservoir (if present) with leakage conductance L_leak
    #   (this works only if 'recycling_flag' is True)
namelist['div_recomb_ratio'] = 0.2
    # fraction the impurity ion flow in the SOL which recombines before reaching the divertor target,
    #   i.e. which enters the divertor neutrals reservoir bypassing the divertor wall reservoir
    #   so, if this is < 1.0, ions can interact with the divertor wall through retention and recycling,
    #   and three new fields in the output are triggered:
    #   - N_divwall, i.e. number of particles permanently stuck at the divertor wall reservoir over time
    #       whose equivalent for the main wall is N_mainwall
    #   - N_divret, i.e. number of particles temporarily retained at the divertor wall reservoir over time
    #       whose equivalent for the main wall is N_mainret
    #   - rcld_rate, i.e. the recycling rate from the divertor wall reservoir 
    #       whose equivalent for the main wall is rclw_rate
namelist['tau_div_SOL_ms'] = 5.0
    # effective divertor retention time,
    #   i.e. time scale of particle loss from the divertor neutrals reservoir towards the core plasma
namelist['tau_rcl_ret_ms'] = 5.0
    # effective wall retention time,
    #   i.e. time scale of particle release from main and divertor walls

# set the options for the recycling model
namelist['phys_volumes'] = True
    # If False, a generic adimensional model for the pumpins is used. In this case, the pumping
    #   is defined through a characteristic pumping time tau_pump, so that pumped_flux = N * tau_pump,
    #   N = number of particles in the neutrals reservoir in front of the pump    
    # If True, the user can define a physical volume for the divertor neutrals reservoir,
    #   e.g. corresponding to the one of an actual device. In this case, the pumping is defined
    #   through an engineering pumping speed S_pump, in cm^3/s, so that pumped_flux = n * S_pump,
    #   with n = N/vol = particle density in the neutrals reservoir in front of the pump
namelist['vol_div'] = 0.4e6  # cm^3
    # Physical volume of the divertor neutrals reservoir, used if 'phys_volumes' = True
namelist['tau_pump_ms'] = 1000  # ms
    # Characteristic pumping time, which defines the pumping if 'phys_volumes' = False
namelist['S_pump'] = 5.0e7  # cm^3/s 
    # Engineering pumping speed, which defines the pumping if 'phys_volumes' = True
namelist['pump_chamber'] = True
    # If False, then the pumping is done directly from the divertor neutrals reservoir
    #   In this case, tau_pump_ms or S_pump are then applied to this reservoir
    # If True, then the particles can travel from the divertor reservoir towards a second
    #   "pump reservoir", and only from there are finally pumped.
    #   This option only works if 'phys_volumes' = True
    #   In this case, the transport between divertor and pump reservoirs is governed by a
    #     conductance L_divpump, in cm^3/s, so that flux_divpump = L_divpump * (n_div-n_pump),
    #     with n_div = N_div/vol_div and n_pump = N_pump/vol_pump particle densities in the two reservoirs
    #   Additionally, a flux of leaking particles from the pump reservoir back towards the main chamber
    #     is governed by a conductance L_leak, in cm^3/s, so that flux_leak = L_leak * n_pump
    #   Therefore, two new fields in the output are triggered:
    #   - N_pump, i.e. number of particles in the pump reservoir over time
    #   - rclp_rate, i.e. the leakage rate from the pump reservoir towards the main chamber
namelist['vol_pump'] = 1.0e6  # cm^3
    # Physical volume of the pump neutrals reservoir, used if 'phys_volumes' = True and 'pump_chamber' = True
namelist['L_divpump'] = 0.5e8 # cm^3/s 
    # Neutral transport conductance between divertor and pump neutrals reservoirs
namelist['L_leak'] = 1.0e6 # cm^3/s
    # Leakage conductance between pump neutrals reservoir and main chamber

# Now get aurora setup
asim = aurora.core.aurora_sim(namelist, geqdsk=geqdsk)

# set time-independent transport coefficients (flat D=1 m^2/s, V=-2 cm/s)
D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -2e2 * np.ones(len(asim.rvol_grid))  # cm/s

# run Aurora forward model and plot results
# note that, since 'physical_volumes' = True, in the resulting plots
# the particle content in the plasma, divertor and pump reservoirs is
# automatically expressed in terms of density (cm^-3) rather than in #
out = asim.run_aurora(D_z, V_z, plot=True)

# extract densities and particle numbers in each simulation reservoir
# mind that, since 'div_recomb_ratio' < 1.0 and 'pump_chamber' = True, more arrays
# will be contained in the tuple out, which must be extracted in the correct order
# see core.py for the order of the fields in the various cases
nz, N_mainwall, N_divwall, N_div, N_pump, N_out, N_mainret, Ndivret, N_tsu, N_dsu, N_dsul, rcld_rate, rclb_rate, rclp_rate, rclw_rate = out

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
