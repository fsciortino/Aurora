Input parameters
================

In this page, we describe some of the most important input parameter for Aurora simulations. Since all Aurora inputs are created in Python, rather than in a low-level language, users are encouraged to browse through the module documentation to get a complete picture; here, we only look at some specific features. 


Namelist for ion transport simulations
--------------------------------------
The table below describes the main input parameters to Aurora's forward model of ion transport. Refer to the following sections for details on spatio-temporal grids, recycling and kinetic profiles specifications.


.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - `imp`
     - Ca
     - Atomic symbol of the simulated ion.
   * - `main_element`
     - D
     - Background ion species, usually hydrogen isotopes.
   * - `source_rate`
     - 1e+21
     - Flux [particles/s] of simulated ions.
   * - `source_type`
     - const
     - Type of ion source, one of ['file','const','step','synth_LBO'], see :py:func:`~aurora.source_utils.get_source_time_history`.
   * - `explicit_source_vals`
     - `None`
     -  2D array for sources on `explicit_source_time` and `explicit_source_rhop` grids
   * - `explicit_source_time`
     - `None`
     -  Time grid for explicit source
   * - `explicit_source_rhop`
     - `None`
     - :math:`\rho_p` grid for explicit source
   * - `source_width_in`
     - 0.0
     - Inner Gaussian source width, only used if >0. See :py:func:`~aurora.source_utils.get_radial_source`.
   * - `source_width_out`
     - 0.0
     - Outer Gaussian source width, only used if >0. See :py:func:`~aurora.source_utils.get_radial_source`.
   * - `imp_source_energy_eV`
     - 3.0
     - Energy of neutral ion source, only used if `source_width_in=source_width_out=0`, see :py:func:`~aurora.source_utils.get_radial_source`.
   * - `prompt_redep_flag`
     - False
     - If True, a simple prompt redeposition model is activated, see :py:func:`~aurora.source_utils.get_radial_source`.
   * - `source_file`
     - None
     - Location of source file, using STRAHL format, only used if `source_type="file"`, see :py:func:`~aurora.source_utils.get_source_time_history`.
   * - `source_cm_out_lcfs`
     - 1.0
     - Source distance in cm from LCFS
   * - `LBO["n_particles"]`
     - 1e+18
     - Number of particles in LBO synthetic source, only used if `source_type="synth_LBO"`
   * - `LBO["t_fall"]`
     - 0.3
     - Decay time of LBO synthetic source, only used if `source_type="synth_LBO"`
   * - `LBO["t_rise"]`
     - 0.05
     - Rise time of LBO synthetic source, only used if `source_type="synth_LBO"`
   * - `LBO["t_start"]`
     - 0.0
     - Start time of LBO synthetic source, only used if `source_type="synth_LBO"`
   * - `timing["dt_increase"]`
     - [1.005 1.   ]
     - `dt` multipliers at every time step change. See detailed description.
   * - `timing["dt_start"]`
     - [1.e-05 1.e-03]
     - `dt` values at the beginning of each interval/cycle. See detailed description.
   * - `timing["steps_per_cycle"]`
     - [1 1]
     - Number of steps before `dt` is multiplied by a `dt_increase` value. See detailed description.
   * - `timing["times"]`
     - [0.  0.1]
     - Times at which intervals/cycles change.
   * - `bound_sep`
     - 2.0
     - Distance between wall boundary and plasma separatrix [cm].
   * - `lim_sep`
     - 1.0
     - Distance between nearest limiter and plasma separatrix [cm].
   * - `clen_divertor`
     - 17.0
     - Connection length to the divertor [cm].
   * - `clen_limiter`
     - 0.5
     - Connection length to the nearest limiter [cm]
   * - `dr_0`
     - 0.3
     - Radial grid spacing on axis. See detailed description.
   * - `dr_1`
     - 0.05
     - Radial grid spacing near the wall. See detailed description.
   * - `K`
     - 6.0
     - Exponential grid resolution factor. See detailed description.
   * - `SOL_decay`
     - 0.05
     - Decay length at the wall bounday, numerical parameter for the last grid point.
   * - `saw_model["saw_flag"]`
     - False
     - If True, activate sawtooth phenomenological model.
   * - `saw_model["rmix"]`
     - 1000.0
     - Mixing radius of sawtooth model. Each charge state density is flattened inside of this.
   * - `saw_model["times"]`
     - [1.0]
     - Times at which sawteeth occur.
   * - `saw_model["crash_width"]`
     - 1.0
     - Smoothing width of sawtooth crash [cm].
   * - `recycling_flag`
     - False
     - If True, particles may return to main chamber, either via flows from the SOL or proper recycling.
   * - `wall_recycling`
     - 0.0
     - If True, recycling is activated: particles from the wall and divertor may return to main chamber.
   * - `divbls`
     - 0.0
     - Fraction of source that is puffed into the divertor, as opposed to into the main chamber.
   * - `tau_div_SOL_ms`
     - 50.0
     - Time scale for transport between the divertor and the open SOL [ms].
   * - `tau_pump_ms`
     - 500.0
     - Time scale for pumping out of divertor reservoir [ms].
   * - `tau_rcl_ret_ms`
     - 50.0
     - Time scale for retention at the wall [ms] before recycling may occur. 
   * - `SOL_mach`
     - 0.1
     - Mach number in the SOL, determining parallel loss rates.
   * - `kin_profs["ne"]`
     - {'fun': 'interpa', 'times': [1.0]}
     - Specification of electron density [:math:`cm^{-3}`]. `fun="interpa"` interpolates data also in the SOL.
   * - `kin_profs["Te"]`
     - {'fun': 'interp', 'times': [1.0], 'decay': [1.0]}
     - Specification of electron temperature [:math:`eV`]. `fun="interp"` sets decay over `decay` length in the SOL.
   * - `kin_profs["Ti"]`
     - {'fun': 'interp', 'times': [1.0], 'decay': [1.0]}
     - Specification of ion temperature [:math:`eV`]. Only used for charge exchange rates.
   * - `kin_profs["n0"]`
     - {'fun': 'interpa', 'times': [1.0]}
     - Specification of background (H-isotope) neutral density [:math:`cm^{-3}`].
   * - `nbi_cxr`
     - {'rhop': None, 'vals': None}
     - Radial profiles of charge exchange rates from NBI neutrals (fast+thermal) for each simulated charge state.
   * - `cxr_flag`
     - False
     - If True, activate charge exchange recombination with background thermal neutrals. Requires `kin_profs["n0"]`.
   * - `nbi_cxr_flag`
     - False
     - If True, activate charge exchange recombination with NBI neutrals (to be specified in :py:class:`~aurora.core.aurora_sim`).
   * - `device`
     - CMOD
     - Name of experimental device, only used by MDS+ if device database can be read via `omfit_eqdsk <https://pypi.org/project/omfit-eqdsk/>`_.
   * - `shot`
     - 99999
     - Shot number, only used in combination with `device` to connect to MDS+ databases.
   * - `time`
     - 1250
     - Time [ms] used to read magnetic equilibrium, if this is fetched via MDS+.
   * - `acd`
     - None
     - ADAS ADF11 ACD file (recombination rates). If left to None, uses defaults in :py:func:`~aurora.adas_files.adas_files_dict` for the chosen ion species.
   * - `scd`
     - None
     - ADAS ADF11 SCD file (ionization rates). If left to None, uses defaults in :py:func:`~aurora.adas_files.adas_files_dict` for the chosen ion species.
   * - `ccd`
     - None
     - ADAS ADF11 CCD file (nl-unresolved charge exchange rates). If left to None, uses defaults in :py:func:`~aurora.adas_files.adas_files_dict` for the chosen ion species.

    
    


Spatio-temporal grids
---------------------

Aurora's spatial and temporal grids are defined in the same way as in STRAHL. Refer to the `STRAHL manual <https://pure.mpg.de/rest/items/item_2143869/component/file_2143868/content>`__ for details. Note that only STRAHL options that have been useful in the authors' experience have been included in Aurora. 

In short, the :py:func:`~aurora.grids_utils.create_radial_grid` function produces a radial grid that is equally-spaced on the :math:`\rho` grid, defined by

    .. math::

        \rho = \frac{r}{\Delta r_{centre}} + \frac{r_{edge}}{k+1} \left(\frac{1}{\Delta r_{edge}}- \frac{1}{\Delta r_{centre}} \right) \left(\frac{r}{r_{edge}} \right)^{k+1}

The corresponding radial step size is given by

    .. math::

        \Delta r = \left[\frac{1}{\Delta r_{centre}} + \left(\frac{1}{\Delta r_{edge}} - \frac{1}{\Delta r_{centre}} \right) \left(\frac{r}{r_{edge}}\right)^k \right]^{-1}

The radial grid above requires a number of user parameters:

#. The `k` factor in the formulae; large values give finer grids at the plasma edge. A value of 6 is usually appropriate.

#. `dr_0` and `dr_1` give the radial spacing (in `rvol` units) at the center and at the last grid point (in cm).

#. The `r_edge` parameter in the formulae above is given by::

     r_edge = namelist['rvol_lcfs'] + namelist['bound_sep']

where `rvol_lcfs` is the distance from the center to the separatrix and `bound_sep` is the distance between the separatrix and the wall boundary, both given in flux-surface-volume normalized units. The `rvol_lcfs` parameter is automatically computed by the :py:class:`~aurora.core.aurora_sim` class initialization, based on the provided `geqdsk`. `bound_sep` can be estimated via the :py:func:`~aurora.grids_utils.estimate_boundary_distance` function, if an `aeqdsk` file can be accessed via `MDSplus` (alternatively, users may set it to anything they find appropriate). Additionally, since the edge model of Aurora simulates the presence of a limiter somewhere in between the LCFS and the wall boundary, we add a `lim_sep` parameter to specify the distance between the LCFS and the limiter surface. 

To demonstrate the creation of a spatial grid, we are going to select some example parameters::

  namelist={}
  namelist['K'] = 6.
  namelist['dr_0'] = 1.0  # 1 cm spacing near axis 
  namelist['dr_1'] = 0.1   # 0.1 cm spacing at the edge
  namelist['rvol_lcfs'] = 50.0 # 50cm minor radius (in rvol units)
  namelist['bound_sep'] = 5.0  # distance between LCFS and wall boundary
  namelist['lim_sep'] = 3.0 # distance between LCFS and limiter

  # now create grid and plot it
  rvol_grid, pro_grid, qpr_grid, prox_param = create_radial_grid(namelist,plot=True)

This will plot the radial spacing over the grid and show the location of the LCFS and the limiter, also specifying the total number of grid points. The larger the number of grid points, the longer simulations will take.

Similarly, to create time grids one needs a dictionary of input parameters, which :py:class:`~aurora.core.aurora_sim` automatically looks for in the dictionary `namelist['timing']`. The contents of this dictionary are

#. `timing['times']`: list of times at which the time grid must change. The first and last time indicate the start and end times of the simulation.

#. `timing['dt_start']`: list of time spacings (dt) at each of the times given by `timing['times']`.

#. `timing['steps_per_cycle']`: number of time steps before adapting the time step size. This defines a "cycle".

#. `timing['dt_increase']`: multiplicative factor by which the time spacing (dt) should change within one "cycle".

Let's test the creation of a grid and plot the result:::

  timing = {}
  timing['times'] = [0.,0.5, 1.]
  timing['dt_start'] = [1e-4,1e-3, 1e-3]  # last value not actually used, except when sawteeth are modelled!
  timing['steps_per_cycle'] = [2, 5, 1]   # last value not actually used, except when sawteeth are modelled!
  timing['dt_increase'] = [1.005, 1.01, 1.0]  # last value not actually used, except when sawteeth are modelled!
  time, save = aurora.create_time_grid(timing, plot=True)

The plot title will show how many time steps are part of the time grid (given by the `time` output). The `save` output is a list of 0's and 1's that is used to indicate which time grid points should be saved to the output. 



Particle sources
----------------

Core sources of particles can be specified in a number of ways. A time- and radially-dependent source can be set by setting `namelist['source_type'] = 'arbitrary_2d_source'` and then providing the parameters

#. `explicit_source_rhop` : radial grid (in square root of normalized poloidal flux)

#. `explicit_source_time` : time grid (in seconds)

#. `explicit_source_vals` : values of source flux (particles/s)

Alternatively, if time and radial dependences of core sources can be effectively separated, source time histories and radial profiles can be described in other ways. The time history of core sources can be created using the :py:func:`~aurora.source_utils.get_source_time_history` function, whereas radial profiles of core sources can be defined by specifying parameters for the :py:func:`~aurora.source_utils.get_radial_source` function. Please refer to the documentation of these functions for explanations of how to call these.


Particle sources can also be specified such that they enter the simulation from the divertor reservoir. This parameter can be useful to simulate divertor puffing. Note that it can only have an effect if `recycling_flag` = True and `wall_recycling` is >=0, so that particles from the divertor are allowed to flow to the main chamber plasma. In order to specify a source into the divertor, one needs to specify 2 parameters:

#. `source_div_time` : time base for the particle source into the divertor;
   
#. `source_div_vals` : values of the particle source into the divertor.




Edge parameters
---------------

A 1.5D transport model such as Aurora cannot accurately model edge transport. Aurora uses a number of parameters to approximate the transport of impurities outside of the LCFS; we recommend that users ensure that their core results don't depend sensitively on these parameters:
   
#. `recycling_flag`: if this is False, no recycling nor communication between the divertor and core plasma particle reservoirs is allowed.

#. `wall_recycling` : if this is 0, particles are allowed to move from the divertor reservoir back into the core plasma, based on the `tau_div_SOL_ms` and `tau_pump_ms` parameters, but no recycling from the wall is enabled. If >0 and <=1, recycling of particles hitting the limiter and wall reservoirs is enabled, with a recycling coefficient equal to this value. 

#. `tau_div_SOL_ms` : time scale with which particles travel from the divertor into the SOL, entering again the core plasma reservoir. Default is 50 ms.

#. `tau_pump_ms` : time scale with which particles are completely removed from the simulation via a pumping mechanism in the divertor. Default is 500 ms (very long)

#. `tau_rcl_ret_ms` : time scale of recycling retention at the wall. This parameter is not present in STRAHL. It is introduced to reproduce the physical observation that after an ELM recycling impurities may return to the plasma over a finite time scale. Default is 50 ms.

#. `SOL_mach`: Mach number in the SOL. This is used to compute the parallel loss rate, both in the open SOL and in the limiter shadow. Default is 0.1.

The parallel loss rate in the open SOL and limiter shadow also depends on the local connection length. This is approximated by two parameters: `clen_divertor` and `clen_limiter`, in the open SOL and the limiter shadow, respectively. These connection lengths can be approximated using the edge safety factor and the major radius from the `geqdsk`, making use of the :py:func:`~aurora.grids_utils.estimate_clen` function.




Kinetic profiles
----------------
In this section, we add a few more details on the specification of kinetic profiles in the Aurora namelist for 1.5D simulations of ion transport. We reproduce here the rows of the previous table that are relevant to this.


.. list-table:: Kinetic profiles specification
   :widths: 20 10 70
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - `kin_profs["ne"]`
     - {'fun': 'interpa', 'times': [1.0]}
     - Specification of electron density [:math:`cm^{-3}`]. `fun=interpa` interpolates data also in the SOL.
   * - `kin_profs["Te"]`
     - {'fun': 'interp', 'times': [1.0], 'decay': [1.0]}
     - Specification of electron temperature [:math:`eV`]. `fun=interp` sets decay over `decay` length in the SOL.
   * - `kin_profs["Ti"]`
     - {'fun': 'interp', 'times': [1.0], 'decay': [1.0]}
     - Specification of ion temperature [:math:`eV`]. Only used for charge exchange rates.
   * - `kin_profs["n0"]`
     - {'fun': 'interpa', 'times': [1.0]}
     - Specification of background (H-isotope) neutral density [:math:`cm^{-3}`].


Simulations that don't include charge exchange will only need electron density (`ne`) and temperature (`Te`). If charge exchange is added, then an ion temperature `Ti` and background H-isotope neutral density must be specified. Note that `Ti` should strictly be :math:`T_{red}=(m_H T_n + m_{imp} T_i)/(T_n+T_i)`, where `m_H` is the background species mass and `T_n` is the background neutral temperature, since only the effective ("reduced") energy of the neutral-impurity interaction enters the evaluation of charge exchange rates. `Ti` is also used to compute parallel loss rates in the SOL; if not provided by users, it is substituted by `Te`.

Each field of `kin_profs` requires specification of `fun`, `times`, `rhop` and `vals`. 

#. `fun` corresponds to a specification of interpolation functions in Aurora. Users should choose whether to interpolate data as given also in the SOL (`fun=interp`) or if SOL profiles should be substituted by an exponential decay. In the latter case, a decay scale length (in :math:`cm` units) should also be provided as `decay`.
#. `times` is a 1D array of times, in seconds, at which time-dependent profiles are given. If only a single value is given, whatever it may be, profiles are taken to be time independent.
#. `rhop` is a 1D array of radial grid values, given as square-root of normalized poloidal flux.
#. `vals` is a 2D array of values of the given kinetic quantity. The first dimension is expected to be time, the second radial coordinate. 
