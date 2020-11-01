Input parameters
================

In this page, we describe some of the most important input parameter for Aurora simulations. Since all Aurora inputs are created in Python, rather than in a low-level language, users are encouraged to browse through the module documentation to get a complete picture; here, we only look at some specific features. 


Spatio-temporal grids
-----------------------------

Aurora's spatial and temporal grids are defined in the same way as in STRAHL. Refer to the [STRAHL manual](https://pure.mpg.de/rest/items/item_2143869/component/file_2143868/content) for details. Note that only STRAHL options that have been useful in the authors' experience have been included in Aurora. 

In short, the :py:func:`~aurora.grids_utils.create_radial_grid` function produces a radial grid that is equally-spaced on the :math:`rho` grid, defined by

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


Recycling
---------

A 1.5D transport model such as Aurora cannot accurately model recycling at walls. Like STRAHL, Aurora uses a number of parameters to approximate the transport of impurities outside of the LCFS; we recommend that users ensure that their core results don't depend sensitively on these parameters:

#. `recycling_flag`: if this is False, no recycling nor communication between the divertor and core plasma particle reservoirs is allowed.

#. `wall_recycling` : if this is 0, particles are allowed to move from the divertor reservoir back into the core plasma, based on the `tau_div_SOL_ms` and `tau_pump_ms` parameters, but no recycling from the wall is enabled. If >0 and <1, recycling of particles hitting the limiter and wall reservoirs is enabled, with a recycling coefficient equal to this value. 

#. `tau_div_SOL_ms` : time scale with which particles travel from the divertor into the SOL, entering again the core plasma reservoir. Default is 50 ms.

#. `tau_pump_ms` : time scale with which particles are completely removed from the simulation via a pumping mechanism in the divertor. Default is 500 ms (very long)

#. `tau_rcl_ret_ms` : time scale of recycling retention at the wall. This parameter is not present in STRAHL. It is introduced to reproduce the physical observation that after an ELM recycling impurities may return to the plasma over a finite time scale. Default is 50 ms.

#. `SOL_mach`: Mach number in the SOL. This is used to compute the parallel loss rate, both in the open SOL and in the limiter shadow. Default is 0.1.

#. `divbls` : fraction of user-specified impurity source that is added to the divertor reservoir rather than the core plasma reservoir. These particles can return to the core plasma only if `recycling_flag=True` and `wall_recycling>=0`. This parameter is useful to simulate divertor puffing. 

The parallel loss rate in the open SOL and limiter shadow also depends on the local connection length. This is approximated by two parameters: `clen_divertor` and `clen_limiter`, in the open SOL and the limiter shadow, respectively. These connection lengths can be approximated using the edge safety factor and the major radius from the `geqdsk`, making use of the :py:func:`~aurora.grids_utils.estimate_clen` function.
