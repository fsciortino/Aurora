Tutorial: 
----------------------------------------------------------------

Some basic :py:mod:`aurora` functionality is demonstrated in the `examples` package directory, where users may find a number of useful scripts. Here, we go through some of the same examples and methods.

If :py:mod:`aurora` is correctly installed, you should be able to do::

  import aurora

and then load a default namelist for impurity transport forward modeling::

  namelist = aurora.load_default_namelist()

Note that you can always look at where this function is defined in the package by using, e.g.::

  aurora.load_default_namelist.__module__

Once you have loaded the default namelist, have a look at the `namelist` dictionary. It contains a number of parameters that are needed for :py:mod:`aurora` runs. Some of them, like the name of the device, are only important if automatic fetching of the EFIT equilibrium through :py:mod:`MDSplus` is required, or else it can be ignored (leaving it to its default value). Most of the parameter names should be fairly self-descriptive, but a detailed description will be available soon. In the meantime, please refer to docstrings through the code documentation.

Next, read in a magnetic equilibrium. You can find an example from a C-Mod discharge in the `examples` directory::
  
  geqdsk = omfit_eqdsk.OMFITgeqdsk('example.gfile')

The output `geqdsk` dictionary contains the contents of the EFIT geqdsk file, with additional processing done by the :py:mod:`omfit_eqdsk` package for flux surfaces. Only some of the dictionary fields are used; refer to the :py:meth:`~aurora.grids_utiles` methods for details. The `geqdsk` dictionary is used to create a mapping between the `rhop` grid (square root of normalized poloidal flux) and a `rvol` grid, defined by the normalized volume of each flux surface. :py:mod:`aurora`, like STRAHL, runs its simulations on the `rvol` grid. 

We next need to read in some kinetic profiles, for example from an `input.gacode` file (available in the `examples` directory)::
  
  inputgacode = omfit_gapy.OMFITgacode('example.input.gacode')

Other file formats (e.g. plasma statefiles, TRANSP outputs, etc.) may also be read with :py:mod:`omfit_gapy` or other OMFIT-distributed packages. It is however not important to :py:mod:`aurora` how the users get kinetic profiles: all that matters is that they are stored in the `namelist['kin_prof']` dictionary. To set up time-independent kinetic profiles we can use::

  kp = namelist['kin_profs']
  kp['Te']['rhop'] = kp['ne']['rhop'] = np.sqrt(inputgacode['polflux']/inputgacode['polflux'][-1])
  kp['ne']['vals'] = inputgacode['ne']*1e13    # 1e19 m^-3 --> cm^-3
  kp['Te']['vals'] = inputgacode['Te']*1e3     # keV --> eV

Note that both electron density (`ne`) and temperature (`Te`) must be saved on a `rhop` grid. This grid is internally used by :py:mod:`aurora` to map to the `rvol` grid. Also note that, unless otherwise stated, :py:mod:`aurora` inputs are always in CGS units, i.e. all spatial quantities are given in :math:`cm`!! (exclamation marks are there to highlight that "I told you").

Next, we specify the ion species that we want to simulate. We can simply do::

  imp = namelist['imp'] = 'Ar'

and :py:mod:`aurora` will internally find ADAS data for that ion (assuming that this is one of the common ones for fusion modeling). The namelist also contains information on what kind of source of impurities we need to simulate; here we are going to select a constant source (starting at t=0) of :math:`10^{24}` particles/second.::

  namelist['source_type'] = 'const'
  namelist['Phi0'] = 1e24

Time dependent time histories of the impurity source may however be given by selecting `namelist['source_type']="step"` (for a series of step functions), `"synth_LBO"` (for an analytic function resembling a laser-blow-off (LBO) time history) or `"file"` (to load a detailed function from a file). Refer to the :py:meth:`~aurora.source_utils.get_source_time_history` method for more details. 

Assuming that we're happy with all the inputs in the namelist at this point (many more could be changed!), we can now go ahead and set up our :py:mod:`aurora` simulation:::

  asim = aurora.aurora_sim(namelist, geqdsk=geqdsk)

The :py:class:`~aurora.core.aurora_sim` class creates a Python object with spatial and temporal grids, kinetic profiles, atomic rates and all other inputs to the forward model. :py:mod:`aurora` uses a diffusive-convective model for particle fluxes, so we need to specify diffusion (D) and convection (V) coefficients next:::

  D_z = 1e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
  V_z = -2e2 * np.ones(len(asim.rvol_grid)) # cm/s

Here we have made use of the `rvol_grid` attribute of the `asim` object, whose name is self-explanatory. This grid has a 1-to-1 correspondence with `asim.rhop_grid`. In the lines above we have created flat profiles of :math:`D=10^4 cm^2/s` and :math:`V=-2\times 10^2 cm/s`, defined on our simulation grids. D's and V's could in principle (and, very often, in practice) be defined with more dimensions to represent a time-dependence and also different values for different charge states. Unless specifed otherwise, :py:mod:`aurora` assumes all points of the time grid (now stored in `asim.time_grid`) and all charge states to have the same D and V. See the :py:meth:`~aurora.core.run_aurora` method for details on how to speficy further dependencies.

At this point, we are ready to run an :py:mod:`aurora` simulation, with::

  out = asim.run_aurora(D_z, V_z)

which is blazing fast! Depending on how many time and radial points you have requested (a few hundreds by default), how many charge states you are simulating, etc., a simulation could take as little as <50 ms, which is a significant improvement with respect to other codes. If you add `use_julia=True` to the :py:meth:`~aurora.core.run_aurora` call the run will be even faster; make sure to wear your seatbelt.

You can easily check the quality of particle conservation in the various reservoirs by using::

  reservoirs = asim.check_conservation()

which will show the results in full detail. The `reservoirs` output list contains information about how many particles are in the plasma, in the wall reservoir, in the pump, etc.. Refer to the  :py:meth:`~aurora.core.run_aurora` docstring for details. 

A plot is worth a thousand words, so let's make one for the charge state densities (on a nice slider!)::

  aurora.plot_tools.slider_plot(
    asim.rvol_grid, asim.time_out, asim.rad['line_rad'].transpose(1,2,0),
    xlabel=r'$r_V$ [cm]', ylabel='time [s]', zlabel='Total radiation [A.U.]',
    labels=[str(i) for i in np.arange(0,nz.shape[1])], plot_sum=True, x_line=asim.rvol_lcfs
    )

Use the slider to go over time, as you look at the distributions over radius of all the charge states. 
