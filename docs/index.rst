aurora: modern 1.5D modeling of particle transport in magnetically-confined plasmas
===================================================================================

Github repo: https://github.com/fsciortino/aurora

Overview
--------

:py:mod:`aurora` is a 1.5D modern forward model for radial particle transport in magnetically confined plasmas. It inherets many of its methods from the historical STRAHL code and has been thoroughly benchmarked with it. The core of :py:mod:`aurora` is written in Python3 and Fortran90, with a Julia interface also under development.

:py:mod:`aurora` includes Python functionality to create inputs and read/plot outputs. OMFIT users may access this functionality via the OMFIT STRAHL module. :py:mod:`aurora` was designed to be as efficient as possible in iterative workflows, where different diffusion and convection coefficients are run through the code in order to match some experimental observations. For this reason, :py:mod:`aurora` keeps all data in memory and avoids any I/O during operation. 

A number of standard tests and examples are provided using a real set of Alcator C-Mod kinetic profiles and geometry. In order to interface with EFIT gEQDSK files, :py:mod:`aurora` makes use of the :py:mod:`omfit_eqdsk` package, which offers flexibility to work with data from many devices worldwide. Users may easily substitute this dependence with different magnetic reconstruction packages and/or postprocessing interfaces, if required.

:py:mod:`aurora` provides convenient interfaces to load a default namelist via :py:func:`~aurora.default_nml`, modify it as required and then pass the resulting namelist dictionary into the simulation setup. This is in the main class of :py:mod:`aurora`, :py:class:`~aurora.core.aurora_sim`, which allows creation of radial and temporal grids, interpolation of atomic rates, preparation of parallel loss rates at the edge, etc.

The library in :py:meth:`~aurora.atomic` provides functions to load and interpolate atomic rates from ADAS ADF-11 files, as well as from ADF-15 photon emissivity coefficients (PEC) files. PEC data can alternatively be computed using the collisional-radiative model of CollRadPy, using methods in :py:func:`~aurora.radiation`.

:py:mod:`aurora` was born as a fast forward model of impurity transport, but it can do much more. For example, it may be helpful for parameter scans in modeling of future devices. The :py:func:`~aurora.radiation_model` allows one to use ADAS atomic rates and given kinetic profiles to compute line radiation, bremsstrahlung, continuum and soft-x-ray-filtered radiation. Ionization equilibria can also be computed using the :py:meth:`~aurora.atomic` methods, thus enabling simple "constant-fraction" models where the total density of an impurity species is fixed to a certain percentage of the electron density. 


Installation
------------

To obtain the latest version of the code, it is recommended to git-clone the repo
https://github.com/fsciortino/aurora
and run the makefile from the command line.

The latest stable version of the code can also be obtained via::

    pip install aurora

of from Anaconda Cloud::

    conda install aurora


Demos/Examples
--------------
A number of demonstrations and examples are available in the :py:mod:`aurora` "examples" directory. These show how to load a default namelist, modify it with specific kinetic profiles, load a magnetic equilibrium, set up an :py:mod:`aurora` object and run. A number of plotting tools and postprocessing scripts to calculate radiation are also presented.


Questions? Suggestions? 
-----------------------
Please contact sciortino-at-psfc.mit.edu for any questions. Suggestions and collaborations are more than welcome!


  
Package Reference
-----------------
.. toctree::
   :maxdepth: 4

   aurora



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
