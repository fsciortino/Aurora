Aurora: a modern toolbox for particle transport, plasma-wall interaction, neutrals and radiation modeling
=========================================================================================================

Github repo: https://github.com/fsciortino/Aurora                                                                            

Paper/presentation in `Plasma Physics & Fusion Energy <https://iopscience.iop.org/article/10.1088/1361-6587/ac2890>`_ and on the `arXiv <https://arxiv.org/abs/2106.04528>`_.


Overview
--------

Aurora is a package to simulate heavy-ion transport, plasma-wall interaction (PWI), neutrals and radiation in magnetically-confined plasmas. It includes a 1.5D impurity transport forward model for the plasma ions, thoroughly benchmarked with the widely-adopted STRAHL code, and a simple multi-reservoir particle balance model including neutrals recycling, pumping and interaction with the material surfaces of the simulated device. A simple interface to plot and process atomic and surface data for fusion plasmas makes it a convenient tool for spectroscopy, PWI and integrated modeling. It also offers routines to analyze neutral states of hydrogen isotopes, both from the edge of fusion plasmas and from neutral beam injection. The spectroscopic and PWI calculations can be not only applied to the output of Aurora's own forward model, but also coupled with other 1D, 2D or 3D transport codes.

Aurora's code is mostly written in Python 3 and Fortran 90. A Julia interface has also recently been added.

.. figure:: figs/guido_reni_aurora.jpg
    :align: center
    :alt: Guido Reni - L'Aurora
    :figclass: align-center

    `Aurora fresco <https://smarthistory.org/reni-aurora/>`_, by Guido Reni (circa 1612-1614)

    

This documentation aims at making Aurora usage as clear as possible. Getting started is easy - see the :ref:`Installation` section.  To learn the basics, head to the :ref:`Tutorial` section. 

   
What is Aurora useful for?
--------------------------


Aurora is useful for modeling of particle transport, impurities, plasma-wall interaction, neutrals and radiation in fusion plasmas.

The package includes Python functionality to create inputs and read/plot outputs of impurity transport simulations. It was designed to be as efficient as possible in iterative workflows, where parameters (particularly the radial transport coefficients) are run through the forward model and repeatedly modified in order to match experimental observations. For this reason, Aurora avoids any disk input-output (I/O) during operation. All data is kept in memory. 

Aurora provides convenient interfaces to load a default namelist via :py:func:`~aurora.default_nml`, modify it as required and then pass the resulting namelist dictionary into the simulation setup. This is in the :py:class:`~aurora.core.aurora_sim` class, which allows creation of radial and temporal grids, interpolation of atomic rates and surface data, preparation of parallel loss rates at the edge, etc.. The function :py:func:`~aurora.transport_utils` contains tools to easily prepare the transport-related input parameters (diffusion and convection coefficients) for a run.

The :py:mod:`aurora.atomic` library provides functions to load and interpolate atomic rates from ADAS ADF-11 files, as well as from ADF-15 photon emissivity coefficients (PEC) files. PEC data can alternatively be computed using the collisional-radiative model of ColRadPy, using methods in :py:mod:`aurora.radiation`.

The :py:mod:`aurora.surface` library provides instead functions to load and interpolate TRIM-generated surface data, namely reflection coefficients, sputtering yields and implantation ranges.

A number of standard tests and examples are provided using a real set of Alcator C-Mod kinetic profiles and geometry. In order to interface with EFIT gEQDSK files, Aurora makes use of the `omfit_eqdsk <https://pypi.org/project/omfit-eqdsk/>`__ package, which offers flexibility to work with data from many devices worldwide. Users may easily substitute this dependence with different magnetic reconstruction packages and/or postprocessing interfaces, if required. Interfacing Aurora with several file formats used throughout the fusion community to store kinetic profiles is simple. 

Aurora was born as a fast forward model of impurity transport, but it can also be useful for synthetic spectroscopic diagnostics and radiation modeling in fusion plasmas. For example, it may be helpful for parameter scans to explore the performance of future devices. The :py:func:`~aurora.radiation.radiation_model` method allows one to use ADAS atomic rates and given kinetic profiles to compute line radiation, bremsstrahlung, continuum and soft-x-ray-filtered radiation. Ionization equilibria can also be computed using the :py:meth:`~aurora.atomic` methods, thus enabling simple "constant-fraction" models where the total density of an impurity species is fixed to a certain percentage of the electron density. Background neutrals, either from the edge or from neutral beam injection, can be analyzed using the :py:mod:`aurora.neutrals` and :py:mod:`aurora.nbi_neutrals` libraries.


.. figure:: https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Virmalised_18.03.15_%284%29.jpg/1920px-Virmalised_18.03.15_%284%29.jpg
   :width: 50%
   :alt: Aurora Borealis, photo by K.Pikner on Wikipedia
   :align: center
   Inspirational photo of the Aurora Borealis by K.Pikner

   
Documentation contents
----------------------
.. toctree::
   :maxdepth: 4

   install
   aurora_req
   model
   params
   tutorial
   atomic_data
   surface_data
   external_codes
   facit
   citing
   contacts
   aurora



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`