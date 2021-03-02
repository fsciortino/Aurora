Requirements
============


Python requirements
-------------------

Aurora uses the latest Python-3 distribution and requires a modern Fortran compiler, available on most Unix systems. Additionally, the following packages are automatically installed (from PyPI) when installing Aurora:

  numpy scipy matplotlib xarray omfit_classes

The latter is part of the OMFIT distribution and will provide lots of capabilities to interact with tokamak modeling tools, with which Aurora can be easily integrated (indeed, Aurora is automatically installed as part of any OMFIT installation).





Julia requirements
------------------

To run the Julia version of the code, Julia must be installed; see::

  https://julialang.org/downloads/

Everything else should be automatically handled by the Aurora installation (see :ref:`Installation`).
