Aurora: a modern toolbox for impurity transport, neutrals and radiation modeling
================================================================================

.. image:: https://badge.fury.io/py/aurorafusion.svg
    :target: https://badge.fury.io/py/aurorafusion
    
.. image:: https://anaconda.org/conda-forge/aurorafusion/badges/version.svg   
    :target: https://anaconda.org/conda-forge/aurorafusion

.. image:: https://anaconda.org/conda-forge/aurorafusion/badges/latest_release_date.svg   
    :target: https://anaconda.org/conda-forge/aurorafusion

.. image:: https://anaconda.org/conda-forge/aurorafusion/badges/platforms.svg   
    :target: https://anaconda.org/conda-forge/aurorafusion

.. image:: https://anaconda.org/conda-forge/aurorafusion/badges/license.svg   
    :target: https://anaconda.org/conda-forge/aurorafusion

.. image:: https://anaconda.org/conda-forge/aurorafusion/badges/downloads.svg   
    :target: https://anaconda.org/conda-forge/aurorafusion

Aurora is a package to simulate heavy-ion transportm neutrals and radiation in magnetically-confined plasmas. It includes a 1.5D impurity transport forward model, thoroughly benchmarked with the widely-adopted STRAHL code. It also offers routines to analyze neutral states of hydrogen isotopes, both from the edge of fusion plasmas and from neutral beam injection. A simple interface to atomic data for fusion plasmas makes it a convenient tool for spectroscopy and integrated modeling. Aurora's code is mostly written in Python 3 and Fortran 90. An experimental Julia interface has also been added. 

Documentation is available at https://aurora-fusion.readthedocs.io.


Development 
-----------

The code is developed and maintained by F. Sciortino (MPI-IPP) in collaboration with T. Odstrcil (GA), D. Fajardo (MPI-IPP), A. Cavallaro (MIT) and R. Reksoatmodjo (W&M), with support from O. Linder (MPI-IPP), C. Johnson (U. Auburn), D. Stanczak (IPPLM) and S. Smith (GA). The STRAHL documentation provided by R.Dux (MPI-IPP) was extremely helpful to guide the initial development of Aurora.

New contributors are more than welcome! Please get in touch at francesco.sciortino-at-ipp.mpg.de or open a pull-request via Github. 

Generally, we would appreciate if you could work with us to merge your features back into the main Aurora distribution if there is any chance that the changes that you made could be useful to others. 

Installation
------------

Aurora can be installed from PyPI using

    pip install aurorafusion --user
    
You can omit the `--user` flag if you have write-access to the default package directory on your system and wish to install there.

Installing via conda is also possible using

    conda install -c conda-forge aurorafusion 
    
    
Both the PyPI and conda installation are automatically updated at every package release. Note that the conda installation does not currently install dependencies on `omfit_classes`, which users may need to install via `pip` (see the `PyPI repo <https://pypi.org/project/omfit-classes/>`_). 

To look at the code and contribute to the Aurora repository, it is recommended to install from source, by git-cloning the  `Aurora repo <https://github.com/fsciortino/aurora>`_ from Github. This will ensure that you can access the latest version of the tools. 

For compilation after git-cloning, users can make use of the `setup.py` file, e.g. using 

    python setup.py -e .

or use the makefile in the package directory to build the Fortran code using 

    make clean; make
   
Note that the makefile will not install any of the dependencies, listed in the `requirements.txt` file in the main directory. You can use this file to quickly install dependencies within a Python virtual environment, or install each dependency one at a time.

The Julia version of the code is not built by default. If you have Julia installed on your system, you can do  

    make julia

from the main package directory. This will build a Julia `sysimage` to speed up access of Julia source code from Python, but it is not strictly necessary. See the documentation to read about interfacing Python and Julia. 


Atomic data
-----------

Aurora offers a simple interface to download, read, process and plot atomic data from the Atomic Data and Structure Analysis (ADAS) database, particularly through the OPEN-ADAS website: www.open-adas.ac.uk . ADAS data files can be fetched remotely and stored within the Aurora distribution directory, or users may choose to fetch ADAS files from a chosen, pre-existing directory by setting

    export AURORA_ADAS_DIR=my_adas_directory
    
within their Linux environment (or analogous). If an ADAS files that is not available in AURORA_ADAS_DIR is requested by a user, Aurora attempts to download it and store it there. If you are using a public installation of Aurora and you do not have write-access to the directory where Aurora is installed, make sure to set AURORA_ADAS_DIR to a directory where you do have write-access before starting.

Several ADAS formats can currently be managed -- please see the docs. Please contact the authors to request and/or suggest expansions of current capabilities.



License
-------

Aurora is distributed under the MIT License. The package is made open-source with the hope that this will speed up research on fusion energy and make further code development easier. However, we kindly ask that all users communicate to us their purposes, difficulties and successes with Aurora, so that we may support users as much as possible and grow the code further. 


Citing Aurora
-------------

Please see the `User Agreement <https://github.com/fsciortino/Aurora/blob/master/USER_AGREEMENT.txt>`_. 
