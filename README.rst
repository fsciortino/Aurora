Aurora: a modern toolbox for impurity transport, neutrals and radiation modeling
================================================================================

Aurora is an expanding package to simulate heavy-ion transportm neutrals and radiation in magnetically-confined plasmas. It includes a 1.5D impurity transport forward model which inherits many of the methods from the historical STRAHL code and has been thoroughly benchmarked with it. It also offers routines to analyze neutral states of hydrogen isotopes, both from the edge of fusion plasmas and from neutral beam injection. Aurora's code is mostly written in Python 3 and Fortran 90. A Julia interface has also recently been added. The package enables radiation calculations using ADAS atomic rates, which can easily be applied to the output of Aurora's own forward model, or coupled with other 1D, 2D or 3D transport codes. 

Documentation is available at https://aurora-fusion.readthedocs.io.


Development 
-----------

The code is developed and maintained by F. Sciortino (MIT-PSFC) in collaboration with T. Odstrcil (GA), A. Cavallaro (MIT) and R. Reksoatmodjo (W&M), with support from O. Linder (MPI-IPP), C. Johnson (U. Auburn), D. Stanczak (IPPLM) and S. Smith (GA). The STRAHL documentation provided by R.Dux (MPI-IPP) was extremely helpful to guide the initial development of Aurora.

New contributors are more than welcome! Please get in touch at sciortino-at-psfc.mit.edu or open a pull-request via Github. 

Generally, we would appreciate if you could work with us to merge your features back into the main Aurora distribution if there is any chance that the changes that you made could be useful to others. 

Installation
------------
.. image:: https://badge.fury.io/py/aurorafusion.svg
    :target: https://badge.fury.io/py/aurorafusion
    
Aurora can be installed from PyPI using

    pip install aurorafusion
    
Add a `--user` flag to the command above if you don't have write-access to the default package directory on your system (i.e. if you don't have root permissions). 

Installing via conda is now also possible using

    conda install -c sciortino aurorafusion 

.. image:: https://anaconda.org/sciortino/aurorafusion/badges/version.svg
    :target: https://anaconda.org/sciortino/aurorafusion
    
.. image:: https://anaconda.org/sciortino/aurorafusion/badges/latest_release_relative_date.svg
    :target: https://anaconda.org/sciortino/aurorafusion

Note that the conda version is NOT updated very regularly. If this kind of installation is your preference, feel free to contact the F.Sciortino to request an update. The conda installation does not currently install dependencies on `omfit_classes`, which users may need to install via `pip` (see the `PyPI repo <https://pypi.org/project/omfit-classes/>`_). 

To look at the code and contribute to the Aurora repository, it is recommended to install from source, by git-cloning the  `Aurora repo <https://github.com/fsciortino/aurora>`_ from Github. This will ensure that you can access the latest version of the tools. 

For compilation after git-cloning, users can make use of the `setup.py` file, e.g. using 

    python setup.py -e .

or use the makefile in the package directory to build the Fortran code using 

    make clean; make
   
Note that the makefile will not install any of the dependencies, listed in the `requirements.txt` file in the main directory. You can use this file to quickly install dependencies within a Python virtual environment, or install each dependency one at a time.

Note that the Julia version of the code is not built by default. If you have Julia installed on your system, you can do  

    make julia

from the main package directory. This will build a Julia `sysimage` to speed up access of Julia source code from Python, but it is not strictly necessary. See the documentation to read about interfacing Python3 and Julia. 



License
-------

The package is made open-source with the hope that this will speed up research on fusion energy and make further code development easier. However, we kindly ask that all users communicate to us their purposes, difficulties and successes with Aurora, so that we may support users as much as possible and grow the code further. 


Citing Aurora
-------------

Please see the `User Agreement <https://github.com/fsciortino/Aurora/blob/master/USER_AGREEMENT.txt>`_. 
