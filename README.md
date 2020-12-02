# Aurora: a modern toolbox for impurity transport, neutrals and radiation modeling

Aurora is an expanding package to simulate heavy-ion transportm neutrals and radiation in magnetically-confined plasmas. It includes a 1.5D impurity transport forward model which inherits many of the methods from the historical STRAHL code and has been thoroughly benchmarked with it. It also offers routines to analyze neutral states of hydrogen isotopes, both from the edge of fusion plasmas and from neutral beam injection. Aurora's code is mostly written in Python 3 and Fortran 90. A Julia interface has also recently been added. The package enables radiation calculations using ADAS atomic rates, which can easily be applied to the output of Aurora's own forward model, or coupled with other 1D, 2D or 3D transport codes. 

.. figure:: https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Virmalised_18.03.15_%284%29.jpg/1920px-Virmalised_18.03.15_%284%29.jpg
   :width: 50%
   :alt: Aurora Borealis, photo by K.Pikner on Wikipedia	   
   :align: left

   Inspirational photo of the Aurora Borealis by K.Pikner

Documentation is available at https://aurora-fusion.readthedocs.io.


# Development 

The code is developed and maintained by F. Sciortino (MIT-PSFC) in collaboration with T. Odstrcil (GA) and A. Cavallaro (MIT), with support from O. Linder (MPI-IPP), C. Johnson (U. Auburn), D. Stanczak (IPPLM) and S. Smith (GA). The STRAHL documentation provided by R.Dux (MPI-IPP) was extremely helpful to guide the initial development of Aurora.

New contributors are more than welcome! Please get in touch at sciortino-at-psfc.mit.edu or open a pull-request via Github. 

Generally, we would appreciate if you could work with us to merge your features back into the main Aurora distribution if there is any chance that the changes that you made could be useful to others. 

# Installation

We recommend installing from source, by git-cloning [this repo](https://github.com/fsciortino/aurora) from Github. This will ensure that you can access the latest version of the tools. Make sure to use the `master` branch to use a stable version. *Make use of the Makefile in the package directory to build the Fortran or Julia code* using 
```
make clean; make aurora
```
Note that the Julia version of the code is not built by default. If you have Julia installed on your system, you can do  
```
make julia
```
from the main package directory. See the documentation to read about interfacing Python3 and Julia. 

In the near future, the latest release of the package will also available on the Anaconda Cloud:

[![Anaconda-Server Badge](https://anaconda.org/sciortino/aurorafusion/badges/latest_release_date.svg)](https://anaconda.org/sciortino/aurorafusion)

and from PyPI using 
```
pip install aurorafusion
```

# License

The package is made open-source with the hope that this will speed up research on fusion energy and make further code development easier. However, we kindly ask that all users communicate to us their purposes, difficulties and successes with Aurora, so that we may support users as much as possible and grow the code further. 


# Citing Aurora

Please see the [User Agreement](https://github.com/fsciortino/Aurora/blob/master/USER_AGREEMENT.txt). 
