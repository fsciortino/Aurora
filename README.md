# Aurora: a modern toolbox for impurity transport and radiation modeling

Aurora is an expanding package to simulate heavy-ion transport and radiation in magnetically-confined plasmas. The package offers a simple interface between Python3 and Fortran 90 -- a solution that ensures fast iterations over a 1.5-D forward model, while maintaining simplicity for users. Recently, a Julia interface has also been developed and is now in beta testing. The package includes extensive Python3 functionality to create inputs and read/plot outputs. Users can read and process ADAS atomic data to make radiation predictions, both using the charge state density distributions from an Aurora simulation or from fractional abundances in ionization equilibrium. This type of radiation predictions can be conveniently made for Aurora's own forward model results, and also for external 1D, 2D or 3D transport codes. 

Documentation is available at https://aurora-fusion.readthedocs.io.

Aurora's 1.5D forward model has been thoroughly benchmarked with the standard STRAHL, based on which it was originally developed. Recently, the algorithm proposed by O.Linder et al, Nuclear Fusion 2020, has been implemented and is now the default numerical scheme, although legacy options remain available. 

<img src="https://user-images.githubusercontent.com/25516628/93692659-f12c4b00-fac3-11ea-817c-d971c6853b8b.jpg" width="500" align="right">

Coupling to [ColRadPy](https://github.com/johnson-c/ColRadPy) is also available to process ADAS ADF04 files and estimate emission from Aurora charge state distributions at specific wavelengths. We recommend installation of ColRadPy from the source via Github cloning. 

A number of examples using Aurora are provided using real Alcator C-Mod kinetic profiles and geometry. In order to interface with EFIT gEQDSK files, we make use of the [omfit-eqdsk](https://gafusion.github.io/OMFIT-source/classes.html) package, but users may easily substitute this if they prefer to adopt different magnetic reconstruction packages and/or postprocessing interfaces. 

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
