# About

`aurora` is a modern heavy-ion transport code for magnetically-confined plasmas. Its main applications are forward modeling of impurity transport and radiation predictions for fusion synthetic diagnostics. `aurora` offers a simple interface between Python3 and Fortran 90 -- a solution that ensures fast iterations, while maintaining simplicity for users. Recently, a Julia interface has also been developed and is now in beta testing. The package includes extensive Python3 functionality to create inputs and read/plot outputs. Moreover, users can read and process ADAS atomic data to make radiation predictions, both using the charge state density distributions from an `aurora` simulation or simply from fractional abundances in ionization equilibrium. 

Documentation for the code is available at https://aurora.readthedocs.io.

`aurora` has been thoroughly benchmarked with the standard STRAHL, based on which it was originally developed. Recently, the algorithm proposed by O.Linder et al, Nuclear Fusion 2020, has been implemented and is now the default numerical scheme, although legacy options remain available. 

<img src="https://user-images.githubusercontent.com/25516628/93692659-f12c4b00-fac3-11ea-817c-d971c6853b8b.jpg" width="500" align="right">

Coupling to [ColRadPy](https://github.com/johnson-c/ColRadPy) is also available to process ADAS ADF04 files and estimate emission from `aurora` charge state distributions at specific wavelengths. We recommend installation of ColRadPy from the source via Github cloning. 

A number of examples using `aurora` are provided using real Alcator C-Mod kinetic profiles and geometry. In order to interface with EFIT gEQDSK files, we make use of the [omfit-eqdsk](https://gafusion.github.io/OMFIT-source/classes.html) package, but users may easily substitute this if they prefer to adopt different magnetic reconstruction packages and/or postprocessing interfaces. 

# Development 

The code is developed and maintained by F. Sciortino (MIT-PSFC) in collaboration with T. Odstrcil (GA) and A. Cavallaro (MIT), with support from O. Linder (MPI-IPP) and C. Johnson (U. Auburn). R. Dux (MPI-IPP) provided invaluable contributions by making the STRAHL code available. 

New contributors are more than welcome! Please get in touch at sciortino-at-psfc.mit.edu or open a pull-request via Github. 

Generally, we would appreciate if you could work with us to merge your features back into the main `aurora` distribution if there is any chance that the changes that you made could be useful to others. 

# Installation

We recommend installing from source, by git-cloning [this repo](https://github.com/fsciortino/aurora) from Github. This will ensure that you can access the latest version of the tools. Make sure to use the `master` branch to use a stable version. *Make use of the Makefile in the package directory to build the Fortran or Julia code* using 
```
make clean; make
```
Note that the Julia version of the code is not built by default. If you have Julia installed on your system, you can do  
```
make julia
```
from the main package directory. See the documentation to read about interfacing Python3 and Julia. 

The latest release of the package is also available on the Anaconda Cloud:

[![Anaconda-Server Badge](https://anaconda.org/sciortino/aurora/badges/latest_release_date.svg)](https://anaconda.org/sciortino/aurora)

and from PyPi using 
```
pip install aurora
```

# License

The package is made open-source with the hope that this will speed up research on fusion energy and make further code development easier. `aurora` is released under a MIT License, which gives great flexibility to users and encourages collaboration with minimal constraints. However, we kindly ask that all users communicate to us their purposes, difficulties and successes with `aurora`, so that we may support users as much as possible and grow the code further. 

