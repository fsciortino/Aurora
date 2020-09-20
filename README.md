# `aurora`
F. Sciortino, 2019-

Contributors: T. Odstrcil, A. Cavallaro

<img src="https://user-images.githubusercontent.com/25516628/93692659-f12c4b00-fac3-11ea-817c-d971c6853b8b.jpg" width="500" align="right">

`aurora` is a modern version of the historical STRAHL code by R.Dux (MPI-IPP). 
The core of `aurora` is written in Python3 and Fortran 90, but Julia interfaces are also under development. The package includes Python functionality to create inputs and read/plot outputs. OMFIT users may access this functionality via the OMFIT STRAHL module.

The code has been thoroughly benchmarked with the standard STRAHL. Over time, new options for impurity transport forward modeling have been included in `aurora`.

A number of standard tests are provided using real Alcator C-Mod kinetic profiles and geometry. In order to interface with EFIT gEQDSK files, we make use of the omfit-eqdsk package, but users may easily substitute this if they prefer to adopt different magnetic reconstruction packages and/or postprocessing interfaces. 


The package is now available on the Anaconda Cloud:

[![Anaconda-Server Badge](https://anaconda.org/sciortino/aurora/badges/latest_release_date.svg)](https://anaconda.org/sciortino/aurora)

Note that the Anaconda version only includes major releases. Use the Github version for the latest. 

