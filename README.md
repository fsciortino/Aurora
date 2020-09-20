# `aurora`
F.Sciortino, 2019-

Contributors: T.Odstrcil, A.Cavallaro

<img src="https://user-images.githubusercontent.com/25516628/93692659-f12c4b00-fac3-11ea-817c-d971c6853b8b.jpg" width="500" align="right">

`aurora` is a modern version of the historical STRAHL code written by R.Dux (MPI-IPP). 
`aurora` is written in f90 (rather than f77) and easily interfaces with Python3+. The package also includes Python functionality to create inputs and read outputs. OMFIT users may access this functionality via the OMFIT STRAHL module.

The code has been thoroughly benchmarked with the standard STRAHL. Over time, new options for impurity transport forward modeling have been included in `aurora`.

`aurora` inputs can either be created directly via Python or via a run of the standard STRAHL code. Working solely in Python, making use of the provided routines, is certainly easier. We provide a simple test.py script as an example using a real Alcator C-Mod geometry. In order to interface with EFIT gEQDSK files, we make use of the omfit-eqdsk package, but users may easily substitute this if they prefer to adopt different methods. In general, users may follow the provided Python code as an example and choose to only make use of the Fortran interface (in the `flib` sub-package) within their simulation framework. 

Note that STRAHL is not provided as part of this package and a request to obtain the source code must be addressed to its author. 

The package is now available on the Anaconda Cloud:

[![Anaconda-Server Badge](https://anaconda.org/sciortino/aurora/badges/latest_release_date.svg)](https://anaconda.org/sciortino/aurora)

Note that the Anaconda version only includes major releases. Use the Github version for the latest. 

