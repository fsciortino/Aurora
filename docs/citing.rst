Citing Aurora
=============

Aurora is released under the MIT License, one of the most common, permissive, open-source software licenses. This licensing option aims at making the package as useful and widely-applicable as possibe, in an effort to support the development of fusion energy. In the spirit of an open-source, collabWe do appreciate users pushing our numbers by giving a star to the Aurora Github repo

  https://github.com/fsciortino/aurora

and by citing the following works:

[1] F Sciortino et al 2021, "Modeling of particle transport, neutrals and radiation in magnetically-confined plasmas with Aurora", Plasma Phys. Control. Fusion 63 112001, https://doi.org/10.1088/1361-6587/ac2890
  
    This paper introduces Aurora and describes the general development philosophy, structure of the forward model for impurity transport, use of atomic data, the theory behind superstaging, and a few example applications for fusion simulation and data analysis.
    
[2] R. Dux, 2004, Habilitation Thesis, MPI-IPP. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.830.8834&rep=rep1&type=pdf

    The work of R. Dux on STRAHL is at the basis of many of the methods adopted by Aurora. While Aurora's code does not depend on STRAHL, it owes to it for laying much of the groundwork. 

If you use Aurora, you will likely need atomic rates of some sort. Aurora offers a simple interface to Atomic Data and Analysis Structure (ADAS) files - if you make use of this, please make sure to cite appropriately. Information about publicly available ADAS data can be found at https://open.adas.ac.uk/ . To cite ADAS, it would be reasonable to use

[3] H.P. Summers et al, 2001, "The ADAS manual" version ** [see the version number at https://www.adas.ac.uk/manual.php]

[4] H.P. Summers et al, 2006 Plasma Phys. Control. Fusion 48 263. https://iopscience.iop.org/article/10.1088/0741-3335/48/2/007

If you know the origin of the ADAS data that you are using, it would also be good to cite the specific works that resulted in the ADAS-distributed results. Other references for specific methods/data used in Aurora are shown in some docstrings.
