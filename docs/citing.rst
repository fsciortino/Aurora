Citing Aurora
=============

While Aurora is released publicly in an effort to support the development of fusion energy, we do appreciate users pushing our numbers by giving a star to the Aurora Github repo

  https://github.com/fsciortino/aurora

and by citing the following works:

[1] F. Sciortino et al 2020 Nucl. Fusion 60 126014, https://doi.org/10.1088/1741-4326/abae85

    This paper presented the original application of Aurora to infer impurity transport coefficients in Alcator C-Mod plasmas. Here, the code is referred to as `pySTRAHL` (what a terrible name).

[2] R. Dux, 2004, Habilitation Thesis, MPI-IPP. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.830.8834&rep=rep1&type=pdf

    The work of R. Dux on STRAHL is at the basis of many of the methods adopted by Aurora. While Aurora's code does not depend on STRAHL in any way, it owes to it for laying much of the groundwork. 

If you use Aurora, you will likely need atomic rates of some sort. Aurora offers a simple interface to Atomic Data and Analysis Structure (ADAS) files - if you make use of this, please make sure to cite appropriately. Information about publicly available ADAS data can be found at https://open.adas.ac.uk/ . To cite ADAS, it would be reasonable to use

[3] H.P. Summers et al, 2001, "The ADAS manual" version ** [see the version number at https://www.adas.ac.uk/manual.php]

[4] H.P. Summers et al, 2006 Plasma Phys. Control. Fusion 48 263. https://iopscience.iop.org/article/10.1088/0741-3335/48/2/007

If you know the origin of the ADAS data that you are using, it would also be good to cite the specific works that resulted in the ADAS-distributed results. Other references for specific methods/data used in Aurora are shown in some docstrings.
