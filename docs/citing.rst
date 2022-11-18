Citing Aurora
=============

Aurora is released under the MIT License, one of the most common, permissive, open-source software licenses. This licensing option aims at making the package as useful and widely-applicable as possibe, in an effort to support the development of fusion energy. In the spirit of an open-source, collabWe do appreciate users pushing our numbers by giving a star to the Aurora Github repo

  https://github.com/fsciortino/aurora

and by citing the following works:

[1] F. Sciortino et al, 2021, "Modeling of particle transport, neutrals and radiation in magnetically-confined plasmas with Aurora", Plasma Phys. Control. Fusion 63 112001, https://doi.org/10.1088/1361-6587/ac2890
  
    This paper introduces Aurora and describes the general development philosophy, structure of the forward model for impurity transport, use of atomic data, the theory behind superstaging, and a few example applications for fusion simulation and data analysis.
    
[2] R. Dux, 2004, "Impurity Transport in Tokamak Plasmas", Habilitation Thesis, MPI-IPP. https://pure.mpg.de/pubman/item/item_2136655_1/component/file_2136654/IPP-10-27.pdf

    This thesis contains a general introduction on the topic of impurity transport in fusion plasmas, and is the basis of the impurity transport model present in Aurora.
    
[3] A. Zito et al, 2023, "Modeling and interpretation of helium exhaust dynamics at the ASDEX Upgrade tokamak with full-tungsten wall", Nucl. Fusion, to be submitted

     This paper explains the introduction of the extended multi-reservoir recyling and pumping model in Aurora, as well as the advanced plasma-wall interaction model, and their application to experiments.
     
Atomic and surface data
-----------------------

If you use Aurora, you will likely need atomic rates of some sort. Aurora offers a simple interface to Atomic Data and Analysis Structure (ADAS) files - if you make use of this, please make sure to cite appropriately. Information about publicly available ADAS data can be found at https://open.adas.ac.uk/ . To cite ADAS, it would be reasonable to use

[4] H.P. Summers et al, 2001, "The ADAS manual" version ** [see the version number at https://www.adas.ac.uk/manual.php]

[5] H.P. Summers et al, 2006, "Ionization state, excited populations and emission of impurities in dynamic finite density plasmas", Plasma Phys. Control. Fusion 48 263. https://iopscience.iop.org/article/10.1088/0741-3335/48/2/007

If you know the origin of the ADAS data that you are using, it would also be good to cite the specific works that resulted in the ADAS-distributed results. Other references for specific methods/data used in Aurora are shown in some docstrings.

If you are also using the advanced plasma-wall interaction model, you will be interfacing with the data produced by the Monte Carlo program trim.sp. Information about trim.sp can be found at https://rmlmcfadden.github.io/ion-implantation/trimsp/ . To cite trim.sp, it would be reasonable to use

[6] W. Eckstein, 1991, "Computer Simulation of Ion-Solid Interactions". https://link.springer.com/book/10.1007/978-3-642-73513-4

[7] W. Eckstein, 1994, "Backscattering and sputtering with the Monte-Carlo program TRIM.SP". Radiat. Eff. Defects Solids 130-131 239-250, https://doi.org/10.1080/10420159408219787

while the origin of most of the actual data used by Aurora is

[8] W. Eckstein, 2002, IPP-Report 9/132, https://pure.mpg.de/rest/items/item_2138250/component/file_2138249/content