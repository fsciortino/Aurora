"""Functions to provide TRIM-generated files for plasma-surface interaction modelling in Aurora.
"""
# MIT License
#
# Copyright (c) 2022 Antonello Zito
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os

# location of the "trim_data" directory relative to this script:
trim_data_dir = (
    os.path.dirname(os.path.realpath(__file__)) + os.sep + "trim_data" + os.sep
)


def get_trim_file_loc(filename, filetype):
    """Find location of requested surface data file for the indicated ion, which are
    contained in the folder Aurora/aurora/trim_data/*filetype*/.

    Parameters
    ----------
    filename : str
        Name of the TRIM file of interest, e.g. 'Ar_W.rn'.
    filetype : str
        TRIM file type. Options:
        "bulk_sputter", "depth", "imp_sputter", "refl"
    Returns
    -------
    file_loc : str
        Full path to the requested file. 
    """

    if filename == 'none':
        # user doesn't want to load this file
        return 
    
    else:
        return trim_data_dir+filetype+os.sep+filename



def trim_files_dict():
    """Selections for TRIM-generated files for Aurora runs and plasma-surface
    interaction calculations.

    Returns
    -------
    files : dict
    
        Dictionary with keys equal to the atomic symbols of many of the most common ions of
        interest in fusion research. For each ion, a sub-dictionary contains the the type of
        interaction, with subkeys equal atomic symbols of the most common wall materials.
        Not all files types are available for all ions.
        
        Reflection data (of the type rn,re) are referred to the species
        which is reflected from the surface, i.e. the projectile, with logic
        [{projectile}]["Reflection"][{target_material}].
        
        Bulk sputtering data (of the type y, ye) are referred to the species
        which is sputtered from the surface, i.e. the bulk target, with logic
        [{target_material}]["Bulk_sputtering"][{projectile}].
        
        Implantation depth data (of the type d) are referred to the species
        which is implanted in the surface, i.e. the implanted impurity, with logic
        [{implanted_impurity}]["Implantation_depth"][target_material].
        
        Impurity sputtering data (of the type y, ye) are referred to the species
        which is sputtered from the surface, i.e. the implanted impurity, with logic
        [{implanted_impurity}]["Impurity_sputtering"][target_material][projectile].
        
    """

    files = {}
    
    
    # Data regarding hydrogen as impurity species
    files["H"] = {}
    
    # Hydrogen reflection from different wall materials
    files["H"]["Reflection"] = {}
    
    files["H"]["Reflection"]["Be"] = {}
    files["H"]["Reflection"]["Be"]["rn"] = "H_Be.rn"
    files["H"]["Reflection"]["Be"]["rn_fit"] = "H_Be_fit.rn"
    files["H"]["Reflection"]["Be"]["re"] = "H_Be.re"
    files["H"]["Reflection"]["Be"]["re_fit"] = "H_Be_fit.re"
    files["H"]["Reflection"]["C"] = {}
    files["H"]["Reflection"]["C"]["rn"] = "H_C.rn"
    files["H"]["Reflection"]["C"]["rn_fit"] = "H_C_fit.rn"
    files["H"]["Reflection"]["C"]["re"] = "H_C.re"
    files["H"]["Reflection"]["C"]["re_fit"] = "H_C_fit.re"   
    files["H"]["Reflection"]["W"] = {}
    files["H"]["Reflection"]["W"]["rn"] = "H_W.rn"
    files["H"]["Reflection"]["W"]["rn_fit"] = "H_W_fit.rn"
    files["H"]["Reflection"]["W"]["re"] = "H_W.re"
    files["H"]["Reflection"]["W"]["re_fit"] = "H_W_fit.re"  
     
    
    # Data regarding deuterium as impurity species
    files["D"] = {}
    
    # Deuterium reflection from different wall materials
    files["D"]["Reflection"] = {}
    
    files["D"]["Reflection"]["Be"] = {}
    files["D"]["Reflection"]["Be"]["rn"] = "D_Be.rn"
    files["D"]["Reflection"]["Be"]["rn_fit"] = "D_Be_fit.rn"
    files["D"]["Reflection"]["Be"]["re"] = "D_Be.re"
    files["D"]["Reflection"]["Be"]["re_fit"] = "D_Be_fit.re"
    files["D"]["Reflection"]["C"] = {}
    files["D"]["Reflection"]["C"]["rn"] = "D_C.rn"
    files["D"]["Reflection"]["C"]["rn_fit"] = "D_C_fit.rn"
    files["D"]["Reflection"]["C"]["re"] = "D_C.re"
    files["D"]["Reflection"]["C"]["re_fit"] = "D_C_fit.re"   
    files["D"]["Reflection"]["W"] = {}
    files["D"]["Reflection"]["W"]["rn"] = "D_W.rn"
    files["D"]["Reflection"]["W"]["rn_fit"] = "D_W_fit.rn"
    files["D"]["Reflection"]["W"]["re"] = "D_W.re"
    files["D"]["Reflection"]["W"]["re_fit"] = "D_W_fit.re" 
         
    
    # Data regarding tritium as impurity species
    files["T"] = {}
    
    # Tritium reflection from different wall materials
    files["T"]["Reflection"] = {}
    
    files["T"]["Reflection"]["Be"] = {}
    files["T"]["Reflection"]["Be"]["rn"] = "T_Be.rn"
    files["T"]["Reflection"]["Be"]["rn_fit"] = "T_Be_fit.rn"
    files["T"]["Reflection"]["Be"]["re"] = "T_Be.re"
    files["T"]["Reflection"]["Be"]["re_fit"] = "T_Be_fit.re"
    files["T"]["Reflection"]["C"] = {}
    files["T"]["Reflection"]["C"]["rn"] = "T_C.rn"
    files["T"]["Reflection"]["C"]["rn_fit"] = "T_C_fit.rn"
    files["T"]["Reflection"]["C"]["re"] = "T_C.re"
    files["T"]["Reflection"]["C"]["re_fit"] = "T_C_fit.re"   
    files["T"]["Reflection"]["W"] = {}
    files["T"]["Reflection"]["W"]["rn"] = "T_W.rn"
    files["T"]["Reflection"]["W"]["rn_fit"] = "T_W_fit.rn"
    files["T"]["Reflection"]["W"]["re"] = "T_W.re"
    files["T"]["Reflection"]["W"]["re_fit"] = "T_W_fit.re"  
     
    
    # Data regarding helium as impurity species
    files["He"] = {}
    
    # Helium reflection from different wall materials
    files["He"]["Reflection"] = {}
    
    files["He"]["Reflection"]["Be"] = {}
    files["He"]["Reflection"]["Be"]["rn"] = "He_Be.rn"
    files["He"]["Reflection"]["Be"]["rn_fit"] = "He_Be_fit.rn"
    files["He"]["Reflection"]["Be"]["re"] = "He_Be.re"
    files["He"]["Reflection"]["Be"]["re_fit"] = "He_Be_fit.re"
    files["He"]["Reflection"]["C"] = {}
    files["He"]["Reflection"]["C"]["rn"] = "He_C.rn"
    files["He"]["Reflection"]["C"]["rn_fit"] = "He_C_fit.rn"
    files["He"]["Reflection"]["C"]["re"] = "He_C.re"
    files["He"]["Reflection"]["C"]["re_fit"] = "He_C_fit.re"   
    files["He"]["Reflection"]["W"] = {}
    files["He"]["Reflection"]["W"]["rn"] = "He_W.rn"
    files["He"]["Reflection"]["W"]["rn_fit"] = "He_W_fit.rn"
    files["He"]["Reflection"]["W"]["re"] = "He_W.re"
    files["He"]["Reflection"]["W"]["re_fit"] = "He_W_fit.re"
    
    # Helium implantation depth in different wall materials
    files["He"]["Implantation_depth"] = {}
    
    files["He"]["Implantation_depth"]["Be"] = {}
    files["He"]["Implantation_depth"]["Be"]["d"] = "He_Be.d"
    files["He"]["Implantation_depth"]["C"] = {}
    files["He"]["Implantation_depth"]["C"]["d"] = "He_C.d"   
    files["He"]["Implantation_depth"]["W"] = {}
    files["He"]["Implantation_depth"]["W"]["d"] = "He_W.d"
    
    # Helium sputtering, implanted in different wall materials, from different plasma species
    files["He"]["Impurity_sputtering"] = {} 
    
    files["He"]["Impurity_sputtering"]["W"] = {}
    files["He"]["Impurity_sputtering"]["W"]["D"] = {}
    files["He"]["Impurity_sputtering"]["W"]["D"]["y"] = "D_He_W.y"
    files["He"]["Impurity_sputtering"]["W"]["D"]["y_fit"] = "D_He_W_fit.y"
    files["He"]["Impurity_sputtering"]["W"]["D"]["ye"] = "D_He_W.ye"
    #files["He"]["Impurity_sputtering"]["W"]["D"]["ye_fit"] = "D_He_W_fit.ye"
    files["He"]["Impurity_sputtering"]["W"]["He"] = {}
    files["He"]["Impurity_sputtering"]["W"]["He"]["y"] = "He_He_W.y"
    files["He"]["Impurity_sputtering"]["W"]["He"]["y_fit"] = "He_He_W_fit.y" 
    files["He"]["Impurity_sputtering"]["W"]["He"]["ye"] = "He_He_W.ye"
    #files["He"]["Impurity_sputtering"]["W"]["He"]["ye_fit"] = "He_He_W_fit.ye"
    files["He"]["Impurity_sputtering"]["W"]["B"] = {}
    files["He"]["Impurity_sputtering"]["W"]["B"]["y"] = "B_He_W.y"
    files["He"]["Impurity_sputtering"]["W"]["B"]["y_fit"] = "B_He_W_fit.y"   
    files["He"]["Impurity_sputtering"]["W"]["B"]["ye"] = "B_He_W.ye"
    #files["He"]["Impurity_sputtering"]["W"]["B"]["ye_fit"] = "B_He_W_fit.ye"
    files["He"]["Impurity_sputtering"]["W"]["N"] = {}
    files["He"]["Impurity_sputtering"]["W"]["N"]["y"] = "N_He_W.y"
    files["He"]["Impurity_sputtering"]["W"]["N"]["y_fit"] = "N_He_W_fit.y"  
    files["He"]["Impurity_sputtering"]["W"]["N"]["ye"] = "N_He_W.ye"
    #files["He"]["Impurity_sputtering"]["W"]["N"]["ye_fit"] = "N_He_W_fit.ye"

    # Data regarding beryllium as impurity species
    files["Be"] = {}
    
    # Berillium reflection (only from beryllium itself)
    files["Be"]["Reflection"] = {}
    
    files["Be"]["Reflection"]["Be"] = {}
    files["Be"]["Reflection"]["Be"]["rn"] = "Be_Be.rn"
    #files["Be"]["Reflection"]["Be"]["rn_fit"] = "Be_Be_fit.rn"
    files["Be"]["Reflection"]["Be"]["re"] = "Be_Be.re"
    #files["Be"]["Reflection"]["Be"]["re_fit"] = "Be_Be_fit.re"

    # Bulk beryllium sputtering from different plasma species, including itself
    files["Be"]["Bulk_sputtering"] = {}
    
    files["Be"]["Bulk_sputtering"]["H"] = {}
    files["Be"]["Bulk_sputtering"]["H"]["y"] = "H_Be.y"
    files["Be"]["Bulk_sputtering"]["H"]["y_fit"] = "H_Be_fit.y"
    files["Be"]["Bulk_sputtering"]["H"]["ye"] = "H_Be.ye"
    #files["Be"]["Bulk_sputtering"]["H"]["ye_fit"] = "H_Be_fit.ye" 
    files["Be"]["Bulk_sputtering"]["D"] = {}
    files["Be"]["Bulk_sputtering"]["D"]["y"] = "D_Be.y"
    files["Be"]["Bulk_sputtering"]["D"]["y_fit"] = "D_Be_fit.y"
    files["Be"]["Bulk_sputtering"]["D"]["ye"] = "D_Be.ye"
    #files["Be"]["Bulk_sputtering"]["D"]["ye_fit"] = "D_Be_fit.ye"  
    files["Be"]["Bulk_sputtering"]["T"] = {}
    files["Be"]["Bulk_sputtering"]["T"]["y"] = "T_Be.y"
    files["Be"]["Bulk_sputtering"]["T"]["y_fit"] = "T_Be_fit.y"
    files["Be"]["Bulk_sputtering"]["T"]["ye"] = "T_Be.ye"
    #files["Be"]["Bulk_sputtering"]["T"]["ye_fit"] = "T_Be_fit.ye" 
    files["Be"]["Bulk_sputtering"]["He"] = {}
    files["Be"]["Bulk_sputtering"]["He"]["y"] = "He_Be.y"
    files["Be"]["Bulk_sputtering"]["He"]["y_fit"] = "He_Be_fit.y"
    files["Be"]["Bulk_sputtering"]["He"]["ye"] = "He_Be.ye"
    #files["Be"]["Bulk_sputtering"]["He"]["ye_fit"] = "He_Be_fit.ye"
    files["Be"]["Bulk_sputtering"]["Be"] = {}
    files["Be"]["Bulk_sputtering"]["Be"]["y"] = "Be_Be.y"
    files["Be"]["Bulk_sputtering"]["Be"]["y_fit"] = "Be_Be_fit.y"
    files["Be"]["Bulk_sputtering"]["Be"]["ye"] = "Be_Be.ye"
    #files["Be"]["Bulk_sputtering"]["Be"]["ye_fit"] = "Be_Be_fit.ye" 
    # files["Be"]["Bulk_sputtering"]["N"] = {}
    # files["Be"]["Bulk_sputtering"]["N"]["y"] = "N_Be.y"
    # files["Be"]["Bulk_sputtering"]["N"]["y_fit"] = "N_Be_fit.y"
    # files["Be"]["Bulk_sputtering"]["N"]["ye"] = "N_Be.ye"
    # files["Be"]["Bulk_sputtering"]["N"]["ye_fit"] = "N_Be_fit.ye" 
    files["Be"]["Bulk_sputtering"]["Ne"] = {}
    files["Be"]["Bulk_sputtering"]["Ne"]["y"] = "Ne_Be.y"
    files["Be"]["Bulk_sputtering"]["Ne"]["y_fit"] = "Ne_Be_fit.y"
    files["Be"]["Bulk_sputtering"]["Ne"]["ye"] = "Ne_Be.ye"
    #files["Be"]["Bulk_sputtering"]["Ne"]["ye_fit"] = "Ne_Be_fit.ye" 
    files["Be"]["Bulk_sputtering"]["Ar"] = {}
    files["Be"]["Bulk_sputtering"]["Ar"]["y"] = "Ar_Be.y"
    files["Be"]["Bulk_sputtering"]["Ar"]["y_fit"] = "Ar_Be_fit.y"
    files["Be"]["Bulk_sputtering"]["Ar"]["ye"] = "Ar_Be.ye"
    #files["Be"]["Bulk_sputtering"]["Ar"]["ye_fit"] = "Ar_Be_fit.ye" 
    
    
    # Data regarding carbon as impurity species
    files["C"] = {}
    
    # Carbon reflection (only from carbon itself)
    files["C"]["Reflection"] = {}
    
    files["C"]["Reflection"]["C"] = {}
    files["C"]["Reflection"]["C"]["rn"] = "C_C.rn"
    #files["C"]["Reflection"]["C"]["rn_fit"] = "C_C_fit.rn"
    files["C"]["Reflection"]["C"]["re"] = "C_C.re"
    #files["C"]["Reflection"]["C"]["re_fit"] = "C_C_fit.re"

    # Bulk carbon sputtering from different plasma species, including itself
    files["C"]["Bulk_sputtering"] = {}
    
    files["C"]["Bulk_sputtering"]["H"] = {}
    files["C"]["Bulk_sputtering"]["H"]["y"] = "H_C.y"
    files["C"]["Bulk_sputtering"]["H"]["y_fit"] = "H_C_fit.y"
    files["C"]["Bulk_sputtering"]["H"]["ye"] = "H_C.ye"
    #files["C"]["Bulk_sputtering"]["H"]["ye_fit"] = "H_C_fit.ye" 
    files["C"]["Bulk_sputtering"]["D"] = {}
    files["C"]["Bulk_sputtering"]["D"]["y"] = "D_C.y"
    files["C"]["Bulk_sputtering"]["D"]["y_fit"] = "D_C_fit.y"
    files["C"]["Bulk_sputtering"]["D"]["ye"] = "D_C.ye"
    #files["C"]["Bulk_sputtering"]["D"]["ye_fit"] = "D_C_fit.ye"  
    files["C"]["Bulk_sputtering"]["T"] = {}
    files["C"]["Bulk_sputtering"]["T"]["y"] = "T_C.y"
    files["C"]["Bulk_sputtering"]["T"]["y_fit"] = "T_C_fit.y"
    files["C"]["Bulk_sputtering"]["T"]["ye"] = "T_C.ye"
    #files["C"]["Bulk_sputtering"]["T"]["ye_fit"] = "T_C_fit.ye" 
    files["C"]["Bulk_sputtering"]["He"] = {}
    files["C"]["Bulk_sputtering"]["He"]["y"] = "He_C.y"
    files["C"]["Bulk_sputtering"]["He"]["y_fit"] = "He_C_fit.y"
    files["C"]["Bulk_sputtering"]["He"]["ye"] = "He_C.ye"
    #files["C"]["Bulk_sputtering"]["He"]["ye_fit"] = "He_C_fit.ye"
    files["C"]["Bulk_sputtering"]["C"] = {}
    files["C"]["Bulk_sputtering"]["C"]["y"] = "C_C.y"
    files["C"]["Bulk_sputtering"]["C"]["y_fit"] = "C_C_fit.y"
    files["C"]["Bulk_sputtering"]["C"]["ye"] = "C_C.ye"
    #files["C"]["Bulk_sputtering"]["C"]["ye_fit"] = "C_C_fit.ye" 
    files["C"]["Bulk_sputtering"]["N"] = {}
    files["C"]["Bulk_sputtering"]["N"]["y"] = "N_C.y"
    files["C"]["Bulk_sputtering"]["N"]["y_fit"] = "N_C_fit.y"
    files["C"]["Bulk_sputtering"]["N"]["ye"] = "N_C.ye"
    #files["C"]["Bulk_sputtering"]["N"]["ye_fit"] = "N_C_fit.ye" 
    files["C"]["Bulk_sputtering"]["Ne"] = {}
    files["C"]["Bulk_sputtering"]["Ne"]["y"] = "Ne_C.y"
    files["C"]["Bulk_sputtering"]["Ne"]["y_fit"] = "Ne_C_fit.y"
    files["C"]["Bulk_sputtering"]["Ne"]["ye"] = "Ne_C.ye"
    #files["C"]["Bulk_sputtering"]["Ne"]["ye_fit"] = "Ne_C_fit.ye" 
    files["C"]["Bulk_sputtering"]["Ar"] = {}
    files["C"]["Bulk_sputtering"]["Ar"]["y"] = "Ar_C.y"
    files["C"]["Bulk_sputtering"]["Ar"]["y_fit"] = "Ar_C_fit.y"
    files["C"]["Bulk_sputtering"]["Ar"]["ye"] = "Ar_C.ye"
    #files["C"]["Bulk_sputtering"]["Ar"]["ye_fit"] = "Ar_C_fit.ye"   
    
    
    # Data regarding nitrogen as impurity species
    files["N"] = {}
    
    # Nitrogen reflection from different wall materials
    files["N"]["Reflection"] = {}
    
    # files["N"]["Reflection"]["Be"] = {}
    # files["N"]["Reflection"]["Be"]["rn"] = "N_Be.rn"
    # files["N"]["Reflection"]["Be"]["rn_fit"] = "N_Be_fit.rn"
    # files["N"]["Reflection"]["Be"]["re"] = "N_Be.re"
    # files["N"]["Reflection"]["Be"]["re_fit"] = "N_Be_fit.re"
    files["N"]["Reflection"]["C"] = {}
    files["N"]["Reflection"]["C"]["rn"] = "N_C.rn"
    files["N"]["Reflection"]["C"]["rn_fit"] = "N_C_fit.rn"
    files["N"]["Reflection"]["C"]["re"] = "N_C.re"
    files["N"]["Reflection"]["C"]["re_fit"] = "N_C_fit.re"   
    files["N"]["Reflection"]["W"] = {}
    files["N"]["Reflection"]["W"]["rn"] = "N_W.rn"
    files["N"]["Reflection"]["W"]["rn_fit"] = "N_W_fit.rn"
    files["N"]["Reflection"]["W"]["re"] = "N_W.re"
    files["N"]["Reflection"]["W"]["re_fit"] = "N_W_fit.re"
    
    
    # Data regarding neon as impurity species
    files["Ne"] = {}
    
    # Neon reflection from different wall materials
    files["Ne"]["Reflection"] = {}
    
    files["Ne"]["Reflection"]["Be"] = {}
    files["Ne"]["Reflection"]["Be"]["rn"] = "Ne_Be.rn"
    files["Ne"]["Reflection"]["Be"]["rn_fit"] = "Ne_Be_fit.rn"
    files["Ne"]["Reflection"]["Be"]["re"] = "Ne_Be.re"
    files["Ne"]["Reflection"]["Be"]["re_fit"] = "Ne_Be_fit.re"
    files["Ne"]["Reflection"]["C"] = {}
    files["Ne"]["Reflection"]["C"]["rn"] = "Ne_C.rn"
    files["Ne"]["Reflection"]["C"]["rn_fit"] = "Ne_C_fit.rn"
    files["Ne"]["Reflection"]["C"]["re"] = "Ne_C.re"
    files["Ne"]["Reflection"]["C"]["re_fit"] = "Ne_C_fit.re"   
    files["Ne"]["Reflection"]["W"] = {}
    files["Ne"]["Reflection"]["W"]["rn"] = "Ne_W.rn"
    files["Ne"]["Reflection"]["W"]["rn_fit"] = "Ne_W_fit.rn"
    files["Ne"]["Reflection"]["W"]["re"] = "Ne_W.re"
    files["Ne"]["Reflection"]["W"]["re_fit"] = "Ne_W_fit.re"  
    
    
    # Data regarding argon as impurity species
    files["Ar"] = {}
    
    # Argon reflection from different wall materials
    files["Ar"]["Reflection"] = {}
    
    files["Ar"]["Reflection"]["Be"] = {}
    files["Ar"]["Reflection"]["Be"]["rn"] = "Ar_Be.rn"
    files["Ar"]["Reflection"]["Be"]["rn_fit"] = "Ar_Be_fit.rn"
    files["Ar"]["Reflection"]["Be"]["re"] = "Ar_Be.re"
    files["Ar"]["Reflection"]["Be"]["re_fit"] = "Ar_Be_fit.re"
    files["Ar"]["Reflection"]["C"] = {}
    files["Ar"]["Reflection"]["C"]["rn"] = "Ar_C.rn"
    files["Ar"]["Reflection"]["C"]["rn_fit"] = "Ar_C_fit.rn"
    files["Ar"]["Reflection"]["C"]["re"] = "Ar_C.re"
    files["Ar"]["Reflection"]["C"]["re_fit"] = "Ar_C_fit.re"   
    files["Ar"]["Reflection"]["W"] = {}
    files["Ar"]["Reflection"]["W"]["rn"] = "Ar_W.rn"
    files["Ar"]["Reflection"]["W"]["rn_fit"] = "Ar_W_fit.rn"
    files["Ar"]["Reflection"]["W"]["re"] = "Ar_W.re"
    files["Ar"]["Reflection"]["W"]["re_fit"] = "Ar_W_fit.re"   
    
    
    # Data regarding tungsten as impurity species
    files["W"] = {}
    
    # Tungsten reflection (only from tungsten itself)
    files["W"]["Reflection"] = {}
    
    files["W"]["Reflection"]["W"] = {}
    files["W"]["Reflection"]["W"]["rn"] = "W_W.rn"
    #files["W"]["Reflection"]["W"]["rn_fit"] = "W_W_fit.rn"
    files["W"]["Reflection"]["W"]["re"] = "W_W.re"
    #files["W"]["Reflection"]["W"]["re_fit"] = "W_W_fit.re"

    # Bulk tungsten sputtering from different plasma species, including itself
    files["W"]["Bulk_sputtering"] = {}
    
    files["W"]["Bulk_sputtering"]["H"] = {}
    files["W"]["Bulk_sputtering"]["H"]["y"] = "H_W.y"
    files["W"]["Bulk_sputtering"]["H"]["y_fit"] = "H_W_fit.y"
    files["W"]["Bulk_sputtering"]["H"]["ye"] = "H_W.ye"
    #files["W"]["Bulk_sputtering"]["H"]["ye_fit"] = "H_W_fit.ye" 
    files["W"]["Bulk_sputtering"]["D"] = {}
    files["W"]["Bulk_sputtering"]["D"]["y"] = "D_W.y"
    files["W"]["Bulk_sputtering"]["D"]["y_fit"] = "D_W_fit.y"
    files["W"]["Bulk_sputtering"]["D"]["ye"] = "D_W.ye"
    #files["W"]["Bulk_sputtering"]["D"]["ye_fit"] = "D_W_fit.ye"  
    files["W"]["Bulk_sputtering"]["T"] = {}
    files["W"]["Bulk_sputtering"]["T"]["y"] = "T_W.y"
    files["W"]["Bulk_sputtering"]["T"]["y_fit"] = "T_W_fit.y"
    files["W"]["Bulk_sputtering"]["T"]["ye"] = "T_W.ye"
    #files["W"]["Bulk_sputtering"]["T"]["ye_fit"] = "T_W_fit.ye" 
    files["W"]["Bulk_sputtering"]["He"] = {}
    files["W"]["Bulk_sputtering"]["He"]["y"] = "He_W.y"
    files["W"]["Bulk_sputtering"]["He"]["y_fit"] = "He_W_fit.y"
    files["W"]["Bulk_sputtering"]["He"]["ye"] = "He_W.ye"
    #files["W"]["Bulk_sputtering"]["He"]["ye_fit"] = "He_W_fit.ye"
    files["W"]["Bulk_sputtering"]["N"] = {}
    files["W"]["Bulk_sputtering"]["N"]["y"] = "N_W.y"
    files["W"]["Bulk_sputtering"]["N"]["y_fit"] = "N_W_fit.y"
    files["W"]["Bulk_sputtering"]["N"]["ye"] = "N_W.ye"
    #files["W"]["Bulk_sputtering"]["N"]["ye_fit"] = "N_W_fit.ye" 
    files["W"]["Bulk_sputtering"]["Ne"] = {}
    files["W"]["Bulk_sputtering"]["Ne"]["y"] = "Ne_W.y"
    files["W"]["Bulk_sputtering"]["Ne"]["y_fit"] = "Ne_W_fit.y"
    files["W"]["Bulk_sputtering"]["Ne"]["ye"] = "Ne_W.ye"
    #files["W"]["Bulk_sputtering"]["Ne"]["ye_fit"] = "Ne_W_fit.ye" 
    files["W"]["Bulk_sputtering"]["Ar"] = {}
    files["W"]["Bulk_sputtering"]["Ar"]["y"] = "Ar_W.y"
    files["W"]["Bulk_sputtering"]["Ar"]["y_fit"] = "Ar_W_fit.y"
    files["W"]["Bulk_sputtering"]["Ar"]["ye"] = "Ar_W.ye"
    #files["W"]["Bulk_sputtering"]["Ar"]["ye_fit"] = "Ar_W_fit.ye"  
    files["W"]["Bulk_sputtering"]["W"] = {}
    files["W"]["Bulk_sputtering"]["W"]["y"] = "W_W.y"
    files["W"]["Bulk_sputtering"]["W"]["y_fit"] = "W_W_fit.y"
    files["W"]["Bulk_sputtering"]["W"]["ye"] = "W_W.ye"
    #files["W"]["Bulk_sputtering"]["W"]["ye_fit"] = "W_W_fit.ye" 


    return files