"""Collection of classes and functions for loading, interpolation and processing of surface data. 
Refer also to the trim_files.py script. 
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

import numpy as np
import matplotlib.pyplot as plt
import os, sys, copy
from . import trim_files
from . import atomic


def get_trim_file_types():
    """Obtain a description of each TRIM file type and its meaning in the context of Aurora.

    Returns
    ------------
    dict
        Dictionary with keys given by the TRIM file types and values giving a description for them.

    Notes
    ---------
    For background on the Monte Carlo plasma-surface interaction model in TRIM, refer to [1]_.

    References
    -----------------
    
    .. [1] Eckstein, "Computer Simulation of Ion-Solid Interactions", Springer-Verlag, 1991

    """

    return {
        "rn": "particle reflection coefficient",
        "re": "energy reflection coefficient",
        "y": "sputtering yield",
        "ye": "energy sputtering yield",
    }


def get_reflection_data(imp, wall, angle, kind):
    """ Collect reflection data for a given impurity.

    Parameters
    ----------
    imp : str
        Atomic symbol of impurity ion.
    wall : str
        Atomic symbol of wall material.
    angle : float
        Angle of incidence of the projectile ion.
        Output data will be referred to the available
        tabulated angle closest to the requested one.
    kind : str
        TRIM file types to be fetched.
        Options:
            "rn" for particle reflection coefficients
            "re" for energy reflection coefficients
    
    Returns
    -------
    refl_data : dict
        Dictionary containing the data for the requested type of coefficient. 
        It contains the type of coefficient, the projectile name, the wall material name,
        the requested incidence angle and the arrays with impact energies and data.
    """

    try:
        # default files dictionary
        all_files = trim_files.trim_files_dict()[
            imp
        ]["Reflection"][wall]  # all reflection files for a given species and wall material
    except KeyError:
        raise KeyError(f"Reflection data not available for ion {imp} on wall material {wall}!")
        
    if kind == "rn":
        coeff_name = "particle reflection coefficient"
    elif kind == "re":
        coeff_name = "energy reflection coefficient"

    refl_data = {}

    # absolute location of the requested files
    fileloc = trim_files.get_trim_file_loc(all_files[kind], filetype="refl")
    
    # full tables
    energies, angles, data = read_trim_table(fileloc)
    
    # index of incidence angle closest to the requested one
    idx = np.argmin(np.abs(np.asarray(angles) - angle))
    
    # coefficients corresponding to the desired angle
    coefficients = data[:,idx]
    
    refl_data = {
        "coefficient_type": coeff_name,
        "projectile": imp,
        "wall_material": wall,
        "angle": angle,
        "energies": energies,
        "data": coefficients,
        }
    
    return refl_data


def reflection_coeff_fit(imp, wall, angle, kind, energies = None):
    """ Read the fitted reflection data for a given impurity according to the Eckstein fit.

    Parameters
    ----------
    imp : str
        Atomic symbol of impurity ion.
    wall : str
        Atomic symbol of wall material.
    angle : float
        Angle of incidence of the projectile ion.
        Output data will be referred to the available
        tabulated angle closest to the requested one.    
    kind : str
        TRIM file types to be fetched.
        Options:
            "rn" for particle reflection coefficients
            "re" for energy reflection coefficients
    energies : list, optional
        Impact energies of the projectile ion in eV.
        If this input is specified, then the coefficients
        will be outputted at the requested energies. If left
        to None, then the entire energy range is used.
        
    Returns
    -------
    refl_data_fit: dict
        Dictionary containing the fitted data for the requested type of coefficient. 
        It contains the type of coefficient, the projectile name, the wall material name,
        the requested incidence angle and the arrays with impact energies and fitted data.
    """    
    
    try:
        # default files dictionary
        all_files = trim_files.trim_files_dict()[
            imp
        ]["Reflection"][wall]  # all reflection files for a given species and wall material
    except KeyError:
        raise KeyError(f"Reflection data not available for ion {imp} on wall material {wall}!")
        
    if kind == "rn":
        coeff_name = "particle reflection coefficient"
    elif kind == "re":
        coeff_name = "energy reflection coefficient"
    
    refl_data_fit = {}

    # absolute location of the requested files
    fileloc = trim_files.get_trim_file_loc(all_files[kind+"_fit"], filetype="refl")
    
    # read the full energy range limits
    temp = get_reflection_data(imp, wall, angle, kind)
    temp = temp["energies"]
    energy_range_limits = np.array([0,0])
    energy_range_limits[0] = temp[0]
    energy_range_limits[1] = temp[-1]
    
    # index of incidence angle closest to the requested one
    angles = np.array([0,15,30,45,55,65,75,80,85])
    idx = np.argmin(np.abs(np.asarray(angles) - angle))
    
    # read the fit coefficients
    file = open(fileloc, "r")
    lines = file.readlines()
    temp = lines[idx+6]
    if "NaN" in temp:
        raise ValueError(f"Fit not available for requested reflection coefficient at incidence angle {angle}!") 
    temp = temp.strip("\n")
    temp = temp.split("  ")
    a1 = float(temp[3])
    a2 = float(temp[4])
    a3 = float(temp[5])
    a4 = float(temp[6])
         
    # build the energy range in which to perform the fit
    if energies is None: # full energy range
        energy_range = np.linspace(energy_range_limits[0],energy_range_limits[1],10000)
    else: # discrete number of energies defined in the input
        energy_range = np.array(energies)
        
    # extract the fitted data for the requested incidence angle using the empirical Eckstein formula
    data_fit = np.zeros(len(energy_range))
    for i in range(0,len(energy_range)):
        data_fit[i] = eckstein_fit(energy_range[i],a1,a2,a3,a4,imp,wall)
    
    refl_data_fit = {
        "coefficient_type": coeff_name,
        "projectile": imp,
        "wall_material": wall,
        "angle": angle,
        "energies": energy_range,
        "data": data_fit,
        }
    
    return refl_data_fit
    

def get_bulk_sputtering_data(imp, wall, angle, kind):
    """ Collect bulk sputtering data for a given impurity.

    Parameters
    ----------
    imp : str
        Atomic symbol of impurity ion.
    wall : str
        Atomic symbol of wall material.
    angle : float
        Angle of incidence of the projectile ion.
        Output data will be referred to the available
        tabulated angle closest to the requested one.
    kind : str
        TRIM file types to be fetched.
        Options:
            "y" for sputtering yields
            "ye" for energy sputtering yields
    
    Returns
    -------
    bulk_sputter_data : dict
        Dictionary containing the data for the requested type of coefficient. 
        It contains the type of coefficient, the projectile name, the wall material name,
        the requested incidence angle and the arrays with impact energies and data.
    """

    try:
        # default files dictionary
        all_files = trim_files.trim_files_dict()[
            wall
        ]["Bulk_sputtering"][imp]  # all bulk sputtering files for a given species and wall material
    except KeyError:
        raise KeyError(f"Bulk sputtering data not available for ion {imp} on wall material {wall}!")
        
    if kind == "y":
        coeff_name = "sputtering yield"
    elif kind == "ye":
        coeff_name = "energy sputtering yield"

    bulk_sputter_data = {}

    # absolute location of the requested files
    fileloc = trim_files.get_trim_file_loc(all_files[kind], filetype="bulk_sputter")
    
    # full tables
    energies, angles, data = read_trim_table(fileloc)
    
    # index of incidence angle closest to the requested one
    idx = np.argmin(np.abs(np.asarray(angles) - angle))
    
    # coefficients corresponding to the desired angle
    coefficients = data[:,idx]
    
    bulk_sputter_data = {
        "coefficient_type": coeff_name,
        "projectile": imp,
        "wall_material": wall,
        "angle": angle,
        "energies": energies,
        "data": coefficients,
        }
    
    return bulk_sputter_data


def bulk_sputtering_coeff_fit(imp, wall, angle, kind, energies = None):
    """ Read the fitted bulk sputtering data for a given impurity according to the Bohdansky fit.

    Parameters
    ----------
    imp : str
        Atomic symbol of impurity ion.
    wall : str
        Atomic symbol of wall material.
    angle : float
        Angle of incidence of the projectile ion.
        Output data will be referred to the available
        tabulated angle closest to the requested one.
    kind : str
        TRIM file types to be fetched.
        Options:
            "y" for sputtering yields
            "ye" for energy sputtering yields
    energies : list, optional
        Impact energies of the projectile ion in eV.
        If this input is specified, then the coefficients
        will be outputted at the requested energies. If left
        to None, then the entire energy range is used.
        
    Returns
    -------
    bulk_sputter_data_fit: dict
        Dictionary containing the fitted data for the requested type of coefficient. 
        It contains the type of coefficient, the projectile name, the wall material name,
        the requested incidence angle and the arrays with impact energies and fitted data.
    """    
    
    try:
        # default files dictionary
        all_files = trim_files.trim_files_dict()[
            wall
        ]["Bulk_sputtering"][imp]  # all bulk sputtering files for a given species and wall material
    except KeyError:
        raise KeyError(f"Bulk sputtering data not available for ion {imp} on wall material {wall}!")
        
    if kind == "y":
        coeff_name = "sputtering yield"
    elif kind == "ye":
        raise KeyError("Fitted curves for energy sputtering yields not availabile yet!")
    
    bulk_sputter_data_fit = {}

    # absolute location of the requested files
    fileloc = trim_files.get_trim_file_loc(all_files[kind+"_fit"], filetype="bulk_sputter")
    
    # read the full energy range limits
    temp = get_bulk_sputtering_data(imp, wall, angle, kind)
    temp = temp["energies"]
    energy_range_limits = np.array([0,0])
    energy_range_limits[0] = temp[0]
    energy_range_limits[1] = temp[-1]
    
    # index of incidence angle closest to the requested one
    angles = np.array([0,15,30,45,55,65,75,80,85])
    idx = np.argmin(np.abs(np.asarray(angles) - angle))
    
    # read the fit coefficients
    file = open(fileloc, "r")
    lines = file.readlines()
    temp = lines[idx+6]
    if "NaN" in temp:
        raise ValueError(f"Fit not available for requested sputtering yield at incidence angle {angle}!") 
    temp = temp.strip("\n")
    temp = temp.split("  ")
    Q = float(temp[3])
    Eth = float(temp[4])
         
    # build the energy range in which to perform the fit
    if energies is None: # full energy range
        energy_range = np.linspace(energy_range_limits[0],energy_range_limits[1],10000)
    else: # discrete number of energies defined in the input
        energy_range = np.array(energies)
        
    # extract the fitted data for the requested incidence angle using the Bohdansky formula
    data_fit = np.zeros(len(energy_range))
    for i in range(0,len(energy_range)):
        data_fit[i] = bohdansky_fit(energy_range[i],Q,Eth,imp,wall)
    
    bulk_sputter_data_fit = {
        "coefficient_type": coeff_name,
        "projectile": imp,
        "wall_material": wall,
        "angle": angle,
        "energies": energy_range,
        "data": data_fit,
        }
    
    return bulk_sputter_data_fit


def get_impurity_sputtering_data(imp, projectile, wall, angle):
    """ Collect impurity sputtering yields for a given impurity.

    Parameters
    ----------
    imp : str
        Atomic symbol of impurity ion.
    projectile : str
        Atomic symbol of projectile ion.   
    wall : str
        Atomic symbol of wall material.
    angle : float
        Angle of incidence of the projectile ion.
        Output data will be referred to the available
        tabulated angle closest to the requested one.
    
    Returns
    -------
    imp_sputter_data : dict
        Dictionary containing the requested sputtering yield data.
        It contains the type of coefficient, the implanted impurity name, the
        projectile name, the wall material name, the requested incidence angle
        and the arrays with energies, impurity concentrations and data.
    """

    try:
        # default files dictionary
        all_files = trim_files.trim_files_dict()[
            imp
        ]["Impurity_sputtering"][wall][projectile]  # all impurity sputtering files for given inpurity, projectile and wall material
    except KeyError:
        raise KeyError(f"Impurity sputtering data not available for ion {imp} on wall material {wall} with projectile {projectile}!")

    imp_sputter_data = {}

    # absolute location of the requested files
    fileloc = trim_files.get_trim_file_loc(all_files["y"], filetype="imp_sputter")
    
    file = open(fileloc, "r")
    lines = file.readlines()
    
    temp = lines[6]
    temp = temp.strip("\n")
    temp = temp.split(" ")
    angle_values = np.zeros(len(temp)-2)
    for i in range(0,len(temp)-2):
        angle_values[i] = float(temp[i+2])
    temp = lines[7]
    temp = temp.strip("\n")
    temp = temp.split(" ")
    energy_values = np.zeros(len(temp)-2)
    for i in range(0,len(temp)-2):
        energy_values[i] = float(temp[i+2])
    temp = lines[8]
    temp = temp.strip("\n")
    temp = temp.split(" ")
    C_imp_values = np.zeros(len(temp)-2)
    for i in range(0,len(temp)-2):
        C_imp_values[i] = float(temp[i+2])
    
    data = np.zeros((len(angle_values),len(energy_values),len(C_imp_values)))
    
    for i in range(0,len(lines)-11):
        temp = lines[i+11]
        temp = temp.split("	")
        ang = float(temp[1])
        angle_index = np.argmin(np.abs(np.asarray(angle_values)-ang))
        en = float(temp[2])
        energy_index = np.argmin(np.abs(np.asarray(energy_values)-en))
        C_imp = float(temp[3])   
        C_imp_index = np.argmin(np.abs(np.asarray(C_imp_values)-C_imp))
        Y_imp = float(temp[5])
        data[angle_index,energy_index,C_imp_index] = Y_imp 
    
    # index of incidence angle closest to the requested one
    idx = np.argmin(np.abs(np.asarray(angle_values) - angle))

    # coefficients corresponding to the desired angle
    coefficients = data[idx,:,:]

    imp_sputter_data = {
        "coefficient_type": "sputtering yield",
        "impurity": imp,
        "projectile": projectile,
        "wall_material": wall,
        "angle": angle,
        "energies": energy_values,
        "impurity_concentrations": C_imp_values,
        "data": coefficients,
        }

    return imp_sputter_data    


def impurity_sputtering_coeff_fit(imp, projectile, wall, angle, energies = None):
    """ Read the fitted impurity sputtering yield for a given impurity (normalized to the
    impurity concentration in the surface) according to the Bohdansky fit.

    Parameters
    ----------
    imp : str
        Atomic symbol of impurity ion implanted in the wall surface.
    projectile : str
        Atomic symbol of projectile ion.
    wall : str
        Atomic symbol of wall material.
    angle : float
        Angle of incidence of the projectile ion.
        Output data will be referred to the available
        tabulated angle closest to the requested one.
    energies : list, optional
        Impact energies of the projectile ion in eV.
        If this input is specified, then the coefficients
        will be outputted at the requested energies. If left
        to None, then the entire energy range is used.
        
    Returns
    -------
    imp_sputter_data_fit: dict
        Dictionary containing the fitted data for the requested type of coefficient. 
        It contains the type of coefficient, the projectile name, the wall material name,
        the requested incidence angle and the arrays with impurity concentrations,
        impact energies and fitted data (the last ones normalized to the impurity concentration).
    """    

    try:
        # default files dictionary
        all_files = trim_files.trim_files_dict()[
            imp
        ]["Impurity_sputtering"][wall][projectile]  # all impurity sputtering files for given inpurity, projectile and wall material
    except KeyError:
        raise KeyError(f"Impurity sputtering data not available for ion {imp} on wall material {wall} with projectile {projectile}!")
    
    imp_sputter_data_fit = {}

    # absolute location of the requested files
    fileloc = trim_files.get_trim_file_loc(all_files["y_fit"], filetype="imp_sputter")
    
    # read the full energy range limits
    temp = get_impurity_sputtering_data(imp, projectile, wall, angle)
    temp = temp["energies"]
    energy_range_limits = np.array([0,0])
    energy_range_limits[0] = temp[0]
    energy_range_limits[1] = temp[-1]
    
    # index of incidence angle closest to the requested one
    angles = np.array([0,20,40,55,65,70,75,80])
    idx = np.argmin(np.abs(np.asarray(angles) - angle))
    
    # read the fit coefficients
    file = open(fileloc, "r")
    lines = file.readlines()
    temp = lines[idx+8]
    if "NaN" in temp:
        raise ValueError(f"Fit not available for requested sputtering yield at incidence angle {angle}!") 
    temp = temp.strip("\n")
    temp = temp.split("	")
    Q = float(temp[1])
    Eth = float(temp[2])
         
    # build the energy range in which to perform the fit
    if energies is None: # full energy range
        energy_range = np.linspace(energy_range_limits[0],energy_range_limits[1],10000)
    else: # discrete number of energies defined in the input
        energy_range = np.array(energies)
        
    # extract the fitted data for the requested incidence angle using the Bohdansky formula
    data_fit = np.zeros(len(energy_range))
    for i in range(0,len(energy_range)):
        data_fit[i] = bohdansky_fit(energy_range[i],Q,Eth,projectile,imp)
    
    imp_sputter_data_fit = {
        "coefficient_type": "sputtering yield",
        "impurity": imp,
        "projectile": projectile,
        "wall_material": wall,
        "angle": angle,
        "energies": energy_range,
        "normalized_data": data_fit,
        }
    
    return imp_sputter_data_fit


def read_trim_table(filename):
    """ Read the TRIM-generated datafiles for reflection (rn,re) and bulk sputtering (y,ye)

    Parameters
    ----------
    filename : str
        Absolute location of the data file to be read.
    
    Returns
    -------
    energies : 1-dimensional array
        Incidence energy values at which the requested coefficient is desired
    angles : 1-dimensional array
        Incidence angle values at which the requested coefficient is desired
    data : 2-dimensional array
        Requested coefficient in function of (energies,angles)
    """
    #TODO: Correct the bug which does not allow to read correctly
    #      values > 1.0 (try e.g. N_C.y)
    
    file = open(filename, "r")
    lines = file.readlines()
    
    temp = lines[2].strip("c ")
    temp = temp.strip("\n")
    temp = temp.split(", ")
    ne_temp = temp[0].split("=")
    na_temp = temp[1].split("=")
    ne = int(ne_temp[1])
    na = int(na_temp[1])
    energies = np.zeros(ne)
    angles = np.zeros(na)
    data = np.zeros((ne,na))
    
    temp = lines[4]
    temp = temp.strip("\n")
    temp = temp.split("      ")
    for j in range(0,na):
        angles[j] = float(temp[j+1])
    
    for i in range(6,6+ne):
        temp = lines[i]
        temp = temp.strip("\n")
        temp = temp.split("  ")
        if not temp[0]:
            energies[i-6] = float(temp[1])
            for j in range(0,na):
                data[i-6,j] = float(temp[j+2])
        else:
            energies[i-6] = float(temp[0])
            for j in range(0,na):
                data[i-6,j] = float(temp[j+1])   

    return energies, angles, data


def eckstein_fit(E0,a1,a2,a3,a4,imp,wall):
    """
    Calculation of the fitted curves for particle and energy reflection
    coefficient using the empirical Eckstein formula.
    """
    
    # mass and atomic number of the projectile
    if imp == 'H':
        m1 = 1.01
        z1 = 1
    elif imp == 'D':
        m1 = 2.01
        z1 = 1
    elif imp == 'T':
        m1 = 3.02
        z1 = 1
    elif imp == 'He':
        m1 = 4.00
        z1 = 2
    elif imp == 'Be':
        m1 = 9.01
        z1 = 4
    elif imp == 'B':
        m1 = 10.81
        z1 = 5
    elif imp == 'C':
        m1 = 12.01
        z1 = 6 
    elif imp == 'N':
        m1 = 14.01
        z1 = 7
    elif imp == 'Ne':
        m1 = 20.18
        z1 = 10
    elif imp == 'Ar':
        m1 = 39.95
        z1 = 18
    elif imp == 'W':
        m1 = 183.85
        z1 = 74
    
    # mass and atomic number of the bulk target material
    if wall == 'Be':
        m2 = 9.01
        z2 = 4
    elif wall == 'C':
        m2 = 12.01
        z2 = 6 
    elif wall == 'W':
        m2 = 183.85
        z2 = 74
            
    # Thomas-Fermi energy (eV)
    ETF = 30.74 * ((m1+m2)/m2) * z1 * z2 * np.sqrt((z1**(2/3)) + (z2**(2/3)));
    
    # Reduced impact energy
    epsilon = E0/ETF
    
    # reflection coefficient (Eckstein formula)
    coeff = np.exp(a1*(epsilon**a2)) / (1.0 + np.exp(a3*(epsilon**a4)))
       
    return coeff


def bohdansky_fit(E0,Q,Eth,imp,wall):
    """
    Calculation of the fitted curves for the sputtering
    yield using the Bohdansky formula.
    """
    
    # mass and atomic number of the projectile
    if imp == 'H':
        m1 = 1.01
        z1 = 1
    elif imp == 'D':
        m1 = 2.01
        z1 = 1
    elif imp == 'T':
        m1 = 3.02
        z1 = 1
    elif imp == 'He':
        m1 = 4.00
        z1 = 2
    elif imp == 'Be':
        m1 = 9.01
        z1 = 4
    elif imp == 'B':
        m1 = 10.81
        z1 = 5
    elif imp == 'C':
        m1 = 12.01
        z1 = 6 
    elif imp == 'N':
        m1 = 14.01
        z1 = 7
    elif imp == 'Ne':
        m1 = 20.18
        z1 = 10
    elif imp == 'Ar':
        m1 = 39.95
        z1 = 18
    elif imp == 'W':
        m1 = 183.85
        z1 = 74
    
    # mass and atomic number of the bulk target material
    if wall == 'Be':
        m2 = 9.01
        z2 = 4
    elif wall == 'C':
        m2 = 12.01
        z2 = 6 
    elif wall == 'W':
        m2 = 183.85
        z2 = 74
        
    # mass and atomic number of the impurity imlpanted into the material
    if wall == 'H':
        m2 = 1.01
        z2 = 1
    elif wall == 'D':
        m2 = 2.01
        z2 = 1
    elif wall == 'T':
        m2 = 3.02
        z2 = 1
    elif wall == 'He':
        m2 = 4.00
        z2 = 2
    elif wall == 'B':
        m2 = 10.81
        z2 = 5
    elif wall == 'N':
        m2 = 14.01
        z2 = 7
    elif wall == 'Ne':
        m2 = 20.18
        z2 = 10
    elif wall == 'Ar':
        m2 = 39.95
        z2 = 18
            
    # Thomas-Fermi energy (eV)
    ETF = 30.74 * ((m1+m2)/m2) * z1 * z2 * np.sqrt((z1**(2/3)) + (z2**(2/3)));
    
    # Reduced impact energy
    epsilon = E0/ETF
    
    # Nuclear stopping cross section
    Sn = (0.5 * np.log(1 + 1.2288 * epsilon)) / (epsilon + 0.1728 * np.sqrt(epsilon) + 0.008 * (epsilon**0.1504))
    
    # sputtering yield (Bohdansky formula)
    y = Q * Sn * (1.0 - (Eth/E0)**(2/3)) * ((1.0-Eth/E0)**2)
       
    return y


def get_impact_energy(Te,projectile,Ti_over_Te = 1.0, gammai = 2.0):
    """
    Calculate the impact energy of a projectile ion onto a material surface
    as ion kinetic energy + sheath acceleration contribute.
    
    Parameters
    ----------
    Te : float or array
        Electron temperature, in eV.
    projectile : str
        Atomic symbol of the projectile ion hitting the surface.
    Ti_over_Te : float, optional
        Ratio of ion to electron temperatures. Default to 1.0.
    gammmai : float, optional
        Ion sheath heat transmission coefficient. Default to 2.0.
        
    Returns
    -------
    E0: array
        Projectile impact energy.
    """    

    me = 9.1093837e-31
    mp = 1.6726217e-27
    
    # ion mass of the projectile
    if projectile == 'H':
        mi = 1.01*mp
    elif projectile == 'D':
        mi = 2.01*mp
    elif projectile == 'T':
        mi = 3.02*mp
    elif projectile == 'He':
        mi = 4.00*mp
    elif projectile == 'Be':
        mi = 9.01*mp
    elif projectile == 'B':
        mi = 10.81*mp
    elif projectile == 'C':
        mi = 12.01*mp
    elif projectile == 'N':
        mi = 14.01*mp
    elif projectile == 'Ne':
        mi = 20.18*mp
    elif projectile == 'Ar':
        mi = 39.95*mp
    elif projectile == 'W':
        mi = 183.85*mp

    # if input arrays is multi-dimensional, flatten them here and restructure at the end
    _Te = np.ravel(Te) if Te is not None else None

    # get the ADAS rates for the projectile ion and calculate its mean charge state
    #   in function of the electron temperature, assuming ionization equilibrium
    atom_data = atomic.get_atom_data(projectile, ["scd", "acd"])
    _, Z_mean = atomic.get_Z_mean(atom_data, [1e14], Te_eV = _Te, plot = False) # precise value of ne irrelevant
   
    Ti = Ti_over_Te*_Te
    
    # ion kinetic energy
    ion_kinetic_energy = gammai*Ti
    
    # sheath acceleration term
    sheath_acceleration = -0.5 * np.log((2*np.pi*(me/mi)) * (1+Ti_over_Te)) * Z_mean * _Te
    
    # impact energy
    E0 = ion_kinetic_energy + sheath_acceleration
    
    if np.size(Te) > 1:
        # re-structure to original array dimensions
        E0 = E0.reshape(np.array(Te).shape)
        
    return E0   
    