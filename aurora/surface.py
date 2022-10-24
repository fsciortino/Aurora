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
import scipy
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


def get_reflection_data(projectile, target, angle, kind):
    """ Collect reflection data for a given impurity.

    Parameters
    ----------
    projectile : str
        Atomic symbol of projectile ion.
    target : str
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
            projectile
        ]["Reflection"][target]  # all reflection files for a given species and wall material
    except KeyError:
        raise KeyError(f"Reflection data not available for ion {projectile} on wall material {target}!")
        
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
        "projectile": projectile,
        "wall_material": target,
        "angle": angle,
        "energies": energies,
        "data": coefficients,
        }
    
    return refl_data


def reflection_coeff_fit(projectile, target, angle, kind, energies = None):
    """ Read the fitted reflection data for a given impurity according to the Eckstein fit.

    Parameters
    ----------
    projectile : str
        Atomic symbol of projectile ion.
    target : str
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
    energies : float or array, optional
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
            projectile
        ]["Reflection"][target]  # all reflection files for a given species and wall material
    except KeyError:
        raise KeyError(f"Reflection data not available for ion {projectile} on wall material {target}!")
        
    if kind == "rn":
        coeff_name = "particle reflection coefficient"
    elif kind == "re":
        coeff_name = "energy reflection coefficient"
    
    refl_data_fit = {}

    # absolute location of the requested files
    fileloc = trim_files.get_trim_file_loc(all_files[kind+"_fit"], filetype="refl")
    
    # read the full energy range limits
    temp = get_reflection_data(projectile, target, angle, kind)
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
        data_fit[i] = eckstein_fit(energy_range[i],a1,a2,a3,a4,projectile,target)
    
    refl_data_fit = {
        "coefficient_type": coeff_name,
        "projectile": projectile,
        "wall_material": target,
        "angle": angle,
        "energies": energy_range,
        "data": data_fit,
        }
    
    return refl_data_fit


def calc_reflected_energy(projectile, target, angle, energies = None):  
    """ Calculate the mean reflection energy data for a given impurity according to the Eckstein fit.

    Parameters
    ----------
    projectile : str
        Atomic symbol of projectile ion.
    target : str
        Atomic symbol of wall material.
    angle : float
        Angle of incidence of the projectile ion.
        Output data will be referred to the available
        tabulated angle closest to the requested one.    
    energies : float or array, optional
        Impact energies of the projectile ion in eV.
        If this input is specified, then the coefficients
        will be outputted at the requested energies. If left
        to None, then the entire energy range is used.
        
    Returns
    -------
    refl_energy_fit: dict
        Dictionary containing the fitted data for the requested type of coefficient. 
        It contains the type of coefficient, the projectile name, the wall material name,
        the requested incidence angle and the arrays with impact energies and fitted data.
    """    
    
    refl_energy_fit = {}

    # Fitted particle reflection coefficients
    rn = reflection_coeff_fit(projectile, target, angle, 'rn', energies = energies)
    
    # Fitted energy reflection coefficients
    re = reflection_coeff_fit(projectile, target, angle, 're', energies = energies)   
    
    energy_range = rn["energies"]
    
    data_rn = rn["data"]
    data_re = re["data"]
    
    mean_reflection_energy = energy_range*(data_re/data_rn)
    
    refl_energy_fit = {
        "coefficient_type": 'Mean reflection energy',
        "projectile": projectile,
        "wall_material": target,
        "angle": angle,
        "energies": energy_range,
        "data": mean_reflection_energy,
        }
    
    return refl_energy_fit


def get_bulk_sputtering_data(projectile, target, angle, kind):
    """ Collect bulk sputtering data for a given impurity.

    Parameters
    ----------
    projectile : str
        Atomic symbol of projectile ion.
    target : str
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
            target
        ]["Bulk_sputtering"][projectile]  # all bulk sputtering files for a given species and wall material
    except KeyError:
        raise KeyError(f"Bulk sputtering data not available for ion {projectile} on wall material {target}!")
        
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
        "projectile": projectile,
        "wall_material": target,
        "angle": angle,
        "energies": energies,
        "data": coefficients,
        }
    
    return bulk_sputter_data


def bulk_sputtering_coeff_fit(projectile, target, angle, kind, energies = None):
    """ Read the fitted bulk sputtering data for a given impurity according to the Bohdansky fit.

    Parameters
    ----------
    projectile : str
        Atomic symbol of projectile ion.
    target : str
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
    energies : float or array, optional
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
            target
        ]["Bulk_sputtering"][projectile]  # all bulk sputtering files for a given species and wall material
    except KeyError:
        raise KeyError(f"Bulk sputtering data not available for ion {projectile} on wall material {target}!")
        
    if kind == "y":
        coeff_name = "sputtering yield"
    elif kind == "ye":
        raise KeyError("Fitted curves for energy sputtering yields not availabile yet!")
    
    bulk_sputter_data_fit = {}

    # absolute location of the requested files
    fileloc = trim_files.get_trim_file_loc(all_files[kind+"_fit"], filetype="bulk_sputter")
    
    # read the full energy range limits
    temp = get_bulk_sputtering_data(projectile, target, angle, kind)
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
        data_fit[i] = bohdansky_fit(energy_range[i],Q,Eth,projectile,target)
    
    bulk_sputter_data_fit = {
        "coefficient_type": coeff_name,
        "projectile": projectile,
        "wall_material": target,
        "angle": angle,
        "energies": energy_range,
        "data": data_fit,
        }
    
    return bulk_sputter_data_fit


#TODO:def calc_bulk_sputtered_energy(projectile, target, angle, energies = None):


def get_implantation_depth_data(projectile, target, angle):
    """ Collect implantation depth data, in angstrom, for a given impurity.
        For now only helium available.

    Parameters
    ----------
    projectile : str
        Atomic symbol of projectile ion.
    target : str
        Atomic symbol of wall material.
    angle : float
        Angle of incidence of the projectile ion.
        Output data will be referred to the available
        tabulated angle closest to the requested one.
    
    Returns
    -------
    depth_data : dict
        Dictionary containing the data for the requested type of coefficient. 
        It contains the type of coefficient, the projectile name, the wall material name,
        the requested incidence angle and the arrays with impact energies and data.
    """

    try:
        # default files dictionary
        all_files = trim_files.trim_files_dict()[
            projectile
        ]["Implantation_depth"][target]  # all implantation depth files for a given species and wall material
    except KeyError:
        raise KeyError(f"Implantation depth data not available for ion {projectile} on wall material {target}!")

    depth_data = {}

    # absolute location of the requested files
    fileloc = trim_files.get_trim_file_loc(all_files['d'], filetype="depth")
    
    # full tables
    energies, angles, data = read_trim_table(fileloc)
    
    # index of incidence angle closest to the requested one
    idx = np.argmin(np.abs(np.asarray(angles) - angle))
    
    # coefficients corresponding to the desired angle
    coefficients = data[:,idx]
    
    depth_data = {
        "coefficient_type": 'mean implantation depth',
        "projectile": projectile,
        "wall_material": target,
        "angle": angle,
        "energies": energies,
        "data": coefficients,
        }
    
    return depth_data


def implantation_depth_fit(projectile, target, angle, energies = None):
    """ Read the fitted implantation depth data for a given impurity,
        in angstrom, according to a simple linear interpolation.

    Parameters
    ----------
    projectile : str
        Atomic symbol of projectile ion.
    target : str
        Atomic symbol of wall material.
    angle : float
        Angle of incidence of the projectile ion.
        Output data will be referred to the available
        tabulated angle closest to the requested one.    
    energies : float or array, optional
        Impact energies of the projectile ion in eV.
        If this input is specified, then the coefficients
        will be outputted at the requested energies. If left
        to None, then the entire energy range is used.
        
    Returns
    -------
    depth_data_fit: dict
        Dictionary containing the fitted data for the requested type of coefficient. 
        It contains the type of coefficient, the projectile name, the wall material name,
        the requested incidence angle and the arrays with impact energies and fitted data.
    """    
    
    try:
        # default files dictionary
        all_files = trim_files.trim_files_dict()[
            projectile
        ]["Implantation_depth"][target]  # all implantation depth files for a given species and wall material
    except KeyError:
        raise KeyError(f"Implantation depth data not available for ion {projectile} on wall material {target}!")
    
    depth_data_fit = {}
    
    # absolute location of the requested files
    fileloc = trim_files.get_trim_file_loc(all_files['d'], filetype="depth")
    
    # full tables
    en, angles, data = read_trim_table(fileloc)
    
    # index of incidence angle closest to the requested one
    idx = np.argmin(np.abs(np.asarray(angles) - angle))
    
    # coefficients corresponding to the desired angle
    coefficients = data[:,idx]
    
    # simple linear interpolation of the data in function of the impact energy
    fun = scipy.interpolate.interp1d(en, coefficients, kind="linear", fill_value="extrapolate", assume_sorted=True)
    
    # energy range limits
    energy_range_limits = np.array([0,0])
    energy_range_limits[0] = en[0]
    energy_range_limits[1] = en[-1]
         
    # build the energy range in which to perform the fit
    if energies is None: # full energy range
        energy_range = np.linspace(energy_range_limits[0],energy_range_limits[1],10000)
    else: # discrete number of energies defined in the input
        energy_range = np.array(energies)
        
    # extract the fitted data for the requested incidence angle using a simple linear interpolation
    data_fit = fun(energy_range)
    
    depth_data_fit = {
        "coefficient_type": 'mean implantation depth',
        "projectile": projectile,
        "wall_material": target,
        "angle": angle,
        "energies": energy_range,
        "data": data_fit,
        }
    
    return depth_data_fit


def get_impurity_sputtering_data(imp, projectile, target, angle):
    """ Collect impurity sputtering yields for a given impurity
        implanted in a given wall material.

    Parameters
    ----------
    imp : str
        Atomic symbol of impurity ion.
    projectile : str
        Atomic symbol of projectile ion.   
    target : str
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
        ]["Impurity_sputtering"][target][projectile]  # all impurity sputtering files for given inpurity, projectile and wall material
    except KeyError:
        raise KeyError(f"Impurity sputtering data not available for ion {imp} on wall material {target} with projectile {projectile}!")

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
        "wall_material": target,
        "angle": angle,
        "energies": energy_values,
        "impurity_concentrations": C_imp_values,
        "data": coefficients,
        }

    return imp_sputter_data    


def impurity_sputtering_coeff_fit(imp, projectile, target, angle, energies = None):
    """ Read the fitted impurity sputtering yield for a given impurity implanted
        in a given wall material (normalized to the impurity 
        concentration in the surface) according to the Bohdansky fit.

    Parameters
    ----------
    imp : str
        Atomic symbol of impurity ion implanted in the wall surface.
    projectile : str
        Atomic symbol of projectile ion.
    target : str
        Atomic symbol of wall material.
    angle : float
        Angle of incidence of the projectile ion.
        Output data will be referred to the available
        tabulated angle closest to the requested one.
    energies : float or array, optional
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
        ]["Impurity_sputtering"][target][projectile]  # all impurity sputtering files for given inpurity, projectile and wall material
    except KeyError:
        raise KeyError(f"Impurity sputtering data not available for ion {imp} on wall material {target} with projectile {projectile}!")
    
    imp_sputter_data_fit = {}

    # absolute location of the requested files
    fileloc = trim_files.get_trim_file_loc(all_files["y_fit"], filetype="imp_sputter")
    
    # read the full energy range limits
    temp = get_impurity_sputtering_data(imp, projectile, target, angle)
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
        "wall_material": target,
        "angle": angle,
        "energies": energy_range,
        "normalized_data": data_fit,
        }
    
    return imp_sputter_data_fit


def get_impurity_sputtered_energy(imp, projectile, target):
    """ Collect mean impurity sputtered energy for a given impurity
        implanted in a given wall material.
        For now values are available only at one (the most realistic)
        incidence angle of the projectile, i.e. the same one callable
        by the function impact_angle.

    Parameters
    ----------
    imp : str
        Atomic symbol of impurity ion.
    projectile : str
        Atomic symbol of projectile ion.   
    target : str
        Atomic symbol of wall material.
    
    Returns
    -------
    imp_sputter_energy_data : dict
        Dictionary containing the requested sputtered energy data.
        It contains the type of coefficient, the implanted impurity name, the
        projectile name, the wall material name, the requested incidence angle
        and the arrays with energies and data.
    """

    try:
        # default files dictionary
        all_files = trim_files.trim_files_dict()[
            imp
        ]["Impurity_sputtering"][target][projectile]  # all impurity sputtering files for given inpurity, projectile and wall material
    except KeyError:
        raise KeyError(f"Impurity sputtering data not available for ion {imp} on wall material {target} with projectile {projectile}!")

    imp_sputter_energy_data = {}

    # absolute location of the requested files
    fileloc = trim_files.get_trim_file_loc(all_files["ye"], filetype="imp_sputter")
    
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
    
    data = np.zeros((len(angle_values),len(energy_values)))
    
    for i in range(0,len(lines)-10):
        temp = lines[i+10]
        temp = temp.split("	")
        ang = float(temp[1])
        angle_index = np.argmin(np.abs(np.asarray(angle_values)-ang))
        en = float(temp[2])
        energy_index = np.argmin(np.abs(np.asarray(energy_values)-en))
        E_mean_imp = float(temp[3])
        data[angle_index,energy_index] = E_mean_imp 

    imp_sputter_energy_data = {
        "coefficient_type": "mean sputtered energy",
        "impurity": imp,
        "projectile": projectile,
        "wall_material": target,
        "angle": angle_values[0],
        "energies": energy_values,
        "data": data,
        }

    return imp_sputter_energy_data 

    
def calc_imp_sputtered_energy(imp, projectile, target, energies = None):
    """ Calculate the mean impurity sputtered energy for a given impurity
        implanted in a given wall material in function of the impact energy
        using a simple linear interpolation on the tabulated data.
        For now values are available only at one (the most realistic)
        incidence angle of the projectile, i.e. the same one callable
        by the function impact_angle.

    Parameters
    ----------
    imp : str
        Atomic symbol of impurity ion.
    projectile : str
        Atomic symbol of projectile ion.   
    target : str
        Atomic symbol of wall material.
    energies : float or array, optional
        Impact energies of the projectile ion in eV.
        If this input is specified, then the coefficients
        will be outputted at the requested energies. If left
        to None, then the entire energy range is used.
    
    Returns
    -------
    imp_sputter_energy_data_fit : dict
        Dictionary containing the fitted data for the requested type of coefficient.
        It contains the type of coefficient, the implanted impurity name, the
        projectile name, the wall material name, the requested incidence angle
        and the arrays with energies and data.
    """    
    
    try:
        # default files dictionary
        all_files = trim_files.trim_files_dict()[
            imp
        ]["Impurity_sputtering"][target][projectile]  # all impurity sputtering files for given inpurity, projectile and wall material
    except KeyError:
        raise KeyError(f"Impurity sputtering data not available for ion {imp} on wall material {target} with projectile {projectile}!")

    imp_sputter_energy_data_fit = {}

    # absolute location of the requested files
    fileloc = trim_files.get_trim_file_loc(all_files["ye"], filetype="imp_sputter")
    
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
    
    data = np.zeros((len(angle_values),len(energy_values)))
    
    for i in range(0,len(lines)-10):
        temp = lines[i+10]
        temp = temp.split("	")
        ang = float(temp[1])
        angle_index = np.argmin(np.abs(np.asarray(angle_values)-ang))
        en = float(temp[2])
        energy_index = np.argmin(np.abs(np.asarray(energy_values)-en))
        E_mean_imp = float(temp[3])
        data[angle_index,energy_index] = E_mean_imp 

    # simple linear interpolation of the data in function of the impact energy
    fun = scipy.interpolate.interp1d(energy_values, data[0,:], kind="linear", fill_value="extrapolate", assume_sorted=True)

    # energy range limits
    energy_range_limits = np.array([0,0])
    energy_range_limits[0] = energy_values[0]
    energy_range_limits[1] = energy_values[-1]
     
    # build the energy range in which to perform the fit
    if energies is None: # full energy range
        energy_range = np.linspace(energy_range_limits[0],energy_range_limits[1],10000)
    else: # discrete number of energies defined in the input
        energy_range = np.array(energies)
    
    # extract the fitted data for the requested incidence angle using a simple linear interpolation
    data_fit = fun(energy_range)

    imp_sputter_energy_data_fit = {
        "coefficient_type": "mean sputtered energy",
        "impurity": imp,
        "projectile": projectile,
        "wall_material": target,
        "angle": angle_values[0],
        "energies": energy_range,
        "data": data_fit,
        }  
    
    return imp_sputter_energy_data_fit
    

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


def eckstein_fit(E0,a1,a2,a3,a4,projectile,target):
    """
    Calculation of the fitted curves for particle and energy reflection
    coefficient using the empirical Eckstein formula.
    """
    
    # mass and atomic number of the projectile
    if projectile == 'H':
        m1 = 1.01
        z1 = 1
    elif projectile == 'D':
        m1 = 2.01
        z1 = 1
    elif projectile == 'T':
        m1 = 3.02
        z1 = 1
    elif projectile == 'He':
        m1 = 4.00
        z1 = 2
    elif projectile == 'Be':
        m1 = 9.01
        z1 = 4
    elif projectile == 'B':
        m1 = 10.81
        z1 = 5
    elif projectile == 'C':
        m1 = 12.01
        z1 = 6 
    elif projectile == 'N':
        m1 = 14.01
        z1 = 7
    elif projectile == 'Ne':
        m1 = 20.18
        z1 = 10
    elif projectile == 'Ar':
        m1 = 39.95
        z1 = 18
    elif projectile == 'W':
        m1 = 183.85
        z1 = 74
    
    # mass and atomic number of the bulk target material
    if target == 'Be':
        m2 = 9.01
        z2 = 4
    elif target == 'C':
        m2 = 12.01
        z2 = 6 
    elif target == 'W':
        m2 = 183.85
        z2 = 74
            
    # Thomas-Fermi energy (eV)
    ETF = 30.74 * ((m1+m2)/m2) * z1 * z2 * np.sqrt((z1**(2/3)) + (z2**(2/3)));
    
    # Reduced impact energy
    epsilon = E0/ETF
    
    # reflection coefficient (Eckstein formula)
    coeff = np.exp(a1*(epsilon**a2)) / (1.0 + np.exp(a3*(epsilon**a4)))
       
    return coeff


def bohdansky_fit(E0,Q,Eth,projectile,target):
    """
    Calculation of the fitted curves for the sputtering
    yield using the Bohdansky formula.
    """
    
    # mass and atomic number of the projectile
    if projectile == 'H':
        m1 = 1.01
        z1 = 1
    elif projectile == 'D':
        m1 = 2.01
        z1 = 1
    elif projectile == 'T':
        m1 = 3.02
        z1 = 1
    elif projectile == 'He':
        m1 = 4.00
        z1 = 2
    elif projectile == 'Be':
        m1 = 9.01
        z1 = 4
    elif projectile == 'B':
        m1 = 10.81
        z1 = 5
    elif projectile == 'C':
        m1 = 12.01
        z1 = 6 
    elif projectile == 'N':
        m1 = 14.01
        z1 = 7
    elif projectile == 'Ne':
        m1 = 20.18
        z1 = 10
    elif projectile == 'Ar':
        m1 = 39.95
        z1 = 18
    elif projectile == 'W':
        m1 = 183.85
        z1 = 74
    
    # mass and atomic number of the bulk target material
    if target == 'Be':
        m2 = 9.01
        z2 = 4
    elif target == 'C':
        m2 = 12.01
        z2 = 6 
    elif target == 'W':
        m2 = 183.85
        z2 = 74
        
    # mass and atomic number of the impurity imlpanted into the material
    if target == 'H':
        m2 = 1.01
        z2 = 1
    elif target == 'D':
        m2 = 2.01
        z2 = 1
    elif target == 'T':
        m2 = 3.02
        z2 = 1
    elif target == 'He':
        m2 = 4.00
        z2 = 2
    elif target == 'B':
        m2 = 10.81
        z2 = 5
    elif target == 'N':
        m2 = 14.01
        z2 = 7
    elif target == 'Ne':
        m2 = 20.18
        z2 = 10
    elif target == 'Ar':
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


def get_impact_energy(Te, projectile, mode = 'sheath' ,Ti_over_Te = 1.0, gammai = 2.0):
    """
    Calculate the impact energy of a projectile ion onto a material surface.
    
    Parameters
    ----------
    Te : float or array
        If mode = "sheath", this is the electron temperature at the
            plasma-wall interface, in eV.
        If mode = "FSM", this is the intra-ELM pedestal electron
            temperature, in eV.
    projectile : str
        Atomic symbol of the projectile ion hitting the surface.
    mode : str
        Assumption done for calculating the impact energy.
        If "sheath" (default) then the existence of the sheath
            is considered, and the impact energy is calculated as
            ion kinetic energy + sheath acceleration contribute.
            This is always valid during inter-ELM phases, but is valid
            only for the contact with the main wall in intra-ELM phases.
        If "FSM" then the existence of the sheath is neglected, and
            the free streaming model is applied, which foresees an
            impact energy proportional (with a constant) to the
            pedestal electron temperature. This is valid during intra-
            ELM phases for the contact with the divertor target, where
            the free streaming assumption can be made after the long
            travel along the entire parallel connection length of the
            ELM filaments from midplane to divertor targets.
            See D. Moulton et al 2013 PPCF 55 085003 for details
    Ti_over_Te : float, optional
        If mode = "sheath", this is the ratio of ion to electron
            temperatures at the plasma-wall interface.
        If mode = "FSM", this is the ratio of the intra-ELM ion
            to electron temperatures at the pedestal
        Default to 1.0.
    gammmai : float, optional
        Ion sheath heat transmission coefficient. Default to 2.0.
        
    Returns
    -------
    E0: array
        Projectile impact energy, in eV.
        If mode = "sheath", then this is the actual impact energy
            in function of the electron temperature at the plasma-wall
            interface.
        If mode = "FSM", then this is maximum value of a time-dependent
            impact energy during an ELM in function of electron
            temperature at the pedestal.
    """    

    me = 9.1093837e-31
    mp = 1.6726217e-27
    
    # ion mass and FS factor of the projectile
    if projectile == 'H':
        mi = 1.01*mp
        z = 1
    elif projectile == 'D':
        mi = 2.01*mp
        z = 1
    elif projectile == 'T':
        mi = 3.02*mp
        z = 1
    elif projectile == 'He':
        mi = 4.00*mp
        z = 2
    elif projectile == 'Be':
        mi = 9.01*mp
        z = 4
    elif projectile == 'B':
        mi = 10.81*mp
        z = 5
    elif projectile == 'C':
        mi = 12.01*mp
        z = 6
    elif projectile == 'N':
        mi = 14.01*mp
        z = 7
    elif projectile == 'Ne':
        mi = 20.18*mp
        z = 10
    elif projectile == 'Ar':
        mi = 39.95*mp
        z = 18
    elif projectile == 'W':
        mi = 183.85*mp
        z = 74

    if mode == 'sheath':

        # if input arrays is multi-dimensional, flatten them here and restructure at the end
        _Te = np.ravel(Te) if Te is not None else None

        # get the ADAS rates for the projectile ion and calculate its mean charge state
        #   in function of the electron temperature at the plasma-wall interface,
        #   assuming ionization equilibrium
        if projectile == 'D' or projectile == 'T':
            atom_data = atomic.get_atom_data('H', ["scd", "acd"])
        else:
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
            
    elif mode == 'FSM':
        
        # if input arrays is multi-dimensional, flatten them here and restructure at the end
        _Te = np.ravel(Te) if Te is not None else None
        
        # get the ADAS rates for the projectile ion and calculate its mean charge state
        #   in function of the electron temperature at the pedestal
        #   assuming ionization equilibrium
        if projectile == 'D' or projectile == 'T':
            atom_data = atomic.get_atom_data('H', ["scd", "acd"])
        else:
            atom_data = atomic.get_atom_data(projectile, ["scd", "acd"])
        _, Z_mean = atomic.get_Z_mean(atom_data, [1e14], Te_eV = _Te, plot = False) # precise value of ne irrelevant

        # time range
        t = np.linspace(0.0001,1,10000)
        
        # function, in a.u., of the ion energy onto the divertor target after an ELM crash
        #   in normalized units of x/cs (with x parallel coordinate)
        Q = np.zeros(len(t))
        for j in range(0,len(t)):
            Q[j] = (1/(t[j]**2)) * np.exp(-1/(2*(t[j]**2))) * (1/((2*(t[j]**2)))*(z+1)+1)
        
        # find the time at which there is the peak value, in normalized units of x/cs
        #   (i.e. the time, along the parallel coordinate x from the midplane, at which
        #    the maximum in Q occurs will t_max = t_max_norm * (x/cs))
        max_index = np.nanargmax(Q)
        t_max_norm = t[max_index]
        
        # now t_max_norm will be a parameter in the equation giving alpha, i.e. the factor
        #   which will multiply T_e_ped to give the maximum impact energy
        alpha = (Z_mean+1)/(2*t_max_norm**2)+1
        
        # peak impact energy
        E0 = alpha * Te
        
        if np.size(Te) > 1:
            # re-structure to original array dimensions
            E0 = E0.reshape(np.array(Te).shape)
        
    return E0   


def incidence_angle(projectile):
    """
    Convenience function for estimating the mean incidence angle of different ion species
    onto a material surface, according to equation-of-motion calculations of long range
    transport trajectories of projectile ions, performed with realistic plasma parameters 
    (B = 1-2 T, field line inclination onto material = 4-6°, Te = 10-30 eV,
    ne = 10^18-10^19 m^-3, mach number = 0.1-1) and taking into account gyromotion
    and sheath acceleration. Source: Schmid et al, NF 2010.
    
    Parameters
    ----------
    projectile : str
        Atomic symbol of the projectile ion hitting the surface.
        
    Returns
    -------
    angle: float
        Assumed incidence angle for the requested projectile wrt the surface normal, in degrees.
    """    
    
    if projectile == 'H':
        angle = 65
    elif projectile == 'D':
        angle = 65
    elif projectile == 'T':
        angle = 65
    elif projectile == 'He':
        angle = 65 
    elif projectile == 'Be':
        angle = 65
    elif projectile == 'B':
        angle = 65
    elif projectile == 'C':
        angle = 65
    elif projectile == 'N':
        angle = 65
    elif projectile == 'Ne':
        angle = 65
    elif projectile == 'Ar':
        angle = 55        
    elif projectile == 'W':
        angle = 45        
        
    return angle


def calc_implanted_impurity_concentration(sigma_imp, imp, bulk_target, implantation_depth):
    """
    Calculate the concentration of the implanted impurity into a bulk material
    starting from a given value of surface implantation density.
    The conversion will of course depend on the implantation depth.

    Parameters
    ----------
    sigma_imp : float or array
        Surface implantation density of the impurity in the bulk material,
        used by the aurora calculations, in
        number of impurity atoms / m^2 of surface.
    imp : str
        Atomic symbol of the impurity.
    bulk_target : str
        Atomic symbol of the bulk taget material.
    wall_type : str
        Type of considered wall ("main_wall" or "divertor_wall").
    implantation_depth : float
        Assumed depth of a uniform implantation profile of the impurity
        in the material, in angstrom.
        
    Returns
    -------
    C_imp: float or array
        Concentration of the impurity in the bulk material,
        used for extracing the sputtering data from the TRIM datafiles,
        in number of impurity atoms / total number of material atoms (bulk+imp)
    """       

    # density and molar mass of the bulk material
    if bulk_target == 'Be':
        rho = 1.80 # g/cm^3
        M = 183.85 # g/mol
    elif bulk_target == 'C':
        rho = 1.85 # g/cm^3
        M = 12.01 # g/mol
    elif bulk_target == 'W':
        rho = 19.29 # g/cm^3
        M = 183.85 # g/mol
        
    avogadro = 6.022e23 # mol^-1
    
    # calculate the molar density
    mol_density = rho / M # mol/cm^3
    
    # calculate the bulk atom density
    n_bulk = avogadro * mol_density * 1e6 # atoms/m^-3
    
    # convert the surface implantation density into volumetric impurity atom density
    #   in the implantation layer
    depth = implantation_depth * 1e-10 # m
    n_imp = sigma_imp/depth
    
    # calculate the resulting impurity concentration
    C_imp = (n_imp) / (n_imp + n_bulk)
    
    return C_imp
    

def calc_implanted_impurity_density(C_imp, imp, bulk_target, implantation_depth):
    """
    Calculate the surface of the implanted impurity into a bulk material
    starting from a given value of implantation concentration.
    The conversion will of course depend on the implantation depth.

    Parameters
    ----------
    C_imp : float or array
        Concentration of the impurity in the bulk material,
        used for extracing the sputtering data from the TRIM datafiles,
        in number of impurity atoms / total number of material atoms (bulk+imp)
    imp : str
        Atomic symbol of the impurity.
    bulk_target : str
        Atomic symbol of the bulk taget material.
    wall_type : str
        Type of considered wall ("main_wall" or "divertor_wall").
    implantation_depth : float
        Assumed depth of a uniform implantation profile of the impurity
        in the material, in angstrom.
        
    Returns
    -------
    sigma_imp : float or array
        Surface implantation density of the impurity in the bulk material,
        used by the aurora calculations, in
        number of impurity atoms / m^2 of surface.
    """       

    # density and molar mass of the bulk material
    if bulk_target == 'Be':
        rho = 1.80 # g/cm^3
        M = 183.85 # g/mol
    elif bulk_target == 'C':
        rho = 1.85 # g/cm^3
        M = 12.01 # g/mol
    elif bulk_target == 'W':
        rho = 19.29 # g/cm^3
        M = 183.85 # g/mol
        
    avogadro = 6.022e23 # mol^-1
    
    # calculate the molar density
    mol_density = rho / M # mol/cm^3
    
    # calculate the bulk atom density
    n_bulk = avogadro * mol_density * 1e6 # atoms/m^-3
    
    # calculate the volumetric impurity atom density
    n_imp = n_bulk * (C_imp/(1-C_imp))
    
    # convert the volumetric impurity atom density into surface implantation density
    #   in the implantation layer
    depth = implantation_depth * 1e-10 # m  
    sigma_imp = n_imp*depth
     
    return sigma_imp    
    

def calc_impurity_saturation_density(imp, bulk_target, E0):
    """
    Calculate the saturation density of an implanted impurity,
    into its implantation layer, according to experimental data.
    Data are for now only available for He implanted in C and W
    (source: Schmid et al, NF 2007).

    Parameters
    ----------
    imp : str
        Atomic symbol of the impurity.
    bulk_target : str
        Atomic symbol of the bulk taget material.
    E0 : float or array
        Characteristic impact energy for the impurity onto the surface
        used for calculate the depth of the implantation layer and to
        build a functional dependence of the experimental data on the
        impact energy.
        
    Returns
    -------
    sigma_imp_sat : float or array
        Saturation value of the surface implantation density of the impurity
        in the bulk material, in number of impurity atoms / m^2 of surface.
    """
    
    if imp == 'H' or imp == 'D' or imp == 'T' or imp == 'B' or imp == 'N' or imp == 'Ne' or imp == 'Ar':
        raise KeyError(f"Saturation implantation data not available yet for ion {imp} on wall material {bulk_target}!")
    if imp == 'He' and (bulk_target == 'Be' or bulk_target == 'C'):
        raise KeyError(f"Saturation implantation data not available yet for ion {imp} on wall material {bulk_target}!")
    
    # Experimental data for He in W (Schmid et al, NF 2010)
    if imp == 'He':
        if bulk_target == 'W':
            #   Max. retention increases with increasing energy of the impinging He flux
            #   as with increasing energy, the implantation layer becomes wider, and the
            #   effects of a higher W erosion due to more energetic He projectiles are
            #   apparently negligible --> No implanted He is promptly lost already during the
            #   implantation itself because of erosion (differently to what happens with He in C)
            E0_exp = [200,600] # eV
            sigma_imp_sat_exp = [5e19,1.5e20] # m^-2
            # Assume a simple linear interpolation
            fun = scipy.interpolate.interp1d(E0_exp, sigma_imp_sat_exp, kind="linear", fill_value="extrapolate", assume_sorted=True)
            # Estimated values of the saturation density of He implanted in W
            #   in function of a characterstic He impact energy E0
            sigma_imp_sat = fun(E0)
            
    return sigma_imp_sat
    