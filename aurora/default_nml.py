# MIT License
#
# Copyright (c) 2021 Francesco Sciortino and 2022 Antonello Zito
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


def load_default_namelist(device=None):
    """ Load default namelist. 
    Users should modify and complement this for a successful forward-model run.
    """
    
    namelist = {
        "imp": "Ca",
        "main_element": "D",  # background
        # --------------------
        "source_rate": 1e21,
        "source_type": "const",
        # explicit source:
        "explicit_source_vals": None,  # provide 2D array on explicit_source_time and explicit_source_rhop grids
        "explicit_source_time": None,
        "explicit_source_rhop": None,
        # other options to specify source:
        "source_width_in": 0.0,  # exponential ionization decay from wall boundary if widths both = 0
        "source_width_out": 0.0,
        "imp_source_energy_eV": 3.0,  # energy of externally injected impurity neutrals, only needed if source_width_in=source_width_out=0.0
        "imp_recycling_energy_eV": 3.0,  # energy of recycled impurity neutrals, only needed if source_width_in=source_width_out=0.0
        "prompt_redep_flag": False,
        "source_file": None,  # required if source_type='file'
        "source_cm_out_lcfs": 1.0,  # source distance in cm from LCFS
        # LBO synthetic model, only used if source_type='synth_LBO'
        "LBO": {"n_particles": 1e18, "t_fall": 0.3, "t_rise": 0.05, "t_start": 0.01},
        # --------------------
        # time grid specifications
        "timing": {
            "dt_increase": np.array([1.005, 1.0]),
            "dt_start": np.array([1e-5, 0.001]),
            "steps_per_cycle": np.array([1, 1]),
            "times": np.array([0.0, 0.1]),
            "time_start_plot": 0.0,
        },
        # --------------------
        # radial grid
        "bound_sep": 2.0, # cm
        "lim_sep": 1.0, # cm
        "clen_divertor": 17.0, # m
        "clen_limiter": 0.5, # m
        "dr_0": 0.3,
        "dr_1": 0.05,
        "K": 6.0,
        "SOL_decay": 0.05,
        # -------------------
        # sawteeth model
        "saw_model": {
            "saw_flag": False,
            "rmix": 1000.0,
            "times": [1.0,],
            "crash_width": 1.0,
        },
        # -------------------
        # ELM model
        "ELM_model": {
            "ELM_flag": False,
            "ELM_time_windows": None,
            "ELM_frequency": [100], # Hz
            "crash_duration": [0.5],  # ms
            "plateau_duration": [1.0],  # ms
            "recovery_duration": [0.5],  # ms
            "ELM_shape_decay_param": 2000, # s^-1
            "ELM_shape_delay_param": 0.001, # s
            "adapt_time_grid": False,
            "dt_intra_ELM": 5e-5, # s
            "dt_increase_inter_ELM": 1.05,
            
        },
        # --------------------
        # edge
        "recycling_flag": False,
        "screening_eff": 0.0,
        "div_recomb_ratio": 1.0,
        "tau_div_SOL_ms": 50.0,  # ms
        "SOL_mach": 0.1,
        "SOL_mach_ELM": 0.1,
        # --------------------
        # plasma-wall interaction
        "phys_surfaces": False,
        "surf_mainwall": 1.0e5,  # cm^2
        "surf_divwall": 1.0e4,  # cm^2
        "mainwall_roughness": 1.0,
        "divwall_roughness": 1.0,
        "wall_recycling": 0.0,
        "tau_rcl_ret_ms": 50.0, # ms
        "advanced_PWI": {
            "advanced_PWI_flag": False,
            "main_wall_material": 'W',
            "div_wall_material": 'W',
            "mode": 'manual',
            "background_species": ['D'],
            "main_wall_fluxes": [0], # s^-1
            "main_wall_fluxes_ELM": [0], # s^-1
            "div_wall_fluxes": [0], # s^-1
            "div_wall_fluxes_ELM": [0], # s^-1
            "files": ['file/location'],
            "characteristic_impact_energy_main_wall": 200, # eV
            "characteristic_impact_energy_div_wall": 500, # eV
            "n_main_wall_sat": 1e20, # m^-2
            "n_div_wall_sat": 1e20, # m^-2
            "energetic_recycled_neutrals": False,
            "Te_ped_intra_ELM": 400.0, # eV
            "Te_div_inter_ELM": 30.0, # eV
            "Te_lim_intra_ELM": 20.0, # eV
            "Te_lim_inter_ELM": 10.0, # eV
            "Ti_over_Te": 1.0,
            "gammai": 2.0,
        },
        # --------------------
        # pumping
        "phys_volumes": False,
        "vol_div": 1.0e6,  # cm^3
        "pump_chamber": False,
        "vol_pump": 1.0e6,  # cm^3
        "tau_pump_ms": 500.0,  # ms
        "S_pump": 5.0e6,  # cm^3/s
        "L_divpump": 1.0e7, # cm^3/s
        "L_leak": 0.0,  # cm^3/s
        # --------------------
        # kinetic profiles
        "kin_profs": {
            "ne": {"fun": "interpa", "times": [1.0]},
            "Te": {"fun": "interp", "times": [1.0], "decay": [1.0]},
            "Ti": {"fun": "interp", "times": [1.0], "decay": [1.0]},
            "n0": {"fun": "interpa", "times": [1.0]},
        },
        # and NBI CXR rates
        "nbi_cxr": {"rhop": None, "vals": None},  # time indpt profiles; (rhop,nZ)
        # ----------------
        # charge exchange flags
        "cxr_flag": False,
        "nbi_cxr_flag": False,
        # -----------------
        # misc
        "device": "CMOD",
        "shot": 99999,
        "time": 1250,  # in ms, for equilibrium
        # atomic data (ADF11)
        "acd": None,  # use default
        "scd": None,  # use default
        "ccd": None,  # use default
        "superstages": [],
    }
    
    if device is None or device=="CMOD": # Default input parameters (CMOD)

        pass
        
    else: # Adapt geometry-related input parameters for other devices
        
        namelist["device"] = device   
    
        if device=="AUG": # Geometry-related input parameters adapted for AUG
            
            namelist["advanced_PWI"]["main_wall_material"] = 'W'
            namelist["advanced_PWI"]["div_wall_material"] = 'W'
            namelist["vol_div"] = 0.8e6 # cm^3
            namelist["vol_pump"] = 1.7e6 # cm^3
            namelist["surf_mainwall"] = 5.0e4 # cm^2
            namelist["surf_divwall"] = 0.5e4 # cm^2            
            namelist["source_cm_out_lcfs"] = 15.0 # cm
            namelist["bound_sep"] = 10.0 # cm
            namelist["lim_sep"] = 6.0 # cm
            namelist["clen_divertor"] = 50.0 # m
            namelist["clen_lim"] = 1.0 # m
            namelist["shot"] = 39148    
            namelist["time"] = 2.7     
            
        else:
           
            print("Warning: default namelist for device " + device + " not existing. Using default namelist.")
        
    return namelist


if __name__ == "__main__":
    # print default namelist for documentation

    nml = load_default_namelist()
    for key in nml:
        if isinstance(nml[key], dict):
            for key2 in nml[key]:
                print(f'   * - `{key}["{key2}"]`')
                print(f"     - {nml[key][key2]}")
                print("     -")
        else:
            print(f"   * - `{key}`")
            print(f"     - {nml[key]}")
            print("     -")
