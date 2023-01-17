# MIT License
#
# Copyright (c) 2021 Francesco Sciortino
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


def load_default_namelist():
    """Load default namelist.
    Users should modify and complement this for a successful forward-model run.
    """

    namelist = {
        # --------------------
        # IMPURITY AND MAIN SPECIES
        "imp": "Ca",  # impurity species
        "main_element": "D",  # main background species
        # --------------------
        # TIME GRID
        "timing": {
            "dt_increase": np.array([1.005, 1.0]),
            "dt_start": np.array([1e-5, 0.001]),
            "steps_per_cycle": np.array([1, 1]),
            "times": np.array([0.0, 0.1]),
        },
        # --------------------
        # RADIAL GRID
        "K": 6.0,
        "dr_0": 0.3,
        "dr_1": 0.05,
        "lim_sep": 1.0,  # cm
        "bound_sep": 2.0,  # cm
        "decay_length_boundary": 0.05,
        # --------------------
        # KINETIC PROFILES 
        "kin_profs": {
            "ne": {"fun": "interpa", "times": [1.0]},
            "Te": {"fun": "interp", "times": [1.0], "decay": [1.0]},
            "Ti": {"fun": "interp", "times": [1.0], "decay": [1.0]},
            "n0": {"fun": "interpa", "times": [1.0]},
        },
        # --------------------
        # ATOMIC PHYSICS
        "acd": None,  # use default
        "scd": None,  # use default
        "ccd": None,  # use default
        "cxr_flag": False,
        "nbi_cxr_flag": False,
        "nbi_cxr": {"rhop": None, "vals": None},  # time indpt profiles; (rhop,nZ)
        "superstages": [],
        # --------------------
        # EXTERNAL PARTICLE SOURCES
        "source_type": "const",
        "source_rate": 1e21,
        "source_file": None,  # required if source_type='file'
        "explicit_source_vals": None,  # provide 2D array on explicit_source_time and explicit_source_rhop grids
        "explicit_source_time": None,
        "explicit_source_rhop": None,
        "LBO": {"n_particles": 1e18, "t_fall": 0.3, "t_rise": 0.05, "t_start": 0.01},
        # --------------------
        # NEUTRAL SOURCE PROFILES 
        "source_width_in": 0.0,  # exponential ionization decay from wall boundary if widths both = 0
        "source_width_out": 0.0,
        "source_cm_out_lcfs": 1.0,  # source distance in cm from LCFS   
        "imp_source_energy_eV": 3.0,  # energy of externally injected impurity neutrals, only needed if source_width_in=source_width_out=0.0
        "imp_recycling_energy_eV": 3.0,  # energy of recycled impurity neutrals, only needed if source_width_in=source_width_out=0.0
        "prompt_redep_flag": False,
        # --------------------
        # SAWTEETH MODEL
        "saw_model": {
            "saw_flag": False,
            "mixing_radius": 1000.0,
            "times": [1.0,],
            "crash_width": 1.0,
        },
        # --------------------
        # EDGE/DIVERTOR
        "clen_divertor": 17.0, # m
        "clen_limiter": 0.5, # m
        "SOL_mach": 0.1,
        "div_recomb_ratio": 1.0,
        "recycling_flag": False, 
        "tau_div_SOL_ms": 50.0,  # ms
        "screening_eff": 0.0,
        # --------------------
        # PLASMA-WALL INTERACTION
        "wall_recycling": 0.0,
        "tau_rcl_ret_ms": 50.0,  # ms
        "phys_surfaces": False,
        "surf_mainwall": 1.0e5,  # cm^2
        "surf_divwall": 1.0e4,  # cm^2
        "mainwall_roughness": 1.0,
        "divwall_roughness": 1.0,
        "advanced_PWI": {
            "advanced_PWI_flag": False,
            "main_wall_material": 'W',
            "div_wall_material": 'W',
            "background_mode": 'manual',
            "background_species": ['D'],
            "background_main_wall_fluxes": [0], # s^-1
            "background_div_wall_fluxes": [0], # s^-1
            "background_files": ['file/location'],
            "characteristic_impact_energy_main_wall": 200, # eV
            "characteristic_impact_energy_div_wall": 500, # eV
            "n_main_wall_sat": 1e20, # m^-2
            "n_div_wall_sat": 1e20, # m^-2
            "energetic_recycled_neutrals": False,
            "Te_div": 30.0, # eV
            "Te_lim": 10.0, # eV
            "Ti_over_Te": 1.0,
            "gammai": 2.0,
        },
        # --------------------
        # PUMPING
        "phys_volumes": False,
        "tau_pump_ms": 500.0,  # ms
        "vol_div": 1.0e6,  # cm^3
        "S_pump": 5.0e6,  # cm^3/s
        "pump_chamber": False,
        "vol_pump": 1.0e6,  # cm^3
        "L_divpump": 1.0e7, # cm^3/s
        "L_leak": 0.0,  # cm^3/s
        # --------------------
        # DEVICE
        "device": "CMOD",
        "shot": 99999,
        "time": 1250,  # in ms, for equilibrium
    }
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
