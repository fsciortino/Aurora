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
    ''' Load default namelist. 
    Users should modify and complement this for a successful forward-model run.
    '''

    namelist = {
        'imp': 'Ca',
        'main_element': 'D',  # background
        # --------------------
        'source_rate': 1e+21,
        'source_type':'const', 
        # explicit source:
        'explicit_source_vals' : None, # provide 2D array on explicit_source_time and explicit_source_rhop grids
        'explicit_source_time' : None, 
        'explicit_source_rhop' : None,
        # other options to specify source:
        'source_width_in' :  0.0,   # exponential ionization decay from wall boundary if widths both = 0
        'source_width_out': 0.0,    
        'imp_source_energy_eV' : 3.0, # only needed if source_width_in=source_width_out=0.0
        'prompt_redep_flag' : False,
        'source_file': None, # required if source_type='file'
        'source_cm_out_lcfs' : 1.0, # source distance in cm from LCFS
        # LBO synthetic model, only used if source_type='synth_LBO'
        'LBO': {'n_particles': 1e+18,
                't_fall': 0.3,
                't_rise': 0.05,
                't_start': 0.01},
        # --------------------
        # time grid specifications
        'timing': {'dt_increase': np.array([1.005, 1.]),
                   'dt_start': np.array([1e-5, 0.001]),
                   'steps_per_cycle': np.array([1, 1]),
                   'times': np.array([0., 0.1])},
        # --------------------
        # radial grid
        'bound_sep': 2.0,
        'lim_sep': 1.0,
        'clen_divertor': 17.0,
        'clen_limiter': 0.5,
        'dr_0': 0.3,
        'dr_1': 0.05,
        'K': 6.0,
        'SOL_decay': 0.05, 
        # -------------------
        'saw_model': {'saw_flag': False,
                      'rmix': 1000.0,
                      'times': [1.0,], 
                      'crash_width' : 1.0},
        # --------------------
        # edge/recycling
        'recycling_flag' : False,
        'wall_recycling' : 0.0, 
        'tau_div_SOL_ms' : 50.0,  # ms
        'tau_pump_ms' : 500.0,  # ms
        'tau_rcl_ret_ms' : 50.0,   # ms
        'SOL_mach': 0.1,
        # --------------------
        # kinetic profiles
        'kin_profs' : {'ne': {'fun': 'interpa',
                              'times': [1.]}, 
                       'Te': {'fun':'interp',
                              'times': [1.],
                              'decay': [1.]},
                       'Ti': {'fun':'interp',
                              'times': [1.],
                              'decay': [1.]},
                       'n0': {'fun':'interpa',
                              'times': [1.]}
                   },
        # and NBI CXR rates
        'nbi_cxr' : {'rhop': None,
                     'vals': None}, # time indpt profiles; (rhop,nZ)
        # ----------------
        # charge exchange flags
        'cxr_flag' : False,
        'nbi_cxr_flag' : False,
        # -----------------
        # misc
        'device': 'CMOD',
        'shot': 99999,
        'time': 1250, # in ms, for equilibrium
        # atomic data (ADF11)
        'acd': None, # use default
        'scd': None, # use default
        'ccd': None, # use default
        'superstages': []
    }
    return namelist


if __name__=='__main__':
    # print default namelist for documentation

    nml = load_default_namelist()
    for key in nml:
        if isinstance(nml[key], dict):
            for key2 in nml[key]:
                print(f'   * - `{key}["{key2}"]`')
                print(f'     - {nml[key][key2]}')
                print('     -')
        else:
            print(f'   * - `{key}`')
            print(f'     - {nml[key]}')
            print('     -')
