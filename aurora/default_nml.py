''' Method to load default namelist. This should be complemented with additional info 
by each user. 

sciortino, July 2020
'''

import numpy as np


def load_default_namelist():
    ''' Load default namelist. 
    Users should modify and complement this for a successful run.
    '''

    namelist = {
        'device': 'CMOD',
        'shot': 99999,
        'time': 1250, # in ms, for equilibrium
        'imp': 'Ca',
        # --------------------
        # background:
        'main_element': 'D',
        'Baxis': 5.5,
        # --------------------
        # source
        'source_width_in' :  0.0,   # exponential ionization decay from wall boundary if widths both = 0
        'source_width_out': 0.0,    
        'imp_source_energy_eV' : 3.0, # only needed if source_width_in=source_width_out=0.0
        'prompt_redep_flag' : False,
        'Phi0': 1e+24,
        'source_type':'file', 
        'source_file': None, # required if source_type='file'
        'source_cm_out_lcfs' : 1.0, # source distance in cm from LCFS
        'lbo_model': {'n_particles': 1e+18,
                      't_fall': 1.71,
                      't_rise': 0.181,
                      't_start': 1.0},
        # --------------------
        # timing
        'timing': {'dt_increase': np.array([1., 1.]),
                   'dt_start': np.array([0.001, 0.001]),
                   'steps_per_cycle': np.array([1, 1]),
                   'times': np.array([1., 2.])},
        # --------------------
        # radial grid
        'K': 6.0,
        'bound_sep': 2.0,
        'clen_divertor': 17.0,
        'clen_limiter': 0.5,
        'dr_0': 0.3,
        'dr_1': 0.05,
        'lim_sep': 1.0,
        'SOL_decay': 0.05, 
        # -------------------
        'saw_model': {'saw_flag': False,
                      'rmix': 1000.0,
                      'times': [1.0,], 
                      'sawtooth_erfc_width' : 1.0},
        # --------------------
        # edge/recycling
        'recycling_flag' : False,
        'wall_recycling' : 0.0, 
        'divbls' : 0.0,
        'tau_div_SOL_ms' : 50.0,  # ms
        'tau_pump_ms' : 500.0,  # ms
        'tau_rcl_ret_ms' : 50.0,   # ms
        'SOL_mach': 0.1,
        # --------------------
        # kinetic profiles
        'average_kin_profs': True,
        'kin_profs' : {'ne': {'fun': 'interpa',
                              'times': [1.]}, 
                       'Te': {'fun':'interp',
                              'times': [1.], 'decay': [1.]},
                       'Ti': {'fun':'interp',
                              'times': [1.], 'decay': [1.]},
                       'n0': {'fun':'interpa',
                              'times': [1.]}
                   },
        # ----------------
        # flags
        'cxr_flag' : False,
        'nbi_cxr_flag' : False,
        'prad_flag' : False,
        'thermal_cx_rad_flag' : False,
        'spectral_brem_flag' : False,
        'sxr_flag' : False,
        'main_ion_brem_flag' : False,
    }
    return namelist
