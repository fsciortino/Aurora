#-*-Python-*-
# Created by sciortino & odstrcil

import scipy.io
import copy,os,sys
import numpy as np
from scipy.interpolate import interp1d
from IPython import embed
from flib import _aurora
from . import interp
from . import atomic
from . import grids_utils
from . import source_utils


def interp_kin_prof(prof, kin_prof, time_grid, rhop_grid, radius_grid):
    ''' Interpolate the given kinetic profile on the provided radial and temporal grids [units of s].
    This function extrapolates in the SOL based on input options using the same methods as
    used in STRAHL.
    '''
    times = kin_prof[prof]['times']

    r_lcfs = np.interp(1,rhop_grid,radius_grid)

    #extrapolate profiles outside of LCFS by exponential decays
    r = interp1d(rhop_grid, radius_grid,fill_value='extrapolate')(kin_prof[prof]['rhop'])
    if kin_prof[prof]['fun'] == 'interp':
        data = interp.interp_quad(r/r_lcfs,kin_prof[prof]['vals'],
                                         kin_prof[prof]['decay'],r_lcfs,radius_grid)
        data[data < 1.01] = 1
    elif kin_prof[prof]['fun'] == 'interpa':
        data = interp.interpa_quad(r/r_lcfs,kin_prof[prof]['vals'],r_lcfs,radius_grid)

    #linear interpolation in time
    if len(times) > 1:  # time-dept
        data = interp1d(times,data,axis=0)(np.clip(time_grid,*times[[0,-1]]))

    return data


def get_aurora_kin_profs(namelist, time_grid, rhop_grid, radius_grid,min_T=1.01):
    ''' Get kinetic profiles on given radial and time grids.  '''

    # get ne, Te (and Ti, if available) on rhop_grid
    kin_profs = namelist['kin_profs']

    Te = interp_kin_prof('Te', kin_profs, time_grid, rhop_grid, radius_grid)
    ne = interp_kin_prof('ne', kin_profs, time_grid, rhop_grid, radius_grid)
    if 'Ti' in kin_profs:
        Ti = interp_kin_prof('Ti', kin_profs, time_grid, rhop_grid, radius_grid)
    else:
        Ti = Te

    # get neutral background ion density
    if namelist.get('cxr_flag',False):
        n0 = interp_kin_prof('n0', kin_profs, time_grid, rhop_grid, radius_grid)
    else:
        n0 = None

    #STRAHL assumes that T cannot be less than 1eV.
    Te[Te < min_T] = min_T
    Ti[Ti < min_T] = min_T
    ne[ne < min_T] = min_T

    # make sure that Te,ne have the same shape at this stage
    ne,Te,Ti = np.broadcast_arrays(ne,Te,Ti)

    return ne,Te,Ti,n0


def get_time_dept_atomic_rates(namelist, time_grid, ne,Te,Ti,n0, nbi_cxr=None):
    ''' Obtain time-dependent ionization and recombination rates for a simulation run.
    If kinetic profiles are given as time-independent, atomic rates for each time slice
    will be set to be the same.
    '''
    # get directory name containing atomic data
    imp = namelist['imp']

    lne = np.log10(ne)
    lTe = np.log10(Te)

    # get TIME-DEPENDENT atomic rates
    atom_data = atomic.get_all_atom_data(imp,['acd','scd'])

    # get electron impact ionization and radiative recombination rates in units of [s^-1]
    S_rates = atomic.interp_atom_prof(atom_data['scd'],lne, lTe)
    alpha_rates = atomic.interp_atom_prof(atom_data['acd'],lne, lTe)

    # define effective recombination rate as R:
    R_rates = copy.deepcopy(alpha_rates)
    if namelist['cxr_flag']:
        # include thermal charge exchange recombination
        atom_data = atomic.get_all_atom_data(imp,['ccd'])

        lTi = np.log10(Ti)
        alpha_CX_rates = atomic.interp_atom_prof(atom_data['ccd'], lne, lTi,
                                                        x_multiply=False)

        # change rates from units of [1/s/cm^3] to [1/s] ---> this is the result of STRAHL's `sas' subroutine
        R_rates += n0[:,None] * alpha_CX_rates[:,:alpha_rates.shape[1],:]   # select only relevant CCD ion stages (useful for Foster scaling)

    if namelist['nbi_cxr_flag']:
        # include charge exchange between NBI neutrals and impurities

        # STEPS (external to this package)
        # get density of fast NBI neutrals (both fast and thermal/halo) ---> n0_nbi, n0_halo
        # get total rates (n-unresolved) for CX with NBI neutrals --> _alpha_CX_NBI_rates
        # thermal rates for the halo are the ccd ones used for the edge
        # sum n0_nbi *  alpha_CX_NBI_rates + n0_halo * alpha_CX_rates

        R_rates += nbi_cxr.transpose(1,0,2)

    # nz=nion of rates arrays must be filled with zeros - final shape: (nr,nion,nt)
    S_rates = np.append(S_rates, np.zeros_like(S_rates[:,[0]]),axis=1).T
    R_rates = np.append(R_rates, np.zeros_like(R_rates[:,[0]]),axis=1).T

    # broadcast in the requested shape
    S_rates,R_rates,_ = np.broadcast_arrays(S_rates,R_rates,time_grid[None, None, :])


    return S_rates, R_rates





def get_parallel_loss(namelist, radius_grid, Te, Ti, time, trust_SOL_Ti=False):
    '''
    Calculate the parallel loss frequency on the radial and temporal grids [1/s].

    trust_SOL_Ti should generally be set to False, unless specific Ti measurements are available
    in the SOL.
    ***
    '''
    # background mass (=2 for D)
    apl = float(namelist['a'])

    # factor for v = machnumber * sqrt((3T_i+T_e)k/m)
    vpf = namelist['SOL_mach']*np.sqrt(1.602e-19/1.601e-27/apl)  # v[m/s]=vpf*sqrt(T[ev])

    # number of points inside of LCFS
    ids = radius_grid.searchsorted(namelist['rvol_lcfs'],side='left')
    idl = radius_grid.searchsorted(namelist['rvol_lcfs']+namelist['lim_sep'],side='left')

    # Calculate parallel loss frequency using different connection lengths in the SOL and in the limiter shadow
    dv = np.zeros_like(Te.T) # space x time

    if not trust_SOL_Ti:
        # Ti is not reliable in SOL, replace it by Te
        Ti = Te

    # open SOL
    dv[ids:idl] = vpf*np.sqrt(3.*Ti.T[ids:idl] + Te.T[ids:idl])/namelist['clen_divertor']

    # limiter shadow
    dv[idl:] = vpf*np.sqrt(3.*Ti.T[idl:] + Te.T[idl:])/namelist['clen_limiter']

    dv,_ = np.broadcast_arrays(dv,time[None])

    return np.asfortranarray(dv)






def aurora_setup(namelist, geqdsk=None):
    ''' Setup simulation input dictionary from the given namelist.

    The `geqdsk' argument should contain the EFIT gfile after postprocessing as in the
    omfit_eqdsk package (OMFITgeqdsk class). If left to None (default), the geqdsk dictionary
    is constructed starting from the gfile in the MDS+ tree.
    '''
    aurora_dict = {}
    aurora_dict['imp'] = imp = namelist['imp']

    if geqdsk is None:
        # Fetch CMOD geqdsk from MDS+ and post-process it using the OMFIT geqdsk format.
        # This requires the omfit-eqdsk Python package to be installed (pip install omfit-eqdsk).
        try:
                import omfit_eqdsk
        except:
                raise ValueError('Could not import omfit_eqdsk! Install with pip install omfit_eqdsk')
        geqdsk = omfit_eqdsk.OMFITgeqdsk('').from_mdsplus(device=namelist['device'],shot=namelist['shot'],
                                                          time=namelist['time'], SNAPfile='EFIT01',
                                                          fail_if_out_of_range=False,time_diff_warning_threshold=20)

    # Get r_V to rho_pol mapping
    rho_pol, r_V_ = grids_utils.get_rhopol_rV_mapping(geqdsk)
    rvol_lcfs = interp1d(rho_pol,r_V_)(1.0)
    rvol_lcfs = np.round(rvol_lcfs,1)
    aurora_dict['rvol_lcfs'] = namelist['rvol_lcfs'] = rvol_lcfs   # store for use in other functions

    # create radial grid
    out = grids_utils.create_aurora_radial_grid(namelist, plot=False)
    aurora_dict['radius_grid'], aurora_dict['pro'], aurora_dict['prox'], aurora_dict['qpr'] = out
    radius_grid = aurora_dict['radius_grid']

    # get rho_poloidal grid corresponding to aurora internal (r_V) grid
    aurora_dict['rhop_grid'] = interp1d(r_V_,rho_pol)(radius_grid)

    # Save R on LFS and HFS like STRAHL does -- useful for some diagnostic synthetic diagnostics
    Rhfs, Rlfs = grids_utils.get_HFS_LFS(geqdsk, rho_pol_arb=aurora_dict['rhop_grid'])
    aurora_dict['large_radius_lfs'] = copy.deepcopy(Rlfs)
    aurora_dict['large_radius_hfs'] = copy.deepcopy(Rhfs)

    # define time grid ('timing' must be in namelist)
    comp_time,  save_time = grids_utils.create_aurora_time_grid(timing=namelist['timing'], plot=False)
    aurora_dict['time'] = comp_time
    aurora_dict['time_out'] = comp_time[save_time]
    aurora_dict['save_time'] = save_time

    # get kinetic profiles on the radial and temporal grids
    ne,Te,Ti,n0 = get_aurora_kin_profs(namelist, comp_time,
                                         aurora_dict['rhop_grid'],radius_grid)

    # cache kinetic profiles for radiation postprocessing
    aurora_dict['ne'] = ne
    aurora_dict['Te'] = Te
    aurora_dict['Ti'] = Ti
    aurora_dict['n0'] = n0

    # Get time-dependent parallel loss rate
    aurora_dict['parallel_loss_rate'] = get_parallel_loss(namelist, radius_grid,Te,Ti,comp_time)

    # Obtain atomic rates on the computational time and radial grids
    s_t, al_t = get_time_dept_atomic_rates(namelist,comp_time,ne,Te,Ti,n0)
    aurora_dict['s_t'], aurora_dict['al_t'] = np.asfortranarray(s_t),np.asfortranarray(al_t)

    # get radial profile of source function
    aurora_dict['sint'] = source_utils.get_radial_source(namelist, radius_grid, s_t[:,0],
                                              aurora_dict['pro'],Ti)

    # mixing radius:
    aurora_dict['mixing_radius'] = namelist['saw_model']['rmix']

    # decay length at the wall boundary
    aurora_dict['decay_length_boundary'] = namelist['SOL_decay']

    # create array of 0's of length equal to comp_time, with 1's where sawteeth must be triggered
    aurora_dict['saw_on'] = np.zeros_like(comp_time)
    saw_times = np.array(namelist['saw_model']['times'])[namelist['saw_model']['times']<comp_time[-1]]
    if namelist['saw_model']['saw_flag'] and len(namelist['saw_model']['times'])>0:
        aurora_dict['saw_on'][comp_time.searchsorted(saw_times)] = 1

    aurora_dict['large_radius'] = namelist['Raxis']*100 #cm

    # source function
    aurora_dict['source_function'] = source_utils.get_aurora_source(namelist, comp_time)

    aurora_dict['Z_imp'] = Z_imp = int(namelist['Z_imp'])

    # initial impurity density array -- set to 0's before source function begins
    aurora_dict['impurity_density'] = np.zeros((len(radius_grid), Z_imp+1 ))

    for key in ['wall_recycling',  'divbls', 'tau_div_SOL', 'tau_pump', 'tau_rcl_ret']:
        aurora_dict[key] = namelist[key]

    # confusing STRAHL nomenclature: recycling_flag enables both recycling and divertor return flows
    if not namelist['recycling_flag']:
        aurora_dict['wall_recycling'] = -1.0  # no divertor return flows

    aurora_dict['bound_sep'] = namelist['bound_sep']
    aurora_dict['lim_sep'] = namelist['lim_sep']
    aurora_dict['sawtooth_erfc_width'] = namelist['saw_model']['sawtooth_erfc_width']
    aurora_dict['Z'] = namelist['Z']
    aurora_dict['a'] = namelist['a']
    aurora_dict['cxr_flag'] = namelist['cxr_flag']
    aurora_dict['nbi_cxr_flag'] = namelist['nbi_cxr_flag']

    return aurora_dict







def run_aurora(aurora_dict, times_DV, D_z, V_z, nz_init=None, method='old',evolneut=False):
    ''' Run a simulation using inputs in the given dictionary and D,v profiles as a function
    of space, time and potentially also ionization state. Users may give an initial state of each
    ion charge state as an input.

    Results can be conveniently visualized with time-slider using

    >> aurora.pylib.slider_plot(rhop,time, nz.transpose(1,2,0),   # charge states in first dimension
                               xlabel=r'$\rho_p$', ylabel='time [s]', zlabel=r'$n_z$ [cm$^{-3}$]', plot_sum=True,
                               labels=[f'Ca$^{{{str(i)}}}$' for i in np.arange(nz_w.shape[1]])
    INPUTS
    -------------
    aurora_dict : dict
        Dictionary containing saved initialization arrays, obtained via the 'aurora_setup'.
    times_DV : 1D array
        Array of times at which D_z and V_z profiles are given. (Note that it is assumed that
        D and V profiles are already on the aurora_dict['radius_grid'] radial grid).
    D_z, V_z: arrays, shape of (space, time,nZ) or (space,time)
        Diffusion and convection coefficients, in units of cm^2/s and cm/s, respectively.
        This may be given as a function of (space,time) or (space,nZ, time), where nZ indicates
        the number of charge states. If inputs are found to be have only 2 dimensions, it is
        assumed that all charge states should be set to have the same transport coefficients.
    nz_init: array, shape of (space, nZ)
        Impurity charge states at the initial time of the simulation. If left to None, this is
        internally set to an array of 0's.
    method : str, optional
        If method='linder', use the Linder algorithm for increased stability and accuracy.
    evolneut : bool, optional
        If True, evolve neutral impurities based on their D,V coefficients. Default is False, in
        which case neutrals are only taken as a source and those that are not ionized immediately after
        injection are neglected.

    OUTPUTS:
    -------------
    out : list
        List containing each particle reservoir of a simulation, i.e.
        nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out
    '''
    if nz_init is None:
        # default: start in a state with no impurity ions
        nz_init = np.zeros((len(aurora_dict['radius_grid']),int(aurora_dict['Z_imp'])+1))

    if D_z.ndim==2:
        # set all charge states to have the same transport
        D_z = np.tile(np.atleast_3d(D_z),(1,1,aurora_dict['Z_imp']+1))  # include elements for neutrals
        # unless specified, D_z for neutrals should be 0
        D_z[:,:,0] = 0.0
    if V_z.ndim==2:
        V_z = np.tile(np.atleast_3d(V_z),(1,1,aurora_dict['Z_imp']+1))  # include elements for neutrals
        # unless specified, V_z for neutrals should be 0
        V_z[:,:,0] = 0.0

    # number of times at which simulation outputs results
    nt_out = len(aurora_dict['time_out'])

    #NOTE: use only f_contiguous arrays for speed!
    return _aurora.run(
        nt_out,
        times_DV,
        D_z, # cm^2/s      #(ir,nt_trans,nion)
        V_z, # cm/s
        aurora_dict['parallel_loss_rate'],  # time dependent
        aurora_dict['sint'],# source profile
        aurora_dict['s_t'], # ioniz_rate,
        aurora_dict['al_t'], # recomb_rate,
        aurora_dict['radius_grid'],
        aurora_dict['pro'],
        aurora_dict['qpr'],
        aurora_dict['mixing_radius'],
        aurora_dict['decay_length_boundary'],
        aurora_dict['time'],
        aurora_dict['saw_on'],
        aurora_dict['source_function'],
        aurora_dict['save_time'],
        aurora_dict['sawtooth_erfc_width'], # dsaw width  [cm, circ geometry]
        aurora_dict['wall_recycling'], # rcl   [fraction]
        aurora_dict['divbls'], # divbls   [fraction of source into divertor]
        aurora_dict['tau_div_SOL'] * 1e-3, # taudiv   [s]
        aurora_dict['tau_pump'] *1e-3, # taupump  [s]
        aurora_dict['tau_rcl_ret'] *1e-3,   # tauwret  [s]
        aurora_dict['rvol_lcfs'],       # rx = rvol_lcfs + dbound
        aurora_dict['bound_sep'],
        aurora_dict['lim_sep'],
        aurora_dict['prox'],
        rn_t0 = nz_init,  # if omitted, internally set to 0's
        linder=True if method=='linder' else False,
        evolneut=evolneut
    )





def run_julia(pystrahl_dict, times_DV, D_z, V_z, nz_init=None):
    ''' Run a single simulation using the Julia version.

    INPUTS
    -------------
    pystrahl_dict : dict
        Dictionary containing saved initialization arrays, obtained via `pySTRAHL_setup`.
    times_DV : 1D array
        Array of times at which D_z and V_z profiles are given. (Note that it is assumed that
        D and V profiles are already on the pystrahl_dict['radius_grid'] radial grid).
    D_z, V_z: arrays, shape of (space, nZ, time)
        Diffusion and convection coefficients, in units of cm^2/s and cm/s, respectively.
        This may be given as a function of (space,time) or (space,nZ, time), where nZ indicates
        the number of charge states. If inputs are found to be have only 2 dimensions, it is
        assumed that all charge states should be set to have the same transport coefficients.
    nz_init: array, shape of (space, nZ)
        Impurity charge states at the initial time of the simulation. If left to None, this is
        internally set to an array of 0's.

    OUTPUTS:
    -------------
    out : list
        List containing each particle reservoir of a simulation, i.e.
        nz, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out
    '''

    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import aurora as aurora_jl

    if nz_init is None:
        # default: start in a state with no impurity ions
        nz_init = np.zeros((len(pystrahl_dict['radius_grid']),int(pystrahl_dict['Z_imp'])+1))

    if D_z.ndim==2:
            # set all charge states to have the same transport
            D_z = np.tile(np.atleast_3d(D_z),(1,1,pystrahl_dict['Z_imp']+1))  # include elements for neutrals
    if V_z.ndim==2:
            V_z = np.tile(np.atleast_3d(V_z),(1,1,pystrahl_dict['Z_imp']+1))  # include elements for neutrals

    # number of times at which simulation outputs results
    nt_out = len(pystrahl_dict['time_out'])

    #NOTE: use only f_contiguous arrays for speed!
    return aurora_jl.run(nt_out,
    times_DV,
    D_z,
    V_z, # cm/s
    pystrahl_dict['parallel_loss_rate'],  # time dependent
    pystrahl_dict['sint'],# source profile
    pystrahl_dict['s_t'], # ioniz_rate,
    pystrahl_dict['al_t'], # recomb_rate,
    pystrahl_dict['radius_grid'],
    pystrahl_dict['pro'],
    pystrahl_dict['qpr'],
    pystrahl_dict['mixing_radius'],
    pystrahl_dict['decay_length_boundary'],
    pystrahl_dict['time'],
    pystrahl_dict['saw_on'],
    pystrahl_dict['source_function'],
    pystrahl_dict['save_time'],
    pystrahl_dict['sawtooth_erfc_width'], # dsaw width  [cm, circ geometry]
    pystrahl_dict['wall_recycling'], # rcl   [fraction]
    pystrahl_dict['divbls'], # divbls   [fraction of source into divertor]
    pystrahl_dict['tau_div_SOL'] * 1e-3, # taudiv   [s]
    pystrahl_dict['tau_pump'] *1e-3, # taupump  [s]
    pystrahl_dict['tau_rcl_ret'] *1e-3,   # tauwret  [s]
    pystrahl_dict['rvol_lcfs'],       # rx = rvol_lcfs + dbound
    pystrahl_dict['bound_sep'],
    pystrahl_dict['lim_sep'],
    pystrahl_dict['prox'],
    nz_init)
