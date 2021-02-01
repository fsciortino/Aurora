"""
Aurora functionality to set up and run KN1D to extract atomic and neutral 
background densities at the edge. 

sciortino, January 2021
"""

from scipy.interpolate import interp1d
import numpy as np
import os
import scipy.io
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

from . import neutrals
from . import coords

thisdir = os.path.dirname(os.path.realpath(__file__))

def _setup_kin_profs(rhop, ne_cm3, Te_eV, Ti_eV, 
                     geqdsk, bound_sep_cm, lim_sep_cm,
                     kin_prof_exp_decay_SOL=False, kin_prof_exp_decay_LS=False,
                     ne_decay_len_cm=1.0, Te_decay_len_cm=1.0, Ti_decay_len_cm=1.0, 
                     ne_min_cm3=1e12, Te_min_eV=1.0, Ti_min_eV=1.0):
    '''Private method to set up kinetic profiles to the format required by 
    :py:fun:`~aurora.kn1d.run_kn1d`. Refer to this function for descriptions of inputs.
    
    This function returns ne, Te and Ti profiles on the rmid_to_wall_cm radial grid, 
    from the core to the wall. 

    Returns:
        rmid_to_wall_cm : 1D array
            Midradius coordinate from the magnetic axis to the wall. Units of [:math:`cm`].
        ne : 1D array
            Electron density [:math:`cm^{-3}`] on the rmid_to_wall_cm grid.
        Te : 1D array
            Electron temperature [:math:`eV`] on the rmid_to_wall_cm grid.
        Ti : 1D array
            Main ion temperature [:math:`eV`] on the rmid_to_wall_cm grid.
    '''

    ne_m3 = ne_cm3 * 1e6  # cm^-3 --> m^-3

    # convert radial coordinate to rmid
    rmid = coords.rad_coord_transform(rhop, 'rhop', 'rmid', geqdsk)

    # define radial regions in the SOL in coordinates centered on the mag axis
    rsep = coords.rad_coord_transform(1.0, 'rhop', 'rmid', geqdsk)
    rwall = rsep + bound_sep_cm * 1e-2  # cm-->m
    rlim = rsep + lim_sep_cm * 1e-2  # cm-->m

    # interpolate profiles on grid extending to wall
    rmid_to_wall_cm = np.linspace(np.min(rmid), rwall, 1001)  # 101) #201)
    _ne = interp1d(rmid, ne_m3, bounds_error=False)(rmid_to_wall_cm)  # extrapolates to nan
    _Te = interp1d(rmid, Te_eV, bounds_error=False)(rmid_to_wall_cm)  # extrapolates to nan
    _Ti = interp1d(rmid, Ti_eV, bounds_error=False)(rmid_to_wall_cm)  # extrapolates to nan

    indLCFS = np.searchsorted(rmid_to_wall_cm, rsep)
    indLS = np.searchsorted(rmid_to_wall_cm, rlim)
    ind_end = np.searchsorted(rmid_to_wall_cm, rmid[-1])

    # if kinetic profiles don't extend far enough in radius, we must set an exp decay depending on the radial region
    if ind_end < indLS:
        # decays in SOL (all the way to the wall)
        ne_sol = _ne[ind_end-1] * np.exp(-(rmid_to_wall_cm[ind_end:]-rmid_to_wall_cm[ind_end-1])/(ne_decay_len_cm[0]/100.0))
        ne_ = np.concatenate((_ne[:ind_end], ne_sol))
        Te_sol = _Te[ind_end-1] * np.exp(-(rmid_to_wall_cm[ind_end:]-rmid_to_wall_cm[ind_end-1])/(Te_decay_len_cm[0]/100.0))
        Te_ = np.concatenate((_Te[:ind_end], Te_sol))
        Ti_sol = _Ti[ind_end-1] * np.exp(-(rmid_to_wall_cm[ind_end:]-rmid_to_wall_cm[ind_end-1])/(Ti_decay_len_cm[0]/100.0))
        Ti_ = np.concatenate((_Ti[:ind_end], Ti_sol))
    else:
        ne_ = copy.deepcopy(_ne)
        Te_ = copy.deepcopy(_Te)
        Ti_ = copy.deepcopy(_Ti)

    if ind_end < len(rmid_to_wall_cm):
        # decays in the LS
        ne_ls = ne_[ind_end-1] * np.exp(-(rmid_to_wall_cm[ind_end:]-rmid_to_wall_cm[ind_end-1])/(ne_decay_len_cm[1]/100.0))
        ne = np.concatenate((ne_[:ind_end], ne_ls))
        Te_ls = Te_[ind_end-1] * np.exp(-(rmid_to_wall_cm[ind_end:]-rmid_to_wall_cm[ind_end-1])/(Te_decay_len_cm[1]/100.0))
        Te = np.concatenate((Te_[:ind_end], Te_ls))
        Ti_ls = Ti_[ind_end-1] * np.exp(-(rmid_to_wall_cm[ind_end:]-rmid_to_wall_cm[ind_end-1])/(Ti_decay_len_cm[1]/100.0))
        Ti = np.concatenate((Ti_[:ind_end], Ti_ls))
    else:
        ne = copy.deepcopy(ne_)
        Te = copy.deepcopy(Te_)
        Ti = copy.deepcopy(Ti_)


    # User may want to set exp decay in SOL or LS in place of dubious experimental data
    if kin_prof_exp_decay_SOL:
        # decays in the SOL
        ne[indLCFS:indLS] = ne[indLCFS - 1] * np.exp(
            -(rmid_to_wall_cm[indLCFS:indLS] - rmid_to_wall_cm[indLCFS-1])/(ne_decay_len_cm[0]/100.0)
        )
        Te[indLCFS:indLS] = Te[indLCFS - 1] * np.exp(
            -(rmid_to_wall_cm[indLCFS:indLS] - rmid_to_wall_cm[indLCFS-1])/(Te_decay_len_cm[0]/100.0)
        )
        Ti[indLCFS:indLS] = Ti[indLCFS - 1] * np.exp(
            -(rmid_to_wall_cm[indLCFS:indLS] - rmid_to_wall_cm[indLCFS-1])/(Ti_decay_len_cm[0]/100.0)
        )

    if kin_prof_exp_decay_LS:
        # decays in the LS
        ne[indLS:] = ne[indLS-1] * np.exp(-(rmid_to_wall_cm[indLS:]-rmid_to_wall_cm[indLS-1])/(ne_decay_len_cm[1]/100.0))
        Te[indLS:] = Te[indLS-1] * np.exp(-(rmid_to_wall_cm[indLS:]-rmid_to_wall_cm[indLS-1])/(Te_decay_len_cm[1]/100.0))
        Ti[indLS:] = Ti[indLS-1] * np.exp(-(rmid_to_wall_cm[indLS:]-rmid_to_wall_cm[indLS-1])/(Ti_decay_len_cm[1]/100.0))
    
    # set minima across radial profiles
    ne[ne < ne_min_cm3] = ne_min_cm3
    Te[Te < Te_min_eV] = Te_min_eV
    Ti[Ti < Ti_min_eV] = Ti_min_eV

    return rmid_to_wall_cm, ne, Te, Ti




def run_kn1d(rhop, ne_cm3, Te_eV, Ti_eV, geqdsk, p_H2_mTorr, 
             clen_divertor_cm, clen_limiter_cm, bound_sep_cm, lim_sep_cm,
             innermost_rmid_cm=5.0, mu=2., pipe_diag_cm=0.0, vx=0.0, collisions={},
             kin_prof_exp_decay_SOL=False, kin_prof_exp_decay_LS=False,
             ne_decay_len_cm=[1.0,1.0], Te_decay_len_cm=[1.0,1.0], Ti_decay_len_cm=[1.,1.], 
             ne_min_cm3=1e12, Te_min_eV=1.0, Ti_min_eV=1.0):
    '''Run KN1D for the given parameters. Refer to the KN1D manual for details. 

    Depending on the provided options, kinetic profiles are extended beyond the Last Closed
    Flux Surface (LCFS) and the Limiter Shadow (LS) via exponential decays with specified 
    decay lengths. It is assumed that the given kinetic profiles extend from the core until
    at least the LCFS. All inputs are taken to be time-independent.

    This function automatically checks if a KN1D repository is available; if it is not,
    it obtains it from the web and compiles the necessary code. 

    Note that an IDL license must be available. Aurora does not currently include a Python
    translation of KN1D -- it only acts as a wrapper.

    Args:
    rhop : 1D array
        Sqrt of poloidal flux grid on which ne_cm3, Te_eV and Ti_eV are given.
    ne_cm3 : 1D array
        Electron density on rhop grid [:math:`cm^{-3}`].
    Te_eV : 1D array
        Electron temperature on rhop grid [:math:`eV`].
    Ti_eV : 1D array
        Main ion temperature on rhop grid [:math:`eV`].
    geqdsk : `omfit_eqdsk.OMFITgeqdsk` class instance
        gEQDSK file as processed by the `omfit_eqdsk.OMFITgeqdsk` class.
    p_H2_mTorr : float
        Pressure of molecular hydrogen-isotopes measured at the wall. This may be estimated
        from experimental pressure gauges. This variable effectively sets the amplitude of the 
        neutral source at the edge. Units of :math:`mTorr`.
    clen_divertor_cm : float
        Connection length from the midplane to the divertor [:math:`cm`].
    clen_limiter_cm : float
        Connection length from the midplane to the limiter [:math:`cm`].
    bound_sep_cm : float
        Distance between the wall/boundary and the separatrix [:math:`cm`].
    lim_sep_cm : float
        Distance between the limiter and the separatrix [:math:`cm`].

    Keyword Args:
    innermost_rmid_cm : float
        Distance from the wall to solve for. Default is 5 cm. 
    mu : float
        Atomic mass number of simulated species. Default is 2.0 (D). 
    pipe_diag_cm : float
        Diameter of the pipe through which H2 pressure is measured (see `p_H2_mTorr` variable). 
        If left to 0, this diameter is effectively set to infinity. Default is 0. 
    vx : float
        Radial velocity imposed on neutrals. This only has a weak effect usually. 
        Default is 0 [:math:`cm/s`].
    collisions : dict
        Collision terms flags. Set each to True or False. If any of the flags are not given, 
        all collision terms are internally set to be active. Possible flags are
        'H2_H2_EL','H2_P_EL','H2_H_EL','H2_HP_CX','H_H_EL','H_P_CX','H_P_EL','Simple_CX'
    kin_prof_exp_decay_SOL : bool
        If True, kinetic profiles are set to exponentially decay over the SOL region.
    kin_prof_exp_decay_LS : bool
        If True, kinetic profiles are set to exponentially decay over the LS region.
    ne_decay_len_cm : list of 2 float
        Exponential decay lengths of electron density in the SOL and LS regions. 
        Default is [1,1] :math:`cm`.
    Te_decay_len_cm : float
        Exponential decay lengths of electron temperature in the SOL and LS regions. 
        Default is [1,1] :math:`cm`.    
    Ti_decay_len_cm : float
        Exponential decay lengths of main ion temperature in the SOL and LS regions. 
        Default is [1,1] :math:`cm`.
    ne_min_cm3 : float
        Minimum electron density across profile. Default is :math:`10^{12} cm^{-3}`.
    Te_min_eV : float
        Minimum electron temperaure across profile. Default is :math:`eV`.
    Ti_min_eV : float
        Minimum main ion temperaure across profile. Default is :math:`eV`.
    
    For an example application, see the examples/aurora_kn1d.py script. 
    '''
    
    if 'IDL_STARTUP' not in os.environ:
        raise ValueError('An IDL installation does not seem to be available! KN1D cannot be run.')

    cwd = os.getcwd()
    
    # make sure that the KN1D source code is accessible. 
    if 'KN1D' not in os.listdir(thisdir):
        #if 'KN1D_DIR' not in os.environ:
        # git clone the KN1D repository
        os.system(f'git clone https://github.com/fsciortino/kn1d {thisdir}/KN1D')

        os.chdir(f'{thisdir}/KN1D')
        # compile fortran libraries
        print(f'export KN1D_DIR={thisdir}/KN1D; make clean; make')
        os.system(f'export KN1D_DIR={thisdir}/KN1D; make clean; make')
        os.chdir(cwd)
        #else:
        # copy KN1D directory locally
        #shutil.copytree(os.environ['KN1D_DIR'],thisdir+'/KN1D')
    else:
        # KN1D directory already available, assumed to be already built
        # NB: users need to have write-access to this directory!
        pass

    kn1d={}
    kn1d.update(collisions)
    if 'H2_H2_EL' not in kn1d:
        kn1d['H2_H2_EL'] = True
    if 'H2_P_EL' not in kn1d:
        kn1d['H2_P_EL'] = True
    if 'H2_H_EL' not in kn1d:
        kn1d['H2_H_EL'] = True
    if 'H2_HP_CX' not in kn1d:
        kn1d['H2_HP_CX'] = True
    if 'H_H_EL' not in kn1d:
        kn1d['H_H_EL'] = True
    if 'H_P_CX' not in kn1d:
        kn1d['H_P_CX'] = True
    if 'H_P_EL' not in kn1d:
        kn1d['H_P_EL'] = True
    if 'Simple_CX' not in kn1d:
        kn1d['Simple_CX'] = False


    rmid_to_wall_cm, ne,Te,Ti = _setup_kin_profs(rhop, ne_cm3, Te_eV, Ti_eV, 
                                              geqdsk, bound_sep_cm, lim_sep_cm,
                                              kin_prof_exp_decay_SOL, kin_prof_exp_decay_LS,
                                              ne_decay_len_cm, Te_decay_len_cm, Ti_decay_len_cm, 
                                              ne_min_cm3, Te_min_eV, Ti_min_eV)
    
    rhop = coords.rad_coord_transform(rmid_to_wall_cm, 'rmid', 'rhop', geqdsk)
    rwall = rmid_to_wall_cm[-1]
    rsep = coords.rad_coord_transform(1.0, 'rhop', 'rmid', geqdsk)
    rlim = rsep + lim_sep_cm * 1e-2  # m


    # KN1D defines coordinates from the wall INWARD. Invert now:
    r_kn1d = np.abs(rmid_to_wall_cm - rwall)[::-1]
    rlim_kn1d = np.abs(rlim - rwall)
    rsep_kn1d = np.abs(rsep - rwall)

    # diameter of pressure gauge pipe. Allows collisions with side-walls to be simulated
    dPipe = pipe_diag_cm*np.ones(len(rmid_to_wall_cm))*1e-2  # m  -- zero values are treated as infinity

    # define the connection length vector
    lc = np.zeros(len(rmid_to_wall_cm))
    lc[(rsep < rmid_to_wall_cm) * (rmid_to_wall_cm < rlim)] = clen_divertor_cm * 1e-2  # m
    lc[rmid_to_wall_cm > rlim] = clen_limiter_cm * 1e-2  # m


    def idl_array(arr, num):
        # set arrays into string form for IDL format
        return '[' + ','.join([str(round(val, num)) for val in arr]) + ']'
    
    def round_arr(arr, num, dtype=str):
        # avoid issues with floating-point precision
        return np.array([round(val, num) for val in arr], dtype=dtype)
    
    # cut all radial profiles to given innermost location
    ind_in = np.searchsorted(r_kn1d, innermost_rmid_cm * 1e-2)  # r_kn1d is from wall inwards
    
    # Save arrays in final forms for KN1D 
    num = 5  # digital point precision
    kn1d['x'] = round_arr(r_kn1d[:ind_in], num, float)
    ne_ = ne[::-1]
    kn1d['ne'] = round_arr(ne_[:ind_in], num, float)
    Te_ = Te[::-1]
    kn1d['Te'] = round_arr(Te_[:ind_in], num, float)
    Ti_ = Ti[::-1]
    kn1d['Ti'] = round_arr(Ti_[:ind_in], num, float)
    dPipe_ = dPipe[::-1]
    kn1d['dPipe'] = round_arr(dPipe_[:ind_in], num, float)
    lc_ = lc[::-1]
    kn1d['lc'] = round_arr(lc_[:ind_in], num, float)
    kn1d['xlim'] = rlim_kn1d
    kn1d['xsep'] = rsep_kn1d
    kn1d['p_H2_mTorr'] =  p_H2_mTorr  # mTorr
    kn1d['mu'] = mu
    

    # finally, set plasma radial velocity profile (negative is towards the wall [m s^-1])
    vx = np.ones_like(kn1d['x']) * vx
    kn1d['vx'] = round_arr(vx, num, float)

    # Create IDL script to run KN1D
    idl_cmd = '''
; Input data for KN1D run
x = {x:}
x_lim = {x_lim:.5f}
xsep = {xsep:.5f}
gaugeH2 = {gaugeH2:}
mu = {mu:}
Ti = {Ti:}
Te = {Te:}
dens = {ne:}
vx = {vx:}
lc = {lc:}
dPipe = {dPipe:}

; collisions options are set via a common block
common KN1D_collisions,H2_H2_EL,H2_P_EL,H2_H_EL,H2_HP_CX,H_H_EL,H_P_EL,H_P_CX,Simple_CX

H2_H2_EL= {H2_H2_EL:}
H2_P_EL = {H2_P_EL:}
H2_H_EL = {H2_H_EL:}
H_H_EL= {H_H_EL:}
H_P_CX = {H_P_CX:}
H_P_EL = {H_P_EL:}
Simple_CX = {Simple_CX:}

; now run KN1D
kn1d,x,x_lim,xsep,gaugeH2,mu,Ti,Te,dens,vx,lc,dPipe, xh2,nh2,gammaxh2,th2,qxh2_total,nhp,thp,sh,sp, xh,nh,gammaxh,th,qxh_total,nethsource,sion,qh_total,sidewallh,lyman,balmer,gammahlim

; save result to an IDL .sav file
save,xh2,nh2,gammaxh2,th2,qxh2_total,nhp,thp,sh,sp, xh,nh,gammaxh,th,qxh_total,nethsource,sion,qh_total,sidewallh,lyman,balmer,gammahlim,filename = "kn1d_out.sav"

exit

    '''.format(
        x=idl_array(kn1d['x'], num),
        x_lim=kn1d['xlim'],
        xsep=kn1d['xsep'],
        gaugeH2=kn1d['p_H2_mTorr'],
        mu=kn1d['mu'],
        Ti=idl_array(kn1d['Ti'], num),
        Te=idl_array(kn1d['Te'], num),
        ne=idl_array(kn1d['ne'], num),
        vx=idl_array(kn1d['vx'], num),
        lc=idl_array(kn1d['lc'], num),
        dPipe=idl_array(kn1d['dPipe'], num),
        H2_H2_EL=int(kn1d['H2_H2_EL']),
        H2_P_EL=int(kn1d['H2_P_EL']),
        H2_H_EL=int(kn1d['H2_H_EL']),
        H_H_EL=int(kn1d['H_H_EL']),
        H_P_CX=int(kn1d['H_P_CX']),
        H_P_EL=int(kn1d['H_P_EL']),
        Simple_CX=int(kn1d['Simple_CX']),
    )

    # write IDL file
    with open(f'{thisdir}/KN1D/new_kn1d_run.pro', 'w') as f:
        f.write(idl_cmd)

    # Run the script 
    os.system(f'cd {thisdir}/KN1D; idl new_kn1d_run.pro')

    #### store all KN1D data for postprocessing  #####
    res = {}
    out = res['out'] = scipy.io.readsav(f'{thisdir}/KN1D/kn1d_out.sav')
    res['kn1d_input'] = scipy.io.readsav(f'{thisdir}/KN1D/.KN1D_input')
    res['kn1d_mesh'] = scipy.io.readsav(f'{thisdir}/KN1D/.KN1D_mesh')
    res['kn1d_H2'] = scipy.io.readsav(f'{thisdir}/KN1D/.KN1D_H2')
    res['kn1d_H'] = scipy.io.readsav(f'{thisdir}/KN1D/.KN1D_H')

    os.chdir(cwd)

    # ---------------------------
    # Additional processed outputs

    # Compute ion flux by integrating over atomic ionization rate
    Sion = out['sion']
    Sion_interp = interp1d(out['xh'], Sion, bounds_error=False, fill_value=0.0)(kn1d['x'])
    out['Gamma_i'] = cumtrapz(Sion_interp, kn1d['x'], initial=0.0)

    # Effective diffusivity
    out['D_eff'] = np.abs(out['Gamma_i'] / np.gradient(kn1d['ne'], kn1d['x']))  # check

    # ensure that x bases are all to the same accuracy to avoid issues in interpolation
    out['xh'] = round_arr(out['xh'], num, dtype=float)
    out['xh2'] = round_arr(out['xh2'], num, dtype=float)

    # gradient length scales
    out['L_ne'] = np.abs(1.0 / np.gradient(np.log(kn1d['ne']), kn1d['x']))
    out['L_Te'] = np.abs(1.0 / np.gradient(np.log(kn1d['Te']), kn1d['x']))
    out['L_Ti'] = np.abs(1.0 / np.gradient(np.log(kn1d['Ti']), kn1d['x']))

    #### Calculate radial profiles of neutral excited states   ####

    # take KN1D neutral density profile to be the ground state (excited states are a small correction wrt this)
    N1 = interp1d(out['xh'], out['nh'], kind='linear')(kn1d['x'])

    # assume pure plasma and quasi-neutrality
    nhp_interp = interp1d(out['xh2'], out['nhp'], bounds_error=False, fill_value=0.0)(kn1d['x'])  # nH2+
    out['ni'] = kn1d['ne'] - nhp_interp

    # get profiles of excited states' density (only n=2 and n=3) ---- densities in cm^-3, Te in eV
    N2, N2_ground, N2_cont = neutrals.get_exc_state_ratio(
        m=2, N1=N1/1e6, ni=out['ni']/1e6, ne=kn1d['ne'] / 1e6, Te=kn1d['Te'], plot=False, rad_prof=kn1d['x']
    )
    N3, N3_ground, N3_cont = neutrals.get_exc_state_ratio(
        m=3, N1=N1/1e6, ni=out['ni']/1e6, ne=kn1d['ne'] / 1e6, Te=kn1d['Te'], plot=False, rad_prof=kn1d['x']
    )

    out['N2'] = N2 * 1e6  # transform back to m^-3
    out['N2_ground'] = N2_ground * 1e6
    out['N2_cont'] = N2_cont * 1e6
    out['N3'] = N3 * 1e6
    out['N3_ground'] = N3_ground * 1e6
    out['N3_cont'] = N3_cont * 1e6

    #################################
    # Store neutral density profiles in a format that can be used for integrated modeling
    out_profs = res['kn1d_profs'] = {}

    # save profiles on (inverted) grid extending all the way to the axis (extrapolating)
    _rhop = coords.rad_coord_transform(rwall - kn1d['x'][::-1], 'rmid', 'rhop', geqdsk)
    out_profs['rhop'] = np.linspace(0.0, 1.1, 200)

    out_profs['n0'] = np.exp(interp1d(_rhop, np.log(N1[::-1]), 
                                      bounds_error=False, fill_value='extrapolate')(out_profs['rhop']))
    out_profs['n0_n2'] = np.exp(interp1d(_rhop, np.log(out['N2'][::-1]), 
                                         bounds_error=False, fill_value='extrapolate')(out_profs['rhop']))
    out_profs['n0_n3'] = np.exp(interp1d(_rhop, np.log(out['N3'][::-1]), 
                                         bounds_error=False, fill_value='extrapolate')(out_profs['rhop']))

    # also save profiles of Ly- and H/D-alpha
    _rhop_emiss = coords.rad_coord_transform(rwall - out['xh'][::-1], 'rmid', 'rhop', geqdsk)
    out_profs['lyman'] = np.exp(
        interp1d(_rhop_emiss, np.log(out['lyman'][::-1]), 
                 bounds_error=False, fill_value='extrapolate')(out_profs['rhop'])
    )
    out_profs['balmer'] = np.exp(
        interp1d(_rhop_emiss, np.log(out['balmer'][::-1]), 
                 bounds_error=False, fill_value='extrapolate')(out_profs['rhop'])
    )
    
    return res
