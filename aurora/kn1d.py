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
from IPython import embed

#from aurora import neutrals, coords
from . import neutrals
from . import coords

thisdir = os.path.dirname(os.path.realpath(__file__))+'/../'

def _setup_kin_profs(rhop, ne_cm3, Te_eV, Ti_eV, 
                     geqdsk, bound_sep_cm, lim_sep_cm,
                     kin_prof_exp_decay_SOL=False, kin_prof_exp_decay_LS=False,
                     ne_decay_len_cm=1.0, Te_decay_len_cm=1.0, Ti_decay_len_cm=1.0, 
                     ne_min_cm3=1e12, Te_min_eV=1.0, Ti_min_eV=1.0):
    '''

    kin_prof_exp_decay_SOL : 
        User may want to set exp decay in SOL or LS in place of dubious experimental data
    kin_prof_exp_decay_LS :
        Don't trust data behind limiter shadow; replace it with exponential decay.

    '''

    ne_m3 = ne_cm3 * 1e6  # cm^-3 --> m^-3

    # convert radial coordinate to rmid
    rmid = coords.rad_coord_transform(rhop, 'rhop', 'rmid', geqdsk)

    # define radial regions in the SOL in coordinates centered on the mag axis
    rsep = coords.rad_coord_transform(1.0, 'rhop', 'rmid', geqdsk)
    rwall = rsep + bound_sep_cm * 1e-2  # cm-->m
    rlim = rsep + lim_sep_cm * 1e-2  # cm-->m

    # interpolate profiles on grid extending to wall
    rmid_to_wall = np.linspace(np.min(rmid), rwall, 1001)  # 101) #201)
    _ne = interp1d(rmid, ne_m3, bounds_error=False)(rmid_to_wall)  # extrapolates to nan
    _Te = interp1d(rmid, Te_eV, bounds_error=False)(rmid_to_wall)  # extrapolates to nan
    _Ti = interp1d(rmid, Ti_eV, bounds_error=False)(rmid_to_wall)  # extrapolates to nan

    indLCFS = np.searchsorted(rmid_to_wall, rsep)
    indLS = np.searchsorted(rmid_to_wall, rlim)
    ind_end = np.searchsorted(rmid_to_wall, rmid[-1])

    # if kinetic profiles don't extend far enough in radius, we must set an exp decay depending on the radial region
    if ind_end < indLS:
        # decays in SOL (all the way to the wall)
        ne_sol = _ne[ind_end-1] * np.exp(-(rmid_to_wall[ind_end:]-rmid_to_wall[ind_end-1])/(ne_decay_len_cm[0]/100.0))
        ne_ = np.concatenate((_ne[:ind_end], ne_sol))
        Te_sol = _Te[ind_end-1] * np.exp(-(rmid_to_wall[ind_end:]-rmid_to_wall[ind_end-1])/(Te_decay_len_cm[0]/100.0))
        Te_ = np.concatenate((_Te[:ind_end], Te_sol))
        Ti_sol = _Ti[ind_end-1] * np.exp(-(rmid_to_wall[ind_end:]-rmid_to_wall[ind_end-1])/(Ti_decay_len_cm[0]/100.0))
        Ti_ = np.concatenate((_Ti[:ind_end], Ti_sol))
    else:
        ne_ = copy.deepcopy(_ne)
        Te_ = copy.deepcopy(_Te)
        Ti_ = copy.deepcopy(_Ti)

    if ind_end < len(rmid_to_wall):
        # decays in the LS
        ne_ls = ne_[ind_end-1] * np.exp(-(rmid_to_wall[ind_end:]-rmid_to_wall[ind_end-1])/(ne_decay_len_cm[1]/100.0))
        ne = np.concatenate((ne_[:ind_end], ne_ls))
        Te_ls = Te_[ind_end-1] * np.exp(-(rmid_to_wall[ind_end:]-rmid_to_wall[ind_end-1])/(Te_decay_len_cm[1]/100.0))
        Te = np.concatenate((Te_[:ind_end], Te_ls))
        Ti_ls = Ti_[ind_end-1] * np.exp(-(rmid_to_wall[ind_end:]-rmid_to_wall[ind_end-1])/(Ti_decay_len_cm[1]/100.0))
        Ti = np.concatenate((Ti_[:ind_end], Ti_ls))
    else:
        ne = copy.deepcopy(ne_)
        Te = copy.deepcopy(Te_)
        Ti = copy.deepcopy(Ti_)


    # User may want to set exp decay in SOL or LS in place of dubious experimental data
    if kin_prof_exp_decay_SOL:
        # decays in the SOL
        ne[indLCFS:indLS] = ne[indLCFS - 1] * np.exp(
            -(rmid_to_wall[indLCFS:indLS] - rmid_to_wall[indLCFS-1])/(ne_decay_len_cm[0]/100.0)
        )
        Te[indLCFS:indLS] = Te[indLCFS - 1] * np.exp(
            -(rmid_to_wall[indLCFS:indLS] - rmid_to_wall[indLCFS-1])/(Te_decay_len_cm[0]/100.0)
        )
        Ti[indLCFS:indLS] = Ti[indLCFS - 1] * np.exp(
            -(rmid_to_wall[indLCFS:indLS] - rmid_to_wall[indLCFS-1])/(Ti_decay_len_cm[0]/100.0)
        )

    if kin_prof_exp_decay_LS:
        # decays in the LS
        ne[indLS:] = ne[indLS-1] * np.exp(-(rmid_to_wall[indLS:]-rmid_to_wall[indLS-1])/(ne_decay_len_cm[1]/100.0))
        Te[indLS:] = Te[indLS-1] * np.exp(-(rmid_to_wall[indLS:]-rmid_to_wall[indLS-1])/(Te_decay_len_cm[1]/100.0))
        Ti[indLS:] = Ti[indLS-1] * np.exp(-(rmid_to_wall[indLS:]-rmid_to_wall[indLS-1])/(Ti_decay_len_cm[1]/100.0))
    
    # set minima across radial profiles
    ne[ne < ne_min_cm3] = ne_min_cm3
    Te[Te < Te_min_eV] = Te_min_eV
    Ti[Ti < Ti_min_eV] = Ti_min_eV

    return rmid_to_wall, ne, Te, Ti




def run_kn1d(rhop, ne_cm3, Te_eV, Ti_eV, geqdsk, p_H2_mTorr, 
             clen_divertor_cm, clen_limiter_cm, bound_sep_cm, lim_sep_cm, rmid_to_wall_cm,
             mu=2., innermost_rmid_cm=5.0, pipe_diag_cm=0.0, vx=0.0, collisions={},
             kin_prof_exp_decay_SOL=False, kin_prof_exp_decay_LS=False,
             ne_decay_len_cm=[1.0,1.0], Te_decay_len_cm=[1.0,1.0], Ti_decay_len_cm=[1.,1.], 
             ne_min_cm3=1e12, Te_min_eV=1.0, Ti_min_eV=1.0):
    '''Run a case of KN1D

    Args:
    rhop :
    
    ne_cm3 :
    
    Te_eV :

    Ti_eV :

    p_H2_mTorr :

    clen_divertor_cm :

    clen_limiter_cm :

    bound_sep_cm, 

    lim_sep_cm : 

    from_wall_cm:

    mu : 
    Default is 2.0 (D plasma)

    innermost_rmid_cm :

    pipe_diag_cm :

    vx

    
    MWE:
    
    %run -i ../examples/basic.py

    rhop = asim.rhop_grid
    ne_cm3 = asim.ne[-1,:]
    Te_eV = asim.Te[-1,:]
    Ti_eV = asim.Te[-1,:]
    p_H2_mTorr=1.0
    clen_divertor_cm=200.
    clen_limiter_cm=50.
    lim_sep_cm=2.
    bound_sep_cm=6.
    rmid_to_wall_cm=8.0
    geqdsk = asim.geqdsk

    import kn1d
    out = kn1d.run_kn1d(rhop, ne_cm3, Te_eV, Ti_eV, geqdsk, p_H2_mTorr, 
             clen_divertor_cm, clen_limiter_cm, bound_sep_cm, lim_sep_cm, rmid_to_wall_cm)
    '''
    
    if 'IDL_STARTUP' not in os.environ:
        raise ValueError('An IDL installation does not seem to be available! KN1D cannot be run.')

    # make sure that the KN1D source code is accessible. 
    if 'KN1D' not in os.listdir(thisdir):
        #if 'KN1D_DIR' not in os.environ:
        # git clone the KN1D repository
        os.system(f'git clone https://github.com/fsciortino/kn1d {thisdir}/aurora/KN1D')
        
        # compile fortran libraries
        os.system(f'export KN1D_DIR={thisdir}/aurora/KN1D; cd KN1D; make clean; make; cd ..')
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


    rmid_to_wall, ne,Te,Ti = _setup_kin_profs(rhop, ne_cm3, Te_eV, Ti_eV, 
                                              geqdsk, bound_sep_cm, lim_sep_cm,
                                              kin_prof_exp_decay_SOL, kin_prof_exp_decay_LS,
                                              ne_decay_len_cm, Te_decay_len_cm, Ti_decay_len_cm, 
                                              ne_min_cm3, Te_min_eV, Ti_min_eV)
    
    rhop = coords.rad_coord_transform(rmid_to_wall, 'rmid', 'rhop', geqdsk)
    rwall = rmid_to_wall[-1]
    rsep = coords.rad_coord_transform(1.0, 'rhop', 'rmid', geqdsk)
    rlim = rsep + lim_sep_cm * 1e-2  # m


    # KN1D defines coordinates from the wall INWARD. Invert now:
    r_kn1d = np.abs(rmid_to_wall - rwall)[::-1]
    rlim_kn1d = np.abs(rlim - rwall)
    rsep_kn1d = np.abs(rsep - rwall)

    # diameter of pressure gauge pipe. Allows collisions with side-walls to be simulated
    dPipe = pipe_diag_cm*np.ones(len(rmid_to_wall))*1e-2  # m  -- zero values are treated as infinity

    # define the connection length vector
    lc = np.zeros(len(rmid_to_wall))
    lc[(rsep < rmid_to_wall) * (rmid_to_wall < rlim)] = clen_divertor_cm * 1e-2  # m
    lc[rmid_to_wall > rlim] = clen_limiter_cm * 1e-2  # m


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

    cwd = os.getcwd()
    os.chdir(f'{thisdir}/aurora/KN1D')

    # write IDL file
    with open('new_kn1d_run.pro', 'w') as f:
        f.write(idl_cmd)

    # Run the script 
    os.system('idl new_kn1d_run.pro')

    #### store all KN1D data for postprocessing  #####
    res = {}
    out = res['out'] = scipy.io.readsav('kn1d_out.sav')
    res['kn1d_input'] = scipy.io.readsav('.KN1D_input')
    res['kn1d_mesh'] = scipy.io.readsav('.KN1D_mesh')
    res['kn1d_H2'] = scipy.io.readsav('.KN1D_H2')
    res['kn1d_H'] = scipy.io.readsav('.KN1D_H')

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
    
    return out, out_profs
