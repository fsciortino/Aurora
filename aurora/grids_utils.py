'''Methods to create radial and time grids.

sciortino, 2020
'''

import matplotlib.pyplot as plt
import numpy as np, sys, os
from scipy.interpolate import interp1d
from . import _aurora



def get_HFS_LFS(geqdsk, rho_pol_arb=None):

    if rho_pol_arb is None:
        rho_pol_arb = np.linspace(0,1.2,70)

    R = geqdsk['AuxQuantities']['R']
    Z = geqdsk['AuxQuantities']['Z']

    Z0 = geqdsk['fluxSurfaces']['Z0']
    R0 = geqdsk['fluxSurfaces']['R0']
    RHORZ = geqdsk['AuxQuantities']['RHOpRZ']

    rho_mid = interp1d(Z,RHORZ,axis=0)(Z0)
    center = R.searchsorted(R0)
    rho_mid[:center] *= -1

    #remove a small discontinuity in gradients on axis
    R = np.delete(R,[center-1,center])
    rho_mid = np.delete(rho_mid,[center-1,center])

    Rhfs = np.interp(rho_pol_arb, -rho_mid[::-1],R[::-1])
    Rlfs = np.interp(rho_pol_arb, rho_mid,R)


    return Rhfs, Rlfs


def get_rhopol_rV_mapping(geqdsk, rho_pol=None):
    ''' Compute arrays allowing 1-to-1 mapping of rho_pol and r_V, both inside and
    outside the LCFS.

    r_V is defined as $\sqrt{V/(2 \pi^2 R_{axis}}$ inside the LCFS. Outside of it,
    we artificially expand the LCFS to fit true equilibrium at the midplane based
    on the rho_pol grid (sqrt of normalized poloidal flux).

    Method:
    #     r(rho,theta) = r0(rho) +  (r_lcfs(theta) - r0_lcfs) * scale
    #     z(rho,theta) = z0      +  (z_lcfs(theta) - z0     ) * scale
    #
    #             r(rho,theta=0) - r(rho,theta=180)
    #     scale = ----------------------------------
    #             r_lcfs(theta=0)- r_lcfs(theta=180)
    #
    #     r0_lcfs = .5*(r_lcfs(theta=0)+ r_lcfs(theta=180))
    #     r0(rho) = .5*(r(rho,theta=0) + r(rho,theta=180) )

    The mapping between rho_pol and r_V allows one to interpolate inputs on a
    rho_pol grid onto the r_V grid (in cm) used internally by the code.
    '''

    if rho_pol is None:
        # use arbitrary rho_pol grid, equally-spaced
        rho_pol = np.linspace(0.0, 1.1, 79)

    # volumes calculated by EFIT
    R0 = geqdsk['RMAXIS']
    Z0 = geqdsk['ZMAXIS']

    V_inner = geqdsk['fluxSurfaces']['geo']['vol']
    rhop_inner = np.sqrt( geqdsk['fluxSurfaces']['geo']['psin'])

    # find Rlfs and Rhfs for each value of rho_pol
    Rhfs, Rlfs = get_HFS_LFS(geqdsk, rho_pol)

    # R and Z along the LCFS
    R_lcfs = geqdsk['RBBBS']
    Z_lcfs = geqdsk['ZBBBS']

    r0_lcfs = 0.5*(np.interp(1,rho_pol,Rhfs) + np.interp(1,rho_pol,Rlfs))
    r0 = 0.5*(Rhfs + Rlfs)

    scale = (Rlfs-Rhfs)/(np.interp(1,rho_pol,Rlfs) - np.interp(1,rho_pol,Rhfs))
    R_outer = r0 + np.outer(R_lcfs - r0_lcfs,scale)
    Z_outer = Z0 + np.outer(Z_lcfs - Z0     ,scale)

    # calculate volume enclosed by these flux surfaces
    V_outer = -sum(2*np.pi*((R_outer+np.roll(R_outer,1,0))/2.)**2*(Z_outer-np.roll(Z_outer,1,0)),0)/2.0

    V = np.interp(rho_pol,rhop_inner, V_inner)
    V[rho_pol > 1] = V_outer[rho_pol > 1]

    #correct errors in V close to magnetic axis inside of rho = .2
    V[rho_pol <.2] = V_outer[rho_pol <.2]/V_outer[rho_pol <.2][-1]*V[rho_pol <.2][-1]

    # compute r_V
    r_V = np.sqrt(V/(2*np.pi**2 * R0)) * 100 # m --> cm
    r_V[0] = 0.0 # enforce 0 on axis

    return rho_pol, r_V






def create_aurora_radial_grid(namelist,plot=False):
    ''' This interfaces the package subroutine to create the radial grid.
    This exactly reproduces STRAHL functionality to produce radial grids, both for dr_0<0 and dr_1>0.
    Refer to the STRAHL manual for details.

    If plot==True, then show the radial grid, else return r,pro and qpr arrays required for simulation runs.
    '''
    # NB: there is currently a hard-coded maximum number of grid points (1000)
    _r, _pro, prox, _qpr = _aurora.get_radial_grid(
        namelist['ng'],namelist['bound_sep'],namelist['K'],namelist['dr_0'],namelist['dr_1'], namelist['rvol_lcfs'])

    # eliminate trailing zeros:
    idxs = _r > 0
    idxs[0] = True
    radius_grid, pro, qpr = _r[idxs], _pro[idxs], _qpr[idxs]

    if plot:

        r_lim  = namelist['rvol_lcfs'] + namelist['lim_sep']
        r_wall = namelist['rvol_lcfs'] + namelist['bound_sep']
        dr = np.gradient(radius_grid)

        if plt.fignum_exists('aurora radial step'):
            plt.figure(num='aurora radial step').clf()
        f,ax = plt.subplots(num='aurora radial step')
        
        ax.plot(radius_grid/namelist['rvol_lcfs'], dr,'-')
        ax.axvline(1,ls='--',c='k')
        ax.text(1+0.01,namelist['dr_0']*0.9 ,'LCFS',rotation='vertical')

        ax.axvline(r_lim/namelist['rvol_lcfs'],ls='--',c='k')
        ax.text(r_lim/namelist['rvol_lcfs']+0.01,namelist['dr_0']*0.9 ,'limiter',rotation='vertical')

        if 'saw_model' in namelist and namelist['saw_model']['saw_flag']:
            ax.axvline( namelist['saw_model']['rmix']/namelist['rvol_lcfs'],ls='--',c='k')
            ax.text(namelist['saw_model']['rmix']/namelist['rvol_lcfs']+0.01,namelist['dr_0']*0.5 ,'Sawtooth mixing radius',rotation='vertical')

        ax.set_xlabel(r'$r/r_{lcfs}$');
        ax.set_ylabel(r'$\Delta$ r [cm]');
        ax.set_ylim(0,None)
        ax.set_xlim(0,r_wall/namelist['rvol_lcfs'])
        ax.set_title('# radial grid points: %d'%len(radius_grid))

    else:
        return radius_grid, pro, prox, qpr



def create_aurora_time_grid(timing=None, plot=False):
    ''' Create time grid for simulations.
    '''

    _time, _save = _aurora.time_steps(
        timing['times'],timing['dt_start'],timing['steps_per_cycle'],timing['dt_increase'])

    # eliminate trailing 0's:
    idxs = np.nonzero(_time)
    time = _time[idxs]
    save = _save[idxs] > 0

    if plot:
        #show timebase
        if plt.fignum_exists('STRAHL time step'):
            plt.figure('STRAHL time step').clf()
        f,ax = plt.subplots(num='STRAHL time step')
        ax.set_title('# time steps: %d    # saved steps %d'%(len(time), sum(save)))
        ax.semilogy(time[1:],np.diff(time),'.-',label='step')
        ax.semilogy(time[1:][save[1:]],np.diff(time)[save[1:]],'o',label='step')
        [ax.axvline(t,c='k',ls='--') for t in timing['times']]
        ax.set_xlim(time[0],time[-1])

    else:
        return time, save
