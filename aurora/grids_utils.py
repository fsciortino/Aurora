'''Methods to create radial and time grids for aurora simulations.
'''
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

import matplotlib.pyplot as plt
import numpy as np, sys, os
from scipy.interpolate import interp1d
import warnings

def create_radial_grid(namelist,plot=False):
    r'''Create radial grid for Aurora based on K, dr_0, dr_1, rvol_lcfs and bound_sep parameters. 
    The lim_sep parameters is additionally used if plotting is requested. 

    Radial mesh points are set to be equidistant in the coordinate :math:`\rho`, with

    .. math::

        \rho = \frac{r}{\Delta r_{centre}} + \frac{r_{edge}}{k+1} \left(\frac{1}{\Delta r_{edge}}- \frac{1}{\Delta r_{centre}} \right) \left(\frac{r}{r_{edge}} \right)^{k+1}

    The corresponding radial step size is 

    .. math::

        \Delta r = \left[\frac{1}{\Delta r_{centre}} + \left(\frac{1}{\Delta r_{edge}} - \frac{1}{\Delta r_{centre}} \right) \left(\frac{r}{r_{edge}}\right)^k \right]^{-1}


    See the STRAHL manual for details. 

    Parameters
    ----------
    namelist : dict
        Dictionary containing Aurora namelist. This function uses the K, dr_0, dr_1, rvol_lcfs 
        and bound_sep parameters. Additionally, lim_sep is used if plotting is requested. 
    plot : bool, optional
        If True, plot the radial grid spacing vs. radial location. 

    Returns
    -------
    rvol_grid : array
        Volume-normalized grid used for Aurora simulations.
    pro : array
        Normalized first derivatives of the radial grid, defined as 
        pro = (drho/dr)/(2 d_rho) = rho'/(2 d_rho)
    qpr : array
        Normalized second derivatives of the radial grid, defined as 
        qpr = (d^2 rho/dr^2)/(2 d_rho) = rho''/(2 d_rho)
    prox_param : float
        Grid parameter used for perpendicular loss rate at the last radial grid point. 
    '''
    
    K = namelist['K']
    dr_0 = namelist['dr_0']
    dr_1 = namelist['dr_1']
    rvol_lcfs = namelist['rvol_lcfs']
    dbound = namelist['bound_sep']

    # radial location of wall boundary:
    r_edge = namelist['rvol_lcfs'] + namelist['bound_sep']

    try:
        assert dr_0>0.0
        assert dr_1>0.0
    except:
        raise ValueError('Both dr_0 and dr_1 must be positive!')
    
    a0=1./dr_0

    # number of radial points:
    ir=int(round(1.5+r_edge*(a0*K+ (1./dr_1))/(K+1.)))

    rvol_grid = np.zeros(ir)
    pro_grid = np.zeros(ir)
    qpr_grid = np.zeros(ir)

    # prox_param is used in calculation of ion perpendicular loss at the last grid point
    prox_param=(K+1.)*(float(ir)-1.5)/r_edge-a0*K

    # form the radial grid iteratively
    rvol_grid[0] = 0.
    for i in np.arange(1,ir):
        temp1 = 0.
        temp2 = r_edge*1.05
        
        for j in np.arange(50):
            rvol_grid[i] = (temp1+temp2)/2.
            temp3 = a0*rvol_grid[i]+(prox_param-a0)*r_edge/(K+1.)*(rvol_grid[i]/r_edge)**(K+1.)
            if temp3 > i:
                temp2 = rvol_grid[i]
            else:
                temp1 = rvol_grid[i]

    # terms related with derivatives of rho
    temp1 = 0.5
    pro_grid[0] = 2./(dr_0**2)
    for i in np.arange(1,ir):
        pro_grid[i] = (a0+(prox_param-a0)*(rvol_grid[i]/r_edge)**K)*temp1
        qpr_grid[i] = pro_grid[i]/rvol_grid[i]+temp1*(prox_param-a0)*K/r_edge*(rvol_grid[i]/r_edge)**(K-1.)

    if plot:

        r_lim  = namelist['rvol_lcfs'] + namelist['lim_sep']
        dr = np.gradient(rvol_grid)

        if plt.fignum_exists('Aurora radial step'):
            plt.figure(num='Aurora radial step').clf()
        f,ax = plt.subplots(num='Aurora radial step')
        
        ax.plot(rvol_grid/namelist['rvol_lcfs'], dr,'-')
        ax.axvline(1,ls='--',c='k')
        ax.text(1+0.01,namelist['dr_0']*0.9 ,'LCFS',rotation='vertical')

        ax.axvline(r_lim/namelist['rvol_lcfs'],ls='--',c='k')
        ax.text(r_lim/namelist['rvol_lcfs']+0.01,namelist['dr_0']*0.9 ,'limiter',rotation='vertical')

        if 'saw_model' in namelist and namelist['saw_model']['saw_flag']:
            ax.axvline( namelist['saw_model']['rmix']/namelist['rvol_lcfs'],ls='--',c='k')
            ax.text(namelist['saw_model']['rmix']/namelist['rvol_lcfs']+0.01,
                    namelist['dr_0']*0.5 ,'Sawtooth mixing radius',rotation='vertical')

        ax.set_xlabel(r'$r/r_{lcfs}$');
        ax.set_ylabel(r'$\Delta$ r [cm]');
        ax.set_ylim(0,None)
        ax.set_xlim(0,r_edge/namelist['rvol_lcfs'])
        ax.set_title(r'$\#$ radial grid points: %d'%len(rvol_grid))

    return rvol_grid, pro_grid, qpr_grid, prox_param




class MissingAuroraBuild(UserWarning):
    pass
    
def create_time_grid(timing=None, plot=False):
    '''Create time grid for simulations using the Fortran implementation
    of the time grid generator. 

    Parameters
    ----------
    timing : dict
        Dictionary containing timing elements: 'times', 'dt_start', 'steps_per_cycle','dt_increase'
        As in STRAHL, the last element in each of these arrays refers to sawtooth events. 
    plot : bool
        If True, plot time grid. 

    Returns
    -------
    time : array
        Computational time grid corresponding to :param:timing input.
    save : array
        Array of zeros and ones, where ones indicate that the time step will be stored in memory
        in Aurora simulations. Points corresponding to zeros will not be returned to spare memory. 
    '''
    # import here to avoid import when building documentation or package
    try:
        from ._aurora import time_steps
    except ModuleNotFoundError:
        raise MissingAuroraBuild('Could not load particle transport forward model! '+\
                      'Use the makefile or setup.py to build sources.')
    
    _time, _save = time_steps(
        timing['times'],timing['dt_start'],timing['steps_per_cycle'],timing['dt_increase'])

    # eliminate trailing 0's:
    idxs = np.nonzero(_time)
    time = _time[idxs]
    save = _save[idxs] > 0
    
    if plot:
        #show timebase
        if plt.fignum_exists('Aurora time step'):
            plt.figure('Aurora time step').clf()
        f,ax = plt.subplots(num='Aurora time step')
        ax.set_title(r'$\#$ time steps: %d   $\#$ saved steps %d'%(len(time), sum(save)))
        ax.semilogy(time[1:],np.diff(time),'.-',label='step')
        ax.semilogy(time[1:][save[1:]],np.diff(time)[save[1:]],'o',label='step')
        [ax.axvline(t,c='k',ls='--') for t in timing['times']]
        ax.set_xlim(time[0],time[-1])

    return time, save



    
def create_time_grid_new(timing, verbose=False, plot=False):
    '''Define time base for Aurora based on user inputs
    This function reproduces the functionality of STRAHL's time_steps.f
    Refer to the STRAHL manual for definitions of the time grid
    
    Parameters
    ----------
    n : int
        Number of elements in time definition arrays
    t : array
        Time vector of the time base changes
    dtstart : array
        dt value at the start of a cycle
    itz : array
        cycle length, i.e. number of time steps before increasing dt
    tinc :
        factor by which time steps should be increasing within a cycle
    verbose : bool
        If Trueprint to terminal a few extra info
    
    Returns
    -------
    t_vals : array
        Times in the time base [s]
    i_save : array
        Array of 0,1 values indicating at which times internal arrays should be stored/returned. 

    
    ~~~~~~~~~~~ THIS ISN'T FUNCTIONAL YET! ~~~~~~~~~~~~


    '''
    t = np.array(timing['times'])
    dtstart = np.array(timing['dt_start'])
    itz = np.array(timing['steps_per_cycle'])
    tinc = np.array(timing['dt_increase'])
    
    t_vals = np.zeros(30000)
    i_save = np.zeros(30000)

    dt = np.zeros(250)
    ncyctot = np.zeros(250)    
    ncyc = np.zeros(250)
    
    # zeross without double time points
    t_s = np.zeros(250)
    dtstart_s = np.zeros(250)
    tinc_s = np.zeros(250)
    itz_s = np.zeros(250)
    
    # sort t-change and related verctors
    idxs = np.argsort(t)
    dtstart = dtstart[idxs]
    tinc = tinc[idxs]
    itz = itz[idxs]
    
    if verbose:
        print('Inputs after sorting:')
        print('t:',t)
        print('dtstart: ', dtstart)
        print('tinc: ', tinc)
        print('itz: ', itz)
        
    # cancel double time points
    nevent = 1
    t_s[0] = t[0]
    dtstart_s[0] = dtstart[0]
    tinc_s[0] = tinc[0]
    itz_s[0] = itz[0]
    for i in np.arange(1,len(t)):
        if abs(t[i]-t[nevent]) > 1e-8:
            nevent = nevent + 1
            t_s[nevent] = t[i]
            dtstart_s[nevent] = dtstart[i]
            tinc_s[nevent] = tinc[i]
            itz_s[nevent] = itz[i]

    # define # of cycles for every interval with a start time step: dtstart[i]
    for i in np.arange(nevent-1):
        f = (t_s[i+1] - t_s[i])/dtstart_s[i]
        if i==0: f = f + 1.
        if tinc_s[i]>1.:
            ncyc[i] = max(2,int(np.log(f/itz_s[i]*(tinc_s[i]-1.)+1.)/np.log(tinc_s[i])))
            
        if (tinc_s[i]==1.):
            ncyc[i] = max(2,int(f/itz_s[i]))

        if i==0: ncyctot[i] = ncyc[i]
        if i>0: ncyctot[i] = ncyctot[i-1] + ncyc[i]

    # sum of all timesteps
    nsteps = 0
    for i in np.arange(nevent-1):
        nsteps = nsteps + ncyc[i] * itz_s[i]
        
    nsteps = nsteps - 1

    # define real start timestep dt[i] to fit the time intervals
    for i in np.arange(nevent-1):
        if tinc_s[i]>1.:
            f = itz_s[i]* (tinc_s[i]**ncyc[i]-1.)/(tinc_s[i]-1.)

        if tinc_s[i]==1.: f = 1. * itz_s[i] * ncyc[i]
        if i==1: f = f - 1.
        dt[i] = (t_s[i+1] - t_s[i])/f

    # Now, define t list
    nn = 1
    n_itz = 1
    m = 1
    tnew = t_s[0]
    nevent=1
    det = dt[0]

    if verbose:
        print('time_steps:')
        print('nsteps:',nsteps)
        for i in np.arange(nevent-1):
            print('Interval:',i,' start step: ',dt[i])
    
    t_vals[0] = tnew
    i_save[0] = 1

    tmp2 = True
    while tmp2:
        tmp = True
        while tmp:
            tnew = tnew + det
            t_vals[nn+1] = tnew

            if np.mod(n_itz+1,itz_s[nevent])!= 0:
                nn = nn+1
                n_itz = n_itz + 1
            else:
                tmp=False

        n_itz = 0
        det = tinc[nevent] * det
        if (m == ncyctot[nevent]) & (nn != nsteps):
            nevent = nevent + 1
            det = dt[nevent]

        m=m+1
        i_save[nn+1] = 1

        if nn<nsteps:
            nn = nn + 1
        else:
            tmp2=False


    # -------------
    # eliminate trailing 0's:
    idxs = np.nonzero(t_vals)
    time = t_vals[idxs]
    save = i_save[idxs] > 0

    if plot:
        #show timebase
        if plt.fignum_exists('Aurora time step'):
            plt.figure('Aurora time step').clf()
        f,ax = plt.subplots(num='Aurora time step')
        ax.set_title('# time steps: %d    # saved steps %d'%(len(time), sum(save)))
        ax.semilogy(time[1:],np.diff(time),'.-',label='step')
        ax.semilogy(time[1:][save[1:]],np.diff(time)[save[1:]],'o',label='step')
        [ax.axvline(t,c='k',ls='--') for t in timing['times']]
        ax.set_xlim(time[0],time[-1])

    return time, save



def get_HFS_LFS(geqdsk, rho_pol=None):
    '''Get high-field-side (HFS) and low-field-side (LFS) major radii from the g-EQDSK data. 
    This is useful to define the rvol grid outside of the LCFS. 
    See the :py:func:`~aurora.grids_utils.get_rhopol_rvol_mapping` for an application. 

    Parameters
    ----------
    geqdsk : dict
        Dictionary containing the g-EQDSK file as processed by the `omfit_classes.omfit_eqdsk`. 
    rho_pol : array, optional
        Array corresponding to a grid in sqrt of normalized poloidal flux for which a 
        corresponding rvol grid should be found. If left to None, an arbitrary grid will be 
        created internally. 
    
    Returns
    -------
    Rhfs : array
        Major radius [m] on the HFS
    Rlfs : array
        Major radius [m] on the LFS

    '''
    if rho_pol is None:
        rho_pol = np.linspace(0,1.2,70)

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

    Rhfs = np.interp(rho_pol, -rho_mid[::-1],R[::-1])
    Rlfs = np.interp(rho_pol, rho_mid,R)

    return Rhfs, Rlfs



def get_rhopol_rvol_mapping(geqdsk, rho_pol=None):
    r'''Compute arrays allowing 1-to-1 mapping of rho_pol and rvol, both inside and
    outside the LCFS.

    rvol is defined as :math:`\sqrt{V/(2 \pi^2 R_{axis}}` inside the LCFS. Outside of it,
    we artificially expand the LCFS to fit true equilibrium at the midplane based
    on the rho_pol grid (sqrt of normalized poloidal flux).

    Method:
    
    .. math::

        r(\rho,\theta) = r_0(\rho) +  (r_{lcfs}(\theta) - r_{0,lcfs}) \times \mathcal{f} \\
        z(\rho,\theta) = z_0      +  (z_{lcfs}(\theta) - z_0     ) \times \mathcal{f} \\
        \mathcal{f} = \frac{ r(\rho,\theta=0) - r(\rho,\theta=180)}{r_{lcfs}(\theta=0)- r_{lcfs}(\theta=180)} \\
        r_{0,lcfs} = \frac{1}{2} (r_{lcfs}(\theta=0)+ r_{lcfs}(\theta=180)) \\
        r_0(\rho) = \frac{1}{2} (r(\rho,\theta=0) + r(\rho,\theta=180))

    The mapping between rho_pol and rvol allows one to interpolate inputs on a
    rho_pol grid onto the rvol grid (in cm) used internally by the code.

    Parameters
    ----------
    geqdsk : dict
        Dictionary containing the g-EQDSK file as processed by `omfit_classes.omfit_eqdsk`. 
    rho_pol : array, optional
        Array corresponding to a grid in sqrt of normalized poloidal flux for which a 
        corresponding rvol grid should be found. If left to None, an arbitrary grid will be 
        created internally. 

    Returns
    -------
    rho_pol : array
        Sqrt of normalized poloidal flux grid
    rvol : array
        Mapping of rho_pol to a radial grid defined in terms of normalized flux surface volume.
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

    # calculate volume enclosed by these "flux surfaces" outside of the LCFS
    #V_outer = -sum(2*np.pi*((R_outer+np.roll(R_outer,1,0))/2.)**2*(Z_outer-np.roll(Z_outer,1,0)),0)/2.0
    V_outer = np.abs(sum(2*np.pi*((R_outer+np.roll(R_outer,1,0))/2.)**2*(Z_outer-np.roll(Z_outer,1,0)),0)/2.0)
    
    V = np.interp(rho_pol,rhop_inner, V_inner)
    V[rho_pol > 1] = V_outer[rho_pol > 1]

    # correct errors in V close to magnetic axis inside of rho = .2
    V[rho_pol <.2] = V_outer[rho_pol <.2]/V_outer[rho_pol <.2][-1]*V[rho_pol <.2][-1]

    # compute rvol
    rvol = np.sqrt(V/(2*np.pi**2 * R0)) * 100 # m --> cm

    # enforce 0 on axis
    rvol[0] = 0.0
    
    return rho_pol, rvol






def create_radial_grid_fortran(namelist,plot=False):
    '''This interfaces the package subroutine to create the radial grid exactly as STRAHL does it.
    Refer to the STRAHL manual for details.

    '''
    # import here to avoid import when building documentation or package
    try:
        from ._aurora import get_radial_grid
    except ModuleNotFoundError:
        raise MissingAuroraBuild('Could not load particle transport forward model!'+\
                      'Use the makefile or setup.py to build sources.')
    
    # NB: there is currently a hard-coded maximum number of grid points (1000)
    _r, _pro, prox, _qpr = get_radial_grid(
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
            ax.text(namelist['saw_model']['rmix']/namelist['rvol_lcfs']+0.01,namelist['dr_0']*0.5 ,
                    'Sawtooth mixing radius',rotation='vertical')

        ax.set_xlabel(r'$r/r_{lcfs}$');
        ax.set_ylabel(r'$\Delta$ r [cm]');
        ax.set_ylim(0,None)
        ax.set_xlim(0,r_wall/namelist['rvol_lcfs'])
        ax.set_title('# radial grid points: %d'%len(radius_grid))

    return radius_grid, pro, prox, qpr



def create_aurora_time_grid(timing, plot=False):
    '''Create time grid for simulations using a Fortran routine for definitions. 
    The same functionality is offered by :py:func:`~aurora.grids_utils.create_time_grid`, which however
    is written in Python. This method is legacy code; it is recommended to use the other. 

    Parameters
    ----------
    timing : dict
        Dictionary containing 
        timing['times'],timing['dt_start'],timing['steps_per_cycle'],timing['dt_increase']
        which define the start times to change dt values at, the dt values to start with,
        the number of time steps before increasing the dt by dt_increase. 
        The last value in each of these arrays is used for sawteeth, whenever these are
        modelled, or else are ignored. This is the same time grid definition as used in STRAHL.
    plot : bool, optional
        If True, display the created time grid.

    Returns
    -------
    time : array
        Computational time grid corresponding to `timing` input.
    save : array
        Array of zeros and ones, where ones indicate that the time step will be stored in memory
        in aurora simulations. Points corresponding to zeros will not be returned to spare memory.    
    '''
    # import here to avoid import when building documentation or package
    from ._aurora import time_steps
    
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

    return time, save




def estimate_clen(geqdsk):
    '''Estimate average connection length in the open SOL and in the limiter shadow
    NB: these are just rough numbers!

    Parameters
    ----------
    geqdsk : dict
        EFIT g-EQDSK as processed by `omfit_classes.omfit_eqdsk`.

    Returns
    -------
    clen_divertor : float
        Estimate of the connection length to the divertor [m]
    clen_limiter : float
        Estimate of the connection length to the limiter [m]
    '''
    # estimate connection legnth in divertor
    q = geqdsk['QPSI']
    rhop = np.sqrt(geqdsk['fluxSurfaces']['geo']['psin'])

    # q at the 95% surface (in sqrtpsinorm)
    q95 = np.abs(interp1d(rhop,q)(0.95))
    R0 = geqdsk['fluxSurfaces']['R0']

    # estimate for connection length to the divertor
    clen_divertor = round(np.pi*R0*q95,5)

    # estimate for connection length to limiter
    zlim = geqdsk['ZLIM']
    h = np.max(zlim) - np.min(zlim)
    clen_limiter = round(h/5.,5) # 1/5th of machine height
    
    return clen_divertor, clen_limiter




def estimate_boundary_distance(shot, device, time_ms):
    '''Obtain a simple estimate for the distance between the LCFS and the wall boundary.
    This requires access to the A_EQDSK on the EFIT01 tree on MDS+. Users who may find that this call
    does not work for their device may try to adapt the OMFITmdsValue TDI string.

    Parameters
    ----------
    shot : int
        Discharge/experiment number
    device : str
        Name of device, e.g. 'C-Mod', 'DIII-D', etc.
    time_ms : int or float
        Time at which results for the outer gap should be taken.

    Returns
    -------
    bound_sep : float
        Estimate for the distance between the wall boundary and the separatrix [cm]
    lim_sep : float
        Estimate for the distance between the limiter and the separatrix [cm]. This is (quite arbitrarily)
        taken to be 2/3 of the bound_sep distance.    
    '''
    # import this here, so that it is not required for the whole package
    from omfit_classes.omfit_mds import OMFITmdsValue

    try:
        tmp = OMFITmdsValue(server=device, treename='EFIT01', shot=shot,
                            TDI='\\EFIT01::TOP.RESULTS.A_EQDSK.ORIGHT')  # CMOD format, take ORIGHT
    except Exception:
        tmp = OMFITmdsValue(server=device, treename='EFIT01', shot=shot,
                            TDI='\\EFIT01::TOP.RESULTS.AEQDSK.GAPOUT') # useful variable for many other devices

    time_vec = tmp.dim_of(0)
    data_r = tmp.data()
    
    ind = np.argmin(np.abs(time_vec - time_ms))
    inds = slice(ind-3, ind+3)
    bound_sep = round(np.mean(data_r[inds]),3)

    # take separation to limiter to be 2/3 of the separation to the wall boundary
    lim_sep = round(bound_sep*2./3.,3)

    return bound_sep, lim_sep



def vol_int(var, rvol_grid, pro_grid, Raxis_cm, rvol_max=None):
    """
    Perform a volume integral of an input variable. If the variable is f(t,x) 
    then the result is f(t). If the variable is f(t,*,x) then the result is f(t,charge)
    when "*" represents charge, line index, etc...

    Parameters
    ----------
    var : 2D+ array (time, ..., radius)
        Data array for which a volume integral should be evaluated.
        The last dimension must be radial, other dimensions are arbitrary.
    rvol_grid : 1D array
        Volume-normalized radial grid.
    pro_grid : 
        Normalized first derivative of the radial grid, see :py:func:`~aurora.grids_utils.create_radial_grid`.
    Raxis_cm : float 
        Major radius on axis [cm]
    rvol_max : float
        Maximum volume-normalized radius for integral. If not provided, integrate
        over the entire simulation radial grid. 

    Returns
    -------
    var_volint : array (nt,)
        Time history of volume integrated variable
    """
    C = 2 * np.pi * Raxis_cm
    zvol = C * np.pi * rvol_grid / pro_grid

    # Determine range
    if rvol_max is not None:
        wh = ( rvol_grid <= rvol_max)
        zvol = zvol[wh]
        var  = var[...,wh]
    
    # 2D or 3D array f(t,x)
    var_volint = np.nansum(var*zvol,axis=-1)

    return var_volint


