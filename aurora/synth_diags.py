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
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.integrate import trapz
import matplotlib.tri as tri
plt.ion()
from scipy.constants import e as q_electron, m_p


def line_int_weights(R_path, Z_path, rhop_path, dist_path, R_axis=None, rhop_out=None, CF_lam=None):
    '''Obtain weights for line integration on a rhop grid, given the 3D path of line integration in the (R,Z,Phi) 
    coordinates, as well as the value of sqrt of normalized poloidal flux at each point along the path. 

    Parameters
    ----------
    R_path : array (np,)
        Values of the R coordinate [m] along the line integration path.
    Z_path : array (np,)
        Values of the Z coordinate [m] along the line integration path.
    rhop_path : array (np,)
        Values of the rhop coordinate along the line integration path. 
    dist_path : array (np,)
        Vector starting from 0 to the maximum distance [m] considered along the line integration. 
    R_axis : float
        R value at the magnetic axis [m]. Only used for centrifugal asymmetry effects if CF_lam is not None.
    rhop_out : array (nr,)
        The sqrt of normalized poloidal flux grid on which weights should be computed. If left to None, an 
        equally-spaced grid with 201 points from the magnetic axis to the LCFS is used. 
    CF_lam : array (nr,)
        Centrifugal (CF) asymmetry exponential factor, returned by the :py:func:`~aurora.synth_diags.centrifugal_asym`
        function. If provided, this is taken to be on an input rhop_out grid. If left to None, no CF asymmetry is considered. 
    '''
    if rhop_out is None:
        rhop_out = np.linspace(0,1,201)

    # response matrix for effective line integration
    response = interp1d( rhop_out, np.eye(len(rhop_out)),  axis=1, 
        bounds_error=False, copy=False, fill_value=0, assume_sorted=True, kind='linear'
    )(rhop_path)
    
    if CF_lam is not None:
        # interpolate CF lambda to ray/beam path
        interp_lam = interp1d(rhop_out, CF_lam, 
                              copy=False, assume_sorted=True, bounds_error=False, fill_value=0)(rhop_path)
        
        asym = np.exp(interp_lam * (R_path ** 2 - R_axis** 2))
    else:
        asym = 1.

    # compute weights by summing over beam/ray path
    weights = trapz(response[None] * asym, dist_path, axis=2)
    
    return weights


def centrifugal_asym(rhop, Rlfs, omega, Zeff, A_imp, Z_imp, Te, Ti, 
                          main_ion_A=2, plot=False, nz=None, geqdsk=None):
    r"""Estimate impurity poloidal asymmetry effects from centrifugal forces. 

    The result of this function is :math:`\lambda`, defined such that

    .. math::

       n(r,\theta) = n_0(r) \times \exp\left(\lambda(\rho) (R(r,\theta)^2- R_0^2)\right)


    See Odstrcil et al. 2018 Plasma Phys. Control. Fusion 60 014003 for details on centrifugal asymmetries. 
    Also see  Appendix A of Angioni et al 2014 Nucl. Fusion 54 083028 for details on these should also be 
    accounted for when comparing transport coefficients used in Aurora (on a rvol grid) to coefficients used 
    in codes that use other coordinate systems (e.g. based on rmid). 

    Parameters
    ----------
    rhop : array (nr,)
        Sqrt of normalized poloidal flux grid.
    Rlfs : array (nr,)
        Major radius on the Low Field Side (LFS), at points corresponding to rhop values
    omega : array (nt,nr) or (nr,) [ rad/s ] 
        Toroidal rotation on Aurora temporal time_grid and radial rhop_grid (or, equivalently, rvol_grid) grids.
    Zeff : array (nt,nr), (nr,) or float
        Effective plasma charge on Aurora temporal time_grid and radial rhop_grid (or, equivalently, rvol_grid) grids.
        Alternatively, users may give Zeff as a float (taken constant over time and space).
    A_imp : float
        Impurity ion atomic mass number (e.g. 40 for Ca)
    Z_imp : array (nr, ) or int 
        Charge state of the impurity of interest. This can be an array, giving the expected charge state at every 
        radial position, or just a float. 
    Te : array (nr,nt)
        Electron temperature (eV)
    Ti : array (nr, nt)
        Background ion temperature (eV)
    main_ion_A : int, optional
        Background ion atomic mass number. Default is 2 for D. 
    plot : bool
        If True, plot asymmetry factor :math:`\lambda` vs. radius and show the predicted 2D impurity density distribution 
        at the last time point.
    nz : array (nr,nZ)
        Impurity charge state densities (output of Aurora at a specific time slice), only used for 2D plotting.
    geqdsk : dict
        Dictionary containing the `omfit_classes.omfit_eqdsk` reading of the EFIT g-file. 

    Returns
    -------
    CF_lam : array (nr,)
        Asymmetry factor, defined as :math:`\lambda` in the expression above. 
    """
    if omega.ndim==1:
        omega = omega[None,:]  # take constant in time
    if isinstance(Zeff,(int,float)):
        Zeff = np.array(Zeff) * np.ones_like(Ti)
    if Zeff.ndim==1:
        Zeff = Zeff[None,:] # take constant in time

    # deuterium mach number
    mach = np.sqrt(2. * m_p / q_electron * (omega * Rlfs[None,:]) ** 2 / (2. * Ti))

    # valid for deuterium plasma with Zeff almost constants on flux surfaces
    CF_lam = A_imp / 2. * (mach / Rlfs[None,:]) ** 2 * (
        1. - Z_imp * main_ion_A / A_imp * Zeff * Te / (Ti + Zeff * Te)
    )

    # centrifugal asymmetry is only relevant on closed flux surfaces
    CF_lam[:, rhop > 1.] = 0  

    if plot:
        # show centrifugal asymmetry lambda as a function of radius
        fig, ax = plt.subplots()
        ax.plot(rhop, CF_lam.T)
        ax.set_xlabel(r'$\rho_p$')
        ax.set_ylabel(r'$\lambda$')

        # plot expected radial impurity density over the poloidal cross section            
        fig,ax = plt.subplots()
        if isinstance(Z_imp,(int,float)):
            # select charge state of interest
            nz_sel = nz[:,int(Z_imp)-1]
        else: 
            # use total impurity density if Z_imp was given as a vector
            nz_sel = nz.sum(1)

        rhop_surfs = np.sqrt(geqdsk['fluxSurfaces']['geo']['psin'])                   

        Rs = []; Zs = []; vals = []
        for ii,surf in enumerate(geqdsk['fluxSurfaces']['flux']):

            # FSA nz on this flux surface at the last time point
            nz_sel_i = interp1d(rhop, nz_sel)(rhop_surfs[ii])
            CF_lam_i = interp1d(rhop, CF_lam[-1,:])(rhop_surfs[ii]) 

            Rs = np.concatenate((Rs, geqdsk['fluxSurfaces']['flux'][ii]['R']))
            Zs = np.concatenate((Zs, geqdsk['fluxSurfaces']['flux'][ii]['Z']))
            vals = np.concatenate( ( vals, 
                                    nz_sel_i * np.exp(
                                        CF_lam_i * (
                                            geqdsk['fluxSurfaces']['flux'][ii]['R']**2 - geqdsk['RMAXIS']**2
                                            )
                                    )
                                ) )

        triang = tri.Triangulation(Rs, Zs)
        cntr1 = ax.tricontourf(triang, vals,levels=300)
        ax.plot(geqdsk['RBBBS'], geqdsk['ZBBBS'], c='k')
        ax.scatter(geqdsk['RMAXIS'], geqdsk['ZMAXIS'], marker='x', c='k')
        ax.axis('equal')
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        plt.tight_layout()

    return CF_lam



