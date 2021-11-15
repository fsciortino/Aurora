'''
Methods for neutral beam analysis, particularly in relation to impurity transport studies.
These script collects functions that should be device-agnostic.
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

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from scipy.interpolate import RectBivariateSpline, interp1d, interp2d
import copy, itertools

from .janev_smith_rates import js_sigma
from .plot_tools import get_ls_cycle
from . import atomic

def get_neutrals_fsa(neutrals, geqdsk, debug_plots=True):
    """Compute charge exchange recombination for a given impurity with neutral beam components,
    obtaining rates in [:math:`s^{-1}`] units. This method expects all neutral components to be given in a
    dictionary with a structure that is independent of NBI model (i.e. coming from FIDASIM, NUBEAM, 
    pencil calculations, etc.).

    Parameters
    ----------
    neutrals : dict
        Dictionary containing fields
        {"beams","names","R","Z", beam1, beam2, etc.}
        Here beam1,beam2,etc. are the names in neutrals["beams"]. "names" are the names of each 
        beam component, e.g. 'fdens','hdens','halo', etc., ordered according to "names". 
        "R","Z" are the major radius and vertical coordinates [cm] on which neutral density components are 
        given in elements such as

        .. code-block:: python
            
            neutrals[beams[0]]["n=0"][name_idx]

        It is currently assumed that n=0,1 and 2 beam components are provided by the user. 

    geqdsk : dictionary output of `omfit_classes.omfit_eqdsk.OMFITgeqdsk` class
        gEQDSK post-processed dictionary, as given by `omfit_classes.omfit_eqdsk`. 
    debug_plots : bool, optional
        If True, various plots are displayed. 

    Returns
    -------
    neut_fsa : dict
        Dictionary of flux-surface-averaged (FSA) neutral densities, in the same units as in the input. 
        Similarly to the input "neutrals", this dictionary has a structure like

        .. code-block:: python

            neutrals_ext[beam][f'n={n_level}'][name_idx]

    """

    beams = neutrals['beams']
    names = neutrals['names']
    zz = neutrals['Z'] / 1e2  # cm --> m
    rr = neutrals['R'] / 1e2  # cm --> m

    if debug_plots:
        # for debugging/plotting:
        RBBBS = geqdsk['RBBBS']
        ZBBBS = geqdsk['ZBBBS']
        fig, ax = plt.subplots()
        ax.contourf(rr, zz, neutrals[beams[0]]['n=0'][0].T)
        ax.plot(RBBBS, ZBBBS, 'w--', lw=5)
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')

    # simple way to find rhop at any combination of R,Z coords:
    RHOpRZ = geqdsk['AuxQuantities']['RHOpRZ']
    Rgrid = geqdsk['AuxQuantities']['R']  # m
    Zgrid = geqdsk['AuxQuantities']['Z']  # m

    # extrapolate beam to cover the entire 2D poloidal cross section,
    # setting all neutral populations to 0 outside of the simulated region
    neutrals_ext = {}
    for beam in beams:
        neutrals_ext[beam] = {}
        for n_level in [0, 1, 2]:
            neutrals_ext[beam][f'n={n_level}'] = {}
            for ii, name in enumerate(names):
                dens = neutrals[beam][f'n={n_level}'][ii]

                f = interp2d(rr, zz, dens.T, kind='linear', bounds_error=False, fill_value=0.0)
                tmp = f(Rgrid, Zgrid)

                tmp[tmp < 0] = 0.0
                neutrals_ext[beam][f'n={n_level}'][name] = tmp

    if debug_plots:
        fig, ax = plt.subplots()
        CS = ax.contourf(Rgrid, Zgrid, neutrals_ext[beam]['n=0']['fdens'])
        cbar = fig.colorbar(CS)
        cbar.ax.set_ylabel(r'f density [$cm^{-3}$]')
        ax.plot(RBBBS, ZBBBS, c='w', lw=5)
        ax.axis('equal')
        CS = plt.contour(Rgrid, Zgrid, RHOpRZ, np.linspace(0.0, 1.2, 13), c='w')
        plt.clabel(CS, inline=1, fontsize=14)
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')

    # Get flux surface average quantities:
    neut_fsa = {}
    rhop = neut_fsa['rhop'] = geqdsk['AuxQuantities']['RHOp']
    neut_fsa['names'] = names

    for beam in beams:
        neut_fsa[beam] = {}

        for n_level in [0, 1, 2]:
            neut_fsa[beam][f'n={n_level}'] = {}

            for ii, name in enumerate(names):
                dens = neutrals_ext[beam][f'n={n_level}'][name]

                # flux surface average
                def avg_function(r, z):
                    return RectBivariateSpline(Zgrid, Rgrid, dens, kx=1, ky=1).ev(z, r)

                neut_fsa[beam][f'n={n_level}'][name] = geqdsk['fluxSurfaces'].surfAvg(function=avg_function)

    for beam in beams:
        # store beam component energies
        neut_fsa[beam]['f_energy'] = neutrals[beam]['f_energy']  # keV
        neut_fsa[beam]['h_energy'] = neutrals[beam]['h_energy']  # keV
        neut_fsa[beam]['t_energy'] = neutrals[beam]['t_energy']  # keV
        neut_fsa[beam]['m_beam'] = neutrals[beam]['m_beam']  # amu
    neut_fsa['m_bckg'] = copy.deepcopy(neutrals['m_bckg'])  # amu

    if debug_plots:
        ls_cycle = get_ls_cycle()

        # plot FSA neutral densities
        fig = plt.figure()
        fig.set_size_inches(9, 6, forward=True)
        a1 = plt.subplot2grid((4, 4), (0, 0), rowspan=4, colspan=3)
        a2 = plt.subplot2grid((4, 4), (0, 3), rowspan=4, colspan=1)
        for beam in beams:
            for n_level in [0, 1, 2]:
                for name in ['fdens', 'hdens', 'tdens', 'dcx+halo']:
                    if name == 'dcx+halo':
                        name = 'halo'  # simple renaming
                    lss = next(ls_cycle)
                    a1.plot(rhop, neut_fsa[beam][f'n={n_level}'][name], lss, lw=2.0)
                    a2.plot([], [], lss, lw=2.0, label=f'{beam} {name}, n={n_level+1}')

        a1.set_xlabel(r'$\rho_{p}$')
        a1.set_ylabel('FSA density')
        a2.axis('off')
        a2.legend(fontsize=14)
        # fig.tight_layout()

    return neut_fsa



def get_NBI_imp_cxr_q(neut_fsa, q, rhop_kp, times_kp, Ti_eV, ne_cm3, include_fast=True, include_halo=True, debug_plots=False):
    """Compute flux-surface-averaged (FSA) charge exchange recombination for a given impurity with
    neutral beam components, applying appropriate Maxwellian averaging of cross sections and
    obtaining rates in [:math:`s^-1`] units. This method expects all neutral components to be given in a
    dictionary with a structure that is independent of NBI model.

    Note that while Ti and ne may be time-dependent, with a time base given by times_kp, the FSA
    neutrals are expected to be time-independent. Hence, the resulting CXR rates will only have
    time dependence that reflects changes in Ti and ne, but not the NBI.

    Parameters
    ----------
    neut_fsa : dict
        Dictionary containing FSA neutral densities in the form that is output by :py:meth:`get_neutrals_fsa`.
    q : int or float
        Charge of impurity species
    rhop_kp : array-like
        Sqrt of poloidal flux radial coordinate for Ti profiles.
    times_kp : array-like
        Time base on which Ti_eV is given [s]. 
    Ti_eV : array-like
        Ion temperature profile on the rhop_kp, times_kp bases, in units of :math:`eV`.
    ne_cm3 : array-like
        Electron density profile on the rhop_kp, times_kp bases, in units of :math:`cm^{-3}`.
    include_fast : bool, optional
        If True, include CXR rates from fast NBI neutrals. Default is True. 
    include_halo : bool, optional
        If True, include CXR rates from themral NBI halo neutrals. Default is True. 
    debug_plots : bool, optional
        If True, plot several plots to assess the quality of the calculation. 

    Returns
    -------
    rates : dict
        Dictionary containing CXR rates from NBI neutrals. This dictionary has analogous form to the 
        :py:meth:`get_neutrals_fsa` function, e.g. we have 

        .. code-block:: python
    
            rates[beam][f'n={n_level}']['halo']

        Rates are on a radial grid corresponding to the input neut_fsa['rhop']. 

    Notes
    -----
    For details on inputs and outputs, it is recommendeded to look at the internal plotting functions. 

    """
    m_bckg = neut_fsa['m_bckg']  # amu
    rhop = neut_fsa['rhop']

    if len(times_kp) == 1:
        Ti_keV_interp = np.atleast_2d(interp1d(rhop_kp, Ti_eV[:, 0] / 1e3, bounds_error=False, fill_value=3e-3)(rhop)).T
        ne_cm3_interp = np.atleast_2d(interp1d(rhop_kp, ne_cm3[:, 0])(rhop)).T
    else:
        Ti_keV_interp = RectBivariateSpline(times_kp, rhop_kp, Ti_eV / 1e3)(times_kp, rhop)
        ne_cm3_interp = RectBivariateSpline(times_kp, rhop_kp, ne_cm3)(times_kp, rhop)

    # collect rates for each energy component and excited state (ONLY fast neutrals here)
    rates = {}
    rates['rhop'] = rhop
    rates['times'] = times_kp
    rates['cxr_total'] = np.zeros((len(rhop), len(times_kp)))

    # setup dictionaries here in case include_fast=False
    for beam in beams:
        rates[beam] = {}
        rates[beam]['cxr_total'] = np.zeros((len(rhop), len(times_kp)))
        for n_level in [0, 1, 2]:
            rates[beam][f'n={n_level}'] = {}

    if include_fast:
        # compute impurity recombination rate with fast neutral populations
        for beam in beams:
            m_beam = neut_fsa[beam]['m_beam']  # amu
            for n_level in [0, 1, 2]:
                for comp in ['f', 'h', 't']:
                    rates[beam][f'n={n_level}'][comp] = {}

                    # energy of each beam component
                    energy = neut_fsa[beam][f'{comp}_energy']  # keV

                    # create cross section function only as a function of energy/amu for Maxwellian average
                    sigma_fun = lambda E_per_amu: js_sigma(E_per_amu, q, n1=n_level + 1, type='cx')  # cm^2

                    rate = bt_rate_maxwell_average(sigma_fun, Ti_keV_interp, energy, m_bckg, m_beam, n_level + 1.0)  # .T
                    rates[beam][f'n={n_level}'][comp] = rate * neut_fsa[beam][f'n={n_level}'][f'{comp}dens'][:, np.newaxis]

                    # we are eventually interested in the total:
                    rates[beam]['cxr_total'] += rates[beam][f'n={n_level}'][comp]

            # sum over all beams:
            rates['cxr_total'] += rates[beam]['cxr_total']

    if include_halo:
        # fetch CCD file for W from ADAS and only pick appropriate charge (Rydberg assumption)
        atom_data = atomic.get_atom_data('W',files=['ccd'])
        alpha_cx_rates = atomic.interp_atom_prof(atom_data['ccd'], np.log10(ne_cm3_interp), np.log10(Ti_keV_interp*1e3), x_multiply=False) # cm^3/s
        rate = alpha_cx_rates[:,q-1,0]   # no recombination of neutral stage
        
        # Now add recombination from halo neutrals (all thermal interactions) - n-unresolved for impurities
        for beam in beams:
            m_beam = neut_fsa[beam]['m_beam']  # amu

            for n_level in [0, 1, 2]:
                rates[beam][f'n={n_level}']['halo'] = rate[:, None] * neut_fsa[beam][f'n={n_level}']['dcx+halo'][:, None]
                rates[beam]['cxr_total'] += rates[beam][f'n={n_level}']['halo']
                rates['cxr_total'] += rates[beam][f'n={n_level}']['halo']

    if debug_plots:
        ls_cycle = get_ls_cycle()

        fig = plt.figure()
        fig.set_size_inches(9,6, forward=True)
        a1 = plt.subplot2grid((4, 4), (0, 0), rowspan=4, colspan=3)
        a2 = plt.subplot2grid((4, 4), (0, 3), rowspan=4, colspan=1)

        for beam in beams:
            for n_level in [0, 1, 2]:
                if include_fast:
                    for comp in ['f', 'h', 't']:
                        # time average to reduce number of lines
                        lss = next(ls_cycle)
                        a1.plot(rhop, np.mean(rates[beam][f'n={n_level}'][comp], axis=-1), lss)
                        a2.plot([], [], lss, lw=2.0, label=f'{beam} {comp}-energy, n={n_level+1}')
                if include_halo:
                    lss = next(ls_cycle)
                    a1.plot(rhop, np.mean(rates[beam][f'n={n_level}']['halo'], axis=-1), lss)
                    a2.plot([], [], lss, lw=2.0, label=f'{beam} halo, n={n_level+1}')

            lss = next(ls_cycle)
            a1.plot(rhop, np.mean(rates[beam]['cxr_total'], axis=-1), lss)
            a2.plot([], [], lss, label=f'beam {beam} CXR total')

        lss = next(ls_cycle)
        a1.plot(rhop, rates['cxr_total'], lss)
        a2.plot([], [], lss, label=f'CXR total')
        a1.set_xlabel(r'$\rho_p$')
        a1.set_ylabel(fr'CXR rate (q={q}) [$s^{{-1}}$]')
        a2.legend(fontsize=14)
        a2.axis('off')
        fig.tight_layout()

    return rates


def beam_grid(uvw_src, axis, max_radius=255.0):
    """Method to obtain the 3D orientation of a beam with respect to the device.
    The uvw_src and (normalized) axis arrays may be obtained from the d3d_beams method
    of fidasim_lib.py in the FIDASIM module in OMFIT. 

    This is inspired by `beam_grid` in fidasim_lib.py of the FIDASIM module (S. Haskey) 
    in OMFIT. 
    """

    pos = uvw_src + 100 * axis
    rsrc = np.sqrt(uvw_src[0] ** 2 + uvw_src[1] ** 2)
    if rsrc < max_radius:
        print("Source radius:{} cannot be less then max_radius:{}".format(rsrc, max_radius))
        raise ValueError()
    dis = np.sqrt(np.sum((uvw_src - pos) ** 2.0))
    beta = np.arcsin((uvw_src[2] - pos[2]) / dis)
    alpha = np.arctan2((pos[1] - uvw_src[1]), (pos[0] - uvw_src[0]))
    gamma = 0.0

    # Find where the origin has to be along the beam injection
    # axis so that x=0 is at a radius of max_radius
    a = axis[0] ** 2 + axis[1] ** 2
    b = 2 * (uvw_src[0] * axis[0] + uvw_src[1] * axis[1])
    c = uvw_src[0] ** 2 + uvw_src[1] ** 2 - max_radius ** 2
    t = (-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    origin = uvw_src + t * axis
    return alpha, beta, gamma, origin


def rotation_matrix(alpha, beta, gamma):
    """See the table of all rotation possiblities, on the Tait Bryan side
    https://en.wikipedia.org/wiki/Euler_angles#Tait.E2.80.93Bryan_angles
    """
    a = alpha
    b = beta
    g = gamma
    sa = np.sin(a)
    ca = np.cos(a)
    sb = np.sin(b)
    cb = np.cos(b)
    sg = np.sin(g)
    cg = np.cos(g)
    R = np.zeros((3, 3), dtype=float)

    R[0, 0] = ca * cb
    R[0, 1] = ca * sb * sg - cg * sa
    R[0, 2] = sa * sg + ca * cg * sb
    R[1, 0] = cb * sa
    R[1, 1] = ca * cg + sa * sb * sg
    R[1, 2] = cg * sa * sb - ca * sg
    R[2, 0] = -sb
    R[2, 1] = cb * sg
    R[2, 2] = cb * cg
    
    return R


def uvw_xyz(u, v, w, origin, R):
    """
    Computes array elements by multiplying the rows of the first
    array by the columns of the second array. The second array
    must have the same number of rows as the first array has
    columns. The resulting array has the same number of rows as
    the first array and the same number of columns as the second
    array.
    
    See uvw_to_xyz in fidasim.f90
    """
    u, v, w = np.atleast_1d(u), np.atleast_1d(v), np.atleast_1d(w)
    orig_shape = u.shape
    order = 'C'
    uvw = np.transpose(np.array([u.flatten(order=order), v.flatten(order=order), w.flatten(order=order)]))
    uvw_shifted = np.transpose(uvw - origin[np.newaxis, :])
    basis = np.linalg.inv(R)

    xyz = np.dot(basis, uvw_shifted)
    x, y, z = xyz[0, :].reshape(orig_shape, order='C'), xyz[1, :].reshape(orig_shape, order='C'), xyz[2, :].reshape(orig_shape, order='C')
    return x, y, z


def xyz_uvw(x, y, z, origin, R):
    """
    Computes array elements by multiplying the rows of the first
    array by the columns of the second array. The second array
    must have the same number of rows as the first array has
    columns. The resulting array has the same number of rows as
    the first array and the same number of columns as the second
    array.
    
    See xyz_to_uvw in fidasim.f90
    """
    x, y, z = np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)
    orig_shape = x.shape
    order = 'C'
    xyz = np.array([x.flatten(order=order), y.flatten(order=order), z.flatten(order=order)])

    basis = R

    uvw = np.dot(basis, xyz) + origin[:, np.newaxis]
    u, v, w = uvw[0, :].reshape(orig_shape, order='C'), uvw[1, :].reshape(orig_shape, order='C'), uvw[2, :].reshape(orig_shape, order='C')
    return u, v, w








def bt_rate_maxwell_average(sigma_fun, Ti_keV, E_beam, m_bckg, m_beam, n_level):
    """Calculates Maxwellian reaction rate for a beam with atomic mass "m_beam", 
    energy "E_beam", firing into a target with atomic mass "m_bckg" and temperature "T".

    The "sigma_fun" argument must be a function for a specific charge and n-level of the beam particles.
    Ref: FIDASIM atomic_tables.f90 bt_maxwellian_n_m.

    Parameters
    ----------
    sigma_fun: :py:meth
        Function to compute a specific cross section [:math:`cm^2`], function of energy/amu ONLY.
        Expected call form: sigma_fun(erel/ared)
    Ti_keV : float, 1D or 2D array
        Target temperature [keV]. Results will be computed for each Ti_keV value in a vectorized manner.
    E_beam : float
        Beam energy [keV]
    m_bckg : float
        Target atomic mass [amu]
    m_beam : float
        Beam atomic mass [amu]
    n_level :int
        n-level of beam. This is used to evaluate the hydrogen ionization potential,
        below which an electron is unlikely to charge exchange with surrounding ions.

    Returns
    -------
    rate : float, 1D or 2D array
        output reaction rate in [cm^3/s] units
    """
    from scipy import constants as consts

    # enforce expected shape
    Ti_keV = np.atleast_2d(Ti_keV)

    # radial and parallel velocity grids, in units of thermal velocity
    vr = np.linspace(0.0, 4.0, 30)
    vz = np.linspace(-4, 4.0, 60)

    # normalized energy and temperature
    E_beam_per_amu = E_beam / m_beam  # E_bar
    T_per_amu = np.maximum(Ti_keV, 1.0e-6) / m_bckg  # T_bar

    # beam/target reduced mass:
    ared = m_bckg * m_beam / (m_bckg + m_beam)
    dE = (13.6e-3) / (n_level ** 2)  # hydrogen ionization potential

    v_therm = np.sqrt(2.0 * T_per_amu * 1.0e3 * consts.e / consts.m_p) * 1e2

    zb = np.sqrt(E_beam_per_amu / T_per_amu)  # sqrt(E_bar/T_bar)
    u2_to_erel = ared * T_per_amu

    if ared <= 0.5:  # for electron interactions
        ared = 1.0

    fr = np.zeros((Ti_keV.shape[0], Ti_keV.shape[1], len(vr)))
    fz = np.zeros((Ti_keV.shape[0], Ti_keV.shape[1], len(vz)))

    for i in np.arange(len(vz)):
        for j in np.arange(len(vr)):
            # relative square velocity:
            u2 = (zb - vz[i]) ** 2 + vr[j] ** 2
            Erel = u2_to_erel * u2

            sig = np.zeros_like(Erel)
            mask = Erel >= dE  # no possible interaction below hydrogen ionization potential
            sig[mask] = sigma_fun(Erel[mask] / ared)

            fr[:, :, j] = sig * np.sqrt(u2) * np.exp(-(vz[i] ** 2.0 + vr[j] ** 2.0)) * vr[j]

        fz[:, :, i] = scipy.integrate.simps(fr, vr, axis=-1)

    # effective maxwellian-averaged rate:
    sig_eff = (2.0 / np.sqrt(np.pi)) * scipy.integrate.simps(fz, vz, axis=-1)
    rate = sig_eff * v_therm

    return rate




def tt_rate_maxwell_average(sigma_fun, Ti_keV, m_i, m_n, n_level):
    """Calculates Maxwellian reaction rate for an interaction between two thermal populations,
    assumed to be of neutrals (mass m_n) and background ions (mass m_i).

    The 'sigma_fun' argument must be a function for a specific charge and n-level of the neutral 
    particles. This allows evaluation of atomic rates for charge exchange interactions between thermal
    beam halos and background ions.

    Parameters
    ----------
    sigma_fun: python function
        Function to compute a specific cross section [:math:`cm^2`], function of energy/amu ONLY.
        Expected call form: sigma_fun(erel/ared)
    Ti_keV: float or 1D array
        background ion and halo temperature [keV]
    m_i: float
        mass of background ions [amu]
    m_n: float 
        mass of neutrals [amu]
    n_level: int
        n-level of beam. This is used to evaluate the hydrogen ionization potential,
        below which an electron is unlikely to charge exchange with surrounding ions.

    Returns
    -------
    rate : float or 1D array
        output reaction rate in [:math:`cm^3/s`] units

    Notes
    -----
    This does not currently account for the effect of rotation! Doing so will require making the integration in this
    function 2-dimensional.
    """
    Ti_keV = np.atleast_1d(Ti_keV)

    vz = np.linspace(0, 4.0, 60)
    Erel = Ti_keV[:, np.newaxis] * vz[np.newaxis, :] ** 2

    # normalized energy and temperature
    T_per_amu = np.maximum(Ti_keV, 1.0e-6) / m_n  # T_bar

    integrand = (
        lambda erel: bt_rate_maxwell_average(sigma_fun, Ti_keV, erel, m_i, m_n, n_level) * (2.0 * m_n * erel) ** (-0.5) * np.exp(-erel / Ti_keV)
    )

    dE = (13.6e-3) / (n_level ** 2)  # hydrogen ionization potential
    sigma = np.zeros_like(Erel)

    for ie in np.arange(len(vz)):  # loop over vz
        mask = Erel[:, ie] >= dE  # no possible interaction below hydrogen ionization potential
        sigma[mask, ie] = bt_rate_maxwell_average(sigma_fun, Ti_keV[mask], Erel[mask, ie], m_i, m_n, n_level)

    prefactor = np.sqrt(2.0 / (np.pi * T_per_amu)) ** (-0.5)
    sigmav = prefactor * scipy.integrate.simps(sigma, Erel, axis=-1)

    return sigmav

