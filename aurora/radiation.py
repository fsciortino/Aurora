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

import os,sys
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
plt.ion()
from scipy import constants
import warnings, copy
from scipy import constants

from . import atomic
from . import adas_files
from . import plot_tools


def compute_rad(imp, nz, ne, Te,
                n0 = None, Ti = None, ni = None, adas_files_sub = {},
                prad_flag=False, sxr_flag=False,
                thermal_cx_rad_flag=False, spectral_brem_flag=False):
    '''Calculate radiation terms corresponding to a simulation result. The nz,ne,Te,n0,Ti,ni arrays
    are normally assumed to be given as a function of (time,nZ,space), but time and space may 
    be substituted by other coordinates (e.g. R,Z)
    
    Result can be conveniently plotted with a time-slider using, for example

    .. code-block:: python

        aurora.slider_plot(rhop,time, res['line_rad'].transpose(1,2,0)/1e6,
            xlabel=r'$\\rho_p$', ylabel='time [s]', 
            zlabel=r'$P_{rad}$ [$MW$]',
            plot_sum=True,
            labels=[f'Ca$^{{{str(i)}}}$' for i in np.arange(res['line_rad'].shape[1])])

    All radiation outputs are given in :math:`W cm^{-3}`, consistently with units of :math:`cm^{-3}`
    given for inputs.

    Parameters
    ----------
    imp : str
        Impurity symbol, e.g. Ca, F, W
    nz : array (time, nZ, space) [:math:`cm^{-3}`]
        Dictionary with impurity density result, as given by :py:func:`~aurora.core.run_aurora` method.
    ne : array (time,space) [:math:`cm^{-3}`]
        Electron density on the output grids.
    Te : array (time,space) [eV]
        Electron temperature on the output grids.
    n0 : array(time,space), optional [:math:`cm^{-3}`]
        Background neutral density (assumed of hydrogen-isotopes).
        This is only used if thermal_cx_rad_flag=True.
    Ti : array (time,space) [eV]
        Main ion temperature (assumed of hydrogen-isotopes). This is only used
        if thermal_cx_rad_flag=True. If not set, Ti is taken equal to Te. 
    adas_files_sub : dict
        Dictionary containing ADAS file names for radiation calculations, possibly including keys
        "plt","prb","prc","pls","prs","pbs","brs"
        Any file names that are needed and not provided will be searched in the 
        :py:meth:`~aurora.adas_files.adas_files_dict` dictionary. 
    prad_flag : bool, optional
        If True, total radiation is computed (for each charge state and their sum)
    sxr_flag : bool, optional
        If True, soft x-ray radiation is computed (for the given 'pls','prs' ADAS files)
    thermal_cx_rad_flag : bool, optional
        If True, thermal charge exchange radiation is computed.
    spectral_brem_flag : bool, optional
        If True, spectral bremstrahlung is computed (based on available 'brs' ADAS file)

    Returns
    -------
    res : dict
        Dictionary containing radiation terms, depending on the activated flags. 

    Notes
    -----
    The structure of the "res" dictionary is as follows.
    
    If prad_flag=True,
    
    res['line_rad'] : array (nt,nZ,nr)- from ADAS "plt" files
        Excitation-driven line radiation for each impurity charge state.
    res['cont_rad'] : array (nt,nZ,nr)- from ADAS "prb" files
        Continuum and line power driven by recombination and bremsstrahlung for impurity ions.
    res['brems'] : array (nt,nr)- analytic formula. 
        Bremsstrahlung produced by electron scarrering at fully ionized impurity 
        This is only an approximate calculation and is more accurately accounted for in the 
        'cont_rad' component.
    res['thermal_cx_cont_rad'] : array (nt,nZ,nr)- from ADAS "prc" files
        Radiation deriving from charge transfer from thermal neutral hydrogen to impurity ions.
        Returned only if thermal_cx_rad_flag=True.
    res['tot'] : array (nt,nZ,nr)
        Total unfilted radiation, summed over all charge states, given by the sum of all known 
        radiation components.
    
    If sxr_flag=True,

    res['sxr_line_rad'] : array (nt,nZ,nr)- from ADAS "pls" files
        Excitation-driven line radiation for each impurity charge state in the SXR range.
    res['sxr_cont_rad'] : array (nt,nZ,nr)- from ADAS "prs" files
        Continuum and line power driven by recombination and bremsstrahlung for impurity ions
        in the SXR range. 
    res['sxr_brems'] : array (nt,nZ,nr)- from ADAS "pbs" files
        Bremsstrahlung produced by electron scarrering at fully ionized impurity in the SXR range.
    res['sxr_tot'] : array (nt,nZ,nr)
        Total radiation in the SXR range, summed over all charge states, given by the sum of all known 
        radiation components in the SXR range. 

    If spectral_brem_flag,

    res['spectral_brems'] : array (nt,nZ,nr) -- from ADAS "brs" files
        Bremsstrahlung at a specific wavelength, depending on provided "brs" file. 
    '''
    res = {}

    Z_imp = nz.shape[1] - 1
    logTe = np.log10(Te)
    logne = np.log10(ne)

    # calculate total radiation
    if prad_flag:

        atom_data = atomic.get_atom_data(imp, files = {'plt': adas_files_sub.get('plt',None)})
        pltt = atomic.interp_atom_prof(atom_data['plt'],logne,logTe) # W

        atom_data = atomic.get_atom_data(imp, files = {'prb': adas_files_sub.get('prb',None)})
        prb = atomic.interp_atom_prof(atom_data['prb'],logne,logTe) # W

        # line radiation for each charge state
        res['line_rad'] = np.maximum(nz[:,:-1] * pltt, 1e-60) # no line rad for fully stripped ion       

        # total continuum radiation (NB: neutrals do not have continuum radiation)
        res['cont_rad'] = nz[:,1:] * prb

        # impurity brems (inaccurate Gaunt factor!) -- already included in 'cont_rad'
        res['brems'] = atomic.impurity_brems(nz, ne, Te)

        # Total unfiltered radiation: 
        res['tot'] = res['line_rad'].sum(1) + res['cont_rad'].sum(1) 

        if thermal_cx_rad_flag:
            if n0 is None:
                raise ValueError(
                    'Requested thermal CX emission to be computed, '
                    'but no background neutral density was provided!')
            if Ti is None:
                warnings.warn('Requested thermal CX emission to be computed '
                              'but no Ti values were provided! Setting Ti=Te',
                              RuntimeWarning)
                Ti = copy.deepcopy(Te)

            # make sure that n0 and Ti are given as 2D:
            assert n0.ndim==2 and Ti.ndim==2
            
            logTi = np.log10(Ti)
            try:
                # thermal CX radiation to total recombination and continuum radiation terms:
                atom_data = atomic.get_atom_data(imp, files = {'prc': adas_files_sub.get('prc',None)})

                # prc has weak dependence on density, so no difference between using ni or ne
                prc = atomic.interp_atom_prof(atom_data['prc'],logne,logTi,x_multiply=False) # W

                # broadcast n0 to dimensions (nt,nZ,nr):
                res['thermal_cx_cont_rad'] = nz[:,1:] * n0[:,None] * prc

                # add to total unfiltered radiation:
                res['tot'] += res['thermal_cx_cont_rad'].sum(1)
            except:
                res['thermal_cx_cont_rad'] = 0
                       
    if sxr_flag: # SXR-filtered radiation (spectral range depends on filter used for files)
        atom_data = atomic.get_atom_data(imp, files = {'pls': adas_files_sub.get('pls',None)})
        pls = atomic.interp_atom_prof(atom_data['pls'],logne,logTe) # W

        atom_data = atomic.get_atom_data(imp, files = {'prs': adas_files_sub.get('prs', None)})
        prs = atomic.interp_atom_prof(atom_data['prs'],logne,logTe) # W

        # SXR line radiation for each charge state
        res['sxr_line_rad'] = np.maximum(nz[:,:-1] * pls, 1e-60)

        # SXR continuum radiation for each charge state
        res['sxr_cont_rad'] = nz[:,1:] * prs

        try:
            # impurity bremsstrahlung in SXR range -- already included in 'sxr_cont_rad'
            atom_data = atomic.get_atom_data(imp, files = {'pbs': adas_files_sub.get('pbs',None)})
            pbs = atomic.interp_atom_prof(atom_data['pbs'],logne,logTe) # W
            res['sxr_brems'] = nz[:,1:] * pbs 
        except:
            # pbs file not available by default for this ion. Users may specify it in adas_files_sub
            pass
        
        # SXR total radiation
        res['sxr_tot'] = res['sxr_line_rad'].sum(1) + res['sxr_cont_rad'].sum(1)

        
    if spectral_brem_flag:  # spectral bremsstrahlung (i.e. brems at a specific wavelength)

        atom_data = atomic.get_atom_data(imp, files = {'brs': adas_files_sub.get('brs',None)})
        x,y,tab = atom_data['brs']
        brs = atomic.interp_atom_prof((x,y,tab.T),None,logTe) # W

        # interpolate on Z grid of impurity of interest
        logZ_rep = np.log10(np.arange(Z_imp)+1)
        brs = interp1d(x, brs,axis=1,copy=False,assume_sorted=True)(logZ_rep)

        # Note: no spectral bremsstrahlung from neutral stage
        res['spectral_brems'] = nz[:,1:] * brs

    return res


def sync_rad(B_T, ne_cm3, Te_eV, r_min, R_maj):
    '''Calculate synchrotron radiation following Trubnikov's formula [1]_.
    We make use of a simplified formulation as given by Zohm [2]_.

    Parameters
    -----------------
    B_T : float or 1D array
        Magnetic field amplitude [T].
    ne_cm3 : float or 1D array
        Electron density [:math:`cm^{-3}`]
    Te_eV : float or 1D array
        Electron temperature [:math:`eV`]
    r_min : float
        Minor radius [m].
    R_maj : float
         Major radius [m].

    Returns
    array
        Rate of synchrotron radiation [:math:`W/cm^3`]

    References
    -----------------
    
    .. [1] Trubnikov, JETP Lett. 16 25 (1972)
    .. [2] Zohm et al., Journal of Fusion Energy (2019) 38:3-10
    
    '''
    return 1.32e-7 * (B_T * Te_eV/1e3)**2.5 * np.sqrt(ne_cm3*1e-14/r_min)*\
        (1. + 18.*r_min/(R_maj * np.sqrt(Te_eV/1e3)))



def radiation_model(imp, rhop, ne_cm3, Te_eV, geqdsk,
                    adas_files_sub={}, n0_cm3=None, Ti_eV=None, nz_cm3=None, frac=None, plot=False):
    '''Model radiation from a fixed-impurity-fraction model or from detailed impurity density
    profiles for the chosen ion. This method acts as a wrapper for :py:func:`~aurora.compute_rad`,
    calculating radiation terms over the radius and integrated over the plasma cross section. 

    Parameters
    ----------
    imp : str (nr,)
        Impurity ion symbol, e.g. W
    rhop : array (nr,)
        Sqrt of normalized poloidal flux array from the axis outwards
    ne_cm3 : array (nr,)
        Electron density in :math:`cm^{-3}` units.
    Te_eV : array (nr,)
        Electron temperature in eV
    geqdsk : dict, optional
        EFIT gfile as returned after postprocessing by the :py:mod:`omfit_classes.omfit_eqdsk` 
        package (OMFITgeqdsk class).
    adas_files_sub : dict
        Dictionary containing ADAS file names for forward modeling and/or radiation calculations.
        Possibly useful keys include
        "scd","acd","ccd","plt","prb","prc","pls","prs","pbs","brs"
        Any file names that are needed and not provided will be searched in the 
        :py:meth:`~aurora.adas_files.adas_files_dict` dictionary. 
    n0_cm3 : array (nr,), optional
        Background ion density (H,D or T). If provided, charge exchange (CX) 
        recombination is included in the calculation of charge state fractional 
        abundances.
    Ti_eV : array (nr,), optional
        Background ion density (H,D or T). This is only used if CX recombination is 
        requested, i.e. if n0_cm3 is not None. If not given, Ti is set equal to Te. 
    nz_cm3 : array (nr,nz), optional
        Impurity charge state densities in :math:`cm^{-3}` units. Fractional abundancies can 
        alternatively be specified via the :param:frac parameter for a constant-fraction
        impurity model across the radius. If provided, nz_cm3 is used. 
    frac : float, optional
        Fractional abundance, with respect to ne, of the chosen impurity. 
        The same fraction is assumed across the radial profile. If left to None,
        nz_cm3 must be given. 
    plot : bool, optional
        If True, plot a number of diagnostic figures. 

    Returns
    -------
    res : dict
        Dictionary containing results of radiation model.  

    '''
    if nz_cm3 is None:
        assert frac is not None
    else:
        if nz_cm3.ndim!=2 or nz_cm3.shape[0]!=len(rhop):
            raise ValueError('Input nz_cm3 must have dimensions (nr,nz)!')

    # limit all considerations to inside LCFS
    ne_cm3 = ne_cm3[rhop<=1.]
    Te_eV = Te_eV[rhop<=1.]
    if nz_cm3 is not None:
        nz_cm3 = nz_cm3[rhop<=1.]
    if n0_cm3 is not None:
        n0_cm3 = n0_cm3[rhop<=1.]
    rhop = rhop[rhop<=1.]

    # extract flux surface volumes from geqdsk
    psin_ref = geqdsk['fluxSurfaces']['geo']['psin']
    rhop_ref = np.sqrt(psin_ref) # sqrt(norm. pol. flux)
    vol = interp1d(rhop_ref, geqdsk['fluxSurfaces']['geo']['vol'])(rhop)
    
    # create results dictionary
    out = {}
    out['rhop'] = rhop
    out['ne_cm3'] = ne_cm3
    out['Te_eV'] = Te_eV
    out['vol'] = vol
    if n0_cm3 is not None:
        out['n0_cm3'] = n0_cm3

    if nz_cm3 is None:
        # load ionization and recombination rates
        atom_files = {}
        atom_files['acd'] = adas_files_sub.get('acd', adas_files.adas_files_dict()[imp]['acd'])
        atom_files['scd'] = adas_files_sub.get('scd', adas_files.adas_files_dict()[imp]['scd'])
        if n0_cm3 is not None:
            atom_files['ccd'] = adas_files_sub.get('ccd',adas_files.adas_files_dict()[imp]['ccd'])
        
        # now load ionization and recombination rates
        atom_data = atomic.get_atom_data(imp,files=atom_files)

        if n0_cm3 is None:
            # obtain fractional abundances without CX:
            _Te, out['fz'] = atomic.get_frac_abundances(atom_data,ne_cm3,Te_eV,rho=rhop, plot=plot)
        else:
            # include CX for ionization balance:
            _Te, out['fz'] = atomic.get_frac_abundances(atom_data,ne_cm3,Te_eV,rho=rhop, plot=plot,
                                                   include_cx=True, n0_by_ne=n0_cm3/ne_cm3)
        
        # Impurity densities
        nz_cm3 = frac * ne_cm3[None,:,None] * out['fz'][None,:,:]  # (time,nZ,space)
    else:
        # set input nz_cm3 into the right shape for compute_rad: (time,space,nz)
        nz_cm3 = nz_cm3[None,:,:]

        # calculate fractional abundances 
        fz = nz_cm3[0,:,:].T/np.sum(nz_cm3[0,:,:],axis=1)
        out['fz'] = fz.T  # (nz,space)


    # compute radiated power components for impurity species
    rad = compute_rad(imp, nz_cm3.transpose(0,2,1), ne_cm3[None,:], Te_eV[None,:],
                      n0=n0_cm3[None,:] if n0_cm3 is not None else None,
                      Ti=Te_eV[None,:] if Ti_eV is None else Ti_eV[None,:],
                      adas_files_sub=adas_files_sub,
                      prad_flag=True, sxr_flag=False,
                      thermal_cx_rad_flag= n0_cm3 is not None,
                      spectral_brem_flag=False)

    # radiation terms -- converted from W/cm^3 to W/m^3
    out['line_rad_dens'] = rad['line_rad'][0,:,:]*1e6
    out['cont_rad_dens'] = rad['cont_rad'][0,:,:]*1e6
    out['brems_dens'] = rad['brems'][0,:,:]*1e6
    out['rad_tot_dens'] = rad['tot'][0,:]*1e6
    
    # cumulative integral over all volume
    out['line_rad'] = cumtrapz(out['line_rad_dens'], vol, initial=0.)
    out['line_rad_tot'] = cumtrapz(out['line_rad_dens'].sum(0), vol, initial=0.)
    out['cont_rad'] = cumtrapz(out['cont_rad_dens'], vol, initial=0.)
    out['brems'] = cumtrapz(out['brems_dens'], vol, initial=0.)
    out['rad_tot'] = cumtrapz(out['rad_tot_dens'], vol, initial=0.)

    if n0_cm3 is not None:
        out['thermal_cx_rad_dens'] = rad['thermal_cx_cont_rad'][0,:,:]*1e6
        out['thermal_cx_rad'] = cumtrapz(out['thermal_cx_rad_dens'].sum(0), vol, initial=0.)
        
    # total power is the last element of the cumulative integral
    out['Prad'] = out['rad_tot'][-1]
    
    #print(f'Total {imp} line radiation power: {out["line_rad_tot"][-1]/1e6:.3f} MW')
    #print(f'Total {imp} continuum radiation power: {out["cont_rad"].sum(0)[-1]/1e6:.3f} MW')
    #print(f'Total {imp} bremsstrahlung radiation power: {out["brems"].sum(0)[-1]/1e6:.3f} MW')
    #if n0_cm3 is not None: print(f'Thermal CX power: {out["thermal_cx_rad"][-1]/1e6:.3f} MW')
    #print(f'Total radiated power: {out["Prad"]/1e6:.3f} MW')

    # calculate average charge state Z across radius
    out['Z_avg'] = np.sum(np.arange(out['fz'].shape[1])[:,None] * out['fz'].T, axis=0)
    
    if plot:
        # plot power in MW/m^3
        fig,ax = plt.subplots()
        ax.plot(rhop, out['line_rad_dens'].sum(0)/1e6, label=r'$P_{rad,line}$')
        ax.plot(rhop, out['cont_rad_dens'].sum(0)/1e6, label=r'$P_{cont}$')
        #ax.plot(rhop, out['brems_dens'].sum(0)/1e6, label=r'$P_{brems}$')
        ax.plot(rhop, out['rad_tot_dens']/1e6, label=r'$P_{rad,tot}$')
        ax.set_xlabel(r'$\rho_p$')
        ax.set_ylabel(fr'$P_{{rad}}$ {imp} [$MW/m^3$]')
        ax.legend().set_draggable(True)
        
        # plot cumulative power in MW 
        fig,ax = plt.subplots()
        ax.plot(rhop, out['line_rad'].sum(0)/1e6, label=r'$P_{rad,line}$')
        ax.plot(rhop, out['cont_rad'].sum(0)/1e6, label=r'$P_{cont}$')
        #ax.plot(rhop, out['brems'].sum(0)/1e6, label=r'$P_{brems}$')
        ax.plot(rhop, out['rad_tot']/1e6, label=r'$P_{rad,tot}$')
        ax.set_xlabel(r'$\rho_p$')
        ax.set_ylabel(fr'$P_{{rad}}$ {imp} [MW]')
        fig.suptitle('Cumulative power')
        ax.legend().set_draggable(True)
        plt.tight_layout()
        
        # plot line radiation for each charge state
        fig = plt.figure(figsize=(10,7))
        colspan = 8 if out['line_rad_dens'].shape[0]<50 else 7
        a_plot = plt.subplot2grid((10,10),(0,0),rowspan = 10, colspan = colspan, fig=fig) 
        a_legend = plt.subplot2grid((10,10),(0,8),rowspan = 10, colspan = 10-colspan, fig=fig) 
        ls_cycle = plot_tools.get_ls_cycle()
        for cs in np.arange(out['line_rad_dens'].shape[0]):
            ls = next(ls_cycle)
            a_plot.plot(rhop, out['line_rad_dens'][cs,:]/1e6, ls)
            a_legend.plot([], [], ls, label=imp+fr'$^{{{cs}+}}$')
        a_plot.set_xlabel(r'$\rho_p$')
        a_plot.set_ylabel(fr'$P_{{rad}}^{{line}}$ {imp} [$MW/m^3$]')
        ncol_leg = 2 if out['line_rad_dens'].shape[0]<25 else 3
        leg=a_legend.legend(loc='center right', fontsize=11, ncol=ncol_leg).set_draggable(True)
        a_legend.axis('off')
        
        # plot average Z over radius
        fig,ax = plt.subplots()
        ax.plot(rhop, out['Z_avg'])
        ax.set_xlabel(r'$\rho_p$')
        ax.set_ylabel(fr'$\langle Z \rangle$ \ {imp}')
        plt.tight_layout()
        
    return out



def get_main_ion_dens(ne_cm3, ions, rhop_plot=None):
    '''Estimate the main ion density via quasi-neutrality. 
    This requires subtracting from ne the impurity charge state density times Z for each 
    charge state of every impurity present in the plasma in significant amounts. 
    
    Parameters
    ----------
    ne_cm3 : array (time,space)
        Electron density in :math:`cm^{-3}`
    ions : dict
        Dictionary with keys corresponding to the atomic symbol of each impurity under 
        consideration. The values in ions[key] should correspond to the charge state 
        densities for the selected impurity ion in :math:`cm^{-3}`, with shape 
        (time,nZ,space). 
    rhop_plot : array (space), optional
        rhop radial grid on which densities are given. If provided, plot densities of 
        all species at the last time slice over this radial grid. 

    Returns
    -------
    ni_cm3 : array (time,space)
        Estimated main ion density in :math:`cm^{-3}`.
    '''
    ni_cm3 = copy.deepcopy(ne_cm3)

    for imp in ions:
        # extract charge state densities for given ion
        nz_cm3 = ions[imp]
        
        Z_imp = nz_cm3.shape[1] -1  # don't include neutral stage
        Z_n_imp = (np.arange(Z_imp+1)[None,:,None]*nz_cm3).sum(1)
        ni_cm3 -= Z_n_imp

    if rhop_plot is not None:
        fig,ax = plt.subplots()
        ax.plot(rhop_plot, ne_cm3[-1,:], label='electrons')
        for imp in ions:
            ax.plot(rhop_plot, ions[imp][-1,:,:].sum(0), label=imp)
        ax.set_xlabel(r'$\rho_p$')
        ax.set_ylabel(r'$cm^{-3}$')

    return ni_cm3



def read_adf15(path, order=1, plot_lines=[], ax=None, plot_3d=False):
    """Read photon emissivity coefficients from an ADAS ADF15 file.

    Returns a dictionary whose keys are the wavelengths of the lines in angstroms. 
    The value is an interpolant that will evaluate the log10 of the PEC at a desired density 
    and temperature. The power-10 exponentiation of this PEC has units of :math:`photons \cdot cm^3/s`

    Units for interpolation: :math:`cm^{-3}` for density; :math:`eV` for temperature.

    Parameters
    ----------
    path : str
        Path to adf15 file to read.
    order : int, opt
        Parameter to control the order of interpolation. Default is 1 (linear interpolation).
    plot_lines : list
        List of lines whose PEC data should be displayed. Lines should be identified
        by their wavelengths. The list of available wavelengths in a given file can be retrieved
        by first running this function ones, checking dictionary keys, and then requesting a
        plot of one (or more) of them.
    ax : matplotlib axes instance
        If not None, plot on this set of axes.
    plot_3d : bool
        Display PEC data as 3D plots rather than 2D ones.

    Returns
    -------
    log10pec_dict : dict
        Dictionary containing interpolation functions for each of the available lines of the
        indicated type (ionization or recombination). Each interpolation function takes as arguments
        the log-10 of ne and Te and returns the log-10 of the chosen PEC.
    
    Examples
    --------
    To plot the Lyman-alpha photon emissivity coefficients for H (or its isotopes), you can use:

    >>> filename = 'pec96#h_pju#h0.dat' # for D Ly-alpha
    >>> # fetch file automatically, locally, from AURORA_ADAS_DIR, or directly from the web:
    >>> path = aurora.get_adas_file_loc(filename, filetype='adf15')  
    >>>
    >>> # plot Lyman-alpha line at 1215.2 A. 
    >>> # see available lines with log10pec_dict.keys() after calling without plot_lines argument
    >>> log10pec_dict = aurora.read_adf15(path, plot_lines=[1215.2])

    Another example, this time also with charge exchange::

    >>> filename = 'pec96#c_pju#c2.dat'
    >>> path = aurora.get_adas_file_loc(filename, filetype='adf15')
    >>> log10pec_dict = aurora.read_adf15(path, plot_lines=[361.7])

    Metastable-resolved files will be automatically identified and parsed accordingly, e.g.::

    >>> filename = 'pec96#he_pjr#he0.dat'
    >>> path = aurora.get_adas_file_loc(filename, filetype='adf15')
    >>> log10pec_dict = aurora.read_adf15(path, plot_lines=[584.4])

    Notes
    -----
    This function expects the format of PEC files produced via the ADAS adas810 or adas218 routines.

    """
    # find out whether file is metastable resolved
    meta_resolved = path.split('#')[-2][-1]=='r'
    if meta_resolved: print('Identified metastable-resolved PEC file')
    
    with open(path, 'r') as f:
        lines = f.readlines()
    cs = path.split('#')[-1].split('.dat')[0]

    header = lines.pop(0)
    # Get the expected number of lines by reading the header:
    num_lines = int(header.split()[0])
    log10pec_dict = {}

    for i in range(0, num_lines):

        while 'isel' not in lines[0].lower():
            # eliminate variable number of label lines at the top
            _ = lines.pop(0)

        # Get the wavelength, number of densities and number of temperatures
        # from the first line of the entry:
        l = lines.pop(0)
        header = l.split()

        # sometimes the wavelength and its units are not separated:
        try:
            header = [hh.split('A')[0] for hh in header]
        except:
            # lam and A are separated. Delete 'A' unit.
            header = np.delete(header, 1)

        lam = float(header[0])

        if header[1]=='':
            # 2nd element was empty -- annoyingly, this happens sometimes
            num_den = int(header[2])
            num_temp = int(header[3])
        else:
            num_den = int(header[1])
            num_temp = int(header[2])            

        if meta_resolved:
            # index of metastable state
            INDM = int(header[-3].split('/')[0].split('=')[-1])

        # Get the densities:
        dens = []
        while len(dens) < num_den:
            dens += [float(v) for v in lines.pop(0).split()]
        dens = np.asarray(dens)

        # Get the temperatures:
        temp = []
        while len(temp) < num_temp:
            temp += [float(v) for v in lines.pop(0).split()]
        temp = np.asarray(temp)

        # Get the PEC's:
        PEC = []
        while len(PEC) < num_den:
            PEC.append([])
            while len(PEC[-1]) < num_temp:
                PEC[-1] += [float(v) for v in lines.pop(0).split()]
        PEC = np.asarray(PEC)
        
        # find what kind of rate we are dealing with
        if 'recom' in l.lower(): rate_type = 'recom'
        elif 'excit' in l.lower(): rate_type = 'excit'
        elif 'chexc' in l.lower(): rate_type = 'chexc'
        elif 'drsat' in l.lower(): rate_type = 'drsat'
        elif 'ion' in l.lower(): rate_type = 'ioniz'
        else:
            # attempt to report unknown rate type -- this should be fairly robust
            rate_type = l.replace(' ','').lower().split('type=')[1].split('/')[0]

        # sometimes multiple lines of the same rate_type can be listed at the same wavelength
        # separate them here by 1e-6 A
        # while True:
        #     if lam in log10pec_dict and rate_type in log10pec_dict[lam]:
        #         lam += 1e-6
        #     else:
        #         break

        # create dictionary with keys for each wavelength:
        if lam not in log10pec_dict:
            log10pec_dict[lam] = {}

        # add a key to the log10pec_dict[lam] dictionary for each type of rate: recom, excit or chexc
        # interpolate PEC on log dens,temp scales
        pec_fun = RectBivariateSpline(
            np.log10(dens),
            np.log10(temp),
            np.log10(PEC),   # NB: interpolation of log10 of PEC to avoid issues at low ne or Te
            kx=order,
            ky=order
        )
        
        if meta_resolved:
            if rate_type not in log10pec_dict[lam]:
                log10pec_dict[lam][rate_type] = {}
            log10pec_dict[lam][rate_type][f'meta{INDM}'] = pec_fun
        else:
            log10pec_dict[lam][rate_type] = pec_fun
            
        if lam in plot_lines:

            # plot PEC values over ne,Te grid given by ADAS, showing interpolation quality
            NE, TE = np.meshgrid(dens, temp)
            
            PEC_eval = 10**pec_fun.ev(np.log10(NE), np.log10(TE)).T

            # plot PEC rates
            _ax = _plot_pec(dens,temp, PEC, PEC_eval, lam,cs,rate_type, ax, plot_3d)

            meta_str = ''
            if meta_resolved: meta_str = f' , meta = {INDM}'
            _ax.set_title(cs + r' , $\lambda$ = '+str(lam) +' $\AA$, '+rate_type+meta_str)
            plt.tight_layout()

    return log10pec_dict



def _plot_pec(dens, temp, PEC, PEC_eval, lam,cs,rate_type, ax=None, plot_3d=False):
    '''Private method to plot PEC data within :py:func:`~aurora.atomic.read_adf15` function.
    '''
    if ax is None:
        f1 = plt.figure(figsize=(9,8))
        if plot_3d:
            ax1 = f1.add_subplot(1,1,1, projection='3d')
        else:
            ax1 = f1.add_subplot(1,1,1)
    else:
        ax1 = ax

    if plot_3d:
        from mpl_toolkits.mplot3d import Axes3D
        
        DENS, TEMP = np.meshgrid(np.log10(dens), np.log10(temp))

        # plot interpolation surface
        ax1.plot_surface(DENS, TEMP, PEC_eval.T, alpha=0.5)
        
        # overplot ADAS data points
        ax1.scatter(DENS.ravel(), TEMP.ravel(), PEC.T.ravel(), color='b')

        if ax is None:
            ax1.set_xlabel('$log_{10}(n_e)$ [cm$^{-3}$]')
            ax1.set_ylabel('$log_{10}(T_e)$ [eV]')
            ax1.set_zlabel('PEC [photons $\cdot cm^3/s$]')

    else:
        # plot in 2D
        labels = ['{:.0e}'.format(ne)+r' $cm^{-3}$' for ne in dens] #ne_eval]

        ls_cycle = plot_tools.get_ls_cycle()
        for ine in np.arange(PEC.shape[0]):
            ls = next(ls_cycle)
            l, = ax1.plot(temp, PEC_eval[ine,:], ls, label=labels[ine])
            #ax1.plot(temp, PEC[ine,:], color=l.get_color(), marker='o', mfc=l.get_color(), ms=5.)
            ax1.plot(temp, PEC[ine,:], ls, marker='o', mfc=l.get_color(), ms=5.)
        ax1.set_xlabel(r'$T_e$ [eV]')
        ax1.set_ylabel('PEC [photons $\cdot cm^3/s$]')
        ax1.set_yscale('log')

        ax1.legend(loc='best').set_draggable(True)

    return ax1


def get_local_spectrum(adf15_file, ion, ne_cm3, Te_eV, ion_exc_rec_dens,
                       Ti_eV=None, n0_cm3=0.0, dlam_A=0.0,
                       plot_spec_tot=True, plot_all_lines=False, no_leg=False, ax=None):
    r'''Plot spectrum based on the lines contained in an ADAS ADF15 file
    at specific values of electron density and temperature. Charge state densities
    can be given explicitely, or alternatively charge state fractions will be automatically 
    computed from ionization equilibrium (no transport). 

    Parameters
    ----------
    adf15_file : str or dict
        Path on disk to the ADAS ADF15 file of interest or dictionary returned when calling the :py:func:`~aurora.radiation.read_adf15`
        with this file path as an argument. All wavelengths and radiating components in the file or dictionary
        will be read/processed.
    ion : str
        Atomic symbol of ion of interest, e.g. 'Ar'
    ne_cm3 : float
        Local value of electron density, in units of :math:`cm^{-3}`.
    Te_eV : float
        Local value of electron temperature, in units of :math:`eV`. This is used to evaluate 
        local values of photon emissivity coefficients.
    ion_exc_rec_dens : list of 3 floats
        Density of ionizing, excited and recombining charge states that may contribute to 
        emission from the given ADF15 file. In the absence of charge state densities from 
        particle transport modeling, these scalars may be taken from the output of 
        :py:func:`aurora.atomic.get_frac_abundances`.
    Ti_eV : float
        Local value of ion temperature, in units of :math:`eV`. This is used to represent the 
        effect of Doppler broadening. If left to None, it is internally set equal to `Te_eV`.
    n0_cm3 : float, optional
        Local density of atomic neutral hydrogen isotopes. This is only used if the provided
        ADF15 file contains charge exchange contributions.
    dlam_A : float or 1D array
        Doppler shift in A. This can either be a scalar or an array of the same shape as the 
        output wavelength array. For the latter option, it is recommended to call this function
        twice to find the wave_final_A array first.         
    plot_spec_tot : bool
        If True, plot total spectrum (sum over all components) from given ADF15 file. 
    plot_all_lines : bool
        If True, plot all individual lines, rather than just the profiles due to different atomic processes.
        If more than 50 lines are included, a down-selection is automatically made to avoid excessive
        memory consumption.
    no_leg : bool
        If True, no plot legend is shown. Default is False, i.e. show legend.
    ax : matplotlib Axes instance
        Axes to plot on if plot=True. If left to None, a new figure is created.

    Returns
    -------
    wave_final_A : 1D array
        Array of wavelengths in units of :math:`\r{A}` on which the total spectrum is returned. 
    spec_ion : 1D array
        Spectrum from ionizing components of the input ADF15 file as a function of wave_final_A.
    spec_exc : 1D array
        Spectrum from excitation components of the input ADF15 file as a function of wave_final_A.
    spec_rr : 1D array
        Spectrum from radiative recombination components of the input ADF15 file as a function of wave_final_A.
    spec_dr : 1D array
        Spectrum from dielectronic recombination components of the input ADF15 file as a function of wave_final_A.
    spec_cx : 1D array
        Spectrum from charge exchange recombination components of the input ADF15 file as a function of wave_final_A.
    ax : matplotlib Axes instance
        Axes on which the plot is returned.

    Notes
    -----
    Including ionizing, excited and recombining charge states allows for a complete description
    of spectral lines that may derive from various atomic processes in a plasma.

    Doppler broadening depends on the local ion temperature and mass of the emitting species. 
    It is modeled here using

    .. math::

        \theta(\nu) = \frac{1}{\sqrt{\pi}\Delta \nu_D} e^{-\left(\frac{\nu - \nu_0}{\Delta \nu_D}\right)^2}

    with the Doppler profile half-width being

    .. math::

        \Delta \nu_D = \frac{1}{\nu_0} \sqrt{\frac{2 T_i}{m}}

    The Doppler shift dlam_A can be calculated from

    .. math::

        \Delta \lambda_v = \lambda \cdot \left( 1 - \frac{v\cdot \cos(\alpha)}{c}\right)

    where :math:`v` is the plasma velocity and :math:`\alpha` is the angle between the line-of-sight 
    and the direction of plasma rotation.

    Refs: S. Loch's and C. Johnson's PhD theses.

    '''
    # ensure input ne,Te,n0 are floats
    ne_cm3=float(ne_cm3)
    Te_eV=float(Te_eV)
    if Ti_eV is None:
        Ti_eV = copy.deepcopy(Te_eV)
    else:
        Ti_eV=float(Ti_eV)
    n0_cm3=float(n0_cm3)

    if isinstance(adf15_file, str):
        # read ADF15 file
        log10pec_dict = read_adf15(adf15_file)
    elif isinstance(adf15_file, dict):
        # user passed dictionary containing a cached log10pec_dict
        log10pec_dict = adf15_file
    else:
        raise ValueError('Unrecognized adf15_file format!')

    # import here to avoid issues when building docs or package
    from omfit_classes.utils_math import atomic_element
    
    # get nuclear charge Z and atomic mass number A
    out = atomic_element(symbol=ion)
    spec = list(out.keys())[0]
    ion_Z = int(out[spec]['Z'])
    ion_A = int(out[spec]['A'])

    nlines = len(list(log10pec_dict.keys()))
    wave_A = np.zeros(nlines)

    pec = {}
    for typ in ['ioniz', 'excit', 'recom','chexc', 'drsat' ]:
        pec[typ] = np.zeros(nlines)
        for ii,lam in enumerate(log10pec_dict):
            wave_A[ii] = lam
            if typ in log10pec_dict[lam]:
                pec[typ][ii] = 0.0
                if isinstance(log10pec_dict[lam][typ], dict): # metastable resolved
                    for metastable in log10pec_dict[lam][typ]: # loop over all metastables
                        pec[typ][ii] += 10**log10pec_dict[lam][typ][metastable].ev(
                            np.log10(ne_cm3),np.log10(Te_eV))
                else:
                    # no metastables
                    pec[typ][ii] += 10**log10pec_dict[lam][typ].ev(np.log10(ne_cm3),np.log10(Te_eV))
                        
    # Doppler broadening
    mass = constants.m_p * ion_A
    dnu_g = np.sqrt(2.*(Ti_eV*constants.e)/mass)*(constants.c/wave_A)/constants.c
    
    # set a variable delta lambda based on the width of the broadening
    _dlam_A = wave_A**2/constants.c* dnu_g * 5 # 5 standard deviations
    
    lams_profs_A =np.linspace(wave_A-_dlam_A, wave_A + _dlam_A, 100, axis=1) 
    
    theta_tmp = 1./(np.sqrt(np.pi)*dnu_g[:,None])*\
                np.exp(-((constants.c/lams_profs_A-constants.c/wave_A[:,None])/dnu_g[:,None])**2)

    # Normalize Gaussian profile
    theta = np.einsum('ij,i->ij', theta_tmp, 1./np.trapz(theta_tmp,x=lams_profs_A,axis=1))
    
    wave_final_A = np.linspace(np.min(lams_profs_A), np.max(lams_profs_A), 100000)
    
    if (plot_all_lines or plot_spec_tot) and ax is None:
        fig,ax = plt.subplots()

    # contributions to spectrum
    source = {'ioniz': 0, 'excit': 1, 'recom': 2, 'chexc': 2, 'drsat': 2}
    spec_ion = np.zeros_like(wave_final_A)
    spec = {}
    spec_tot = np.zeros_like(wave_final_A)
    for typ,c in zip(['ioniz', 'excit', 'recom','drsat','chexc'],['r','b','g','m','c']):
        spec[typ] = np.zeros_like(wave_final_A)
        for ii in np.arange(lams_profs_A.shape[0]):
            comp = interp1d(lams_profs_A[ii,:], ne_cm3*ion_exc_rec_dens[source[typ]]*pec[typ][ii]*theta[ii,:],
                                bounds_error=False, fill_value=0.0)(wave_final_A)
            if plot_all_lines and pec[typ][ii]>np.max(pec[typ])/1000:
                ax.plot(wave_final_A+dlam_A, comp, c=c)
            spec[typ] += comp
        spec_tot += spec[typ]
  
    
    if plot_spec_tot:
        # plot contributions from different processes
        ax.plot(wave_final_A+dlam_A, spec['ioniz'], c='r', ls='--', label='' if no_leg else 'ionization')
        ax.plot(wave_final_A+dlam_A, spec['excit'], c='b', ls='--', label='' if no_leg else 'excitation')
        ax.plot(wave_final_A+dlam_A, spec['recom'], c='g', ls='--', label='' if no_leg else 'radiative recomb')
        ax.plot(wave_final_A+dlam_A, spec['drsat'], c='m', ls='--', label='' if no_leg else 'dielectronic recomb')
        ax.plot(wave_final_A+dlam_A, spec['chexc'], c='c', ls='--', label='' if no_leg else 'charge exchange recomb')

        # total envelope
        ax.plot(wave_final_A+dlam_A, spec_tot, c='k', ls='--', label='' if no_leg else 'total')


    if plot_all_lines or plot_spec_tot:
        if not no_leg: ax.legend(loc='best').set_draggable(True)
        ax.set_xlabel(r'$\lambda$ [$\AA$]')
        ax.set_ylabel(r'$\epsilon$ [A.U.]')
    else:
        ax=None

    # return Doppler-shifted wavelength if dlam_A was given as non-zero
    return wave_final_A+dlam_A, spec['ioniz'], spec['excit'], spec['recom'], spec['drsat'], spec['chexc'], ax



def get_cooling_factors(imp, ne_cm3, Te_eV, n0_cm3=0.0, ion_resolved=False, superstages = [],
                        line_rad_file=None, cont_rad_file=None, sxr=False, plot=True, ax=None):
    '''Calculate cooling coefficients for the given fractional abundances and kinetic profiles.

    Parameters
    ----------
    imp : str
        Atomic symbol of ion of interest
    ne_cm3 : 1D array
        Electron density [:math:`cm^{-3}`], used to find charge state fractions at ionization equilibrium.
    Te_eV : 1D array
        Electron temperature [:math:`eV`] at which cooling factors should be obtained. 
    n0_cm3 : 1D array or float
        Background H/D/T neutral density [:math:`cm^{-3}`] used to account for charge exchange 
        when calculating ionization equilibrium. 
        If left to 0, charge exchange effects are not included.
    ion_resolved : bool
        If True, cooling factors are returned for each charge state. If False, they are summed over charge states.
        The latter option is useful for modeling where charge states are assumed to be in ionization equilibrium 
        (no transport). Default is False.
    superstages : list or 1D array
        List of superstages to consider. An empty list (default) corresponds to the inclusion of all charge states.
        Note that when ion_resolved=False, cooling coefficients are independent of whether superstages are being used or not.
    line_rad_file : str or None
        Location of ADAS ADF11 file containing line radiation data. This can be a PLT (unfiltered) or 
        PLS (filtered) file. If left to None, the default file given in :py:func:`~aurora.adas_files.adas_files_dict` 
        will be used.
    cont_rad_file : str or None
        Location of ADAS ADF11 file containing recombination and bremsstrahlung radiation data. 
        This can be a PRB (unfiltered) or PRS (filtered) file. 
        If left to None, the default file given in :py:func:`~aurora.adas_files.adas_files_dict` 
        will be used.
    sxr : bool
        If True, line radiation, recombination and bremsstrahlung radiation are taken to be from SXR-filtered
        ADAS ADF11 files, rather than from unfiltered files. 
    plot : bool
        If True, plot all radiation components, summed over charge states.
    ax : matplotlib.Axes instance
        If provided, plot results on these axes. 
    
    Returns
    -------
    line_rad_tot : 1D array
        Cooling coefficient from line radiation [:math:`W\cdot m^3`]. 
        Depending on whether :param:`sxr`=True or False, this indicates filtered 
        or unfiltered radiation, respectively.
    cont_rad_tot : 1D array
        Cooling coefficient from continuum radiation [:math:`W\cdot m^3`]. 
        Depending on whether `sxr`=True or False, this indicates filtered
        or unfiltered radiation, respectively. 

    '''
    files = ['scd','acd']
    if n0_cm3 != 0.0: files+=['ccd']
    atom_data_eq = atomic.get_atom_data(imp,files)

    if superstages is None:
        superstages = []

    _Te, fz = atomic.get_frac_abundances(atom_data_eq, ne_cm3, Te_eV, plot=False,
                                           n0_by_ne=n0_cm3/ne_cm3)

    if superstages:
        fz_full = copy.deepcopy(fz)
        _Te, fz = atomic.get_frac_abundances(atom_data_eq, ne_cm3, Te_eV, plot=False,
                                               n0_by_ne=n0_cm3/ne_cm3, superstages=superstages)
    

    # line radiation
    atom_data = atomic.get_atom_data(imp,{'pls' if sxr else 'plt': line_rad_file})
    PLT= atomic.interp_atom_prof(atom_data['pls' if sxr else 'plt'], None, np.log10(Te_eV)) # line radiation [W.cm^3]

    # recombination and bremsstrahlung radiation
    atom_data = atomic.get_atom_data(imp,{'prs' if sxr else 'prb': cont_rad_file})
    PRB = atomic.interp_atom_prof(atom_data['prs' if sxr else 'prb'],None, np.log10(Te_eV)) # continuum radiation [W.cm^3]

    # zero bremstrahlung of neutral stage
    PRB = np.hstack(( np.zeros((_Te.size,1)), PRB))

    # zero line radiation of fully stripped ion stage
    PLT = np.hstack((PLT, np.zeros((_Te.size,1))))
    
    if len(superstages) and fz_full is not None:
        # superstage radiation data
        Z_imp = PRB.shape[1] - 1
        
        # check input superstages
        if 1 not in superstages: # needed to match dimensions with superstage_rates
            print('Warning: 1th superstage was included')
            superstages = np.r_[1,superstages]
        if 0 not in superstages:
            print('Warning: 0th superstage for neutral was included')
            superstages = np.r_[0,superstages]
        if np.any(np.diff(superstages)<=0):
            print('Warning: sorting superstages in increasing order')
            superstages = np.sort(superstages)
        if superstages[-1] > Z_imp:
            raise Exception('The highest superstage must be less than Z_imp = %d'%Z_imp)

        _PLT, _PRB = np.copy(PLT), np.copy(PRB)
        PLT, PRB  =  _PLT[:,superstages],_PRB[:,superstages]
        
        _superstages = np.r_[superstages, Z_imp+1]

        for i in range(len(_superstages)-1):
            if _superstages[i]+1 < _superstages[i+1]:
                weight = np.copy(fz_full[:,_superstages[i]:_superstages[i+1]])
                weight /= np.maximum(weight.sum(1),1e-20)[:,None]

                PRB[:,i] = (_PRB[:,_superstages[i]:_superstages[i+1]]*weight).sum(1)
                PLT[:,i] = (_PLT[:,_superstages[i]:_superstages[i+1]]*weight).sum(1)
    
    PLT *= fz*1e-6  # W.cm^3-->W.m^3
    PRB *= fz*1e-6  # W.cm^3-->W.m^3

    if plot:
        if ax is None:
            fig, ax = plt.subplots()

        if ion_resolved:
            # plot contributions from each charge state at ionization equilibrium
            ls_cycle = plot_tools.get_ls_cycle()
            
            lss = next(ls_cycle)
            for cs in np.arange(fz.shape[1]-1):
                ax.loglog(Te_eV/1e3, PLT[:,cs], lss, lw=2.0, label=f'{imp}{cs+1}+')

                # change line style here because there's no line rad for fully-ionized stage or recombination from neutral stage
                lss = next(ls_cycle)
                
                # show line and continuum recombination components separately
                ax.loglog(Te_eV/1e3, PRB[:,cs], lss, lw=1.0) #, label=f'{imp}{cs}+')

        else:

            # total radiation (includes hard X-ray, visible, UV, etc.)
            l, = ax.loglog(Te_eV/1e3, PRB.sum(1)+ PLT.sum(1), ls='-',  # W.cm^3-->W.m^3
                           label=f'{imp} $L_z$ (total)')
            
            # show line and continuum recombination components separately
            ax.loglog(Te_eV/1e3,  PLT.sum(1), c=l.get_color(), ls='--', # W.cm^3-->W.m^3
                      label='line radiation')
            ax.loglog(Te_eV/1e3, PRB.sum(1), c=l.get_color(), ls='-.', # W.cm^3-->W.m^3
                      label='continuum radiation')
    
        ax.legend(loc='best').set_draggable(True)
        ax.grid('on', which='both')
        ax.set_xlabel('T$_e$ [keV]')
        ax.set_ylabel('$L_z$ [$W$ $m^3$]')
        plt.tight_layout()

    if ion_resolved:
        return PLT[:,:-1], PRB[:,1:]
    else:
        return PLT.sum(1), PRB.sum(1)




def adf15_line_identification(pec_files, lines=None, Te_eV = 1e3, ne_cm3=5e13, mult=[]):
    '''Display all photon emissivity coefficients from the given list of ADF15 files and (optionally) compare to a set
    of chosen wavelengths, given in units of Angstrom.

    Parameters
    -----------------
    pec_files : str or list of str
        Path to a single ADF15 file or a list of files.
    lines : dict, list or 1D array
        Lines to overplot with the loaded PECs to consider overlap within spectrum. This argument may be a dictionary, with
        keys corresponding to line names and values corresponding to wavelengths (in units of Angstrom). If provided as a 
        list or array, this is assumed to contain only wavelengths in Angstrom.
    Te_eV : float
        Single value of electron temperature at which PECs should be evaluated [:math:`eV`].
    ne_cm3 : float
        Single value of electron density at which PECs should be evaluated [:math:`cm^{-3}`].
    mult : list or array
        Multiplier to apply to lines from each PEC file. This could be used for example to rescale the results of
        multiple ADF15 files by the expected fractional abundance or density of each element/charge state.

    Notes
    --------
    To attempt identification of spectral lines, one can load a set of ADF15 files, calculate approximate fractional
    abundances at equilibrium and overplot expected emissivities in a few steps::

    >>> pec_files = ['mypecs1','mypecs2','mypecs3']
    >>> Te_eV=500; ne_cm3=5e13; ion='Ar'   # examples
    >>> atom_data = aurora.atomic.get_atom_data(ion,['scd','acd'])
    >>> _Te, fz = aurora.atomic.get_frac_abundances(atom_data, ne_cm3, Te_eV, plot=False)
    >>> mult = [fz[0,10], fz[0,11], fz[0,12]] # to select charge states 11+, 12+ and 13+, for example
    >>> adf15_line_identification(pec_files, Te_eV=Te_eV, ne_cm3=ne_cm3, mult=mult)
    '''

    fig = plt.figure(figsize=(10,7))
    a_id = plt.subplot2grid((10,10),(0,0),rowspan = 2, colspan = 8, fig=fig)
    ax = plt.subplot2grid((10,10),(2,0),rowspan = 8, colspan = 8, fig=fig, sharex=a_id)
    a_legend = plt.subplot2grid((10,10),(0,8),rowspan = 10, colspan = 2, fig=fig) 
    a_id.axis('off')
    a_legend.axis('off')
    
    ymin = np.inf
    ymax= -np.inf

    if isinstance(pec_files,str):
        pec_files = [pec_files,]

    if len(mult) and len(pec_files)!=len(mult):
        raise ValueError('Different number of ADF15 files and multipliers detected!')
    
    cols = iter(plt.cm.rainbow(np.linspace(0,1,len(pec_files))))

    def _plot_line(log10pec_interp_fun, lam, ymin, ymax, c):
        _pec = _mult * 10 ** log10pec_interp_fun(np.log10(ne_cm3), np.log10(Te_eV))[0, 0]
        if _pec > 1e-20:  # plot only stronger lines
            ymin = min(_pec, ymin)
            ymax = max(_pec, ymax)
            ax.semilogy([lam, lam], [1e-70, _pec], 'o-.', c=c)
        return ymin,ymax

    lams = []
    for pp, pec_file in enumerate(pec_files):
        
        # load single PEC file from given list
        log10_pecs = read_adf15(pec_file)

        _mult = mult[pp] if len(mult) else 1.

        c = next(cols)

        # now plot all ionization-, excitation- and recombination-driven components
        for lam, log10pec_interps in log10_pecs.items():

            for process in ['ioniz','excit','recom']:  # loop over populating processes
                
                if process in log10pec_interps: # often, only excit is relevant
                    if isinstance(log10pec_interps[process],dict):  # metastable resolved
                        for metastable in log10pec_interps[process]: # loop over all metastables
                            ymin,ymax= _plot_line(log10pec_interps[process][metastable], lam, ymin, ymax, c)
                    else: # no metastables
                        ymin,ymax=_plot_line(log10pec_interps[process], lam, ymin, ymax, c)

        lams += log10_pecs.keys()
        if len(pec_files)>1:
            a_legend.plot([],[], c=c, label=pec_file.split('/')[-1])

    
    ax.set_ylim(ymin, ymax*2)
    ax.set_xlim(min(lams) / 1.5, max(lams) * 1.5)

    ax.set_xlabel(r'$\lambda$ [$\AA$]')
    ax.set_ylabel('PEC [phot $\cdot$ cm$^3$/s]')
    a_id.set_title(r'$T_e$ = %d eV, $n_e$ = %.2e cm$^{-3}$' % (Te_eV, ne_cm3))

    # plot location of certain lines of interest
    if lines is not None:
        a_legend.axvline(np.nan, label='Input lines')
        if isinstance(lines, dict):
            for name, wvl in lines.items():
                ax.axvline(wvl, c='k', lw=2.0)
                a_id.text(wvl, 0.1, name, rotation=90) #, clip_on=True)
        else:
            for i, wvl in enumerate(lines):
                ax.axvline(wvl, c='k', lw=2.0)
                a_id.text(wvl, 0.1, str(i), rotation=90) #, clip_on=True)
                
    a_legend.plot([], [], 'o-.', label='PEC ionization')
    a_legend.plot([], [], 'o-', label='PEC excitation')
    a_legend.plot([], [], 'o--', label='PEC recombination')
    
    if len(pec_files)==1:
        # show path of single file that was passed
        a_legend.text(1.05, -0.05, pec_files[0], rotation=90, transform=a_legend.transAxes)
        
    a_legend.legend(loc='best').set_draggable(True)



def adf04_files():
    '''Collection of trust-worthy ADAS ADF04 files. 
    This function will be moved and expanded in ColRadPy in the near future. 
    '''
    files = {}
    files['Ca'] = {}
    files['Ca']['8'] = 'mglike_lfm14#ca8.dat'
    files['Ca']['9'] = 'nalike_lgy09#ca9.dat'
    files['Ca']['10'] = 'nelike_lgy09#ca10.dat'
    files['Ca']['11'] = 'flike_mcw06#ca11.dat'
    files['Ca']['14'] = 'clike_jm19#ca14.dat'
    files['Ca']['15'] = 'blike_lgy12#ca15.dat'
    files['Ca']['16'] = 'belike_lfm14#ca16.dat'
    files['Ca']['17'] = 'lilike_lgy10#ca17.dat'
    files['Ca']['18'] = 'helike_adw05#ca18.dat'

    # TODO: check quality
    files['Al'] = {}
    files['Al']['11'] = 'helike_adw05#al11.dat'
    files['Al']['10'] = 'lilike_lgy10#al10.dat'
    files['Al']['9'] = 'belike_lfm14#al9.dat'
    files['Al']['8'] = 'blike_lgy12#al8.dat'
    files['Al']['7'] = 'clike_jm19#al7.dat'
    files['Al']['6'] = ''
    files['Al']['5'] = ''
    files['Al']['4'] = 'flike_mcw06#al4.dat'
    files['Al']['3'] = 'nelike_lgy09#al3.dat'
    files['Al']['2'] = 'nalike_lgy09#al2.dat'
    files['Al']['1'] = 'mglike_lfm14#al1.dat'

    # TODO: check quality
    files['F'] = {}
    files['F']['8'] = 'copha#h_hah96f.dat'
    files['F']['7'] = 'helike_adw05#f7.dat'
    files['F']['6'] = 'lilike_lgy10#f6.dat'
    files['F']['5'] = 'belike_lfm14#f5.dat'
    files['F']['4'] = 'blike_lgy12#f4.dat'
    files['F']['3'] = 'clike_jm19#f3.dat'

    return files




def get_colradpy_pec_prof(ion, cs, rhop, ne_cm3, Te_eV, 
                          lam_nm, lam_width_nm, adf04_loc,
                          meta_idxs=[0], pec_threshold=1e-20, pec_units=2, plot=True):
    '''Compute radial profile for Photon Emissivity Coefficients (PEC) for lines within the chosen
    wavelength range using the ColRadPy package. This is an alternative to the option of using 
    the :py:func:`~aurora.radiation.read_adf15` function to read PEC data from an ADAS ADF-15 file and 
    interpolate results on ne,Te grids. 

    Parameters
    ----------
    ion : str
        Ion atomic symbol
    cs : str
        Charge state, given in format like '17', indicating total charge of ion (e.g. '17' would be for Li-like Ca)
    rhop : array (nr,)
        Sqrt of normalized poloidal flux radial array
    ne_cm3 : array (nr,)
        Electron density in :math:`cm^{-3}` units
    Te_eV : array (nr,)
        Electron temperature in eV units
    lam_nm : float
        Center of the wavelength region of interest [nm]
    lam_width_nm : float
        Width of the wavelength region of interest [nm]
    adf04_loc : str
        Location from which ADF04 files listed in :py:func:`adf04_files` should be fetched.
    meta_idxs : list of integers
        List of levels in ADF04 file to be treated as metastable states. Default is [0] (only ground state).
    prec_threshold : float
        Minimum value of PECs to be considered, in :math:`photons \cdot cm^3/s`
    pec_units : int
        If 1, results are given in :math:`photons \cdot cm^3/s`; if 2, they are given in :math:`W.cm^3`. 
        Default is 2.
    plot : bool
        If True, plot lines profiles and total.

    Returns
    -------
    pec_tot_prof : array (nr,)
        Radial profile of PEC intensity, in units of :math:`photons \cdot cm^3/s` (if pec_units=1) or 
        :math:`W \cdot cm^3` (if pec_units=2).
    '''
    try:
        # temporarily import this here, until ColRadPy dependency can be set up properly
        from colradpy import colradpy
    except ImportError:
        raise ValueError('Could not import colradpy. Install this from the Github repo!')
    
    files = adf04_files()

    filepath = adf04_repo+files[ion][cs]
    
    crm = colradpy(filepath, meta_idxs, Te_eV, ne_cm3,temp_dens_pair=True,
                   use_recombination=False,
                   use_recombination_three_body=False)
    
    crm.make_ioniz_from_reduced_ionizrates()
    crm.suppliment_with_ecip()
    crm.make_electron_excitation_rates()
    crm.populate_cr_matrix()   # time consuming step
    crm.solve_quasi_static()

    lams = crm.data['processed']['wave_vac']
    lam_sel_idxs = np.where((lams>lam_nm-lam_width_nm/2.)&(lams<lam_nm+lam_width_nm/2.))[0]
    _lam_sel_nm = lams[lam_sel_idxs]

    pecs = crm.data['processed']['pecs'][lam_sel_idxs,0,:]  # 0 index is for excitation component
    pecs_sel_idxs = np.where((np.max(pecs,axis=1)<pec_threshold))[0]
    pecs_sel = pecs[pecs_sel_idxs,:]
    lam_sel_nm = _lam_sel_nm[pecs_sel_idxs]
    
    # calculate total PEC profile
    pec_tot_prof = np.sum(pecs_sel,axis=0)

    if pec_units==2:
        # convert from photons.cm^3/s to W.cm^3
        mults = constants.h * constants.c / (lam_sel_nm*1e-9)
        pecs_sel *= mults[:,None]
        pec_tot_prof *= mults[:,None]
        
    if plot:
        fig,ax = plt.subplots()
        for ll in np.arange(len(lam_sel_nm)):
            ax.plot(rhop, pecs_sel[ll,:], label=fr'$\lambda={lam_sel_nm[ll]:.5f}$ nm')
        ax.plot(rhop, pec_tot_prof, lw=3.0, c='k', label='Total')
        fig.suptitle(fr'$\lambda={lam_nm} \pm {lam_width_nm/2.}$ nm')
        ax.set_xlabel(r'$\rho_p$')
        ax.set_ylabel('PEC [W cm$^3$]' if phot2energy else 'PEC [ph cm$^3$ s$^{-1}$]')
        ax.legend(loc='best').set_draggable(True)

    return pec_tot_prof
