import os,sys
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
plt.ion()
from scipy import constants
import warnings, copy
from scipy.constants import e as q_electron,k as k_B, h, m_p, c as c_speed

from . import atomic
from . import adas_files
from . import plot_tools


def compute_rad(imp, nz, ne, Te,
                n0 = None, Ti = None, ni = None, adas_files_sub = {},
                prad_flag=False, sxr_flag=False,
                thermal_cx_rad_flag=False, spectral_brem_flag=False, ):
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

        if 'plt' in adas_files_sub:  # check if user requested use of a specific file
            atom_data = atomic.get_atom_data(imp, ['plt'],[adas_files_sub['plt']])
        else:  # use default file from adas_files.adas_files_dict()
            atom_data = atomic.get_atom_data(imp, ['plt'])
        pltt = atomic.interp_atom_prof(atom_data['plt'],logne,logTe) # W

        if 'prb' in adas_files_sub:
            atom_data = atomic.get_atom_data(imp, ['prb'],[adas_files_sub['prb']])
        else:
            atom_data = atomic.get_atom_data(imp, ['prb'])
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
            
            # thermal CX radiation to total recombination and continuum radiation terms:
            if 'prc' in adas_files_sub:
                atom_data = atomic.get_atom_data(imp, ['prc'],[adas_files_sub['prc']])
            else:
                atom_data = atomic.get_atom_data(imp, ['prc'])

            # prc has weak dependence on density, so no difference between using ni or ne
            prc = atomic.interp_atom_prof(atom_data['prc'],logne,logTi,x_multiply=False) # W

            # broadcast n0 to dimensions (nt,nZ,nr):
            res['thermal_cx_cont_rad'] = nz[:,1:] * n0[:,None] * prc

            # add to total unfiltered radiation:
            res['tot'] += res['thermal_cx_cont_rad'].sum(1)
                       
    if sxr_flag: # SXR-filtered radiation (spectral range depends on filter used for files)

        if 'pls' in adas_files_sub:
            atom_data = atomic.get_atom_data(imp, ['pls'],[adas_files_sub['pls']])
        else:
            atom_data = atomic.get_atom_data(imp, ['pls'])
        pls = atomic.interp_atom_prof(atom_data['pls'],logne,logTe) # W

        if 'prs' in adas_files_sub:
            atom_data = atomic.get_atom_data(imp, ['prs'],[adas_files_sub['prs']])
        else:
            atom_data = atomic.get_atom_data(imp, ['prs'])
        prs = atomic.interp_atom_prof(atom_data['prs'],logne,logTe) # W

        # SXR line radiation for each charge state
        res['sxr_line_rad'] = np.maximum(nz[:,:-1] * pls, 1e-60)

        # SXR continuum radiation for each charge state
        res['sxr_cont_rad'] = nz[:,1:] * prs

        try:
            # impurity bremsstrahlung in SXR range -- already included in 'sxr_cont_rad'
            if 'pbs' in adas_files_sub:
                atom_data = atomic.get_atom_data(imp, ['pbs'],[adas_files_sub['pbs']])
            else:
                atom_data = atomic.get_atom_data(imp, ['pbs'])
            pbs = atomic.interp_atom_prof(atom_data['pbs'],logne,logTe) # W
            res['sxr_brems'] = nz[:,1:] * pbs 
        except IndexError:
            # pbs file not available by default for this ion. Users may specify it in adas_files_sub
            pass
        
        # SXR total radiation
        res['sxr_tot'] = res['sxr_line_rad'].sum(1) + res['sxr_cont_rad'].sum(1)

        
    if spectral_brem_flag:  # spectral bremsstrahlung (i.e. brems at a specific wavelength)

        if 'brs' in adas_files_sub:
            atom_data = atomic.get_atom_data(imp, ['brs'],[adas_files_sub['brs']])
        else:
            atom_data = atomic.get_atom_data(imp, ['brs'])
        x,y,tab = atom_data['brs']
        brs = atomic.interp_atom_prof((x,y,tab.T),None,logTe) # W

        # interpolate on Z grid of impurity of interest
        logZ_rep = np.log10(np.arange(Z_imp)+1)
        brs = interp1d(x, brs,axis=1,copy=False,assume_sorted=True)(logZ_rep)

        # Note: no spectral bremsstrahlung from neutral stage
        res['spectral_brems'] = nz[:,1:] * brs

    return res





def radiation_model(imp,rhop, ne_cm3, Te_eV, vol,
                    adas_files_sub={}, n0_cm3=None, Ti_eV=None, nz_cm3=None, frac=None, plot=False):
    '''Model radiation from a fixed-impurity-fraction model or from detailed impurity density
    profiles for the chosen ion. This method acts as a wrapper for :py:method:compute_rad(), 
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
    vol : array (nr,)
        Volume of each flux surface in :math:`m^3`. Note the units! We use :math:`m^3` here
        rather than :math:`cm^3` because it is more common to work with :math:`m^3` for 
        flux surface volumes of fusion devices.
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
    
    # limit all considerations to inside LCFS
    ne_cm3 = ne_cm3[rhop<=1.]
    Te_eV = Te_eV[rhop<=1.]
    vol = vol[rhop<=1.]
    if n0_cm3 is not None:
        n0_cm3 = n0_cm3[rhop<=1.]
    rhop = rhop[rhop<=1.]

    # create results dictionary
    out = {}
    out['rhop'] = rhop
    out['ne_cm3'] = ne_cm3
    out['Te_eV'] = Te_eV
    out['vol'] = vol
    if n0_cm3 is not None:
        out['n0_cm3'] = n0_cm3
        
    # load ionization and recombination rates
    filetypes = ['acd','scd']
    filenames = []
    def_adas_files_dict = adas_files.adas_files_dict()
    for filetype in filetypes:
        if filetype in adas_files_sub:
            filenames.append(adas_files_sub[filetype])
        else:
            filenames.append(def_adas_files_dict[imp][filetype])

    # if background neutral density was given, load thermal CX rates too
    if n0_cm3 is not None:
        filetypes.append('ccd')
        if 'ccd' in adas_files_sub:
            filenames.append(adas_files_sub['ccd'])
        else:
            filenames.append(def_adas_files_dict[imp]['ccd']) 

    if nz_cm3 is None:
        # obtain fractional abundances via a constant-fraction model 
        atom_data = atomic.get_atom_data(imp,filetypes,filenames)

        if n0_cm3 is None:
            # obtain fractional abundances without CX:
            logTe, out['fz'],rates = atomic.get_frac_abundances(atom_data,ne_cm3,Te_eV,rho=rhop, plot=plot)
        else:
            # include CX for ionization balance:
            logTe, out['fz'],rates = atomic.get_frac_abundances(atom_data,ne_cm3,Te_eV,rho=rhop, plot=plot,
                                                   include_cx=True, n0_by_ne=n0_cm3/ne_cm3)
        out['logTe'] = logTe
        
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
                      thermal_cx_rad_flag=False if n0_cm3 is None else True,
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
        ax.set_ylabel(fr'{imp} $P_{{rad}}$ [$MW/m^3$]')
        ax.legend().set_draggable(True)
        
        # plot cumulative power in MW 
        fig,ax = plt.subplots()
        ax.plot(rhop, out['line_rad'].sum(0)/1e6, label=r'$P_{rad,line}$')
        ax.plot(rhop, out['cont_rad'].sum(0)/1e6, label=r'$P_{cont}$')
        #ax.plot(rhop, out['brems'].sum(0)/1e6, label=r'$P_{brems}$')
        ax.plot(rhop, out['rad_tot']/1e6, label=r'$P_{rad,tot}$')
        ax.set_xlabel(r'$\rho_p$')
        ax.set_ylabel(fr'{imp} $P_{{rad}}$ [MW]')
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
        a_plot.set_ylabel(fr'{imp} $P_{{rad}}$ [$MW/m^3$]')
        ncol_leg = 2 if out['line_rad_dens'].shape[0]<25 else 3
        leg=a_legend.legend(loc='center right', fontsize=11, ncol=ncol_leg).set_draggable(True)
        a_legend.axis('off')
        
        # plot average Z over radius
        fig,ax = plt.subplots()
        ax.plot(rhop, out['Z_avg'])
        ax.set_xlabel(r'$\rho_p$')
        ax.set_ylabel(fr'{imp} $\langle Z \rangle$')
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
        
        if '----' in lines[0]: 
            _ = lines.pop(0) # separator may exist before each transition

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
            
        for ine in np.arange(PEC.shape[0]):
            l, = ax1.plot(temp, PEC_eval[ine,:], label=labels[ine])
            ax1.plot(temp, PEC[ine,:], color=l.get_color(), marker='o', mfc=l.get_color(), ms=5.)

        ax1.set_xlabel(r'$T_e$ [eV]')
        ax1.set_ylabel('PEC [photons $\cdot cm^3/s$]')
        ax1.set_yscale('log')

        ax1.legend(loc='best').set_draggable(True)

    return ax1


def get_local_spectrum(adf15_filepath, ion, ne_cm3, Te_eV,
                       Ti_eV=None, n0_cm3=0.0, ion_exc_rec_dens=None, dlam_A=0.0,
                       plot=True, ax=None, plot_spec_tot=True, no_leg=False, plot_all_lines=False):
    r'''Plot spectrum based on the lines contained in an ADAS ADF15 file
    at specific values of electron density and temperature. Charge state densities
    can be given explicitely, or alternatively charge state fractions will be automatically 
    computed from ionization equilibrium (no transport). 

    Parameters
    ----------
    adf15_filepath : str
        Path on disk to the ADAS ADF15 file of interest. All wavelengths and radiating
        components will be read. 
    ion : str
        Atomic symbol of ion of interest, e.g. 'Ar'
    ne_cm3 : float
        Local value of electron density, in units of :math:`cm^{-3}`.
    Te_eV : float
        Local value of electron temperature, in units of :math:`eV`. This is used to evaluate 
        local values of photon emissivity coefficients.
    Ti_eV : float
        Local value of ion temperature, in units of :math:`eV`. This is used to represent the 
        effect of Doppler broadening. If left to None, it is internally set equal to `Te_eV`.
    n0_cm3 : float, optional
        Local density of atomic neutral hydrogen isotopes. This is only used if the provided
        ADF15 file contains charge exchange contributions.
    ion_exc_rec_dens : list of 3 floats or None
        Density of ionizing, excited and recombining charge states that may contribute to 
        emission from the given ADF15 file. If left to None, ionization equilibrium is assumed.
    dlam_A : float or 1D array
        Doppler shift in A. This can either be a scalar or an array of the same shape as the 
        output wavelength array. For the latter option, it is recommended to call this function
        twice to find the wave_final_A array first.         
    plot : bool
        If True, all spectral emission components are plotted.
    ax : matplotlib Axes instance
        Axes to plot on if plot=True. If left to None, a new figure is created.
    plot_spec_tot : bool
        If True, plot total spectrum (sum over all components) from given ADF15 file. 
    no_leg : bool
        If True, no plot legend is shown. Default is False, i.e. show legend.
    plot_all_lines : bool
        If True, plot all individual lines, rather than just the profiles due to different atomic processes.
        If more than 50 lines are included, a down-selection is automatically made to avoid excessive
        memory consumption.

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
    
    # read ADF15 file
    log10pec_dict = read_adf15(adf15_filepath)

    # get charge state from file name -- assumes standard nomenclature, {classifier}#{ion}{charge}.dat
    cs = adf15_filepath.split('#')[-1].split('.dat')[0]
    
    # import here to avoid issues when building docs or package
    from omfit_classes.utils_math import atomic_element
    
    # get nuclear charge Z and atomic mass number A
    out = atomic_element(symbol=ion)
    spec = list(out.keys())[0]
    ion_Z = int(out[spec]['Z'])
    ion_A = int(out[spec]['A'])

    if ion_exc_rec_dens is None: 
        # use ionization equilibrium fractional abundances as densities

        # get charge state distributions from ionization equilibrium
        files = ['scd','acd','ccd']
        atom_data = atomic.get_atom_data(ion,files)

        # always include charge exchange, although n0_cm3 may be 0
        logTe, fz, rates = atomic.get_frac_abundances(
            atom_data, np.array([ne_cm3,]), np.array([Te_eV,]),
            n0_by_ne=np.array([n0_cm3/ne_cm3,]), include_cx=True, plot=False)
        ion_exc_rec_dens = [fz[0][-4], fz[0][-3], fz[0][-2]] # Li-like, He-like, H-like

    wave_A = np.zeros((len(list(log10pec_dict.keys()))))
    pec_ion = np.zeros((len(list(log10pec_dict.keys()))))
    pec_exc = np.zeros((len(list(log10pec_dict.keys()))))
    pec_rr = np.zeros((len(list(log10pec_dict.keys()))))
    pec_cx = np.zeros((len(list(log10pec_dict.keys()))))
    pec_dr = np.zeros((len(list(log10pec_dict.keys()))))
    for ii,lam in enumerate(log10pec_dict):
        wave_A[ii] = lam
        if 'ioniz' in log10pec_dict[lam]:
            pec_ion[ii] = 10**log10pec_dict[lam]['ioniz'].ev(np.log10(ne_cm3),np.log10(Te_eV))
        if 'excit' in log10pec_dict[lam]:
            pec_exc[ii] = 10**log10pec_dict[lam]['excit'].ev(np.log10(ne_cm3),np.log10(Te_eV))
        if 'recom' in log10pec_dict[lam]:
            pec_rr[ii] = 10**log10pec_dict[lam]['recom'].ev(np.log10(ne_cm3),np.log10(Te_eV))
        if 'chexc' in log10pec_dict[lam]:
            pec_cx[ii] = 10**log10pec_dict[lam]['checx'].ev(np.log10(ne_cm3),np.log10(Te_eV))
        if 'drsat' in log10pec_dict[lam]:
            pec_dr[ii] = 10**log10pec_dict[lam]['drsat'].ev(np.log10(ne_cm3),np.log10(Te_eV))
    
    # Doppler broadening
    mass = m_p * ion_A
    dnu_g = np.sqrt(2.*(Ti_eV*q_electron)/mass)*(c_speed/wave_A)/c_speed
    
    # set a variable delta lambda based on the width of the broadening
    _dlam_A = wave_A**2/c_speed* dnu_g * 5 # 5 standard deviations
    
    lams_profs_A =np.linspace(wave_A-_dlam_A, wave_A + _dlam_A, 100, axis=1) 
    
    theta_tmp = 1./(np.sqrt(np.pi)*dnu_g[:,None])*\
                np.exp(-((c_speed/lams_profs_A-c_speed/wave_A[:,None])/dnu_g[:,None])**2)

    # Normalize Gaussian profile
    theta = np.einsum('ij,i->ij', theta_tmp, 1./np.trapz(theta_tmp,x=lams_profs_A,axis=1))
    
    wave_final_A = np.linspace(np.min(lams_profs_A), np.max(lams_profs_A), 100000)
    
    if plot and ax is None:
        fig,ax = plt.subplots()

    many_lines=False
    if len(wave_A)>50:
        many_lines=True

    # contributions to spectrum
    spec_ion = np.zeros_like(wave_final_A)
    for ii in np.arange(lams_profs_A.shape[0]):
        comp_ion = interp1d(lams_profs_A[ii,:], ne_cm3*ion_exc_rec_dens[0]*pec_ion[ii]*theta[ii,:],
                               bounds_error=False, fill_value=0.0)(wave_final_A)
        if plot_all_lines and many_lines and pec_ion[ii]>np.max(pec_ion)/100:
            ax.plot(wave_final_A+dlam_A, comp_ion, c='r')
        spec_ion += comp_ion

    spec_exc = np.zeros_like(wave_final_A)
    for ii in np.arange(lams_profs_A.shape[0]):
        comp_exc = interp1d(lams_profs_A[ii,:], ne_cm3*ion_exc_rec_dens[1]*pec_exc[ii]*theta[ii,:],
                               bounds_error=False, fill_value=0.0)(wave_final_A)
        if plot_all_lines and many_lines and pec_exc[ii]>np.max(pec_exc)/100:
            ax.plot(wave_final_A+dlam_A, comp_exc, c='b')
        spec_exc += comp_exc

    spec_rr = np.zeros_like(wave_final_A)
    for ii in np.arange(lams_profs_A.shape[0]):
        comp_rr = interp1d(lams_profs_A[ii,:], ne_cm3*ion_exc_rec_dens[2]*pec_rr[ii]*theta[ii,:],
                               bounds_error=False, fill_value=0.0)(wave_final_A)
        if plot_all_lines and many_lines and pec_rr[ii]>np.max(pec_rr)/100:
            ax.plot(wave_final_A+dlam_A, comp_rr, c='g')
        spec_rr += comp_rr

    spec_dr = np.zeros_like(wave_final_A)
    for ii in np.arange(lams_profs_A.shape[0]):
        comp_dr = interp1d(lams_profs_A[ii,:], ne_cm3*ion_exc_rec_dens[2]*pec_dr[ii]*theta[ii,:],
                               bounds_error=False, fill_value=0.0)(wave_final_A)
        if plot_all_lines and many_lines and pec_dr[ii]>np.max(pec_dr)/100:
            ax.plot(wave_final_A+dlam_A, comp_dr, c='g')
        spec_dr += comp_dr

    spec_cx = np.zeros_like(wave_final_A)
    for ii in np.arange(lams_profs_A.shape[0]):
        comp_cx = interp1d(lams_profs_A[ii,:], n0_cm3*ion_exc_rec_dens[2]*pec_cx[ii]*theta[ii,:],
                               bounds_error=False, fill_value=0.0)(wave_final_A)
        if plot_all_lines and many_lines and pec_cx[ii]>np.max(pec_cx)/100:
            ax.plot(wave_final_A+dlam_A, comp_cx, c='g')
        spec_cx += comp_cx

    spec_tot = spec_ion+spec_exc+spec_rr+spec_dr+spec_cx
    
    if plot:
        # plot contributions from different processes
        ax.plot(wave_final_A+dlam_A, spec_ion, c='r', label='' if no_leg else 'ionization')
        ax.plot(wave_final_A+dlam_A, spec_exc, c='b', label='' if no_leg else 'excitation')
        ax.plot(wave_final_A+dlam_A, spec_rr, c='g', label='' if no_leg else 'radiative recomb')
        ax.plot(wave_final_A+dlam_A, spec_dr, c='m', label='' if no_leg else 'dielectronic recomb')
        ax.plot(wave_final_A+dlam_A, spec_cx, c='c', label='' if no_leg else 'charge exchange recomb')
        if plot_spec_tot:
            ax.plot(wave_final_A+dlam_A, spec_tot, c='k', label='' if no_leg else 'total')

        if not no_leg: ax.legend(loc='best').set_draggable(True)
        ax.set_xlabel(r'$\lambda$ [$\AA$]')
        ax.set_ylabel(r'$\epsilon$ [A.U.]')
    else:
        ax=None

    # return Doppler-shifted wavelength if dlam_A was given as non-zero
    return wave_final_A+dlam_A, spec_ion, spec_exc, spec_rr, spec_dr, spec_cx, ax



def get_cooling_factors(imp, ne_cm3, Te_eV, n0_cm3=0.0,
                        line_rad_file=None, cont_rad_file=None, sxr=False, plot=True, show_components=False, ax=None):
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
    if n0_cm3 is not 0.0: files+=['ccd']
    atom_data_eq = atomic.get_atom_data(imp,files)

    logTe, fz, rates = atomic.get_frac_abundances(atom_data_eq, ne_cm3, Te_eV,plot=False,
                                           n0_by_ne=n0_cm3/ne_cm3,
                                           include_cx=True if n0_cm3!=0.0 else False)

    # line radiation
    atom_data = atomic.get_atom_data(imp,['pls' if sxr else 'plt'], 
                                     filenames=[] if line_rad_file is None else [line_rad_file])
    pltt= atomic.interp_atom_prof(atom_data['pls' if sxr else 'plt'],None, np.log10(Te_eV)) # line radiation [W.cm^3]

    # recombination and bremsstrahlung radiation
    atom_data = atomic.get_atom_data(imp,['prs' if sxr else 'prb'], 
                                     filenames=[] if cont_rad_file is None else [cont_rad_file])
    prb = atomic.interp_atom_prof(atom_data['prs' if sxr else 'prb'],None, np.log10(Te_eV)) # continuum radiation [W.cm^3]

    pltt*= fz[:,:-1]
    prb *= fz[:, 1:]

    line_rad_tot  = pltt.sum(1) *1e-6  # W.cm^3-->W.m^3
    cont_rad_tot = prb.sum(1) *1e-6    # W.cm^3-->W.m^3

    if plot:
        if ax is None:
            fig, ax = plt.subplots()

        # total radiation (includes hard X-ray, visible, UV, etc.)
        l, = ax.loglog(Te_eV/1e3, cont_rad_tot+line_rad_tot, ls='-',
                       label=f'{imp} $L_z$ (total)' if show_components else f'{imp}')
        col = l.get_color()
        
        if show_components:
            # show line and continuum recombination components separately
            ax.loglog(Te_eV/1e3, line_rad_tot,c=col, ls='--',label='line radiation')
            ax.loglog(Te_eV/1e3, cont_rad_tot,c=col, ls='-.',label='continuum radiation')
    
        ax.legend(loc='best').set_draggable(True)
        ax.grid('on', which='both')
        ax.set_xlabel('T$_e$ [keV]')
        ax.set_ylabel('$L_z$ [$W$ $m^3$]')
        plt.tight_layout()

    return line_rad_tot, cont_rad_tot



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
