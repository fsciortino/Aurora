import os,sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
plt.ion()
from scipy import constants
import warnings, copy

if not np.any([('sphinx' in k and not 'sphinxcontrib' in k) for k in sys.modules]):
    # this if statement prevents issues with sphinx when building docs
    from omfit_commonclasses.utils_math import atomic_element

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

    Args:
        imp : str
             Impurity symbol, e.g. Ca, F, W
        nz : array (time, nZ, space) [:math:`cm^{-3}`]
            Dictionary with impurity density result, as given by :py:func:`~aurora.core.run_aurora` method.
        ne : array (time,space) [:math:`cm^{-3}`]
            Electron density on the output grids.
        Te : array (time,space) [eV]
            Electron temperature on the output grids.


    Keyword Args:
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

    Returns:
        res : dict
            Dictionary containing radiation terms, depending on the activated flags. 
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
    
        # impurity bremsstrahlung in SXR range -- already included in 'sxr_cont_rad'
        if 'pbs' in adas_files_sub:
            atom_data = atomic.get_atom_data(imp, ['pbs'],[adas_files_sub['pbs']])
        else:
            atom_data = atomic.get_atom_data(imp, ['pbs'])
        pbs = atomic.interp_atom_prof(atom_data['pbs'],logne,logTe) # W
        res['sxr_brems'] = nz[:,1:] * pbs 

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



def plot_radiation_profs(imp, nz_prof, logne_prof, logTe_prof, xvar_prof,
                         xvar_label='', atom_data=None):
    '''Compute profiles of predicted radiation, both SXR-filtered and unfiltered.
    This function offers a simplified interface to radiation calculation with respect to 
    :py:meth:`~aurora.radiation.compute_rad`, which is more complete.

    This function can be used to plot radial profiles (setting xvar_prof to a radial grid)
    or profiles as a function of any variable on which the logne_prof and logTe_prof
    may depend.

    The variable "nz_prof" may be a full description of impurity charge state densities
    (e.g. the output of aurora), or profiles of fractional abundances from ionization 
    equilibrium.

    Args: 
        imp : str, optional
            Impurity ion atomic symbol.
        nz_prof : array (TODO for docs: check dimensions)
            Impurity charge state densities
        logne_prof : array (TODO for docs: check dimensions)
            Electron density profiles in :math:`cm^{-3}`
        logTe_prof : array (TODO for docs: check dimensions)
            Electron temperature profiles in eV
        xvar_prof : array (TODO for docs: check dimensions)
            Profiles of a variable of interest, on the same grid as kinetic profiles. 
        xvar_label : str, optional
            Label for x-axis. 
        atom_data : dict, optional
            Dictionary containing atomic data as output by :py:meth:`~aurora.atomic.get_atom_data`
            for the atomic processes of interest. "prs","pls","plt" and "prb" are required by this function.
            If not provided, this function loads these files internally. 

    Returns:
        pls : array (TODO for docs: check dimensions)
            SXR line radiation.
        prs : array (TODO for docs: check dimensions)
            SXR continuum radiation.
        pltt : array (TODO for docs: check dimensions)
            Unfiltered line radiation.
        prb : array (TODO for docs: check dimensions)
            Unfiltered continuum radiation.        
    '''
    if atom_data is None:
        # if atom_data dictionary was not given, load appropriate default files
        atom_data = atomic.get_atom_data(imp,['pls','prs','plt','prb'])

    # use "pltt" nomenclature rather than "plt" to avoid issues with matplotlib.pyplot imported as plt
    pls, prs, pltt, prb = atomic.get_cooling_factors(atom_data, logTe_prof, nz_prof, ion_resolved = True, plot=False)

    emiss_sxr = np.zeros((len(xvar_prof),nion))
    emiss_tot = np.zeros((len(xvar_prof),nion))
    emiss_sxr[:, 1: ] += prs
    emiss_sxr[:, :-1] += pls
    emiss_tot[:, 1: ] += prb
    emiss_tot[:, :-1] += pltt

    # plot radiation components
    fig,axx = plt.subplots(2,2,figsize=(12,8),sharex=True)
    ax = axx.flatten()
    nion = prs.shape[1]+1
    colors = cm.plasma(np.linspace(0,1, nion))
    for a in ax:
        a.set_prop_cycle('color',colors)
        a.grid(True)

    ax[0].plot([],[]) #empty plot for bremstrahlung of neutral ion
    ax[0].plot(xvar_prof,prs); ax[0].set_title('PRS: cont SXR rad')
    ax[1].plot(xvar_prof,pls); ax[1].set_title('PLS: SXR line rad')
    
    ax[2].plot(xvar_prof,pltt); ax[2].set_title('PLT: tot line rad')
    ax[3].plot([],[]) #empty plot for bremstrahlung of neutral ion
    ax[3].plot(xvar_prof,prb); ax[3].set_title('PRB: tot cont rad')

    ax[2].set_xlabel(xvar_label)
    ax[3].set_xlabel(xvar_label)
    
    labels = [r'$%s^{%d\!+}$'%(imp,cc) for cc in range(nion)]
    ax[0].legend(labels)
    ax[0].set_xlim(xvar_prof[0], xvar_prof[-1])

    # plot total power (in SXR and whole range)
    fig,axx = plt.subplots(2,2,figsize=(12,8),sharex=True)
    ax = axx.flatten()

    for a in ax:
        a.set_prop_cycle('color',colors)
        a.grid(True)

    ax[0].plot(xvar_prof,emiss_sxr); ax[0].set_title('SXR power [W]')
    ax[1].plot(xvar_prof,emiss_tot); ax[1].set_title('Tot. rad. power [W]')
    ax[2].plot(xvar_prof,emiss_sxr*10**logne_prof[:,None]); ax[2].set_title(r'SXR power [W/m$^{-3}$]')
    ax[3].plot(xvar_prof,emiss_tot*10**logne_prof[:,None]); ax[3].set_title(r'Tot. rad. power [W/m$^{-3}$]')
    ax[2].set_xlabel(xvar_label)
    ax[3].set_xlabel(xvar_label)
    ax[0].legend(labels)
    ax[0].set_xlim(xvar_prof[0], xvar_prof[-1])

    return pls, prs, pltt, prb



def radiation_model(imp,rhop, ne_cm3, Te_eV, vol,
                    adas_files_sub={}, n0_cm3=None, Ti_eV=None, nz_cm3=None, frac=None, plot=False):
    '''Model radiation from a fixed-impurity-fraction model or from detailed impurity density
    profiles for the chosen ion. This method acts as a wrapper for :py:method:compute_rad(), 
    calculating radiation terms over the radius and integrated over the plasma cross section. 

    Args:
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

    Keyword Args:
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

    Returns:
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
            logTe, out['fz'] = atomic.get_frac_abundances(atom_data,ne_cm3,Te_eV,rho=rhop, plot=plot)
        else:
            # include CX for ionization balance:
            logTe, out['fz'] = atomic.get_frac_abundances(atom_data,ne_cm3,Te_eV,rho=rhop, plot=plot,
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

    print('------------------------------------')
    print(f'Total {imp} line radiation power: {out["line_rad_tot"][-1]/1e6:.3f} MW')
    print(f'Total {imp} continuum radiation power: {out["cont_rad"].sum(0)[-1]/1e6:.3f} MW')
    print(f'Total {imp} bremsstrahlung radiation power: {out["brems"].sum(0)[-1]/1e6:.3f} MW')
    if n0_cm3 is not None:
        print(f'Thermal CX power: {out["thermal_cx_rad"][-1]/1e6:.3f} MW')
    print(f'Total radiated power: {out["Prad"]/1e6:.3f} MW')

    # calculate average charge state Z across radius
    out['Z_avg'] = np.sum(np.arange(out['fz'].shape[1])[:,None] * out['fz'].T, axis=0)
    
    if plot:
        # plot power in MW/m^3
        fig,ax = plt.subplots()
        ax.plot(rhop, out['line_rad_dens'].sum(0)/1e6, label=r'$P_{rad,line}$')
        ax.plot(rhop, out['cont_rad_dens'].sum(0)/1e6, label=r'$P_{cont}$')
        ax.plot(rhop, out['brems_dens'].sum(0)/1e6, label=r'$P_{brems}$')
        ax.plot(rhop, out['rad_tot_dens']/1e6, label=r'$P_{rad,tot}$')
        ax.set_xlabel(r'$\rho_p$')
        ax.set_ylabel(fr'{imp} $P_{{rad}}$ [$MW/m^3$]')
        ax.legend().set_draggable(True)
        
        # plot power in MW 
        fig,ax = plt.subplots()
        ax.plot(rhop, out['line_rad'].sum(0)/1e6, label=r'$P_{rad,line}$')
        ax.plot(rhop, out['cont_rad'].sum(0)/1e6, label=r'$P_{cont}$')
        ax.plot(rhop, out['brems'].sum(0)/1e6, label=r'$P_{brems}$')
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
    
    Args:
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

    Returns:
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



def adf04_files():
    '''Collection of trust-worthy ADAS ADF04 files. 
    This function will be moved and expanded in ColRadPy in the near future. 
    '''
    files = {}
    files['Ca'] = {}
    files['Ca']['Ca8+'] = 'mglike_lfm14#ca8.dat'
    files['Ca']['Ca9+'] = 'nalike_lgy09#ca9.dat'
    files['Ca']['Ca10+'] = 'nelike_lgy09#ca10.dat'
    files['Ca']['Ca11+'] = 'flike_mcw06#ca11.dat'
    files['Ca']['Ca14+'] = 'clike_jm19#ca14.dat'
    files['Ca']['Ca15+'] = 'blike_lgy12#ca15.dat'
    files['Ca']['Ca16+'] = 'belike_lfm14#ca16.dat'
    files['Ca']['Ca17+'] = 'lilike_lgy10#ca17.dat'
    files['Ca']['Ca18+'] = 'helike_adw05#ca18.dat'

    return files




def get_pec_prof(ion, cs, rhop, ne_cm3, Te_eV, lam_nm=1.8705, lam_width_nm=0.002, meta_idxs=[0],
                 adf04_repo = os.path.expanduser('~')+'/adf04_files/ca/ca_adf04_adas/',
                 pec_threshold=1e-20, phot2energy=True, plot=True):
    '''Compute radial profile for Photon Emissivity Coefficients (PEC) for lines within the chosen
    wavelength range using the ColRadPy package. This is an alternative to the option of using 
    the :py:func:`~atomic.read_adf15` function to read PEC data from an ADAS ADF-15 file and 
    interpolate results on ne,Te grids. 

    Args:
        ion : str
            Ion atomic symbol
        cs : str
            Charge state, given in format like 'Ca18+'
        rhop : array (nr,)
            Srt of normalized poloidal flux radial array
        ne_cm3 : array (nr,)
            Electron density in :math:`cm^{-3}` units
        Te_eV : array (nr,)
            Electron temperature in eV units
        lam_nm : float
            Center of the wavelength region of interest [nm]
        lam_width_nm : float
            Width of the wavelength region of interest [nm]
        meta_idxs : list of integers
            List of levels in ADF04 file to be treated as metastable states. 
        adf04_repo : str
            Location where ADF04 file from :py:method:adf04_files() should be fetched.
        prec_threshold : float
            Minimum value of PECs to be considered, in :math:`photons \cdot cm^3/s`
        phot2energy : bool
            If True, results are converted from :math:`photons \cdot cm^3/s` to :math:`W.cm^3`
        plot : bool
            If True, plot lines profiles and total

    Returns:
        pec_tot_prof : array (nr,)
            Radial profile of PEC intensity, in units of :math:`photons \cdot cm^3/s` (if phot2energy=False) or 
            :math:`W \cdot cm^3` depending (if phot2energy=True). 
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

    if phot2energy:
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
