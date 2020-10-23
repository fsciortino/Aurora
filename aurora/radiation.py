import os,sys
import numpy as np
from scipy.interpolate import interp1d
from omfit_commonclasses.utils_math import atomic_element
from . import atomic
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from colradpy import colradpy
plt.ion()
from scipy import constants
from IPython import embed

def compute_rad(imp, rhop, time, imp_dens, ne, Te,
                n0 = None, nD = None, main_ion_name='D',
                sxr_pls_file=None, sxr_prs_file=None, 
                prad_flag=False,thermal_cx_rad_flag=False, spectral_brem_flag=False,
                sxr_flag=False, main_ion_brem_flag=False):
    '''Calculate radiation terms corresponding to a simulation result. 
    Results are in SI units (NB: inputs are not).
    
    Result can be conveniently plotted with a time-slider using, for example

    .. code-block:: python

        zmax = imp_dens.shape[1]  # number of charge states (including neutrals)
        rad = res['impurity_radiation'][:,:zmax,:]   # no fully-stripped line radiation
        aurora.slider_plot(rhop,time, rad.transpose(1,2,0)/1e6,
            xlabel=r'$\\rho_p$', ylabel='time [s]', 
            zlabel=r'$P_{rad}$ [$MW$]',
            plot_sum=True,
            labels=[f'Ca$^{{{str(i)}}}$' for i in np.arange(nz_w.shape[1]-1)])

    Note that, when sxr_flag=True, SXR radiation will be computed using the default ADAS 'pls' and 'prs' files
    (line and continuum radiation) given by the atomic.adas_files_dict() unless the sxr_pls_file and sxr_prs_file
    parameters are provided. 

    All radiation outputs are given in :math:`W cm^-3`, consistently with units of :math:`cm^-3` given for inputs.

    Args:
        imp : str
             Impurity symbol, e.g. Ca, F, W
        rhop : array (space,)
             Sqrt of poloidal flux radial grid of simulation output. 
        time : array (time,)
             Time array of simulation output.
        imp_dens : array (time, nZ, space)
            Dictionary with impurity density result, as given by :py:func:`~aurora.core.run_aurora` method.
        ne : array (time,space) [cm^-3]
            Electron density on the output grids.
        Te : array (time,space) [eV]
            Electron temperature on the output grids.
        n0 : array(time,space), optional [cm^-3]
             Background neutral density (assumed of hydrogen-isotopes). 
             This is only used if thermal_cx_rad_flag=True.
        nD : array (time,space), optional [cm^-3]
             Main ion density. This is only used if main_ion_brem_flag=True.
        main_ion_name : str, optional
            Atomic symbol of main ion species. Default is 'D'.
            This is only used if main_ion_brem_flag=True.
        sxr_pls_file : str
            ADAS file used for SXR line radiation calculation if sxr_flag=True. If left to None, the 
            default in :py:func:`~aurora.atomic.adas_files_dict` is used. 
        sxr_prs_file : str
            ADAS file used for SXR continuum radiation calculation if sxr_flag=True. If left to None, 
            the default in :py:func:`~aurora.atomic.adas_files_dict` is used. 
        prad_flag : bool, optional
            If True, total radiation is computed (for each charge state and their sum)
        thermal_cx_rad_flag : bool, optional
            If True, thermal charge exchange radiation is computed.
        spectral_brem_flag : bool, optional
            If True, spectral bremstrahlung is computed (based on available 'brs' ADAS file)
        sxr_flag : bool, optional
            If True, soft x-ray radiation is computed (for the given 'pls','prs' ADAS files)
        main_ion_brem_flag : bool, optional
            If True, main ion bremstrahlung (all contributions) is computed. 

    Returns:
        res : dict
            Dictionary containing the radiation terms, depending on the activated flags.
            The structure of this output is intentionally left to be the same as in STRAHL
            for convenience. 
    
            If all flags were on, the dictionary would include
            {'impurity_radiation','spectral_bremsstrahlung','sxr_radiation'}

            Impurity_radiation and sxr_radiation:
            index 0: total line radiation of neutral impurity
            index 1: total line radiation of singly ionised impurity
            ....
            index n-1: total line radiation of hydrogen-like ion
            index n: bremsstrahlung due to electron scattering at main ion (if requested)
            index n+1: total continuum radiation of impurity (bremsstrahlung and recombination continua)
            index n+2: bremsstrahlung due to electron scattering at impurity
            index n+3: total radiation of impurity (and main ion, if set in Xx.atomdat)

            Spectral_bremsstrahlung:
            index 0: = 0
            index 1: bremsstrahlung due to electron scattering at singly ionised impurity
            ....
            index n: bremsstrahlung due to electron scattering at fully ionised impurity
            index n+1: bremsstrahlung due to electron scattering at main ion
            index n+2: total bremsstrahlung of impurity (and main ion, if set in Xx.atomdat)

    '''
    res = {}

    Z_imp = imp_dens.shape[1] - 1
    logTe = np.log10(Te)
    logne = np.log10(ne)

    # now, calculate radiation components as in emissiv.f:
    # nion = Z_imp    #bremsstrahlung due to electron scattering at main ion
    _np = Z_imp+1# total continuum radiation of impurity (bremsstrahlung and recombination continua)
    nq = Z_imp+2  #  bremsstrahlung due to electron scattering at impurity
    no = Z_imp+3  #   index n+4: total radiation of impurity (and main ion, if set in Xx.atomdat)

    # calculate total radiation
    if prad_flag:

        atom_data = atomic.get_atom_data(imp,['plt','prb'])

        # get radial profiles of all radiation terms
        plt = atomic.interp_atom_prof(atom_data['plt'],logne,logTe) # W
        prb = atomic.interp_atom_prof(atom_data['prb'],logne,logTe) # W

        if thermal_cx_rad_flag:
            atom_data = atomic.get_atom_data(imp,['prc'])
            # add to total recombination and cont rad terms:
            logn0 = np.log10(n0)
            prb += atomic.interp_atom_prof(atom_data['prc'],logn0,logTe) # W

        res['impurity_radiation'] = rad = np.zeros((len(time), Z_imp+4, len(rhop)),dtype='single')

        ####### Populate radiation arrays for the entire spectrum #######
        # line radiation for each charge state (NB: fully stripped has no line radiation)
        rad[:,:Z_imp,:] = np.maximum(imp_dens[:,:-1] * plt, 1e-60)

        # rad[:,Z_imp,:] (bremsstrahlung due to electron scattering at main ion) is filled below
        # only if main_ion_brem_flag=True

        rad[:,no,:] = rad.sum(1) # total line radiation

        # total continuum radiation (NB: neutrals do not have continuum radiation)
        rad[:,_np,:] = (imp_dens[:,1:] * prb).sum(1)

        # add continuum radiation to total
        rad[:,no,:] += rad[:,_np,:]

        # impurity brems (inaccurate Gaunt factor!)
        rad[:,nq,:] = atomic.impurity_brems(imp_dens, ne, Te).sum(1)


    if spectral_brem_flag:  # this if-statement is missing in STRAHL's emissiv.f...
        # spectral bremsstrahlung
        res['spectral_bremsstrahlung'] = rad_bs = np.zeros((len(time), Z_imp+3, len(rhop)),dtype='single')

        logZ_rep = np.log10(np.arange(Z_imp)+1)
        atom_data = atomic.get_atom_data(imp,['brs'])
        x,y,tab = atom_data['brs']
        brs = atomic.interp_atom_prof((x,y,tab.T),None,logTe) # W
        brs = interp1d(x, brs,axis=1,copy=False,assume_sorted=True)(logZ_rep)

        rad_bs[:,1:Z_imp+1,:] = imp_dens[:,1:] * brs
        rad_bs[:,nq,:] = rad_bs[:,1:Z_imp,:].sum(1)


    if sxr_flag:

        atom_data = {}
        if sxr_pls_file is not None:
            # load SXR line radiation file requested by user
            atomdat_dir = atomic.get_atomdat_info()
            res = atomic.adas_file(atomdat_dir+sxr_pls_file)
            atom_data['pls'] = res.logNe, res.logT, res.data
        else:
            # default SXR line radiation file
            atom_data['pls'] = atomic.get_atom_data(imp,['pls'])['pls']

        if sxr_prs_file is not None:
            atomdat_dir = atomic.get_atomdat_info()
            res = atomic.adas_file(atomdat_dir+sxr_prs_file)
            atom_data['prs'] = res.logNe, res.logT, res.data
        else:
            # default SXR continuum radiation file 
            atom_data['prs'] = atomic.get_atom_data(imp,['prs'])['prs']
            
        pls = atomic.interp_atom_prof(atom_data['pls'],logne,logTe) # W
        prs = atomic.interp_atom_prof(atom_data['prs'],logne,logTe) # W

        res['sxr_radiation'] = radsxr = np.zeros((len(time), Z_imp+4, len(rhop)),dtype='single')

        # line radiation for each charge state
        radsxr[:,:Z_imp] = np.maximum(imp_dens[:,:-1] * pls, 1e-60)
        radsxr[:,no,:] = radsxr[:,:Z_imp].sum(1) # total line radiation

        # total continuum radiation
        radsxr[:,_np] = (imp_dens[:,1:] * prs).sum(1)

        # add continuum radiation to total
        radsxr[:,no,:] += radsxr[:,_np,:]

        # impurity brems (inaccurate Gaunt factor!)
        radsxr[:,nq,:] = atomic.impurity_brems(imp_dens,ne,Te).sum(1) #BUG this is unfiltered bremstrahlung!! not SXR



    # main ion bremsstrahlung (inaccurate Gaunt factor...)
    if main_ion_brem_flag:

        # get main-ion Z
        elem = atomic_element(symbol=main_ion_name)
        main_ion_Z = elem[list(elem.keys())[0]]['Z']

        if main_ion_name in ['D','T']:
            main_ion_name='H' # same atomic data

        if prad_flag:
            
            # get total radiation for main ion:
            main_ion_rad = atomic.get_adas_continuum_rad(main_ion_name, nD, logne, logTe) # W/m^3

            rad[:,Z_imp] = main_ion_rad/1e6   #W/cm^3
            rad[:,no,:] += rad[:,Z_imp] #add to total

            # less accurate:
            #rad[:,Z_imp,:] = atomic.main_ion_brems(main_ion_Z, nD, ne, Te)
            #rad[:,no,:] += rad[:,Z_imp] #add to total

        if spectral_brem_flag:            
            rad_bs[:,_np,:] = nD * brs[:,main_ion_Z-1,:]   # only main ion brems
            rad_bs[:,nq,:] += rad_bs[:,_np,:]  # total

        if sxr_flag:
            # Bremsstrahlung in the SXR range
            # STRAHL uses brs files. Here we use prs files for background species
            bckg_D = atomic.get_adas_continuum_rad(bckg_ion_name, nD, logne, logTe, sxr=True)  # W/m^3

            radsxr[:,Z_imp] = bckg_D * ne
            radsxr[:,no] += radsxr[:,Z_imp]

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
    (e.g. the output of aurora), or profiles of fractional abundances from ionization equilibrium.

    Args: 
        imp : str, optional
            Impurity ion atomic symbol.
        nz_prof : array (TODO for docs: check dimensions)
            Impurity charge state densities
        logne_prof : array (TODO for docs: check dimensions)
            Electron density profiles in cm^-3
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
        # if atom_data dictionary was not given, load appropriate files
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
                    n0_cm3=None, nz_cm3=None, frac=None, plot=False):
    '''Model radiation from a fixed-impurity-fraction model or from detailed impurity density
    profiles for the chosen ion. This method acts as a wrapper for :py:method:compute_rad(), 
    calculating radiation terms over the radius and integrated over the plasma cross section. 

    Args:
        imp : str (nr,)
            Impurity ion symbol, e.g. W
        rhop : array (nr,)
            Sqrt of normalized poloidal flux array from the axis outwards
        ne_cm3 : array (nr,)
            Electron density in :math:`cm^-3` units.
        Te_eV : array (nr,)
            Electron temperature in eV
        vol : array (nr,)
            Volume of each flux surface in :math:`m^3`.
        n0_cm3 : array (nr,), optional
            Background ion density (H,D or T). If provided, charge exchange (CX) 
            recombination is included in the calculation of charge state fractional 
            abundances. 
        nz_cm3 : array (nr,nz), optional
            Impurity charge state densities in cm^-3 units. Fractional abundancies can 
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
    
    if n0_cm3 is not None: n0_cm3 = n0_cm3[rhop<=1.]
    rhop = rhop[rhop<=1.]

    # load ionization and recombination rates
    filetypes = ['acd','scd']
    if n0_cm3 is not None:
        filetypes.append('ccd')
    
    if nz_cm3 is None:
        # obtain fractional abundances via a constant-fraction model 
        atom_data = atomic.get_atom_data(imp,filetypes)
        
        # get_frac_abundances takes inputs in m^-3 and eV
        logTe, fz = atomic.get_frac_abundances(atom_data, ne_cm3*1e6, Te_eV,rho=rhop, plot=plot)
        if n0_cm3 is not None:
            # compute result with CX and overplot
            logTe, fz = atomic.get_frac_abundances(atom_data, ne_cm3*1e6, Te_eV,rho=rhop, plot=plot,ls='--',
                                                              include_cx=True, n0_by_ne=n0_cm3/ne_cm3,ax=plt.gca())

        # Impurity densities
        nz_cm3 = frac * ne_cm3[None,:,None] * fz[None,:,:]  # (time,nZ,space)
    else:
        # set input nz_cm3 into the right shape for compute_rad: (time,space,nz)
        nz_cm3 = nz_cm3[None,:,:]

    Z_imp = nz_cm3.shape[-1] -1  # don't include neutral stage
    
    # Estimate D/T ion density via quasi-neutrality, subtracting impurity density times Z values from ne
    nD = ne_cm3[None,:]
    Z_n_imp = (np.arange(Z_imp+1)[None,:,None]*nz_cm3.transpose(0,2,1)).sum(1)
    nD -= Z_n_imp
    
    # basic total radiated power
    rad = compute_rad(imp, rhop, [1.0], nz_cm3.transpose(0,2,1), ne_cm3[None,:], Te_eV[None,:],
                                         n0=n0_cm3, nD=nD,
                                         prad_flag=True, thermal_cx_rad_flag=False, 
                                         spectral_brem_flag=False, sxr_flag=False, 
                                         main_ion_brem_flag=True)

    # create results dictionary
    res = {}
    res['rhop'] = rhop
    
    # radiation terms -- converted from W/cm^3 to W/m^3
    res['line_rad_dens'] = rad['impurity_radiation'][0,:Z_imp-1,:]*1e6  # no line radiation from fully-stripped impurity
    res['brems_dens'] = rad['impurity_radiation'][0,Z_imp,:]*1e6
    res['cont_rad_dens'] = rad['impurity_radiation'][0,Z_imp+1,:]*1e6
    res['rad_tot_dens'] = rad['impurity_radiation'][0,Z_imp+3,:]*1e6
    
    # cumulative integral over all volume
    res['line_rad'] = cumtrapz(res['line_rad_dens'], vol, initial=0.)
    res['line_rad_tot'] = cumtrapz(res['line_rad_dens'].sum(0), vol, initial=0.)
    res['brems'] = cumtrapz(res['brems_dens'], vol, initial=0.)
    res['cont_rad'] = cumtrapz(res['cont_rad_dens'], vol, initial=0.)
    res['rad_tot'] = cumtrapz(res['rad_tot_dens'], vol, initial=0.)

    # total power is the last element of the cumulative integral
    res['Prad'] = res['rad_tot'][-1]
    print(f'Total {imp} radiated power: {res["Prad"]/1e6:.3f} MW')

    # calculate average charge state Z across radius
    res['Z_avg'] = np.sum(np.arange(fz.shape[1])[:,None] * fz.T, axis=0)
    
    if plot:
        # plot power in MW/m^3
        fig,ax = plt.subplots()
        ax.plot(rhop, res['line_rad_dens'].sum(0)/1e6, label=r'$P_{rad,line}$')
        ax.plot(rhop, res['brems_dens']/1e6, label=r'$P_{brems}$')
        ax.plot(rhop, res['cont_rad_dens']/1e6, label=r'$P_{cont}$')
        ax.plot(rhop, res['rad_tot_dens']/1e6, label=r'$P_{rad,tot}$')
        ax.set_xlabel(r'$\rho_p$')
        ax.set_ylabel(fr'{imp} $P_{{rad}}$ [$MW/m^3$]')
        ax.legend().set_draggable(True)
        
        # plot power in MW 
        fig,ax = plt.subplots()
        ax.plot(rhop, res['line_rad'].sum(0)/1e6, label=r'$P_{rad,line}$')
        ax.plot(rhop, res['brems']/1e6, label=r'$P_{brems}$')
        ax.plot(rhop, res['cont_rad']/1e6, label=r'$P_{cont}$')
        ax.plot(rhop, res['rad_tot']/1e6, label=r'$P_{rad,tot}$')
        
        ax.set_xlabel(r'$\rho_p$')
        ax.set_ylabel(fr'{imp} $P_{{rad}}$ [MW]')
        fig.suptitle('Cumulative power')
        ax.legend().set_draggable(True)

        # power per charge state
        fig,ax = plt.subplots()
        for cs in np.arange(res['line_rad_dens'].shape[0]):
            ax.plot(rhop, res['line_rad_dens'][cs,:]/1e6, label=imp+fr'$^{{{cs+1}+}}$')
        ax.set_xlabel(r'$\rho_p$')
        ax.set_ylabel(fr'{imp} $P_{{rad}}$ [MW]')
        ax.legend().set_draggable(True)

        # plot average Z over radius
        fig,ax = plt.subplots()
        ax.plot(rhop, res['Z_avg'])
        ax.set_xlabel(r'$\rho_p$')
        ax.set_ylabel(fr'{imp} $\langle Z \rangle$')
    
    return res






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
    the :py:method:atomic.read_adf15() function to read PEC data from an ADAS ADF-15 file and 
    interpolate results on ne,Te grids. 

    Args:
        ion : str
            Ion atomic symbol
        cs : str
            Charge state, given in format like 'Ca18+'
        rhop : array (nr,)
            Srt of normalized poloidal flux radial array
        ne_cm3 : array (nr,)
            Electron density in cm^-3 units
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
            Minimum value of PECs to be considered, in photons.cm^3/s  
        phot2energy : bool
            If True, results are converted from photons.cm^3/s to W.cm^3
        plot : bool
            If True, plot lines profiles and total

    Returns:
        pec_tot_prof : array (nr,)
            Radial profile of PEC intensity, in units of :math:`photons cm^3/s` (if phot2energy=False) or 
            :math:`W \cdot cm^3` depending (if phot2energy=True). 
    '''
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
