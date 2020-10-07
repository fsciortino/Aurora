import os,sys
import numpy as np
from scipy.interpolate import interp1d
from omfit_commonclasses.utils_math import atomic_element
from . import atomic

def compute_rad(imp, rhop, time, imp_dens, ne, Te,
                n0 = None, nD = None, nBckg=None, main_ion_AZ=(1,1), bckg_imp_AZ=(12,6),
                sxr_pls_file=None, sxr_prs_file=None, 
                prad_flag=False,thermal_cx_rad_flag=False, spectral_brem_flag=False,
                sxr_flag=False, main_ion_brem_flag=False):
    ''' Calculate radiation terms corresponding to a simulation result. 
    Results are in SI units (NB: inputs are not).
    
    Result can be conveniently plotted with a time-slider using, for example, 
    
    zmax = imp_dens.shape[1]  # number of charge states (including neutrals)
    rad = res['impurity_radiation'][:,:zmax,:]   # no fully-stripped line radiation, so use zmax rather than zmax+1
    aurora.pylib.slider_plot(rhop,time, rad.transpose(1,2,0)/1e6,
                               xlabel=r'$\rho_p$', ylabel='time [s]', zlabel=r'$P_{rad}$ [$MW$]', plot_sum=True,
                               labels=[f'Ca$^{{{str(i)}}}$' for i in np.arange(nz_w.shape[1]-1)])

    Note that, when sxr_flag=True, SXR radiation will be computed using the default ADAS 'pls' and 'prs' files
    (line and continuum radiation) given by the atomic.adas_files_dict() unless the sxr_pls_file and sxr_prs_file
    parameters are provided. 

    INPUTS
    -----------
    imp : str
         Impurity symbol, e.g. Ca, F, W
    rhop : array (space,)
         Sqrt of poloidal flux radial grid of simulation output. 
    time : array (time,)
         Time array of simulation output.
    imp_dens : array (time, nZ, space)
        Dictionary with impurity density result, as given by run_aurora() method.
    ne : array (time,space) [cm^-3]
        Electron density on the output grids.
    Te : array (time,space) [eV]
        Electron temperature on the output grids.
    ------- OPTIONAL
    n0 : array(time,space), optional [cm^-3]
         Background neutral density (assumed of hydrogen-isotopes). 
         This is only used if 
    nD : array (time,space), optional [cm^-3]
         Main ion density. This is only used if main_ion_brem_flag=True.
         Note that the impurity density of imp_dens times its Z value is internally 
         automatically subtracted from this main ion density. 
    nBckg : array (time,space), optional [cm^-3]
         Background impurity density. This is only used if main_ion_brem_flag=True.
         Note that this can be of any impurity for which atomic data is available. The atomic 
         symbol of this ion is taken to be 'bckg_imp_name'. 
    main_ion_AZ : 2-tuple, optional
        Mass number (number of neutrons+protons in nucleus) and Z for the main ion 
        (background) species. Default is (1,1), corresponding to hydrogen. 
        This is only used if main_ion_brem_flag=sxr_flag=True.
    bckg_imp_AZ : 2-tuple, optional
        Mass number (number of neutrons+protons in nucleus) and Z for the background 
        impurity species. Default is (12,6), corresponding to carbon.
        This is only used if main_ion_brem_flag=sxr_flag=True.
        Note that atomic data must be available for this calculation to be possible. 
    sxr_pls_file : str
        ADAS file used for SXR line radiation calculation if sxr_flag=True. If left to None, the 
        default in atomic.adas_files_dict() is used. 
    sxr_prs_file : str
        ADAS file used for SXR continuum radiation calculation if sxr_flag=True. If left to None, 
        the default in atomic.adas_files_dict() is used. 
    ------ FLAGS
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
        This is currently incomplete and untested!!

    OUTPUTS
    --------------
    res : dict
        Dictionary containing the radiation terms, depending on the activated flags. 
        If all flags were on, the dictionary would include
        {'impurity_radiation','spectral_bremsstrahlung','sxr_radiation'}
        The structure of each of these arrays is the same as in STRAHL (here with Python indexing):

        ** impurity_radiation and sxr_radiation **
        index 0: total line radiation of neutral impurity
        index 1: total line radiation of singly ionised impurity
        ....
        index n-1: total line radiation of hydrogen-like ion
        index n: bremsstrahlung due to electron scattering at main ion (if requested)
        index n+1: total continuum radiation of impurity (bremsstrahlung and recombination continua)
        index n+2: bremsstrahlung due to electron scattering at impurity
        index n+3: total radiation of impurity (and main ion, if set in Xx.atomdat)

        ** spectral_bremsstrahlung **
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

        atom_data = atomic.get_all_atom_data(imp,['plt','prb'])

        # get radial profiles of all radiation terms
        plt = atomic.interp_atom_prof(atom_data['plt'],logne,logTe) # W
        prb = atomic.interp_atom_prof(atom_data['prb'],logne,logTe) # W

        if thermal_cx_rad_flag:
            atom_data = atomic.get_all_atom_data(imp,['prc'])
            # add to total recombination and cont rad terms:
            logn0 = np.log10(n0)
            prb += atomic.interp_atom_prof(atom_data['prc'],logn0,logTe) # W

        res['impurity_radiation'] = rad = np.zeros((len(time), Z_imp+4, len(rhop)),dtype='single')

        ####### Populate radiation arrays for the entire spectrum #######
        # line radiation for each charge state (NB: fully stripped has no line radiation)
        rad[:,:Z_imp,:] = np.maximum(imp_dens[:,:-1] * plt, 1e-60)

        # rad[:,Z_imp,:] (bremsstrahlung due to electron scattering at main ion) is filled below only if main_ion_brem_flag=True

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
        atom_data = atomic.get_all_atom_data(imp,['brs'])
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
            atom_data['pls'] = atomic.get_all_atom_data(imp,['pls'])['pls']

        if sxr_prs_file is not None:
            atomdat_dir = atomic.get_atomdat_info()
            res = atomic.adas_file(atomdat_dir+sxr_prs_file)
            atom_data['prs'] = res.logNe, res.logT, res.data
        else:
            # default SXR continuum radiation file 
            atom_data['prs'] = atomic.get_all_atom_data(imp,['prs'])['prs']
            
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

        # get main ion name so that we can fetch appropriate ADAS file:
        main_ion_name = list(atomic_element(A=main_ion_AZ[0],
                                            Z_ion=main_ion_AZ[1]).values())[0]['symbol']
        if len(main_ion_name)==1: main_ion_name+='_' # to match ADAS file nomenclature

        # get expected background densities of main ions and carbon/lumped impurity
        Z_nimp = (np.arange(Z_imp+1)[None,:,None]*imp_dens).sum(1)
        # remove  density of simulated impurities (scaled by mass number of background)
        nD -= Z_nimp/main_ion_AZ[0]  

        if prad_flag:
            
            # get total radiation for main ion:
            bckg_D_rad = atomic.get_adas_ion_rad(main_ion_name, nD, logne, logTe) # W/m^3

            # if background ion density was specified, include it:
            if nBckg is not None:
                # NB: STRAHL is not adding background impurity!
                bckg_imp_name = list(atomic_element(A=bckg_imp_AZ[0],
                                                    Z_ion=bckg_imp_AZ[1]).values())[0]['symbol']
                bckg_imp_rad = atomic.get_adas_ion_rad(bckg_imp_name, nBckg, logne, logTe)  # W/m^3

                bckg_D_rad += bckg_imp_rad
                
            rad[:,nion] = bckg_D_rad/1e6   #W/cm^3
            rad[:,no,:] += rad[:,nion] #add to total

            # less accurate:
            #rad[:,Z_imp,:] = atomic.main_ion_brems(main_ion_AZ[1], nD, ne, Te)
            #rad[:,no,:] += rad[:,Z_imp] #add to total

        if spectral_brem_flag:
            rad_bs[:,_np,:] = nD * brs[:,int(main_ion_AZ[1])-1,:]   # only main ion brems
            rad_bs[:,nq,:] += rad_bs[:,_np,:]  # total

        if sxr_flag:
            # Bremsstrahlung in the SXR range
            # STRAHL uses brs files. Here we use prs files for background species
            if main_ion_name in ['D_','T_']: bckg_ion_name='H_' # same atomic data
            bckg_D = atomic.get_adas_ion_rad(bckg_ion_name, nD, logne, logTe, sxr=True)  # W/m^3

            # add radiation from background impurity (e.g. C in DIII-D, Mo in C-Mod, etc.)
            bckg_imp_name = list(atomic_element(A=bckg_imp_AZ[0],
                                                Z_ion=bckg_imp_AZ[1]).values())[0]['symbol']
            bckg_C = atomic.get_adas_ion_rad(bckg_imp_name, nBckg, logne, logTe, sxr=True)  # W/m^3

            radsxr[:,Z_imp] = (bckg_D+bckg_C)*ne
            radsxr[:,no] += radsxr[:,Z_imp]

    return res
