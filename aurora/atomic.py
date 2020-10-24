#-*-Python-*-
# Created by sciortinof at 05 Feb 2020  14:59
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from matplotlib import cm
import os
import scipy.ndimage
from scipy.linalg import svd
from IPython import embed
from scipy.constants import m_p, e as q_electron


def get_file_types():
    '''Returns main types of ADAS atomic data of interest '''
    return {'acd':'recombination',
            'scd':'ionization',
            'prb':'continuum radiation',
            'plt':'line radiation',
            'ccd':'thermal charge exchange',
            'prc':'thermal charge exchange continuum radiation',
            'pls':'line radiation in sxr range',
            'prs':'continuum radiation in sxr range',
            'brs':'continuum spectral bremstrahlung',
            'fis':'sensitivity of sxr'}


def get_atomdat_info():
    '''Function to identify location of ADAS atomic data in a generalized fashion
    and to obtain the list of file_types of interest.
    '''
    if 'STRAHL_DIR' not in os.environ:
        # STRAHL_DIR env variable can in principle be used by user to point to a personal installation
        os.environ['STRAHL_DIR'] = os.environ['HOME'] + '/strahl'

    # Define hierarchy of directories that should be searched in order
    paths = [os.environ['HOME'] + '/atomlib/atomdat_master/newdat/',
             os.environ['HOME'] + '/atomAI/atomdat_master/newdat/',
             '/fusion/projects/codes/strahl/{}/intel-18/strahl20/atomdat/newdat/'.format(os.environ['USER']),
             '/fusion/projects/codes/strahl/{}/strahl/atomdat/newdat/'.format(os.environ['USER']),
             '/fusion/projects/codes/strahl/public20/atomdat/newdat/',
             os.environ['STRAHL_DIR'] + '/atomdat/newdat/',
             os.environ['HOME'] + '/strahl/atomdat/newdat/',
             '']
    
    for atomdat_dir in paths:
        if os.path.isdir(atomdat_dir):
            break

    return atomdat_dir



class adas_file():
    '''Read ADAS file in ADF11 format over the given ne, T. '''

    def __init__(self, filepath):

        self.filepath = filepath
        self.filename=filepath.split('/')[-1]
        self.file_type = self.filename[:3]

        if self.file_type not in ['brs','sxr']:
            self.imp = self.filename.split('_')[1].split('.')[0]

        # get data
        self.load()

        # settings for plotting
        self.n_ion  = self.data.shape[0]
        self.ncol = np.ceil(np.sqrt(self.n_ion))
        self.nrow = np.ceil(self.n_ion/self.ncol)


    def load(self):

        with open(self.filepath) as f:
            header = f.readline()
            n_ions, n_ne, n_T = header.split()[:3]
            details = ' '.join(header.split()[3:])

            f.readline()

            n_ions,n_ne,n_T = int(n_ions),int(n_ne),int(n_T)
            logT = []
            logNe = []
            while len(logNe)< n_ne:
                line = f.readline()
                logNe = logNe+[float(n) for n in line.split()]
            while len(logT)< n_T:
                line = f.readline()
                logT = logT+[float(t) for t in line.split()]

            logT = np.array(logT)
            logNe = np.array(logNe)

            data = []
            for i_ion in range(n_ions):
                f.readline()
                plsx = []
                while len(plsx)< n_ne*n_T:
                    line = f.readline()
                    plsx = plsx+[float(L) for L in line.split()]
                plsx = np.array(plsx).reshape(n_T, n_ne)
                data.append(np.array(plsx))

        self.logNe = logNe; self.logT = logT; self.data = np.array(data)


    def plot(self, fig=None, axes=None):

        if fig is None or axes is None:
            fig,axes = plt.subplots(int(self.ncol),int(self.nrow), sharex=True, sharey=True)

        axes = np.atleast_2d(axes)
        colormap = cm.rainbow
        fig.suptitle(self.filename+'  '+ get_file_types()[self.file_type])

        for i,ax in enumerate(axes.flatten()):
            if i >= self.n_ion: break
            if all(self.data[i,:,:].std(axis=1)==0): #independent of density
                ax.plot(self.logT, self.data[i,:,0])
            else:
                ax.set_prop_cycle('color', colormap( np.linspace(0,1,self.data.shape[2])))
                ax.plot(self.logT, self.data[i])
            ax.grid(True)
            if self.file_type != 'brs':
                charge = i+1 if self.file_type in ['scd','prs','ccd','prb'] else i
                ax.set_title(self.imp+'$^{%d\!+}$'% (charge-1))   # check?

        for ax in axes[-1]:
            ax.set_xlabel('$\log T_e\ \mathrm{[eV]}$')
        for ax in axes[:,0]:
            if self.file_type in ['scd','acd','ccd']:
                ax.set_ylabel('$\log('+self.file_type+')\ \mathrm{[cm^3/s]}$')
            elif self.file_type in ['prb','plt','prc','pls','brs','prs']:
                ax.set_ylabel('$\log('+self.file_type+')\ \mathrm{[W\cdot cm^3]}$')


def get_atom_data(imp, files = None):
    ''' Collect atomic data for a given impurity from all types of ADAS files available or
    for only those requested. 

    Args:
        imp : str
            Atomic symbol of impurity ion.
        files : list or array-like
            ADAS file names to be fetched. 
    
    Returns:
        atom_data : dict
           Dictionary containing data for each of the requested files (or all files returned by
           :py:meth:`~aurora.atomic.adas_files_dict` for the impurity ion of interest). 
           Each entry of the dictionary gives log-10 of ne, log-10 of Te and log-10 of the data
           as attributes atom_data[key].logNe, atom_data[key].logT, atom_data[key].data
    '''
    atomdat_dir = get_atomdat_info()

    atom_data = {}

    if files is None:
        # fetch all file types
        files = get_file_types().keys()

    # get dictionary containing default list of ADAS atomic files
    files_dict =  adas_files_dict()

    for key in files:
        file = files_dict[imp][key]

        # load specific file and add it to output dictionary
        res = adas_file(atomdat_dir+file)
        atom_data[key] = res.logNe, res.logT, res.data

    return atom_data


def read_adf15(path, order=1, Te_max = None, ne_max = None,
               ax=None, plot_log=False, plot_3d=False,
               recomb=False,
               pec_plot_min=None,  pec_plot_max = None,
               plot_lines=[]):
    """Read photon emissivity coefficients from an ADF15 file.

    Returns a dictionary whose keys are the wavelengths of the lines in
    angstroms. The value is an interp2d instance that will evaluate the PEC at
    a desired dens, temp.

    Args:
        path : str
            Path to adf15 file to read.
        order : int, opt
            Parameter to control the order of interpolation.
        recomb : bool, opt
            If True, fetch recombination contributions to available lines. If False,
            fetch only ionization contributions.

        To plot PEC data:
        
        plot_lines : list
            List of lines whose PEC data should be displayed. Lines should be identified
            by their wavelengths. The list of available wavelengths in a given file can be retrieved
            by first running this function ones, checking dictionary keys, and then requesting a
            plot of one (or more) of them.
        plot_log : bool
            When plotting, set a log scale
        plot_3d : bool
            Display PEC data as a 3D plot rather than a 2D one.
        pec_plot_min : float
            Minimum value of PEC to visualize in a plot
        pec_plot_max : float
            Maximum value of PEC to visualize in a plot
        ax : matplotlib axes instance
            If not None, plot on this set of axes
        Te_max : float
            Maximum Te value to plot when len(plot_lines)>1
        ne_max : float
            Maximum ne value to plot when len(plot_lines)>1

    Returns:
        pec_dict : dict
            Dictionary containing interpolation functions for each of the available lines of the
            indicated type (ionization or recombination). Each interpolation function takes as arguments
            the log-10 of ne and Te.

    MWE:
        path='/home/sciortino/atomlib/atomdat_master/adf15/h/pju#h0.dat'
        pec = read_adf15(path, recomb=False)
        pec = read_adf15(path, plot_lines=[list(pec.keys())[0]], recomb=False)

    This function should work with PEC files produced via adas810 or adas218.

    """
    if plot_3d:
        from mpl_toolkits.mplot3d import Axes3D

    with open(path, 'r') as f:
        lines = f.readlines()
    cs = path.split('#')[-1].split('.dat')[0]

    header = lines.pop(0)
    # Get the expected number of lines by reading the header:
    num_lines = int(header.split()[0])
    pec_dict = {}

    for i in range(0, num_lines):
        # Get the wavelength, number of densities and number of temperatures
        # from the first line of the entry:
        l = lines.pop(0)
        header = l.split()

        # sometimes the wavelength and its units are not separated. Rough solution:
        try:
            #header = np.concatenate( [hh.split('A')[0] for hh in header])
            header = [hh.split('A')[0] for hh in header]
        except:
            # lam and A are separated. Delete 'A' unit.
            header = np.delete(header, 1)

        try:
            lam = float(header[0])
        except ValueError:
            # These lines appear to occur when lam has more digits than the
            # allowed field width. For the moment, ignore these.
            lam = None

        if recomb and ('recom' not in l.lower()):
            lam=None
        elif not recomb and ('excit' not in l.lower()):
            lam = None

        try:
            num_den = int(header[1])
            num_temp = int(header[2])
        except:
            # 2nd element was empty -- annoyingly, this happens sometimes
            num_den = int(header[2])
            num_temp = int(header[3])

        # Get the densities:
        dens = []
        while len(dens) < num_den:
            dens += [float(v) for v in lines.pop(0).split()]
        dens = scipy.asarray(dens)

        # Get the temperatures:
        temp = []
        while len(temp) < num_temp:
            temp += [float(v) for v in lines.pop(0).split()]
        temp = scipy.asarray(temp)

        # Get the PEC's:
        PEC = []
        while len(PEC) < num_den:
            PEC.append([])
            while len(PEC[-1]) < num_temp:
                PEC[-1] += [float(v) for v in lines.pop(0).split()]
        PEC = scipy.asarray(PEC)

        if lam is not None:
            if lam not in pec_dict:
                pec_dict[lam] = []

            pec_dict[lam].append(
                scipy.interpolate.RectBivariateSpline(
                    np.log10(dens),
                    np.log10(temp),
                    PEC,
                    kx=order,
                    ky=order
                )
            )

            # {'dens': dens, 'temp': temp, 'PEC': PEC}
            if lam in plot_lines:

                # use log spacing for ne values
                if ne_max is not None:
                    # avoid overflow inside of np.logspace
                    ne_eval = 10** np.linspace(np.log10(dens.min()), np.log10(ne_max), 10)
                else:
                    ne_eval = 10** np.linspace(np.log10(dens.min()), np.log10(dens.max()), 10)
                    #scipy.logspace(dens.min(), dens.max(), 10)

                # linear spacing for Te values
                if Te_max is not None:
                    Te_eval = scipy.linspace(temp.min(), Te_max*1000, 100)
                else:
                    Te_eval = scipy.linspace(temp.min(), temp.max(), 100)

                NE, TE = scipy.meshgrid(ne_eval, Te_eval)
                PEC_eval = pec_dict[lam][-1].ev(np.log10(NE), np.log10(TE))

                if ax is None:
                    f1 = plt.figure()
                    if plot_3d:
                        ax1 = f1.add_subplot(1,1,1, projection='3d')
                    else:
                        ax1 = f1.add_subplot(1,1,1)
                else:
                    ax1 = ax

                if plot_3d:

                    # linear scales (log doesn't work in matplotlib 3D)
                    # but can do all plotting on log quantities
                    if plot_log:
                        logNE, logTE = scipy.meshgrid(np.log10(ne_eval), np.log10(Te_eval))
                        ax1.plot_surface(logNE, logTE, PEC_eval, alpha=0.5)
                    else:
                        ax1.plot_surface(NE, TE, PEC_eval, alpha=0.5)

                    if plot_log:
                        dens = np.log10(dens); temp = np.log10(temp)

                    if Te_max is None and ne_max is None:
                        DENS, TEMP = scipy.meshgrid(dens, temp)
                        ax1.scatter(DENS.ravel(), TEMP.ravel(), PEC.T.ravel(), color='b')
                    elif Te_max is not None and ne_max is None:
                        Te_max_ind = np.argmin(np.abs(temp - Te_max*1000))
                        DENS, TEMP = scipy.meshgrid(dens, temp[:Te_max_ind+1])
                        ax1.scatter(DENS.ravel(), TEMP.ravel(), PEC[:,:Te_max_ind+1].T.ravel(), color='b')
                    elif Te_max is None and ne_max is not None:
                        ne_max_ind = np.argmin(np.abs(dens - ne_max))
                        DENS, TEMP = scipy.meshgrid(dens[:ne_max_ind+1], temp)
                        ax1.scatter(DENS.ravel(), TEMP.ravel(), PEC[:ne_max_ind+1,:].T.ravel(), color='b')
                    elif Te_max is not None and ne_max is not None:
                        Te_max_ind = np.argmin(np.abs(temp - Te_max*1000))
                        ne_max_ind = np.argmin(np.abs(dens - ne_max))
                        DENS, TEMP = scipy.meshgrid(dens[:ne_max_ind+1], temp[:Te_max_ind+1])
                        ax1.scatter(DENS.ravel(), TEMP.ravel(), PEC[:ne_max_ind+1,:Te_max_ind+1].T.ravel(), color='b')

                    if ax is None:
                        ax1.set_xlabel('$n_e$ [cm$^{-3}$]')
                        ax1.set_ylabel('$T_e$ [eV]')

                        ax1.set_zlabel('PEC')
                        ax1.set_title(cs + r' --- $\lambda$ = '+str(lam) +' A')
                        plt.tight_layout()

                else:

                    # plot in 2D
                    labels = ['{:.0e}'.format(ne)+r' $cm^{-3}$' for ne in ne_eval]
                    for ine in np.arange(PEC_eval.shape[1]):
                        ax1.plot(Te_eval, PEC_eval[:,ine], label=labels[ine])
                    ax1.set_xlabel(r'$T_e$ [kev]')
                    ax1.set_ylabel('PEC')
                    ax1.set_yscale('log')
                    if pec_plot_min is not None:
                        ax1.set_ylim([pec_plot_min, ax1.get_ylim()[1]])
                    if pec_plot_max is not None:
                        ax1.set_ylim([ax1.get_ylim()[0], pec_plot_max ])

                    ax1.legend(loc='best').set_draggable(True)
                    ax1.set_title(cs + r' --- $\lambda$ = '+str(lam) +' A')


    return pec_dict



def null_space(A):
    ''' Find null space of matrix A '''
    u, s, vh = svd(A, full_matrices=True)
    Q = vh[-1,:].T.conj()    # -1 index is after infinite time/equilibration
    # return the smallest singular eigenvalue
    return Q, s[-2]  # matrix is singular, so last index is ~0


def get_frac_abundances(atom_data, ne_cm3,Te_eV=None, n0_by_ne=1e-5, include_cx=False,
                        plot=True, ax = None, rho = None, rho_lbl=None,ls='-',compute_rates=False):
    '''Calculate fractional abundances from ionization and recombination equilibrium.
    If include_cx=True, radiative recombination and thermal charge exchange are summed.

    Args:
        atom_data : dictionary of atomic ADAS files (only acd, scd are required; ccd is 
            necessary only if include_cx=True
        ne_cm3 : float or array
            Electron density in units of cm^-3
        Te_eV : float or array, optional
            Electron temperature in units of eV. If left to None, the Te grid given in the 
            atomic data is used.
        n0_by_ne: float or array, optional
            Ratio of background neutral hydrogen to electron density, used if include_cx=True. 
        include_cx : bool
            If True, charge exchange with background thermal neutrals is included. 
        plot : bool, optional
            Show fractional abundances as a function of ne,Te profiles parameterization.
        ax : matplotlib.pyplot Axes instance
            Axes on which to plot if plot=True. If False, it creates new axes
        rho : list or array, optional
            Vector of radial coordinates on which ne,Te (and possibly n0_by_ne) are given. 
            This is only used for plotting, if given. 
        rho_lbl: str, optional
            Label to be used for rho. If left to None, defaults to a general "rho".
        ls : str, optional
            Line style for plots. Continuous lines are used by default. 
        compute_rates : bool
            If True, compute rate coefficients for ionization/recombination equilibrium on top
            of fractional abundances (which should be the same regardless of the method used). 

    Returns:
        logTe : array
            log10 of electron temperatures as a function of which the fractional abundances and
            rate coefficients are given.
        fz : array, (space,nZ)
            Fractional abundances across the same grid used by the input ne,Te values. 
        rate_coeff : array, (space, nZ)
            Rate coefficients in units of [s^-1]. 

    '''

    logTe, logS,logR,logcx = get_cs_balance_terms(atom_data,ne_cm3,Te_eV,maxTe=10e3,include_cx=include_cx)
    
    if include_cx:
        # Get an effective recombination rate by summing radiative & CX recombination rates
        logR= np.logaddexp(logR,np.log(n0_by_ne)[:,None] +logcx)
        
    # analytical formula for fractional abundance
    rate_ratio = np.hstack((np.zeros_like(logTe)[:,None], logS-logR))
    rate_ratio[rate_ratio<-10.]=-10.  # to avoid underflow in exponential
    fz = np.exp(np.cumsum(rate_ratio,axis=1))
    fz /= fz.sum(1)[:,None]
    
    if compute_rates:
        # numerical method which calculates also rate_coeff
        nion = logR.shape[1]
        fz  = np.zeros((logTe.size,nion+1))
        rate_coeff = np.zeros(logTe.size)
        for it,t in enumerate(logTe):

            A = -np.diag(np.r_[np.exp(logS[it]),0])+np.diag(np.exp(logS[it]),-1)+\
                np.diag(np.exp(logR[it]),1)- np.diag(np.r_[0,np.exp(logR[it])])
            
            N,rate_coeff[it] = null_space(A)
            fz[it] = N/np.sum(N)

        rate_coeff*=(ne_cm3 * 1e-6)
    
    if plot:
        # plot fractional abundances
        if ax is None:
            fig,axx = plt.subplots()
        else:
            axx = ax

        if rho is None:
            x = 10**logTe
            axx.set_xlabel('T$_e$ [eV]')
            axx.set_xscale('log')
        else:
            if rho_lbl is None: rho_lbl=r'$\rho$'
            x = rho
            axx.set_xlabel(rho_lbl)

        axx.set_prop_cycle('color',cm.plasma(np.linspace(0,1,fz.shape[1])))
        axx.plot(x,fz,ls=ls)
        for i in range(len(fz.T)):
            imax = np.argmax(fz[:,i])
            axx.text(np.max([0.05,x[imax]]), fz[imax,i], i, horizontalalignment='center', clip_on=True)
        axx.grid('on')
        axx.set_ylim(0,1.05)
        axx.set_xlim(x[0],x[-1])
        axx.set_title(r'Fractional abundances')
        if ax is None:
            axx.legend()
    else:
        axx = None    

    if compute_rates:
        return logTe, fz, rate_coeff
    else:
        return logTe, fz





def get_cs_balance_terms(atom_data, ne_cm3=5e13, Te_eV=None, maxTe=10e3, include_cx=True):
    '''Get S, R and cx on the same logTe grid. 
    
    Args:
        atom_data : dictionary of atomic ADAS files (only acd, scd are required; ccd is 
            necessary only if include_cx=True
        ne_cm3 : float or array
            Electron density in units of cm^-3
        Te_eV : float or array
            Electron temperature in units of eV. If left to None, the Te grid
            given in the atomic data is used.
        maxTe : float
            Maximum temperature of interest; only used if Te is left to None. 
        include_cx : bool
            If True, obtain charge exchange terms as well. 
    
    Returns:
        logTe : array (n_Te)
            log10 Te grid on which atomic rates are given
        logS, logR (,logcx): arrays (n_ne,n_Te)
            atomic rates for effective ionization, radiative+dielectronic
            recombination (+ charge exchange, if requested). After exponentiation, all terms
            will be in units of s^-1. 
    '''
    if Te_eV is None:
        # find smallest Te grid from all files
        logne1, logTe1,_ = atom_data['scd']  # ionization
        logne2, logTe2,_ = atom_data['acd']  # radiative recombination

        minTe = max(logTe1[0],logTe2[0])
        maxTe = np.log10(maxTe)# don't go further than some given temperature keV
        maxTe = min(maxTe,logTe1[-1],logTe2[-1])  # avoid extrapolation

        if include_cx:
            logne3, logTe3,_ = atom_data['ccd']  # thermal cx recombination
            minTe = max(minTe,logTe3[0])
            maxTe = min(maxTe,logTe3[-1])  # avoid extrapolation

        logTe = np.linspace(minTe,maxTe,200)

    else:
        logTe = np.log10(Te_eV)

    logne = np.log10(ne_cm3)

    logS = interp_atom_prof(atom_data['scd'],logne, logTe,log_val=True, x_multiply=False)
    logR = interp_atom_prof(atom_data['acd'],logne, logTe,log_val=True, x_multiply=False)
    if include_cx:
        logcx = interp_atom_prof(atom_data['ccd'],logne, logTe,log_val=True, x_multiply=False)
    else:
        logcx = None

    return logTe, logS, logR, logcx



def plot_relax_time(logTe, rate_coeff, ax = None):
    ''' Plot relaxation time of the ionization equilibrium corresponding
    to the inverse of the given rate coefficients.

    Args:
        logTe : array (nr,)
            log-10 of Te [eV], on an arbitrary grid (same as other arguments, but not
            necessarily radial)
        rate_coeff : array (nr,)
            Rate coefficients from ionization balance. See :py:meth:`~aurora.atomic.get_frac_abundances`
            to obtain these via the "compute_rates" argument. 
            N.B.: these rate coefficients will depend also on electron density, which does affect 
            relaxation times. 
        ax : matplotlib axes instance, optional
            If provided, plot relaxation times on these axes.    
    '''
    if ax is None:
        ax = plt.subplot(111)

    ax.loglog(10**logTe,1e3/rate_coeff,'b' )
    ax.set_xlim(10**logTe[0],10**logTe[-1])
    ax.grid('on')
    ax.set_xlabel('T$_e$ [eV]')
    ax.set_ylabel(r'$\tau_\mathrm{relax}$ [ms]')





class CartesianGrid(object):
    """
    Linear multivariate Cartesian grid interpolation in arbitrary dimensions
    This is a regular grid with equal spacing.
    """
    def __init__(self, grids, values):
        ''' grids: list of arrays or ranges of each dimension
        values: array with shape (ndim, len(grid[0]), len(grid[1]),...)
        '''
        self.values = np.ascontiguousarray(values)
        for g,s in  zip(grids,values.shape[1:]):
            if len(g) != s: raise Exception('wrong size of values array')

        self.offsets = [g[0] for g in grids]
        self.scales  = [(g[-1]-g[0])/(n-1) for g,n in zip(grids, values.shape[1:])]
        self.N = values.shape[1:]

    def __call__(self, *coords):
        ''' Transform coords into pixel values '''
        out_shape = coords[0].shape

        coords = np.array(coords).T
        coords -= self.offsets
        coords /= self.scales
        coords = coords.T

        #clip dimension - it will extrapolation by a nearest value
        for coord, n in zip(coords, self.N):
            np.clip(coord,0,n,coord)

        #prepare output array
        inter_out = np.empty((self.values.shape[0],)+out_shape, dtype=self.values.dtype)

        #fast interpolation on a regular grid
        for out,val in zip(inter_out,self.values):
            scipy.ndimage.map_coordinates(val, coords,
                                            order=1, output=out)

        return inter_out


def interp_atom_prof(atom_table,x_prof, y_prof,log_val=False, x_multiply=True):
    ''' Fast interpolate atomic data in atom_table onto the x_prof and y_prof profiles.
    This function assume that x_prof, y_prof, x,y, table are all base-10 logarithms,
    and x_prof, y_prof are equally spaced.

    Args:
        atom_table : list
            List with x,y, table = atom_table, containing atomic data from one of the ADAS files. 
        x_prof : array (nt,nr)
            Spatio-temporal profiles of the first coordinate of the ADAS file table (usually 
            electron density in cm^-3)
        y_prof : array (nt,nr)
            Spatio-temporal profiles of the second coordinate of the ADAS file table (usually 
            electron temperature in eV)
        log_val : bool
            If True, return natural logarithm of the data
        x_multiply : bool
            If True, multiply output by 10**x_prof. 

    Returns:
        interp_vals : array (nt,nion,nr)
            Interpolated atomic data on time,charge state and spatial grid that correspond to the 
            ion of interest and the spatiotemporal grids of x_prof and y_prof. 
    '''
    x,y, table = atom_table

    if (abs(table-table[...,[0]]).all()  < .05) or x_prof is None:
        # 1D interpolaion if independent of the last dimension - like SXR radiation data

        reg_interp = CartesianGrid((y, ),table[:,:,0]*np.log(10))
        interp_vals = reg_interp(y_prof) 

        # multipling of logarithms is just adding
        if x_multiply and x_prof is not None:
            interp_vals += x_prof*np.log(10)

    else: # 2D interpolation
        if x_multiply: #multipling of logarithms is just adding
            table += x
        # broadcast both variables in the sae shape
        x_prof, y_prof = np.broadcast_arrays(x_prof, y_prof)
        #perform fast linear interpolation
        reg_interp = CartesianGrid((x, y),table.swapaxes(1,2)*np.log(10))
        interp_vals = reg_interp(x_prof,y_prof) 

    # reshape to shape(nt,nion,nr)
    interp_vals = interp_vals.swapaxes(0,1)

    if not log_val:
        # return actual value, not logarithm
        np.exp(interp_vals, out=interp_vals)

    return interp_vals





def gff_mean(Z,Te):
    '''
    Total free-free gaunt factor yielding the total radiated bremsstrahlung power
    when multiplying with the result for gff=1.
    Data originally from Karzas & Latter, extracted from STRAHL's atomic_data.f.
    '''
    from scipy.constants import e,h,c,Rydberg
    thirteenpointsix = h*c*Rydberg/e

    log_gamma2_grid =[-3.000,-2.833,-2.667,-2.500,-2.333,-2.167,-2.000,-1.833,-1.667,-1.500,
                        -1.333,-1.167,-1.000,-0.833,-0.667,-0.500,-0.333,-0.167, 0.000, 0.167,
                        0.333, 0.500, 0.667, 0.833, 1.000]

    gff_mean_grid = [1.140, 1.149, 1.159, 1.170, 1.181, 1.193, 1.210, 1.233, 1.261, 1.290,
                     1.318, 1.344, 1.370, 1.394, 1.416, 1.434, 1.445, 1.448, 1.440, 1.418,
                     1.389, 1.360, 1.336, 1.317, 1.300]

    # set min Te here to 10 eV (as in STRAHL), because the grid above does not extend to lower temperatures
    Te = np.maximum(Te, 10.0)

    log_gamma2 = np.log10(Z**2*thirteenpointsix/Te)

    # dangerous/inaccurate extrapolation...??
    return np.interp(log_gamma2, log_gamma2_grid,gff_mean_grid)



def impurity_brems(nz, ne, Te):
    '''Impurity bremsstrahlung in units of mW/nm/sr/m^3.cm^3.

    This is only approximate and may not be very useful, since this contribution
    is already included in the continuum in x2.

    This estimate does not have the correct ne-dependence of the Gaunt factor... use with care!
    '''

    # neutral stage doesn't produce brems
    Z_imp = nz.shape[1]-1
    Z = np.arange(Z_imp)[None,:,None]+1

    gff = gff_mean(Z,Te[:,None])
    imp_brems = Z**2 * nz[:,1:] * gff * (1.69e-32 *np.sqrt(Te) * ne) [:,None]

    return imp_brems


def main_ion_brems(Zi, ni, ne, Te):
    '''Main-ion bremsstrahlung in units of  mW/nm/sr/m^3.cm^3.

    It is likely better to calculate this from H/D/T plt files, which will have more accurate 
    Gaunt factors with the correct density dependence.
    '''
    return 1.69e-32 * ni * Zi**2. * ne * np.sqrt(Te) * gff_mean(Zi,Te)




def get_adas_continuum_rad(ion_name, n_ion, logne_prof, logTe_prof, sxr=False):
    '''Convenience function to get ADAS estimate for continuum radiation in [M/m^3] for a 
    given background ion at given (log) density and temperature profiles. 

    If sxr=True, 'prs' files are used instead of 'prb' ones, thus giving SXR-filtered
    continuum radiation for the SXR filter indicated by the atomic data dictionary.

    Args:
        ion_name : str
            Atomic symbol of ion.
        n_ion : array (nt,nr,nz)
            Density of each charge state of the ion of interest
        logne_prof : array (nt,nr)
            Log-10 of electron density profile (in cm^-3)
        logTe_prof : array (nt,nr)
            Log-10 of electron temperature profile (in eV)
        sxr : bool, optional
            If True, compute continuum radiation in the SXR range rather than the total 
            continuum radiation.

    Returns:
        cont_rad : array (nt,nr)
            Continuum radiation, either total or in the SXR-range, depending on the 
            'sxr' input variable.    
    '''        
    # get continuum radiation data for chosen ion (either total or SXR-filtered)
    atom_data = get_atom_data(ion_name,['prs' if sxr else 'prb'])
    x,y,tab = atom_data['prs' if sxr else 'prb']

    # recombination and bremstrahlung of fully stripped ion
    atom_rates = interp_atom_prof((x,y,tab[[-1]]),logne_prof,logTe_prof,x_multiply=False)
    
    # Now compute expected continuum radiation from this ion:
    cont_rad = atom_rates[:,0] * n_ion * 1e6 # bulk fully-stripped ion radiation [W/m^3]

    return cont_rad


def get_cooling_factors(atom_data, logTe_prof, fz, plot=True,ax=None):
    '''Calculate cooling coefficients for the given fractional abundances and kinetic profiles.

    Args:
        atom_data : dict
            Dictionary containing atomic data as output by :py:meth:`~aurora.atomic.get_atom_data`
            for the atomic processes of interest. "prs","pls","plt" and "prb" are required by this function.
        logTe_prof : array (nt,nr)
            Log-10 of electron temperature profile (in eV)
        fz : array (nt,nr)
            Fractional abundances for all charge states of the ion of "atom_data"
        plot : bool
            If True, plot all radiation components, summed over charge states.
        ax : matplotlib.Axes instance
            If provided, plot results on these axes. 
    
    Returns:
        pls : array (nt,nr)
            Line radiation in the SXR range for each charge state
        prs : array (nt,nr)
            Continuum radiation in the SXR range for each charge state
        pltt : array (nt,nr)
            Line radiation (unfiltered) for each charge state.
            NB: this corresponds to the ADAS "plt" files. An additional "t" is added to the name to avoid
            conflict with the common matplotlib.pyplot short form "plt"
        prb : array (nt,nr)
            Continuum radiation (unfiltered) for each charge state
    '''
    try:
        atom_data['prs']
        atom_data['pls']
        atom_data['plt']
        atom_data['prb']
    except:
        raise ValueError('prs, plt and/or prb files not available!')

    prs = interp_atom_prof(atom_data['prs'],None,logTe_prof)#continuum radiation in SXR range
    pls = interp_atom_prof(atom_data['pls'],None,logTe_prof)#line radiation in SXR range
    pltt= interp_atom_prof(atom_data['plt'],None,logTe_prof) #line radiation
    prb = interp_atom_prof(atom_data['prb'],None,logTe_prof)#continuum radiation

    pls *= fz[:,:-1]
    prs *= fz[:, 1:]
    pltt*= fz[:,:-1]
    prb *= fz[:, 1:]

    line_rad_sxr  = pls.sum(1)
    brems_rad_sxr = prs.sum(1)
    line_rad_tot  = pltt.sum(1)
    brems_rad_tot = prb.sum(1)

    if plot:
        # plot cooling factors
        if ax is None:
            ax = plt.subplot(111)

        # SXR radiation components
        ax.loglog(10**logTe_prof, line_rad_sxr,'b',label='SXR line radiation')   
        ax.loglog(10**logTe_prof, brems_rad_sxr,'r',label='SXR bremsstrahlung and recombination')
        ax.loglog(10**logTe_prof, brems_rad_sxr+line_rad_sxr,'k',label='total SXR radiation',lw=2)

        # total radiation (includes hard X-ray, visible, UV, etc.)
        ax.loglog(10**logTe_prof, line_rad_tot,'g--',label='Unfiltered line radiation')
        ax.loglog(10**logTe_prof, brems_rad_tot,'y--',label='Unfiltered continuum radiation')
        ax.loglog(10**logTe_prof, brems_rad_tot+line_rad_tot,'y--',label='Unfiltered total continuum radiation')

        ax.legend(loc='best')
        
        # Set xlims to visualize scales better
        ax.set_xlim(50,10**logTe_prof[-1])
        ax.set_ylim(line_rad_sxr[np.argmin(np.abs(10**logTe_prof - 50))], np.nanmax( line_rad_tot)*10)

        ax.grid('on')
        ax.set_xlabel('T$_e$ [eV]')
        ax.set_ylabel('L$_z$ [Wm$^3$]')
        ax.set_title('Cooling factors')

    # ion-resolved radiation terms:
    return pls, prs, pltt,prb

        
        

def balance(logTe_val, cs, n0_by_ne, logTe_, S,R,cx):
    '''Evaluate balance of effective ionization, recombination and charge exchange at a given temperature. '''

    a = R +n0_by_ne * cx
    SS_0 = 10**(interp1d(logTe_, np.log10(S[cs-1,:]), kind='cubic', bounds_error=False)(logTe_val))
    aa_0 = 10**(interp1d(logTe_, np.log10(a[cs-1,:]), kind='cubic', bounds_error=False)(logTe_val))
    SS_1 = 10**(interp1d(logTe_, np.log10(S[cs,:]), kind='cubic', bounds_error=False)(logTe_val))
    aa_1 = 10**(interp1d(logTe_, np.log10(a[cs,:]), kind='cubic', bounds_error=False)(logTe_val))

    val = SS_0 - SS_1 - aa_0 + aa_1
    return val*1e20 # get this to large-ish units to avoid tolerance issues due to small powers of 10




def plot_norm_ion_freq(S_z, q_prof, R_prof, imp_A, Ti_prof,
                       nz_profs=None, rhop=None, plot=True, eps_prof=None):
    '''
    Compare effective ionization rate for each charge state with the 
    characteristic transit time that a non-trapped and trapped impurity ion takes
    to travel a parallel distance L = q R. 

    If the normalized ionization rate is less than 1, then flux surface averaging of
    background asymmetries (e.g. from edge or beam neutrals) can be considered in a 
    "flux-surface-averaged" sense; otherwise, local effects (i.e. not flux-surface-averaged)
    may be too important to ignore. 

    This function is inspired by Dux et al. NF 2020. Note that in this paper the ionization 
    rate averaged over all charge state densities is considered. This function avoids the 
    averaging over charge states, unless these are provided as an input. 

    Args:
        S_z : array (r,cs) [s^-1]
             Effective ionization rates for each charge state as a function of radius. 
             Note that, for convenience within aurora, cs includes the neutral stage.
        q_prof : array (r,)
             Radial profile of safety factor
        R_prof : array (r,) or float [m]
             Radial profile of major radius, either given as an average of HFS and LFS, or also
             simply as a scalar (major radius on axis)
        imp_A : float [amu]
             Atomic mass number, i.e. number of protons + neutrons (e.g. 2 for D)
        Ti_prof : array (r,)
             Radial profile of ion temperature [eV]
        nz_profs : array (r,cs), optional
             Radial profile for each charge state. If provided, calculate average normalized 
             ionization rate over all charge states.
        rhop : array (r,), optional
             Sqrt of poloidal flux radial grid. This is used only for (optional) plotting. 
        plot : bool, optional
             If True, plot results.
        eps_prof : array (r,), optional
             Radial profile of inverse aspect ratio, i.e. r/R, only used if plotting is requested.  

    Returns:
        nu_ioniz_star : array (r,cs) or (r,)
             Normalized ionization rate. If nz_profs is given as an input, this is an average over
             all charge state; otherwise, it is given for each charge state.
    '''

    nu = np.zeros_like(S_z)
    for cs in np.arange(S_z.shape[1]): # exclude neutral states
        nu[:,cs] = S_z[:,cs] * q_prof * R_prof * np.sqrt((imp_A * m_p)/(2*Ti_prof))

    if nz_profs is not None:
        # calculate average nu_ioniz_star 
        nu_ioniz_star = np.sum(nz_profs[:,1:]*nu[:,1:],axis=1)/np.sum(nz_profs[:,1:],axis=1)
    else:
        # return normalized ionization rate for each charge state
        nu_ioniz_star = nu[:,1:]

    if plot:
        if rhop is None:
            rhop = np.arange(nu.shape[0])
            
        fig,ax = plt.subplots()
        if nu_ioniz_star.ndim==1:
            ax.semilogy(rhop,nu_ioniz_star, label=r'$\nu_{ion}^*$')
        else:
            for cs in np.arange(S_z.shape[1]-1):
                ax.semilogy(rhop, nu_ioniz_star[:,cs], label=f'q={cs+1}')
            ax.set_ylabel(r'$\nu_{ion}^*$')

        ax.set_xlabel(r'$\rho_p$')

        if eps_prof is not None:
            ax.semilogy(rhop, np.sqrt(eps_prof), label=r'$\sqrt{\epsilon}$')

        ax.legend().set_draggable(True)



def adas_files_dict():
    '''Selections for ADAS files for Aurora runs and radiation calculations.
    This function can be called to fetch a set of default files, which can then be modified (e.g. to 
    use a new file for a specific SXR filter) before running a calculation. 
    '''
            
    files={}
    files["H"] = {}   #1
    files["H"]['acd'] = "acd96_h.dat"
    files["H"]['scd'] = "scd96_h.dat"
    files["H"]['prb'] = "prb96_h.dat"
    files["H"]['plt'] = "plt96_h.dat"
    files["H"]['ccd'] = "ccd96_h.dat"
    files["H"]['prc'] = "prc96_h.dat"
    files["H"]['pls'] = "pls_H_14.dat"
    files["H"]['prs'] = "prs_H_14.dat"
    files["H"]['fis'] = "sxrfil14.dat"
    files["H"]['brs'] = "brs05360.dat"
    files["He"] = {}   #2
    files["He"]['acd'] = "acd96_he.dat"
    files["He"]['scd'] = "scd96_he.dat"
    files["He"]['prb'] = "prb96_he.dat"
    files["He"]['plt'] = "plt96_he.dat"
    files["He"]['ccd'] = "ccd07_he.dat"
    files["He"]['prc'] = "prc96_he.dat"
    files["He"]['pls'] = "pls_He_14.dat"
    files["He"]['prs'] = "prs_He_14.dat"
    files["He"]['fis'] = "sxrfil14.dat"
    files["He"]['brs'] = "brs05360.dat"
    files["Li"] = {}   #3
    files["Li"]['acd'] = "acd96_li.dat"
    files["Li"]['scd'] = "scd96_li.dat"
    files["Li"]['ccd'] = "ccd89_li.dat"
    files["Li"]['prb'] = "prb96_li.dat"
    files["Li"]['plt'] = "plt96_li.dat"
    files["Li"]['prc'] = "prc89_li.dat"
    files["Li"]['pls'] = "pls89_li.dat"
    files["Be"] = {}   #4
    files["Be"]['acd'] = "acd96_be.dat"
    files["Be"]['scd'] = "scd96_be.dat"
    files["Be"]['prb'] = "prb96_be.dat"
    files["Be"]['plt'] = "plt96_be.dat"
    files["Be"]['ccd'] = "ccd89_be.dat"
    files["Be"]['prc'] = "prc89_be.dat"
    files["Be"]['pls'] = "plsx5_be.dat"
    files["Be"]['prs'] = "prsx5_be.dat"
    files["B"] = {}   #5
    files["B"]['acd'] = "acd89_b.dat"
    files["B"]['scd'] = "scd89_b.dat"
    files["B"]['ccd'] = "ccd89_b.dat"
    files["B"]['prb'] = "prb89_b.dat"
    files["B"]['plt'] = "plt89_b.dat"
    files["B"]['prc'] = "prc89_b.dat"
    files["B"]['pls'] = "plsx5_b.dat"
    files["B"]['prs'] = "prsx5_b.dat"
    files["C"] = {}    #6
    files["C"]['acd'] = "acd96_c.dat"
    files["C"]['scd'] = "scd96_c.dat"
    files["C"]['prb'] = "prb96_c.dat"
    files["C"]['plt'] = "plt96_c.dat"
    files["C"]['ccd'] = "ccd96_c.dat"
    files["C"]['prc'] = "prc96_c.dat"
    files["C"]['pls'] = "pls_C_14.dat"
    files["C"]['prs'] = "prs_C_14.dat"
    files["C"]['fis'] = "sxrfil14.dat"
    files["C"]['brs'] = "brs05360.dat"
    files["N"] = {}    #7
    files["N"]['acd'] = "acd96_n.dat"
    files["N"]['scd'] = "scd96_n.dat"
    files["N"]['ccd'] = "ccd89_n.dat"
    files["N"]['prb'] = "prb96_n.dat"
    files["N"]['plt'] = "plt96_n.dat"
    files["N"]['pls'] = "plsx8_n.dat"
    files["N"]['prs'] = "prsx8_n.dat"
    files["N"]['fis'] = "sxrfilD1.dat"
    files["N"]['brs'] = "brs05360.dat"
    files["N"]['ccd'] = "ccd96_n.dat"
    files["O"] = {}    #8
    files["O"]['acd'] = "acd96_o.dat"
    files["O"]['scd'] = "scd96_o.dat"
    files["O"]['ccd'] = "ccd89_o.dat"
    files["O"]['prb'] = "prb96_o.dat"
    files["O"]['plt'] = "plt96_o.dat"
    files["O"]['pls'] = "plsx5_o.dat"
    files["O"]['prs'] = "prsx5_o.dat"
    files["F"] = {}    #9
    files["F"]['acd'] = "acd89_f.dat"
    files["F"]['scd'] = "scd89_f.dat"
    files["F"]['ccd'] = "ccd89_f.dat"
    files["F"]['prb'] = "prb89_f.dat"
    files["F"]['plt'] = "plt89_f.dat"
    files["F"]['fis'] = "sxrfil14.dat"
    files["F"]['brs'] = "brs05360.dat"
    files["F"]['pls'] = "pls_F_14.dat"
    files["F"]['prs'] = "prs_F_14.dat"
    files["F"]['prc'] = "prc89_f.dat"
    files["Ne"] = {}   #10
    files["Ne"]['acd'] = "acd96_ne.dat"
    files["Ne"]['scd'] = "scd96_ne.dat"
    files["Ne"]['prb'] = "prb96_ne.dat"
    files["Ne"]['plt'] = "plt96_ne.dat"
    files["Ne"]['ccd'] = "ccd89_ne.dat"
    files["Ne"]['prc'] = "prc89_ne.dat"
    files["Ne"]['pls'] = "plsx8_ne.dat"
    files["Ne"]['prs'] = "prsx8_ne.dat"
    files["Ne"]['fis'] = "sxrfilD1.dat"
    files["Ne"]['brs'] = "brs05360.dat"
    files["Al"] = {}    #13
    files["Al"]['acd'] = "acd00_al.dat"
    files["Al"]['scd'] = "scd00_al.dat"
    files["Al"]['prb'] = "prb00_al.dat"
    files["Al"]['plt'] = "plt00_al.dat"
    files["Al"]['ccd'] = "ccd89_al.dat"
    files["Al"]['prc'] = "prc89_al.dat"
    files["Al"]['pls'] = "pls_Al_14.dat"
    files["Al"]['prs'] = "prs_Al_14.dat"
    files["Al"]['fis'] = "sxrfil14.dat"
    files["Al"]['brs'] = "brs05360.dat"
    files["Si"] = {}     #14
    files["Si"]['acd'] = "acd00_si.dat"
    files["Si"]['scd'] = "scd00_si.dat"
    files["Si"]['prb'] = "prb00_si.dat"
    files["Si"]['plt'] = "plt97_si.dat"
    files["Si"]['pls'] = "pls_Si_14.dat"
    files["Si"]['prs'] = "prs_Si_14.dat"
    files["Si"]['fis'] = "sxrfil14.dat"
    files["Si"]['brs'] = "brs05360.dat"
    files["Si"]['ccd'] = "ccd89_si.dat"
    files["Ar"] = {}     #18
    files["Ar"]['acd'] = "acd00_ar.dat"
    files["Ar"]['scd'] = "scd00_ar.dat"
    files["Ar"]['prb'] = "prb00_ar.dat"
    files["Ar"]['plt'] = "plt00_ar.dat"
    files["Ar"]['ccd'] = "ccd89_ar.dat"
    files["Ar"]['prc'] = "prc89_ar.dat"
    files["Ar"]['pls'] = "pls_Ar_14.dat"
    files["Ar"]['prs'] = "prs_Ar_14.dat"
    files["Ar"]['fis'] = "sxrfil14.dat"
    files["Ar"]['brs'] = "brs05360.dat"
    files["Ca"] = {}     #20
    files["Ca"]['acd'] = "acd85_ca.dat"
    files["Ca"]['scd'] = "scd85_ca.dat"
    files["Ca"]['ccd'] = "ccd89_w.dat"  # file not available, use first 20 ion stages using Foster scaling
    files["Ca"]['prb'] = "prb85_ca.dat"
    files["Ca"]['plt'] = "plt85_ca.dat"
    files["Ca"]['pls'] = "pls_Ca_14.dat"
    files["Ca"]['prs'] = "prs_Ca_14.dat"
    files["Ca"]['fis'] = "sxrfil14.dat"
    files["Ca"]['brs'] = "brs05360.dat"
    files["Fe"] = {}     #26
    files["Fe"]['acd'] = "acd89_fe.dat"
    files["Fe"]['scd'] = "scd89_fe.dat"
    files["Fe"]['prb'] = "prb89_fe.dat"
    files["Fe"]['plt'] = "plt89_fe.dat"
    files["Fe"]['pls'] = "pls_Fe_14.dat"
    files["Fe"]['prs'] = "prs_Fe_14.dat"
    files["Fe"]['fis'] = "sxrfil14.dat"
    files["Fe"]['brs'] = "brs05360.dat"
    files["Fe"]['ccd'] = "ccd89_fe.dat"
    files["Ni"] = {}     #28
    files["Ni"]['acd'] = "acd00_ni.dat"
    files["Ni"]['scd'] = "scd00_ni.dat"
    files["Ni"]['prb'] = "prb00_ni.dat"
    files["Ni"]['plt'] = "plt00_ni.dat"
    files["Ni"]['pls'] = "pls_Ni_14.dat"
    files["Ni"]['prs'] = "prs_Ni_14.dat"
    files["Ni"]['fis'] = "sxrfil14.dat"
    files["Ni"]['brs'] = "brs05360.dat"
    files["Ni"]['ccd'] = "ccd89_ni.dat"
    files["Kr"] = {}     #36
    files["Kr"]['acd'] = "acd89_kr.dat"
    files["Kr"]['scd'] = "scd89_kr.dat"
    files["Kr"]['ccd'] = "ccd89_kr.dat"
    files["Kr"]['prb'] = "prb89_kr.dat"
    files["Kr"]['plt'] = "plt89_kr.dat"
    files["Kr"]['pls'] = "plsx5_kr.dat"
    files["Kr"]['prs'] = "prsx5_kr.dat"
    files["Mo"] = {}     #42
    files["Mo"]['acd'] = "acd89_mo.dat"
    files["Mo"]['scd'] = "scd89_mo.dat"
    files["Mo"]['ccd'] = "ccd89_mo.dat"
    files["Mo"]['plt'] = "plt89_mo.dat"
    files["Mo"]['prb'] = "prb89_mo.dat"
    files["Mo"]['prc'] = "prc89_mo.dat"
    files["Xe"] = {}     #56
    files["Xe"]['acd'] = "acd89_xe.dat"
    files["Xe"]['scd'] = "scd89_xe.dat"
    files["Xe"]['ccd'] = "ccd89_xe.dat"
    files["Xe"]['plt'] = "plt89_xe.dat"
    files["Xe"]['prb'] = "prb89_xe.dat"
    files["Xe"]['prs'] = "prsx1_xe.dat"
    files["Xe"]['pls'] = "prsx1_xe.dat"
    files["Xe"]['prc'] = "prc89_xe.dat"
    files["W"] = {}     #74
    files["W"]['acd'] = "acd89_w.dat"
    files["W"]['scd'] = "scd89_w.dat"
    files["W"]['prb'] = "prb89_w.dat"
    files["W"]['plt'] = "plt89_w.dat"
    files["W"]['fis'] = "sxrfil14.dat"
    files["W"]['brs'] = "brs05360.dat"
    files["W"]['pls'] = "pls_W_14.dat"
    files["W"]['prs'] = "prs_W_14.dat"
    files["W"]['ccd'] = "ccd89_w.dat"

    return files
