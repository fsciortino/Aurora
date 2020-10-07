#-*-Python-*-
# Created by sciortinof at 05 Feb 2020  14:59
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from matplotlib import cm
import os
import scipy.ndimage
from scipy.linalg import svd

def get_file_types():
    ''' Returns main types of ADAS atomic data of interest '''
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
    ''' Function to identify location of ADAS atomic data in a generalized fashion
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
    ''' Read ADAS file in ADF11 format over the given ne, T. '''

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


def get_all_atom_data(imp, files = None):
    ''' Collect atomic data for a given impurity from all types of ADAS files available.

    imp: str
        Atomic symbol of impurity ion.
    files : list or array-like
        ADAS file names to be fetched. 
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

    INPUTS:
    ------
    path : str
        Path to adf15 file to read.
    order : int, opt
        Parameter to control the order of interpolation.
    recomb : bool, opt
        If True, fetch recombination contributions to available lines. If False,
        fetch only ionization contributions.

    To plot PEC data:
    ------
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

    OUTPUTS:
    pec_dict : dict
        Dictionary containing interpolation functions for each of the available lines of the
        indicated type (ionization or recombination). Each interpolation function takes as arguments
        the log-10 of ne and Te.

    MWE:
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

        # only for testing:
        interpolate_log=True

        if lam is not None:
            if lam not in pec_dict:
                pec_dict[lam] = []

            pec_dict[lam].append(
                scipy.interpolate.RectBivariateSpline(
                    np.log10(dens) if interpolate_log else dens,
                    np.log10(temp) if interpolate_log else temp,
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
                if interpolate_log:
                    PEC_eval = pec_dict[lam][-1].ev(np.log10(NE), np.log10(TE))
                else:
                    PEC_eval = pec_dict[lam][-1].ev(NE,TE)


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

                    # log scales don't work well in 3D...
                    #ax2.set_zscale('log')
                    #ax2.locator_params(axis='z', nbins=5)

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
                    #ax1.legend(loc='best').draggable(True)
                    ax1.set_title(cs + r' --- $\lambda$ = '+str(lam) +' A')


    return pec_dict


# -------------------------------------------------------------------------------
#
#   Routines to examine ionization equilibrium
#
# -------------------------------------------------------------------------------

def null_space(A):
    ''' Find null space of matrix A '''
    u, s, vh = svd(A, full_matrices=True)
    Q = vh[-1,:].T.conj()    # -1 index is after infinite time/equilibration
    # return the smallest singular eigenvalue
    return Q, s[-2]  # matrix is singular, so last index is ~0


def get_frac_abundances(atom_data, ne,Te=None, n0_by_ne=1e-5, include_cx=False,
                          plot=True, ax = None, rho = None, rho_lbl=None, compute_rates=False):
    '''
    Calculate fractional abundances from ionization and recombination equilibrium.
    If include_cx=True, radiative recombination and thermal charge exchange are summed.

    INPUTS:
    -------
    atom_data : dictionary of atomic ADAS files (only acd, scd are required; ccd is 
        necessary only if include_cx=True
    ne : float or array
        Electron density in units of m^-3
    Te : float or array
        Electron temperature in units of eV. If left to None, the Te grid
        given in the atomic data is used.
    n0_by_ne: float or array
        Ratio of background neutral hydrogen to electron density, used if include_cx=True. 
    include_cx : bool
        If True, charge exchange with background thermal neutrals is included. 
    plot : bool
        Show fractional abundances as a function of ne,Te profiles parameterization.
    ax : matplotlib.pyplot Axes instance
        Axes on which to plot if plot=True. If False, it creates new axes
    rho : list or array
        Vector of radial coordinates on which ne,Te (and possibly n0_by_ne) are given. 
        This is only used for plotting, if given. 
    rho_lbl: str
        Label to be used for rho. If left to None, defaults to a general "rho".
    compute_rates : bool
        If True, compute rate coefficients for ionization/recombination equilibrium on top
        of fractional abundances (which should be the same regardless of the method used). 

    OUTPUTS:
    -------
    logTe_ : array
        log10 of electron temperatures as a function of which the fractional abundances and
        rate coefficients are given.
    fz : array, (space,nZ)
        Fractional abundances across the same grid used by the input ne,Te values. 
    rate_coeff : array, (space, nZ)
        Rate coefficients in units of [s^-1]. 

    '''

    logTe_, logS,logR,logcx = get_cs_balance_terms(atom_data, ne,Te, maxTe=10e3, include_cx=include_cx)
    if include_cx:
        # Get an effective recombination rate by summing radiative & CX recombination rates
        logR = np.logaddexp(logR,np.log(n0_by_ne) +logcx)

    # analytical formula for fractional abundance
    rate_ratio = np.hstack((np.zeros_like(logTe_)[:,None], logS-logR))
    rate_ratio[rate_ratio<-10.]=-10.  # to avoid underflow in exponential
    fz = np.exp(np.cumsum(rate_ratio,axis=1))
    fz /= fz.sum(1)[:,None]

    out = [logTe_,fz]
    
    if compute_rates:
        # numerical method which calculates also rate_coeff
        nion = logR.shape[1]
        fz  = np.zeros((logTe_.size,nion+1))
        rate_coeff = np.zeros(logTe_.size)
        for it,t in enumerate(logTe_):
            
            A = -np.diag(np.r_[np.exp(logS[it]),0])+np.diag(np.exp(logS[it]),-1)+\
                np.diag(np.exp(logR[it]),1)- np.diag(np.r_[0,np.exp(logR[it])])
            
            N,rate_coeff[it] = null_space(A)
            fz[it] = N/np.sum(N)

        rate_coeff*=ne
        out.append(rate_coeff)
        
    if plot:
        # plot fractional abundances
        if ax is None:
            fig,ax = plt.subplots()

        if rho is None:
            x = 10**logTe_
            ax.set_xlabel('T$_e$ [eV]')
            ax.set_xscale('log')
        else:
            if rho_lbl is None: rho_lbl=r'$\rho$'
            x = rho
            ax.set_xlabel(rho_lbl)

        ax.set_prop_cycle('color',cm.plasma(np.linspace(0,1,fz.shape[1])))
        ax.plot(x,fz)
        for i in range(len(fz.T)):
            imax = np.argmax(fz[:,i])
            ax.text(np.max([0.05,x[imax]]), fz[imax,i], i, horizontalalignment='center', clip_on=True)
        ax.grid('on')
        ax.set_ylim(0,1.05)
        ax.set_xlim(x[0],x[-1])
        ax.set_title(r'Fractional abundances')
        if ax is None:
            ax.legend()

    return out





def get_cs_balance_terms(atom_data, ne=5e19,Te = None, maxTe=10e3, include_cx=True):
    ''' Get S, R and cx on the same logTe grid. 
    
    INPUTS
    ------
    atom_data : dictionary of atomic ADAS files (only acd, scd are required; ccd is 
        necessary only if include_cx=True
    ne : float or array
        Electron density in units of m^-3
    Te : float or array
        Electron temperature in units of eV. If left to None, the Te grid
        given in the atomic data is used.
    maxTe : float
        Maximum temperature of interest; only used if Te is left to None. 
    include_cx : bool
        If True, obtain charge exchange terms as well. 
    
    OUTPUTS
    -------
    logTe : log10 Te grid on which atomic rates are given
    logS, logR (,logcx): atomic rates for effective ionization, radiative+dielectronic
        recombination (+ charge exchange, if requested). After exponentiation, all terms
        will be in units of s^-1. 
    '''

    if Te is None:
        #find smallest Te grid from all files
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
        logTe = np.log10(Te)


    logne = np.log10(ne)-6

    logS = interp_atom_prof(atom_data['scd'],logne, logTe,log_val=True, x_multiply=False)
    logR = interp_atom_prof(atom_data['acd'],logne, logTe,log_val=True, x_multiply=False)
    if include_cx:
        logcx = interp_atom_prof(atom_data['scd'],logne, logTe,log_val=True, x_multiply=False)
    else:
        logcx = None

    return logTe, logS, logR, logcx



def plot_relax_time(logTe, rate_coeff, ne_mean, ax = None):
    ''' Plot relaxation time of the ionization equilibrium corresponding
    to the inverse of the given rate coefficients '''

    if ax is None:
        ax = plt.subplot(111)

    ax.loglog(10**logTe,1e3/rate_coeff,'b' )
    ax.set_xlim(10**logTe[0],10**logTe[-1])
    ax.grid('on')
    ax.set_xlabel('T$_e$ [eV]')
    ax.set_ylabel(r'$\tau_\mathrm{relax}$ [ms]')
    ax.set_title(r'n$_e$ = %.2g m$^{-3}$'%ne_mean)




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
    assume that x_prof, y_prof, x,y, table are all decadic logarithms
    and x_prof, y_prof are equally spaced(always for ADAS data)
    log_val bool: return natural logarithm of the data
    x_multiply bool: multiply output by 10**x_prof , it will not not multiplied if x_prof is None

    return data interpolated on shape(nt,nion,nr)
    '''
    x,y, table = atom_table

    if (abs(table-table[...,[0]]).all()  < .05) or x_prof is None:
        #1D interpolaion if independent of the last dimension - like SXR radiation data

        reg_interp = CartesianGrid((y, ),table[:,:,0]*np.log(10))
        interp_vals = reg_interp(y_prof) 

        #multipling of logarithms is just adding
        if x_multiply and x_prof is not None:
            interp_vals += x_prof*np.log(10)

    else:#2D interpolation
        if x_multiply: #multipling of logarithms is just adding
            table += x
        #breadcast both variables in the sae shape
        x_prof, y_prof = np.broadcast_arrays(x_prof, y_prof)
        #perform fast linear interpolation
        reg_interp = CartesianGrid((x, y),table.swapaxes(1,2)*np.log(10))
        interp_vals = reg_interp(x_prof,y_prof) 

    #reshape to shape(nt,nion,nr)
    interp_vals = interp_vals.swapaxes(0,1)

    if not log_val:
        #return actual value, not logarithm
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



def impurity_brems( nz, ne, Te):
    '''mW/nm/sr/m^3 *cm^3
    This is only approximate and not very useful, since the bremsstrahlung contribution
    is already included in the continuum in x2.
    This estimate does not have the correct ne-dependence of the Gaunt factor...'''

    # neutral stage doesn't produce brems
    Z_imp = nz.shape[1]-1
    Z = np.arange(Z_imp)[None,:,None]+1

    gff = gff_mean(Z,Te[:,None])
    imp_brems = Z**2 * nz[:,1:] * gff * (1.69e-32 *np.sqrt(Te) * ne) [:,None]

    return imp_brems


def main_ion_brems(Zi, ni, ne, Te):
    ''' mW/nm/sr/m^3 *cm^3
    Better to calculate this from H/D/T plt files, which will have more accurate Gaunt factors
    with density dependence.'''
    return 1.69e-32 * ni * Zi**2. * ne * np.sqrt(Te) * gff_mean(Zi,Te)




def get_adas_ion_rad(ion_name, n_ion, logne_prof, logTe_prof, sxr=False):
    ''' Get ADAS estimate for total radiation in [M/m^3] for the given ion
    with the given (log) density and temperature profiles. 

    If sxr=True, 'prs' files are used instead of 'prb' ones, thus giving SXR-filtered
    radiation for the SXR filter indicated by the atomic data dictionary.
    '''        
    # get all atomic data for chosen ion
    atom_data = get_all_atom_data(ion_name,['prs' if sxr else 'prb'])

    # load filtered data in the SXR range
    x,y,tab = atom_data['prs' if sxr else 'prb']
    
    #recombination and bremstrahlung of fully striped ion
    prs = interp_atom_prof((x,y,tab[[-1]]),logne_prof,logTe_prof,x_multiply=False)

    # Now compute expected radiation expected from this background ion:
    bckg = prs[:,0] * n_ion * 1e6 # bulk fully-stripped ion radiation [W/m^3]

    return bckg


def get_cooling_factors(atom_data, logTe_, fz, ion_resolved = False, plot=True,ax=None):
    '''
    Calculate cooling coefficients from fz and prs,  pls (if these are available).
    Also plot the inverse of rate coefficients, which gives an estimate for relaxation times.
    '''
    try:
        atom_data['prs']
        atom_data['pls']
        atom_data['plt']
        atom_data['prb']
    except:
        raise ValueError('prs, plt and/or prb files not available!')


    prs = interp_atom_prof(atom_data['prs'],None,logTe_)#continuum radiation in SXR range
    pls = interp_atom_prof(atom_data['pls'],None,logTe_)#line radiation in SXR range
    pltt= interp_atom_prof(atom_data['plt'],None,logTe_) #line radiation
    prb = interp_atom_prof(atom_data['prb'],None,logTe_)#continuum radiation

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

        ax.loglog(10**logTe_, line_rad_sxr,'b',label='SXR line radiation ')   # can be many orders of magnitude different from brms_rad_sxr (e.g. lower in F)
        ax.loglog(10**logTe_, brems_rad_sxr,'r',label='SXR bremsstrahlung and recombination')

        # total SXR rad = brems + line rad in SXR range:
        ax.loglog(10**logTe_, brems_rad_sxr+line_rad_sxr,'k',label='total SXR radiation',lw=2)

        # total radiation (includes hard X-ray, visible, UV, etc.)
        ax.loglog(10**logTe_, line_rad_tot,'g--',label='total line radiation')
        ax.loglog(10**logTe_, brems_rad_tot,'y--',label='total continuum radiation')

        ax.legend(loc='best')
        #### set xlims to visualize scales better
        ax.set_xlim(50,10**logTe_[-1])
        ax.set_ylim( line_rad_sxr[np.argmin(np.abs(10**logTe_ - 50))], np.nanmax( line_rad_tot)*10)

        ax.grid('on')
        ax.set_xlabel('T$_e$ [eV]')
        ax.set_ylabel('L$_z$ [Wm$^3$]')
        ax.set_title('Cooling factors')

    elif ion_resolved:
        return pls, prs, pltt,prb


    else:
        return line_rad_sxr, brems_rad_sxr, line_rad_tot, brems_rad_tot




def plot_radiation_profs(atom_data, nz_prof, logne_prof, logTe_prof, xvar, imp='F', plot=False):
    ''' Obtain profiles of predicted radiation.

    This function can be used to plot radial profiles (setting xvar to a radial grid)
    or profiles as a function of any variable on which the logne_prof and logTe_prof
    may depend.

    The variable `nz_prof' may be a full description of impurity charge state densities
    (e.g. the output of Aurora), or profiles of fractional abundances from ionization
    equilibrium.

    Note that the variables `xvar' (array) and imp (str) are only needed if plotting is requested.
    '''


    pls, prs, pltt, prb = get_cooling_factors(atom_data, logTe_prof, nz_prof, ion_resolved = True, plot=False)


    emiss_sxr = np.zeros((len(xvar),nion))
    emiss_tot = np.zeros((len(xvar),nion))
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
    ax[0].plot(xvar,prs); ax[0].set_title('PRS: cont SXR rad')
    ax[1].plot(xvar,pls); ax[1].set_title('PLS: SXR line rad')
    ax[2].plot(xvar,pltt); ax[2].set_title('PLT: tot line rad')
    ax[3].plot([],[])#empty plot for bremstrahlung of neutral ion
    ax[3].plot(xvar,prb); ax[3].set_title('PRB: tot cont rad')

    ax[2].set_xlabel('$r/a$'); ax[3].set_xlabel('$r/a$')
    labels = [r'$%s^{%d\!+}$'%(imp,cc) for cc in range(nion)]
    ax[0].legend(labels)
    ax[0].set_xlim(xvar[0], xvar[-1])

    # plot total power (in SXR and whole range)
    fig,axx = plt.subplots(2,2,figsize=(12,8),sharex=True)
    ax = axx.flatten()

    for a in ax:
        a.set_prop_cycle('color',colors)
        a.grid(True)


    ax[0].plot(xvar,emiss_sxr); ax[0].set_title('SXR power [W]')
    ax[1].plot(xvar,emiss_tot); ax[1].set_title('Tot. rad. power [W]')
    ax[2].plot(xvar,emiss_sxr*10**logne_prof[:,None]); ax[2].set_title(r'SXR power [W/m$^{-3}$]')
    ax[3].plot(xvar,emiss_tot*10**logne_prof[:,None]); ax[3].set_title(r'Tot. rad. power [W/m$^{-3}$]')
    ax[2].set_xlabel('$r/a$'); ax[3].set_xlabel('$r/a$')
    ax[0].legend(labels)
    ax[0].set_xlim(xvar[0], xvar[-1])





def balance(logTe_val, cs, n0_by_ne, logTe_, S,R,cx):
    ''' Evaluate balance of effective ionization, recombination and charge exchange at a given temperature. '''

    a = R +n0_by_ne * cx
    SS_0 = 10**(interp1d(logTe_, np.log10(S[cs-1,:]), kind='cubic', bounds_error=False)(logTe_val))
    aa_0 = 10**(interp1d(logTe_, np.log10(a[cs-1,:]), kind='cubic', bounds_error=False)(logTe_val))
    SS_1 = 10**(interp1d(logTe_, np.log10(S[cs,:]), kind='cubic', bounds_error=False)(logTe_val))
    aa_1 = 10**(interp1d(logTe_, np.log10(a[cs,:]), kind='cubic', bounds_error=False)(logTe_val))

    val = SS_0 - SS_1 - aa_0 + aa_1
    return val*1e20 # get this to large-ish units to avoid tolerance issues due to small powers of 10





def adas_files_dict():
    ''' Selections for ADAS files for Aurora runs and radiation calculations.
    This function can be called to fetch a set of default files, which can then be modified (e.g. to 
    use a new file for a specific SXR filter) before running a calculation. 
    '''
            
    files={}
    files["H"] = {}
    files["H"]['acd'] = "acd96_h.dat"
    files["H"]['scd'] = "scd96_h.dat"
    files["H"]['prb'] = "prb96_h.dat"
    files["H"]['plt'] = "plt96_h.dat"
    files["H"]['ccd'] = "ccd96_h.dat"
    files["H"]['prc'] = "prc96_h.dat"  # available?
    files["H"]['pls'] = "pls_H_14.dat"
    files["H"]['prs'] = "prs_H_14.dat"
    files["H"]['fis'] = "sxrfil14.dat"
    files["H"]['brs'] = "brs05360.dat"
    files["He"] = {}
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
    files["C"] = {}
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
    files["N"] = {}
    files["N"]['acd'] = "acd96_n.dat"
    files["N"]['scd'] = "scd96_n.dat"
    files["N"]['prb'] = "prb96_n.dat"
    files["N"]['plt'] = "plt96_n.dat"
    files["N"]['pls'] = "plsx8_n.dat"
    files["N"]['prs'] = "prsx8_n.dat"
    files["N"]['fis'] = "sxrfilD1.dat"
    files["N"]['brs'] = "brs05360.dat"
    files["N"]['ccd'] = "ccd96_n.dat"
    files["F"] = {}
    files["F"]['acd'] = "acd00_f.dat"
    files["F"]['scd'] = "scd00_f.dat"
    files["F"]['prb'] = "prb00_f.dat"
    files["F"]['plt'] = "plt00_f.dat"
    files["F"]['fis'] = "sxrfil14.dat"
    files["F"]['brs'] = "brs05360.dat"
    files["F"]['pls'] = "pls_F_14.dat"
    files["F"]['prs'] = "prs_F_14.dat"
    files["F"]['ccd'] = "ccd89_f.dat"
    files["F"]['prc'] = "prc89_f.dat"
    files["Ar"] = {}
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
    files["Fe"] = {}
    files["Fe"]['acd'] = "acd00_fe.dat"
    files["Fe"]['scd'] = "scd00_fe.dat"
    files["Fe"]['prb'] = "prb00_fe.dat"
    files["Fe"]['plt'] = "plt00_fe.dat"
    files["Fe"]['pls'] = "pls_Fe_14.dat"
    files["Fe"]['prs'] = "prs_Fe_14.dat"
    files["Fe"]['fis'] = "sxrfil14.dat"
    files["Fe"]['brs'] = "brs05360.dat"
    files["Fe"]['ccd'] = "ccd89_f.dat"
    files["W"] = {}
    files["W"]['acd'] = "acd89_w.dat"
    files["W"]['scd'] = "scd89_w.dat"
    files["W"]['prb'] = "prb89_w.dat"
    files["W"]['plt'] = "plt89_w.dat"
    files["W"]['fis'] = "sxrfil14.dat"
    files["W"]['brs'] = "brs05360.dat"
    files["W"]['pls'] = "pls_W_14.dat"
    files["W"]['prs'] = "prs_W_14.dat"
    files["W"]['ccd'] = "ccd89_w.dat"
    files["Ne"] = {}
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
    files["Si"] = {}
    files["Si"]['acd'] = "acd00_si.dat"
    files["Si"]['scd'] = "scd00_si.dat"
    files["Si"]['prb'] = "prb00_si.dat"
    files["Si"]['plt'] = "plt97_si.dat"
    files["Si"]['pls'] = "pls_Si_14.dat"
    files["Si"]['prs'] = "prs_Si_14.dat"
    files["Si"]['fis'] = "sxrfil14.dat"
    files["Si"]['brs'] = "brs05360.dat"
    files["Si"]['ccd'] = "ccd89_si.dat"
    files["Al"] = {}
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
    files["Ca"] = {}
    files["Ca"]['acd'] = "acd85_ca.dat"
    files["Ca"]['scd'] = "scd85_ca.dat"
    files["Ca"]['ccd'] = "ccd89_w.dat"  # file not available, use first 20 ion stages using Foster scaling
    files["Ca"]['prb'] = "prb85_ca.dat"
    files["Ca"]['plt'] = "plt85_ca.dat"
    files["Ca"]['pls'] = "pls_Ca_14.dat"
    files["Ca"]['prs'] = "prs_Ca_14.dat"
    files["Ca"]['fis'] = "sxrfil14.dat"
    files["Ca"]['brs'] = "brs05360.dat"
    files["Ni"] = {}
    files["Ni"]['acd'] = "acd00_ni.dat"
    files["Ni"]['scd'] = "scd00_ni.dat"
    files["Ni"]['prb'] = "prb00_ni.dat"
    files["Ni"]['plt'] = "plt00_ni.dat"
    files["Ni"]['pls'] = "pls_Ni_14.dat"
    files["Ni"]['prs'] = "prs_Ni_14.dat"
    files["Ni"]['fis'] = "sxrfil14.dat"
    files["Ni"]['brs'] = "brs05360.dat"
    files["Ni"]['ccd'] = "ccd89_ni.dat"
    files["Mo"] = {}
    files["Mo"]['acd'] = "acd89_mo.dat"
    files["Mo"]['scd'] = "scd89_mo.dat"
    files["Mo"]['ccd'] = "ccd89_mo.dat"
    files["Mo"]['plt'] = "plt89_mo.dat"
    files["Mo"]['prb'] = "prb89_mo.dat"
    files["Mo"]['prc'] = "prc89_mo.dat"
    
    return files
