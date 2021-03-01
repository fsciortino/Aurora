'''Collection of classes and functions for loading, interpolation and processing of atomic data. 
Refer also to the adas_files.py script. 
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, interp1d
from matplotlib import cm
import os, sys, copy
import scipy.ndimage
from scipy.linalg import svd
from scipy import constants

from . import adas_files


def get_adas_file_types():
    '''Obtain a description of each ADAS file type and its meaning in the context of Aurora.

    Returns
    ------------
    dict
        Dictionary with keys given by the ADAS file types and values giving a description for them.

    Notes
    ---------
    For background on ADAS generalized collisional-radiative modeling and data formats, refer to 
    [1]_.

    References
    -----------------
    
    .. [1] Summers et al., "Ionization state, excited populations and emission of impurities 
       in dynamic finite density plasmas: I. The generalized collisional-radiative model for 
       light elements", Plasma Physics and Controlled Fusion, 48:2, 2006

    '''
    
    return {'acd':'effective recombination',
            'scd':'effective ionization',
            'prb':'continuum radiation',
            'plt':'line radiation',
            'ccd':'thermal charge exchange',
            'prc':'thermal charge exchange continuum radiation',
            'pls':'line radiation in the SXR range',
            'prs':'continuum radiation in the SXR range',
            'brs':'continuum spectral bremstrahlung',
            'fis':'sensitivity in the SXR range',
            'pbs':'impurity bremsstrahlung in SXR range, also included in prs files'
    }


class adas_file:
    '''Read ADAS file in ADF11 format over the given density and temperature grids. 
    Note that such grids vary between files, and the species they refer to may too.

    Refer to ADAS documentation for details on each file.

    Parameters
    ----------
    filepath : str
        Path to location where ADAS file is located.
    '''
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
        '''Plot data from input ADAS file. If provided, the arguments allow users to overplot
        and compare data from multiple files. 

        Parameters
        ----------
        fig : matplotlib Figure object
            If provided, add specification as to which ADAS file is being plotted.
        axes : matplotlib Axes object (or equivalent)
            If provided, plot on these axes. Note that this typically needs to be a set of axes
            for each plotted charge state. Users may want to call this function once first to get
            some axes, and then pass those same axes to a second call for another file to compare with.
        '''
        if fig is None or axes is None:
            fig,axes = plt.subplots(int(self.ncol),int(self.nrow), sharex=True, sharey=True)

        axes = np.atleast_2d(axes)
        colormap = cm.rainbow
        if fig is not None:
            fig.suptitle(self.filename+'  '+ get_adas_file_types()[self.file_type])

        for i,ax in enumerate(axes.flatten()):
            if i >= self.n_ion: break
            if all(self.data[i,:,:].std(axis=1)==0): #independent of density
                ax.plot(self.logT, self.data[i,:,0])
            else:
                ax.set_prop_cycle('color', colormap( np.linspace(0,1,self.data.shape[2])))
                ax.plot(self.logT, self.data[i])
            ax.grid(True)
            if self.file_type != 'brs':
                charge = i+1 if self.file_type in ['scd','prs','ccd','prb'] else i+2
                ax.set_title(self.imp+'$^{%d\!+}$'% (charge-1))   # check?

        for ax in axes[-1]:
            ax.set_xlabel('$\log T_e\ \mathrm{[eV]}$')
        for ax in axes[:,0]:
            if self.file_type in ['scd','acd','ccd']:
                ax.set_ylabel('$\log('+self.file_type+')\ \mathrm{[cm^3/s]}$')
            elif self.file_type in ['prb','plt','prc','pls','brs','prs']:
                ax.set_ylabel('$\log('+self.file_type+')\ \mathrm{[W\cdot cm^3]}$')


def read_filter_response(filepath, adas_format=True, plot=False):
    '''Read a filter response function over energy. 

    This function attempts to read the data checking for the following formats (in this order):
    
    #. The ADAS format. Typically, this data is from obtained from http://xray.uu.se and produced
       via ADAS routines. 

    #. The format returned by the `Center for X-Ray Optics website  <https://henke.lbl.gov/optical_constants/filter2.html>`__ .

    Note that filter response functions are typically a combination of a filter transmissivity 
    and a detector absorption. 

    Parameters
    ----------
    filepath : str
        Path to filter file of interest.
    plot : bool
        If True, the filter response function is plotted. 
    
    '''
    E_eV=[]
    response=[]
    try:
        # Attempt to read ADAS format
        with open(filepath) as f:

            header = f.readline()
            num = int(header.split()[0])

            # *****
            f.readline()

            while len(E_eV)< num:
                line = f.readline()
                E_eV += [float(n) for n in line.split()]
            while len(response)< num:
                line = f.readline()
                response += [float(t) for t in line.split()]

        # energy and response function are written in natural logs
        E_eV = np.concatenate(([0.,], np.array(np.exp(E_eV))))
        response = np.concatenate(([0.,], np.array(np.exp(response))))
    except ValueError:
        try:
            # Attempt to read CXRO format
            with open(filepath) as f:
                contents = f.readlines()

            for line in contents[2:]:
                tmp = line.strip().split()
                E_eV.append(float(tmp[0]))
                response.append(float(tmp[1]))
            E_eV=np.concatenate(([0.,], np.array(E_eV)))
            response=np.concatenate(([0.,],np.array(response)))
        except ValueError:
            raise ValueError('Unrecognized filter function format...')
    
    if plot:
        fig,ax = plt.subplots()
        ax.semilogx(E_eV, response)
        ax.set_xlabel('Photon energy [eV]')
        ax.set_ylabel('Detector response efficiency')
        plt.tight_layout()

    return E_eV, response

    
def get_atom_data(imp, filetypes=['acd','scd'], filenames=[]):
    ''' Collect atomic data for a given impurity from all types of ADAS files available or
    for only those requested. 

    Parameters
    ----------
    imp : str
        Atomic symbol of impurity ion.
    filetypes : list or array-like 
        ADAS file types to be fetched. Default is ["acd","scd"] for effective ionization 
        and recombination rates (excluding CX).
    filenames : list or array-like, optional
        ADAS file names to be used in place of the defaults given by 
        :py:meth:`~aurora.adas_files.adas_file_dict`.
        If left empty, such defaults are used. Note that the order of filenames must be 
        the same as the one in the "filetypes" list.
    
    Returns
    -------
    atom_data : dict
        Dictionary containing data for each of the requested files. 
        Each entry of the dictionary gives log-10 of ne, log-10 of Te and log-10 of the data
        as attributes atom_data[key].logNe, atom_data[key].logT, atom_data[key].data
    '''
    atom_data = {}
    
    if filenames:
        files = {}
        for ii,typ in enumerate(filetypes):
            files[typ] = filenames[ii]
    else:
        # get dictionary containing default list of ADAS atomic files
        files = adas_files.adas_files_dict()[imp]

    for filetype in filetypes:
        filename = files[filetype]

        # find location of required ADF11 file
        fileloc = adas_files.get_adas_file_loc(filename,filetype='adf11')

        # load specific file and add it to output dictionary
        res = adas_file(fileloc)
        atom_data[filetype] = res.logNe, res.logT, res.data

    return atom_data


def null_space(A):
    '''Find null space of matrix `A`.
    '''
    u, s, vh = svd(A, full_matrices=True)
    Q = vh[-1,:].T.conj()    # -1 index is after infinite time/equilibration
    # return the smallest singular eigenvalue
    return Q, s[-2]  # matrix is singular, so last index is ~0


def get_frac_abundances(atom_data, ne_cm3, Te_eV=None, n0_by_ne=1e-5,
                        include_cx=False, ne_tau=np.inf,
                        plot=True, ax = None, rho = None, rho_lbl=None,ls='-'):
    r'''Calculate fractional abundances from ionization and recombination equilibrium.
    If include_cx=True, radiative recombination and thermal charge exchange are summed.

    This method can work with ne,Te and n0_by_ne arrays of arbitrary dimension, but plotting 
    is only supported in 1D (defaults to flattened arrays).

    Parameters
    ----------
    atom_data : dictionary of atomic ADAS files (only acd, scd are required; ccd is 
        necessary only if include_cx=True)
    ne_cm3 : float or array
        Electron density in units of :math:`cm^{-3}`
    Te_eV : float or array, optional
        Electron temperature in units of eV. If left to None, the Te grid given in the 
        atomic data is used.
    n0_by_ne: float or array, optional
        Ratio of background neutral hydrogen to electron density, used if include_cx=True. 
    include_cx : bool
        If True, charge exchange with background thermal neutrals is included. 
    ne_tau : float, opt
        Value of electron density in :math:`m^{-3}\cdot s` :math:`\times` particle residence time. 
        This is a scalar value that can be used to model the effect of transport on ionization equilibrium. 
        Setting ne_tau=np.inf (default) corresponds to no effect from transport. 
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

    Returns
    -------
    logTe : array
        log10 of electron temperatures as a function of which the fractional abundances and
        rate coefficients are given.
    fz : array, (space,nZ)
        Fractional abundances across the same grid used by the input ne,Te values. 
    rate_coeffs : array, (space, nZ)
        Rate coefficients in units of [:math:`s^{-1}`]. 
    '''
    # if input arrays are multi-dimensional, flatten them here and restructure at the end
    if not isinstance(ne_cm3,float):
        _ne = ne_cm3.flatten()
    else:
        _ne = copy.deepcopy(ne_cm3)
    if not isinstance(Te_eV,float): 
        _Te = Te_eV.flatten()
    else:
        _Te = copy.deepcopy(Te_eV)
    if not isinstance(n0_by_ne,float):
        _n0_by_ne = n0_by_ne.flatten()
    else:
        _n0_by_ne = copy.deepcopy(n0_by_ne)

    logTe, logS,logR,logcx = get_cs_balance_terms(
        atom_data, _ne, _Te, maxTe=10e3, include_cx=include_cx)
    
    if include_cx:
        # Get an effective recombination rate by summing radiative & CX recombination rates
        logR= np.logaddexp(logR,np.log(_n0_by_ne)[:,None] +logcx)

    # numerical method that calculates also rate_coeffs
    nion = logR.shape[1]
    fz  = np.zeros((logTe.size,nion+1))
    rate_coeffs = np.zeros(logTe.size)

    for it,t in enumerate(logTe):

        A = (
              - np.diag(np.r_[np.exp(logS[it]), 0] + np.r_[0, np.exp(logR[it])] + 1e6 / ne_tau)
              + np.diag(np.exp(logS[it]), -1)
              + np.diag(np.exp(logR[it]), 1)
              )

        N,rate_coeffs[it] = null_space(A)
        fz[it] = N/np.sum(N)

    rate_coeffs*=(_ne * 1e-6)

    if plot:
        # plot fractional abundances (only 1D)
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

        # cubic interpolation for smoother visualization:
        x_fine = np.linspace(np.min(x), np.max(x),10000)
        for cs in range(fz.shape[1]):
            fz_i = interp1d(x, fz[:,cs], kind='cubic')(x_fine)
            axx.plot(x_fine, fz_i, ls=ls)
            imax = np.argmax(fz_i)
            axx.text(np.max([0.05,x_fine[imax]]), fz_i[imax], cs,
                     horizontalalignment='center', clip_on=True)
        axx.grid('on')
        axx.set_ylim(0,1.05)
        axx.set_xlim(x[0],x[-1])

    # re-structure to original array dimensions
    logTe = logTe.reshape(ne_cm3.shape)
    fz = fz.reshape(*ne_cm3.shape, fz.shape[1])
    rate_coeffs = rate_coeffs.reshape(*ne_cm3.shape)
    
    return logTe, fz, rate_coeffs





def get_cs_balance_terms(atom_data, ne_cm3=5e13, Te_eV=None, maxTe=10e3, include_cx=True):
    '''Get S, R and cx on the same logTe grid. 
    
    Parameters
    ----------
    atom_data : dictionary of atomic ADAS files (only acd, scd are required; ccd is 
        necessary only if include_cx=True)
    ne_cm3 : float or array
        Electron density in units of :math:`cm^{-3}`
    Te_eV : float or array
        Electron temperature in units of eV. If left to None, the Te grid
        given in the atomic data is used.
    maxTe : float
        Maximum temperature of interest; only used if Te is left to None. 
    include_cx : bool
        If True, obtain charge exchange terms as well. 
    
    Returns
    -------
    logTe : array (n_Te)
        log10 Te grid on which atomic rates are given
    logS, logR (,logcx): arrays (n_ne,n_Te)
        atomic rates for effective ionization, radiative+dielectronic
        recombination (+ charge exchange, if requested). After exponentiation, all terms
        will be in units of :math:`s^{-1}`. 
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

        # select appropriate number of charge states -- this allows use of CCD files from higher-Z ions because of simple CX scaling
        logcx = logcx[:,:logS.shape[1]]
    else:
        logcx = None

    return logTe, logS, logR, logcx



def plot_relax_time(logTe, rate_coeff, ax = None):
    ''' Plot relaxation time of the ionization equilibrium corresponding
    to the inverse of the given rate coefficients.

    Parameters
    ----------
    logTe : array (nr,)
        log-10 of Te [eV], on an arbitrary grid (same as other arguments, but not
        necessarily radial)
    rate_coeff : array (nr,)
        Rate coefficients from ionization balance, as returned by the :py:fun:`~aurora.atomic.get_frac_abundances` function.
        N.B.: these rate coefficients will depend also on electron density, which does affect relaxation times. 
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



class CartesianGrid:
    """Fast linear interpolation for 1D and 2D vecctor data on equally spaced grids.
    This offers optimal speed in Python for interpolation of atomic data tables such
    as the ADAS ones.

    Parameters
    ----------
    grids: list of arrays, N=len(grids), N=1 or N=2
        List of 1D arrays with equally spaced grid values for each dimension
    values: N+1 dimensional array of values used for interpolation
        Values to interpolate. The first dimension typically refers to different ion stages, for which 
        data is provided on the input grids. Other dimensions refer to values on the density and temperature grids.
    """
    def __init__(self, grids, values):

        values = np.ascontiguousarray(np.moveaxis(values, 0, -1))
        self.N = values.shape[:-1]

        if len(self.N) > 2:
            raise OMFITexception('Only 1 and 2 dimensional interpolation is supported')

        for g, s in zip(grids, self.N):
            if len(g) != s:
                raise OMFITexception('wrong size of values array')

        self.offsets = [g[0] for g in grids]
        self.scales = [(g[-1] - g[0]) / (n - 1) for g, n in zip(grids, self.N)]

        A = []
        if len(self.N) == 1:
            A.append(values[:-1])
            A.append(values[1:] - A[0])

        if len(self.N) == 2:
            A.append(values[:-1, :-1])
            A.append(values[1:, :-1] - A[0])
            A.append(values[:-1, 1:] - A[0])
            A.append(values[1:, 1:] - A[2] - A[1] - A[0])

        self.A = A

    def __call__(self, *coords):
        """Evaluate the interpolation at the input `coords` values.
        
        This offers optimally-fast linear interpolation for 1D and  2D dimensional vector data 
        on a equally spaced grids
        
        Parameters
        ----------
        coords:  list of arrays
            List of 1D arrays for the N coordines (N=1 or N=2). These arrays must be of the same shape.
 
        """
        coords = np.array(coords).T
        coords -= self.offsets
        coords /= self.scales
        coords = coords.T

        # clip dimension - it will extrapolation by a nearest value
        for coord, n in zip(coords, self.N):
            np.clip(coord, 0, n - 1.00001, coord)

        #  en.wikipedia.org/wiki/Bilinear_interpolation#Unit_square

        # get indicies x and weights dx
        x = np.int16(coords)  # fast floor(x) function
        coords -= x  # frac(x)
        dx = coords[..., None]

        # prepare coefficients
        if len(self.N) == 1:  # linear interpolation
            # inplace evaluate linear  interpolation
            inter_out = self.A[1][x[0]]
            inter_out *= dx[0]
            inter_out += self.A[0][x[0]]

        else:  # bilinear interpolation
            # inplace evaluate linear  interpolation
            inter_out = self.A[1][x[0], x[1]]
            inter_out *= dx[0]
            inter_out += self.A[0][x[0], x[1]]

            _tmp = self.A[3][x[0], x[1]]
            _tmp *= dx[0]
            _tmp += self.A[2][x[0], x[1]]
            _tmp *= dx[1]
            inter_out += _tmp

        return np.moveaxis(inter_out, -1, 0)


class CartesianGrid_Ndim:
    """
    Linear multivariate Cartesian grid interpolation in arbitrary dimensions
    This is a regular grid with equal spacing.

    Use :py:class:`~aurora.atomic.CartesianGrid` for a version that is optimally fast
    in 2 dimensions only.

    Parameters
    ----------
    grids: list of arrays, N=len(grids), arbitrary N
        List of 1D arrays with equally spaced grid values for each dimension
    values: N+1 dimensional array of values used for interpolation
        Values to interpolate. The first dimension typically refers to different ion stages, for which 
        data is provided on the input grids. Other dimensions refer to values on the N-dimensional grids.

    """
    def __init__(self, grids, values):

        self.values = np.ascontiguousarray(values)
        for g,s in  zip(grids,values.shape[1:]):
            if len(g) != s: raise Exception('wrong size of values array')

        self.offsets = [g[0] for g in grids]
        self.scales  = [(g[-1]-g[0])/(n-1) for g,n in zip(grids, values.shape[1:])]
        self.N = values.shape[1:]

    def __call__(self, *coords):
        '''Transform coords into pixel values and provide interpolated result.

        Parameters
        ----------
        coords:  list of arrays
            List of 1D arrays for the N coordines (N=1 or N=2). These arrays must be of the same shape.
        '''
        out_shape = coords[0].shape

        coords = np.array(coords).T
        coords -= self.offsets
        coords /= self.scales
        coords = coords.T

        # clip dimension - gives extrapolation by nearest value
        for coord, n in zip(coords, self.N):
            np.clip(coord,0,n-1,coord)

        #prepare output array
        inter_out = np.empty((self.values.shape[0],)+out_shape, dtype=self.values.dtype)

        # fast interpolation on a N-dimensional regular grid
        for out,val in zip(inter_out,self.values):
            scipy.ndimage.map_coordinates(val, coords, order=1, output=out)

        return inter_out


def interp_atom_prof(atom_table,xprof, yprof,log_val=False, x_multiply=True):
    r''' Fast interpolate atomic data in atom_table onto the xprof and yprof profiles.
    This function assume that xprof, yprof, x,y, table are all base-10 logarithms,
    and xprof, yprof are equally spaced.

    Parameters
    ----------
    atom_table : list
        List with x,y, table = atom_table, containing atomic data from one of the ADAS files. 
    xprof : array (nt,nr)
        Spatio-temporal profiles of the first coordinate of the ADAS file table (usually 
        electron density in :math:`cm^{-3}`)
    yprof : array (nt,nr)
        Spatio-temporal profiles of the second coordinate of the ADAS file table (usually 
        electron temperature in :math:`eV`)
    log_val : bool
        If True, return natural logarithm of the data
    x_multiply : bool
        If True, multiply output by :math:`10^{xprof}`. 

    Returns
    -------
    interp_vals : array (nt,nion,nr)
        Interpolated atomic data on time,charge state and spatial grid that correspond to the 
        ion of interest and the spatiotemporal grids of xprof and yprof. 
    '''
    x,y, table = atom_table

    if (abs(table-table[...,[0]]).all()  < 0.05) or xprof is None:
        # 1D interpolation if independent of the last dimension - like SXR radiation data

        reg_interp = CartesianGrid((y, ),table[:,:,0]*np.log(10))
        interp_vals = reg_interp(yprof) 

        # multipling of logarithms is just adding
        if x_multiply and xprof is not None:
            interp_vals += xprof*np.log(10)

    else: # 2D interpolation
        if x_multiply: #multipling of logarithms is just adding
            table += x
        # broadcast both variables in the sae shape
        xprof, yprof = np.broadcast_arrays(xprof, yprof)
        #perform fast linear interpolation
        reg_interp = CartesianGrid((x, y),table.swapaxes(1,2)*np.log(10))
        interp_vals = reg_interp(xprof,yprof) 
    
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

    # set min Te here to 10 eV, because the grid above does not extend to lower temperatures
    Te = np.maximum(Te, 10.0)

    log_gamma2 = np.log10(Z**2*thirteenpointsix/Te)

    # dangerous/inaccurate extrapolation...
    return np.interp(log_gamma2, log_gamma2_grid,gff_mean_grid)



def impurity_brems(nz, ne, Te):
    '''Approximate bremsstrahlung in units of :math:`mW/nm/sr/m^3 \cdot cm^3`, or
    equivalently :math:`W/m^3` for full spherical emission. 

    Note that this may not be very useful, since this contribution is already included 
    in the continuum radiation component in ADAS files. The bremsstrahlung estimate in 
    ADAS continuum radiation files is more accurate than the one give by this function, 
    which uses a simpler interpolation of the Gaunt factor with weak ne-dependence. 
    Use with care!

    Parameters
    ----------
    nz : array (time,nZ,space)
        Densities for each charge state [:math:`cm^{-3}`]
    ne : array (time,space)
        Electron density [:math:`cm^{-3}`]
    Te : array (time,space)
        Electron temperature [:math:`cm^{-3}`]

    Returns
    -------
    brems : array (time,nZ,space)
        Bremsstrahlung for each charge state 
    '''
    # neutral stage doesn't produce brems
    Z_imp = nz.shape[1]-1
    Z = np.arange(Z_imp)[None,:,None]+1

    gff = gff_mean(Z,Te[:,None])
    brems = Z**2 * nz[:,1:] * gff * (1.69e-32 *np.sqrt(Te) * ne) [:,None]

    return brems
       
        

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
    r'''Compare effective ionization rate for each charge state with the characteristic
    transit time that passing and trapped impurity ions take to travel a parallel distance 
    :math:`L = q R`, defining

    .. math::

        \nu_{ion}^* \equiv \nu_{ion} \tau_t = \nu_{ion} \frac{q R}{v_{th}} = \frac{\sum_z n_z \nu_z^{ion}}{\sum_z n_z} q R \sqrt{\frac{m_{imp}}{2 k_B T_i}}

    following Ref.[1]_. If the normalized ionization rate (:math:`\nu_{ion}^*`) is less than 1, 
    then flux surface averaging of background asymmetries (e.g. from edge or beam neutrals) may 
    be taken as a good approximation of reality; in this case, 1.5D simulations of impurity transport
    are expected to be valid. If, on the other hand, :math:`\nu_{ion}^*>1` then local effects 
    may be too important to ignore. 

    Parameters
    ----------
    S_z : array (r,cs) [:math:`s^{-1}`]
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
         Radial profile of ion temperature [:math:`eV`]
    nz_profs : array (r,cs), optional
         Radial profile for each charge state. If provided, calculate average normalized 
         ionization rate over all charge states.
    rhop : array (r,), optional
         Sqrt of poloidal flux radial grid. This is used only for (optional) plotting. 
    plot : bool, optional
         If True, plot results.
    eps_prof : array (r,), optional
         Radial profile of inverse aspect ratio, i.e. r/R, only used if plotting is requested.  

    Returns
    -------
    nu_ioniz_star : array (r,cs) or (r,)
         Normalized ionization rate. If nz_profs is given as an input, this is an average over
         all charge state; otherwise, it is given for each charge state.

    References
    ----------
    .. [1] R.Dux et al. Nucl. Fusion 60 (2020) 126039

    '''
    nu = np.zeros((S_z.shape[0],S_z.shape[1]-1))  # exclude neutral states, which have no parallel transport
    for cs in np.arange(nu.shape[1]):
        nu[:,cs] = S_z[:,cs+1] * q_prof * R_prof * np.sqrt((imp_A * constants.m_p)/(2*Ti_prof*constants.e))

    if nz_profs is not None:
        # calculate average nu_ioniz_star 
        nu_ioniz_star = np.sum(nz_profs[:,1:]*nu,axis=1)/np.sum(nz_profs[:,1:],axis=1)
    else:
        # return normalized ionization rate for each charge state
        nu_ioniz_star = nu

    if plot:
        if rhop is None:
            rhop = np.arange(nu.shape[0])
            
        fig,ax = plt.subplots()
        if nu_ioniz_star.ndim==1:
            ax.semilogy(rhop,nu_ioniz_star, label=r'$\nu_{ion}^*$')
        else:
            for cs in np.arange(nu.shape[1]):
                ax.semilogy(rhop, nu_ioniz_star[:,cs], label=f'q={cs+1}')
            ax.set_ylabel(r'$\nu_{ion}^*$')

        ax.set_xlabel(r'$\rho_p$')

        if eps_prof is not None:
            ax.semilogy(rhop, np.sqrt(eps_prof), label=r'$\sqrt{\epsilon}$')

        ax.legend().set_draggable(True)
        ax.set_xlim([0,1])


