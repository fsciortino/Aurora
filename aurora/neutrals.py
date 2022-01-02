'''Aurora functionality for edge neutral modeling. 
The ehr5 file from DEGAS2 is used. See https://w3.pppl.gov/degas2/ for details.
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

from matplotlib import cm
from scipy.optimize import curve_fit, least_squares
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import urllib
import shutil, os, copy
from scipy.constants import e,h,c as c_light,Rydberg
from scipy.interpolate import interp2d
import requests

from . import plot_tools
from . import radiation
from . import adas_files

if 'AURORA_ADAS_DIR' in os.environ:
    # if user indicated a directory for atomic data, use that
    atomic_data_dir = os.environ['AURORA_ADAS_DIR']
else:
    # location of the "adas_data" directory relative to this script:
    atomic_data_dir =  os.path.dirname(os.path.realpath(__file__))+os.sep+'adas_data'+os.sep



def download_ehr5_file():
    '''Download the ehr5.dat file containing atomic data describing the multi-step ionization and 
    recombination of hydrogen. 

    See https://w3.pppl.gov/degas2/ for details. 
    '''
    filename='ehr5.dat'
    url = 'https://w3.pppl.gov/degas2/ehr5.dat'    
    r = requests.get(url)
    
    # write file to directory where ADAS data is also located
    with open(atomic_data_dir+'/ehr5.dat', 'wb') as f:
        f.write(r.content)

    print('Successfully downloaded the DEGAS2 ehr5.dat file.')
    

class ehr5_file:
    ''' Read ehr5.dat file from DEGAS2. 
    Returns a dictionary containing

    - Ionization rate Seff in :math:`cm^3 s^{-1}`
    - Recombination rate Reff in :math:`cm^3 s^{-1}`
    - Neutral electron losses :math:`E_{loss}^{(i)}` in :math:`erg s^{-1}`
    - Continuum electron losses :math:`E_{loss}^{(ii)}` in :math:`erg s^{-1}`
    - Neutral “n=2 / n=1”, :math:`N_2^{(i)}/N_1`
    - Continuum “n=2 / n=1”, :math:`N_2^{(ii)}/N_11
    - Neutral “n=3 / n=1”, :math:`N_3^{(i)}/N_1`
    - Continuum “n=3 / n=1”, :math:`N_3^{(ii)}/N_1`
    
    ... and similarly for n=4 to 9. Refer to the DEGAS2 manual for details.
    '''

    def __init__(self, filepath=None):
        '''Load ehr5.dat file, either from the indicated path or by downloading it locally. 

        Keyword Args:
            filepath : str, optional
                Path of ehr5.dat file to use. If left to None, the file is downloaded from the web
                and saved locally. 

        Results for each of the fields in the `fields` attribute will be available in the `res`
        attribute in the form of a dictionary. Refer to the DEGAS2 manual for a description of these
        fields.
        '''

        if filepath is None:
            if not os.path.exists(atomic_data_dir+'/ehr5.dat'):
                # if ehr5.dat file is not available, download it
                download_ehr5_file()
            self.filepath = atomic_data_dir+'/ehr5.dat'
        else:
            self.filepath = filepath

        self.ne = 10 ** np.array([10 + (jn - 1.0) / 2.0 for jn in np.arange(1, 16)])  # cm^{-3}
        self.Te = 10 ** np.array([-1.2 + (jt - 1.0) / 10.0 for jt in np.arange(1, 61)])  # eV

        self.fields = [
            'Seff',
            'Reff',
            'Ei_loss',
            'Eii_loss',
            'n3i_n1',
            'n3ii_n1',
            'n2i_n1',
            'n2ii_n1',
            'n4i_n1',
            'n4ii_n1',
            'n5i_n1',
            'n5ii_n1',
            'n6i_n1',
            'n6ii_n1',
            'n7i_n1',
            'n7ii_n1',
            'n8i_n1',
            'n8ii_n1',
            'n9i_n1',
            'n9ii_n1',
        ]

        # get data
        self.load()

    def load(self):

        self.res = {}
        with open(self.filepath) as f:
            for field in self.fields:

                data = np.zeros((15, 60))

                # read header
                header = f.readline()

                for jn in np.arange(15):
                    # loop over 15 densities
                    _jn_index = f.readline()
                    arr = []
                    for jt_row in np.arange(10):
                        # 10 rows of 6 values each for Te
                        elems = [val for val in f.readline().strip().split(' ') if val != '']

                        line = [float(val) for val in elems]
                        data[jn, jt_row * 6 : (jt_row + 1) * 6] = np.array(line)
                    _dum = f.readline()  # empty line at the end

                self.res[field] = copy.deepcopy(data)

    def plot(self, field='Seff', fig=None, axes=None):
        colormap = cm.rainbow
        if fig is None or axes is None:
            fig, ax = plt.subplots()

        ls_cycle = plot_tools.get_ls_cycle()

        labels = ['{:.2e}'.format(val) + ' cm${-3}$' for val in self.ne]
        for i in np.arange(len(labels)):
            ax.plot(np.log10(self.Te), self.res[field][i, :], next(ls_cycle), label=labels[i])

        ax.set_xlabel('$\log T_e\ \mathrm{[eV]}$', fontsize=16)
        ax.set_ylabel(field, fontsize=16)
        ax.legend()



def get_exc_state_ratio(m, N1, ni, ne, Te, rad_prof=None, rad_label=r'rmin [cm]', plot=False):
    r"""Compute density of excited states in state `m` (m>1), given the density of ground state atoms.
    This function is not l-resolved.

    The function returns

    .. math::

        N_m/N_1 = \left(\frac{N_m^i}{N_1} \right) N_m + \left(\frac{N_m^{ii}}{n_i} \right) n_i


    where :math:`N_m` is the number of electrons in the excited state :math:`m`, :math:`N_1` 
    is the number in the ground state, and :math:`n_i` is the density of ions that could recombine.
    :math:`i` and :math:`ii` indicate terms corresponding to coupling to the ground state and to 
    the continuum, respectively.

    Ref.: DEGAS2 manual.

    Parameters
    ----------
    m : int
        Principal quantum number of excited state of interest. 2<m<10
    N1 : float, list or 1D-array [:math:`cm^{-3}`]
        Density of ions in the ground state. This must have the same shape as ni!
    ni : float, list or 1D-array [:math:`cm^{-3}`]
        Density of ions corresponding to the atom under consideration. This must
        have the same shape as N1!
    ne : float, list or 1D-array [:math:`cm^{-3}`]
        Electron density to evaluate atomic rates at.
    Te : float, list or 1D-array [:math:`eV`]
        Electron temperature to evaluate atomic rates at.
    rad_prof : list, 1D array or None
        If None, excited state densities are evaluated at all the combinations of ne,Te and zip(Ni,ni).
        If a 1D array (same length as ne,Te,ni and N1), then this is taken to be a radial coordinate
        for radial profiles of ne,Te,ni and N1.
    rad_label : str
        When rad_prof is not None, this is the label for the radial coordinate.
    plot : bool
        Display the excited state ratio

    Returns
    -------
    Nm : array of shape [len(ni)=len(N1),len(ne),len(Te)]
        Density of electrons in excited state `n`  [:math:`cm^{-3}`]
    """
    if m <= 1:
        raise ValueError('Excited state principal quantum number must be greater than 1!')
    if m > 9:
        raise ValueError('Selected excited state value not available!')

    ne = np.atleast_1d(ne)
    Te = np.atleast_1d(Te)
    ni = np.atleast_1d(ni)
    N1 = np.atleast_1d(N1)

    if rad_prof is not None:
        # if a radial profile is being computed, ne, Te, ni and N1 must all have the same length
        assert len(ne) == len(Te) and len(ne) == len(ni) and len(ne) == len(N1)

    # get CR model results:
    atom = ehr5_file()

    # coupling to the ground state:
    ground_coupling = atom.res['n{}i_n1'.format(m)]
    cont_coupling = atom.res['n{}ii_n1'.format(m)]

    if rad_prof is not None:
        # evaluate along radial profiles
        gc_interp = interp2d(atom.ne, atom.Te, ground_coupling.T)
        gc = np.array([float(gc_interp(XX, YY)) for XX, YY in zip(ne, Te)])
        cc_interp = interp2d(atom.ne, atom.Te, cont_coupling.T)
        cc = np.array([float(cc_interp(XX, YY)) for XX, YY in zip(ne, Te)])
    else:
        # evaluate at all combinations of points
        gc = interp2d(atom.ne, atom.Te, ground_coupling.T)(ne, Te).T
        cc = interp2d(atom.ne, atom.Te, cont_coupling.T)(ne, Te).T

        N1 = np.rollaxis(np.tile(N1, (cc.shape[0], cc.shape[1], 1)), axis=2)
        ni = np.rollaxis(np.tile(ni, (cc.shape[0], cc.shape[1], 1)), axis=2)
        gc = np.tile(gc, (len(N1), 1, 1))
        cc = np.tile(cc, (len(N1), 1, 1))

    # combine coupling to ground state and to continuum
    Nm = gc * N1 + cc * ni
    Nm_ground = gc * N1
    Nm_cont = cc * ni

    if plot:
        # plot only the first provided value of value of N1 and ni
        ls_style = plot_tools.get_ls_cycle()

        if rad_prof is not None:
            fig, ax = plt.subplots()
            ax.plot(rad_prof, Nm / N1, next(ls_style), lw=2)
            ax.set_ylabel(r'$N_{}/N_1$'.format(m), fontsize=16)
            ax.set_xlabel(rad_label, fontsize=16)
        else:
            fig, ax = plt.subplots(1, 2, figsize=(15, 8))

            labels_Te = ['{:.2e}'.format(val) + ' eV' for val in Te]
            for jt in np.arange(len(Te)):
                ax[0].semilogx(ne, Nm[0, :, jt] / N1[0], next(ls_style), lw=2, label=labels_Te[jt])
            ax[0].legend(loc='best')
            ax[0].set_title('$N_1$=%.2e m$^{-3}$, $n_i$=%.2e m$^{-3}$' % (N1[0], ni[0]))
            ax[0].set_ylabel(r'$N_{}/N_1$'.format(m), fontsize=16)
            ax[0].set_xlabel(r'$n_e$ [cm$^{-3}$]', fontsize=16)

            labels_ne = ['{:.2e}'.format(val) + ' cm$^{-3}$' for val in ne]
            for jn in np.arange(len(ne)):
                ax[1].semilogx(Te, Nm[0, jn, :] / N1[0], next(ls_style), lw=2, label=labels_ne[jn])
            ax[1].legend(loc='best')
            ax[1].set_title('$N_1$=%.2e m$^{-3}$, $n_i$=%.2e m$^{-3}$' % (N1[0], ni[0]))

            ax[1].set_xlabel(r'$T_e$ [eV]', fontsize=16)

        plt.tight_layout()

    return Nm, Nm_ground, Nm_cont



def plot_exc_ratios(n_list=[2, 3, 4, 5, 6, 7, 8, 9], ne=1e13, ni=1e13, Te=50, N1=1e12, 
                    ax=None, ls='-', c='r', label=None):
    """Plot :math:`N_i/N_1`, the ratio of hydrogen neutral density in the excited state `i`
    and the ground state, for given electron density and temperature.

    Parameters
    ----------
    n_list : list of integers
        List of excited states (principal quantum numbers) to consider.
    ne : float
        Electron density in :math:`cm^{-3}`.
    ni : float
        Ionized hydrogen density [:math:`cm^{-3}`]. This may be set equal to ne for a pure plasma.
    Te : float
        Electron temperature in :math:`eV`.
    N1 : float
        Density of ground state hydrogen [:math:`cm^{-3}`]. This is needed because the excited
        state fractions depend on the balance of excitation from the ground state and
        coupling to the continuum.
    ax : matplotlib.axes instance, optional
        Axes instance on which results should be plotted. 
    ls : str
        Line style to use
    c : str or other matplotlib color specification
        Color to use in plots
    label : str
        Label to use in scatter plot. 

    Returns
    -------
    Ns : list of arrays
        List of arrays for each of the n-levels requested, each containing excited state 
        densities at the chosen densities and temperatures for the given ground state density.
    """

    Ns = np.zeros(len(n_list))
    for i, n in enumerate(n_list):
        Ns[i], _, _ = get_exc_state_ratio(m=n, N1=N1, ni=ne, ne=ne, Te=Te, plot=False)

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(n_list, Ns / N1, c=c, label=label, s=50.0)
    ax.set_xlabel('n', fontsize=16)
    ax.set_ylabel(r'$N_i/N_1$', fontsize=16)
    ax.set_ylim([0, np.max(Ns / N1) * 1.1])

    return Ns




def Lya_to_neut_dens(emiss_prof, ne, Te, ni=None, plot=True, rhop=None,
                     rates_source='adas', axs=None):
    ''' Estimate ground state neutral density from measured emissivity profiles.
    This ignores possible molecular dynamics and effects that may be captured via
    forward modeling of neutral transport.

    Parameters
    ----------
    emiss_prof : 1D array
        Emissivity profile, units of :math:`W/cm`
    ne : 1D array
        Electron density, units of :math:`cm^{-3}`
    Te : 1D array
        Electron temperature, units of :math:`eV`
    ni : 1D array
        Main ion (H/D/T) density, units of :math:`cm^{-3}`. 
        If left to None, this is internally set to ni=ne.
    plot : bool
        If True, plot some of the key density profiles.
    rhop : 1D array
        Sqrt of normalized poloidal flux radial coordinate. Used only for plotting.
    rates_source : str
        Source of atomic rates. Possible choices are 'adas' or 'colrad'
    axs : Axes instance
        If given, plot on these axes.

    Returns
    ------------
    N1 : 1D array
        Radial profile of estimated ground state atomic neutral density on the same grid 
        as the input arrays. Units of :math:`cm^{-3}`.

    Examples
    --------------
    
    >>> N2_colrad,axs = Lya_to_neut_dens_basic(emiss_prof, ne, Te, ni, 
    >>>                                  plot=True, rhop=rhop, rates_source='colrad')
    
    >>> N2_adas,axs = Lya_to_neut_dens_basic(emiss_prof, ne, Te, ni, plot=True, rhop=rhop, 
    >>>                       rates_source='adas',axs=axs)
    
    '''
    assert len(emiss_prof)==len(ne) and len(ne)==len(Te)
    if ni is None:
        ni = copy.deepcopy(ne)
    else:
        assert len(ne)==len(ni)

    thirteenpointsix = h*c_light*Rydberg/e
    A_21 = 4.699E8   # s^-1
    E_21 = thirteenpointsix *(1.0 - 2.**(-2.)) * e # J

    # n=2 population in cm^-3
    N2 = emiss_prof/(A_21 * E_21)

    if rates_source=='colrad':
        # use rates from COLRAD code
        atom = ehr5_file()
        ground_coupling = atom.res['n2i_n1']
        cont_coupling = atom.res['n2ii_n1']

        # atomic data is on a grid in units of cm^-3, eV
        gc_interp = interp2d(atom.ne,atom.Te,ground_coupling.T)
        gc = np.array([float(gc_interp(XX,YY)) for XX,YY in zip(ne,Te)])  # Nm/N1 from exc
        cc_interp = interp2d(atom.ne,atom.Te,cont_coupling.T)
        cc = np.array([float(cc_interp(XX,YY)) for XX,YY in zip(ne,Te)])  # Nm/N1 from recomb

        # ground state density
        #N1 = (N2 - cc * ni)/gc
        N1 = N2/gc # continuum recomb term not well constrained, but should be small. Ignore it


    elif rates_source=='adas':
        filename = 'pec96#h_pju#h0.dat' # for D Ly-alpha

        # fetch file automatically, locally, from AURORA_ADAS_DIR, or directly from the web:
        path = adas_files.get_adas_file_loc(filename, filetype='adf15')
        log10pec_dict = radiation.read_adf15(path) #, plot_lines=[1215.2])
        
        # evaluate these interpolations on our profiles
        pec_recomb = 10**log10pec_dict[1215.2]['recom'].ev(np.log10(ne), np.log10(Te))
        pec_exc = 10**log10pec_dict[1215.2]['excit'].ev(np.log10(ne), np.log10(Te))

        N1 = emiss_prof/E_21/(ne*pec_exc+ni*pec_recomb)


    if plot:
        if rhop is None:
            print('No rhop array was given!')
            rhop = np.arange(len(N1))
        if axs is None:
            fig,ax = plt.subplots(2,1, figsize=(10,7))
        else:
            ax = axs
        sel = np.argmin(np.abs(rhop - 0.9))
        if axs is None:
            # plot this only the first time
            ax[0].plot(rhop[sel:], N2[sel:], label=r'$N_2$')
        ax[0].semilogy(rhop[sel:], N1[sel:], label=fr'{rates_source} $N_1$')
        ax[1].semilogy(rhop[sel:], N1[sel:]/ne[sel:], label=fr'{rates_source} $N_1/n_e$')
        ax[0].set_ylim([1e10,None])
        ax[1].set_ylim([1e-5,None])
        ax[0].set_ylabel(r'Neutral density [$cm^{{-3}}$]')
        ax[1].set_ylabel(r'Density ratio')
        ax[0].legend()
        ax[1].legend()
        ax[1].set_xlabel(r'$\rho_p$')
    else:
        ax = None
        
    return N1,ax

