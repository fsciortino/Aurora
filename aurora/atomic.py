"""Collection of classes and functions for loading, interpolation and processing of atomic data. 
Refer also to the adas_files.py script. 
"""
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

from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, interp1d, RegularGridInterpolator
from matplotlib import cm
import os, sys, copy
import scipy.ndimage
from scipy.linalg import svd
from scipy import constants
from . import adas_files


def get_adas_file_types():
    """Obtain a description of each ADAS file type and its meaning in the context of Aurora.

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

    """

    return {
        "acd": "effective recombination",
        "scd": "effective ionization",
        "prb": "continuum radiation",
        "plt": "line radiation",
        "ccd": "thermal charge exchange",
        "prc": "thermal charge exchange continuum radiation",
        "xcd": "Parent cross-coupling coefficients",
        "qcd": "Cross-coupling coefficients",
        "pls": "line radiation in the SXR range",
        "prs": "continuum radiation in the SXR range",
        "brs": "continuum spectral bremstrahlung",
        "fis": "sensitivity in the SXR range",
        "pbs": "impurity bremsstrahlung in SXR range, also included in prs files",
    }


class adas_file:
    """Read ADAS file in ADF11 format over the given density and temperature grids.
    Note that such grids vary between files, and the species they refer to may too.

    Refer to ADAS documentation for details on each file.

    Parameters
    ----------
    filepath : str
        Path to location where ADAS file is located.
    """

    def __init__(self, filepath):

        self.filepath = filepath
        self.filename = filepath.split("/")[-1]
        self.file_type = self.filename[:3]

        if self.file_type not in ["brs", "sxr"]:
            try:
                self.imp = self.filename.split("_")[1].split(".")[0]
            except:
                self.imp = None  # soem old files have a different naming convenction

        # get data
        self.load()

    def load(self):
        """ADF11 format description: https://www.adas.ac.uk/man/appxa-11.pdf"""
        with open(self.filepath) as f:
            header = f.readline()
            self.n_ion, n_ne, n_T = np.int_(header.split()[:3])
            details = " ".join(header.split()[3:])

            f.readline()  # skip empty line
            line = f.readline()
            # metastable resolved file
            if all([a.isdigit() for a in line.split()]):
                self.metastables = np.int_(line.split())
                f.readline()  # skip empty line
                line = f.readline()
            else:
                self.metastables = np.ones(self.n_ion + 1, dtype=int)

            logNe = []
            while len(logNe) < n_ne:
                logNe += [float(n) for n in line.split()]
                line = f.readline()

            logT = []
            while len(logT) < n_T:
                logT += [float(t) for t in line.split()]
                line = f.readline()

            subheader = line

            ldata, self.Z, self.MGRD, self.MPRT = [], [], [], []
            ind = 0
            while True:
                ind += 1

                try:
                    iprt, igrd, typ, z = subheader.split("/")[1:5]
                    self.Z.append(int(z.split("=")[1]))
                    self.MGRD.append(int(igrd.split("=")[1]))
                    self.MPRT.append(int(iprt.split("=")[1]))
                except:
                    # some old files have different header
                    self.Z.append(ind)
                    self.MGRD.append(1)
                    self.MPRT.append(1)

                # drcofd = log10(generalised collisional radiative coefficients) (units according to class)
                drcofd = []
                while len(drcofd) < n_ne * n_T:
                    line = f.readline()
                    drcofd += [float(L) for L in line.split()]

                ldata.append(np.array(drcofd).reshape(n_T, n_ne))

                subheader = f.readline().replace("-", " ")
                # end of the file
                if len(subheader) == 0 or subheader.isspace() or subheader[0] == "C":
                    break

        self.logNe = np.array(logNe)
        self.logT = np.array(logT)
        self.logdata = np.array(ldata)

        self.meta_ind = list(zip(self.Z, self.MGRD, self.MPRT))

    def plot(self, fig=None, axes=None):
        """Plot data from input ADAS file. If provided, the arguments allow users to overplot
        and compare data from multiple files.

        Parameters
        ----------
        fig : matplotlib Figure object
            If provided, add specification as to which ADAS file is being plotted.
        axes : matplotlib Axes object (or equivalent)
            If provided, plot on these axes. Note that this typically needs to be a set of axes
            for each plotted charge state. Users may want to call this function once first to get
            some axes, and then pass those same axes to a second call for another file to compare with.
        """

        # settings for plotting
        self.ncol = np.ceil(np.sqrt(len(self.Z)))
        self.nrow = np.ceil(len(self.Z) / self.ncol)

        if fig is None or axes is None:
            fig, axes = plt.subplots(
                int(self.ncol), int(self.nrow), sharex=True, sharey=True
            )

        axes = np.atleast_2d(axes)
        colormap = cm.rainbow
        colors = cm.rainbow(np.linspace(0, 1, len(self.logNe)))

        if fig is not None:
            fig.suptitle(self.filename + "  " + get_adas_file_types()[self.file_type])

        for i, ax in enumerate(axes.flatten()):
            if i >= len(self.Z):
                break
            if all(self.logdata[i].std(1) == 0):  # independent of density
                ax.plot(self.logT, self.logdata[i, :, 0])
            else:
                ax.set_prop_cycle("color", colors)
                ax.plot(self.logT, self.logdata[i])
                ax.text(
                    0.1,
                    0.8,
                    "$n_e = 10^{%.0f-%.0f}\mathrm{[cm^{-3}]}$"
                    % (self.logNe[0], self.logNe[-1]),
                    horizontalalignment="left",
                    transform=ax.transAxes,
                )

            ax.grid(True)

            if self.file_type != "brs":
                charge = self.Z[i]
                meta = self.MPRT[i], self.MGRD[i]
                if self.file_type in ["scd", "prs", "ccd", "prb", "qcd"]:
                    charge -= 1
                title = self.imp + "$^{%d\!+}$" % charge
                if any(self.metastables > 1):
                    title += str(meta)
                ax.set_title(title)

        for ax in axes[-1]:
            ax.set_xlabel("$\log\ T_e\ \mathrm{[eV]}$")
        for ax in axes[:, 0]:
            if self.file_type in ["scd", "acd", "ccd"]:
                ax.set_ylabel("$\log(" + self.file_type + ")\ \mathrm{[cm^3/s]}$")
            elif self.file_type in ["prb", "plt", "prc", "pls", "brs", "prs"]:
                ax.set_ylabel("$\log(" + self.file_type + ")\ \mathrm{[W\cdot cm^3]}$")


def read_filter_response(filepath, adas_format=True, plot=False, ax = None):
    """Read a filter response function over energy.

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

    """
    E_eV = []
    response = []
    try:
        # Attempt to read ADAS format
        with open(filepath) as f:

            header = f.readline()
            num = int(header.split()[0])

            # *****
            f.readline()

            while len(E_eV) < num:
                line = f.readline()
                E_eV += [float(n) for n in line.split()]
            while len(response) < num:
                line = f.readline()
                response += [float(t) for t in line.split()]

        # energy and response function are written in natural logs
        E_eV = np.r_[0,np.exp(E_eV)]
        response = np.r_[0, np.exp(response)]
        
    except ValueError:
        try:
            # Attempt to read CXRO format
            with open(filepath) as f:
                contents = f.readlines()

            for line in contents[2:]:
                tmp = line.strip().split()
                E_eV.append(float(tmp[0]))
                response.append(float(tmp[1]))
            E_eV = np.r_[0, E_eV]
            response = np.r_[0, response]
        except ValueError:
            raise ValueError("Unrecognized filter function format...")

    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        ax.semilogx(E_eV, response)
        ax.set_xlabel("Photon energy [eV]")
        ax.set_ylabel("Detector response efficiency")

    return E_eV, response

@lru_cache(maxsize=1000)
def load_adas_file_with_cache(fileloc):
    """Wrapper around the adas_file loading mechanism to enable implementation
    of a granular caching system at file level. Each file would on average
    account for less than 100kB, as such I have arbitrarily choosen to cap the
    cache size to 1000, therefore preventing the cache to ever occupy more than
    100MB in worst case scenario.

    Parameters
    ----------
    fileloc : str containing the address of the adas file to load.

    Returns
    -------
    The in-memory representation of the loaded adas file.
    """
    return adas_file(fileloc)


def get_atom_data(imp, files=["acd", "scd"]):
    """Collect atomic data for a given impurity from all types of ADAS files available or
    for only those requested.

    Parameters
    ----------
    imp : str
        Atomic symbol of impurity ion.
    files : list or dict
        ADAS file types to be fetched. Default is ["acd","scd"] for effective ionization
        and recombination rates (excluding CX) using default files, listed in :py:func:`~aurora.adas_files_adas_files_dict`.
        If users prefer to use specific files, they may pass a dictionary instead, of the form

        .. code-block:: python

            {'acd': 'acd89_ar.dat', 'scd': 'scd89_ar.dat'}

        or

        .. code-block:: python

            {'acd': 'acd89_ar.dat', 'scd': None}

        if only some of the files need specifications and others (given as None) should be
        taken from the default files.

    Returns
    -------
    atom_data : dict
        Dictionary containing data for each of the requested files.
        Each entry of the dictionary gives log-10 of ne, log-10 of Te and log-10 of the data
        as attributes res.logNe, res.logT,  res.logdata, res.meta_ind, res.metastables
    """

    all_files = {}
    atom_dict = adas_files.adas_files_dict().get(imp, None)
    for filename in files:
        if isinstance(files, dict) and files[filename] is not None:
            all_files[filename] = files[filename] 
            
        elif atom_dict is None:
            raise KeyError(f"Atomics data not found for impurity {imp}!")
            
        elif filename not in atom_dict:
            raise ValueError(f"Could not fetch {imp} {filecheck.upper()} file! Please specify file locations using 'files' argument, for example files=dict(acd='user/acd89_ar.dat')")
        else:
            all_files[filename] = atom_dict[filename]

    atom_data = {}
    for filetype in all_files:

        # find location of required ADF11 file
        fileloc = adas_files.get_adas_file_loc(all_files[filetype], filetype="adf11")

        # load specific file and add it to output dictionary
        atom_data[filetype] = load_adas_file_with_cache(fileloc)


    return atom_data

    return atom_data


def null_space(A):
    """Find null space of matrix `A`."""
    u, s, vh = svd(A, full_matrices=True)
    Q = vh[-1, :].T.conj()  # -1 index is after infinite time/equilibration
    # return the smallest singular eigenvalue
    return Q, s[-2]  # matrix is singular, so last index is ~0


def superstage_rates(R, S, superstages, save_time=None):
    """Compute rate for a set of ion superstages.
    Input and output rates are log-values in arbitrary base.

    Parameters
    ----------
    R : array (time,nZ,space)
        Array containing the effective recombination rates for all ion stages,
        These are typically combinations of radiative and dielectronic recombination,
        possibly also of charge exchange recombination.
    S : array (time,nZ,space)
        Array containing the effective ionization rates for all ion stages.
    superstages : list or 1D array
        Indices of charge states of chosen ion that should be included.
    save_time : list or 1D array of bools
        Indices of the timeslcies which are actually returned by AURORA

    Returns
    -------
    superstages : array
        Set of superstages including 0,1 and final stages if these were missing in the input.
    R_rates_super : array (time,nZ-super,space)
        effective recombination rates for superstages
    S_rates_super : array (time,nZ-super,space)
        effective ionization rates for superstages
    fz_upstage : array (space, nZ, time_
        fractional abundances of stages within superstages

    """
    Z_imp = S.shape[1]

    # check input superstages
    if 1 not in superstages:
        print("Warning: 1th superstage was included")
        superstages = np.r_[1, superstages]
    if 0 not in superstages:
        print("Warning: 0th superstage was included")
        superstages = np.r_[0, superstages]
    if np.any(np.diff(superstages) <= 0):
        print("Warning: sorted superstages in increasing order")
        superstages = np.sort(superstages)
    if superstages[-1] > Z_imp:
        raise Exception("The highest superstage must be less than Z_imp = %d" % Z_imp)
    superstages = np.array(superstages)

    # indexing that accounts for no recomb of neutral stage or ionization of full-stripped stage
    R_rates_super = R[:, superstages[1:] - 1]
    S_rates_super = S[:, superstages[1:] - 1]

    if len(S) == 1 or save_time is None:  # time averaged kinetic profiles
        t_slice = slice(None, None)
        nt = 1
    else:  # time resolved kinetic profiles
        t_slice = save_time
        nt = save_time.sum()

    # fractional abundance of supestages used for upstaging.
    fz_upstage = np.ones((R.shape[-1], Z_imp + 1, nt))

    # add fully-stripped charge state
    _superstages = np.r_[superstages, Z_imp + 1]

    for i in range(len(_superstages) - 1):
        if _superstages[i] + 1 != _superstages[i + 1]:
            sind = slice(_superstages[i] - 1, _superstages[i + 1] - 1)

            # calculate fractional abundances within the superstage
            rate_ratio = S[:, sind] / R[:, sind]
            fz = np.cumprod(rate_ratio, axis=1)
            fz /= np.maximum(1e-60, fz.sum(1))[:, None]  # prevents zero division

            # superstage rates
            R_rates_super[:, i - 1] *= np.maximum(fz[:, 0], 1e-60)

            if i < len(_superstages) - 2:  # last superstage cannot ionize further
                S_rates_super[:, i] *= np.maximum(fz[:, -1], 1e-60)

            # fractional abundances inside of each superstage
            fz_upstage[:, _superstages[i] : _superstages[i + 1]] = fz.T[:, :, t_slice]

    return superstages, R_rates_super, S_rates_super, fz_upstage


def get_frac_abundances(
    atom_data,
    ne_cm3,
    Te_eV=None,
    Ti_eV=None,
    n0_by_ne=0.0,
    superstages=[],
    plot=True,
    ax=None,
    rho=None,
    rho_lbl=None,
):
    r"""Calculate fractional abundances from ionization and recombination equilibrium.
    If n0_by_ne is not 0, radiative recombination and thermal charge exchange are summed.

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
    Ti_eV : float or array, optional
        Bulk ion temperature in units of eV. If left to None, Ti is set to be equal to Te
    n0_by_ne: float or array, optional
        Ratio of background neutral hydrogen to electron density. If not 0, CX is considered.
    superstages : list or 1D array
        Indices of charge states of chosen ion that should be included. If left empty, all ion stages
        are included. If only some indices are given, these are modeled as "superstages".
    plot : bool, optional
        Show fractional abundances as a function of ne,Te profiles parameterization.
    ax : matplotlib.pyplot Axes instance
        Axes on which to plot if plot=True. If False, it creates new axes
    rho : list or array, optional
        Vector of radial coordinates on which ne,Te (and possibly n0_by_ne) are given.
        This is only used for plotting, if given.
    rho_lbl: str, optional
        Label to be used for rho. If left to None, defaults to a general "x".

    Returns
    -------
    Te : array
        electron temperatures as a function of which the fractional abundances and
        rate coefficients are given.
    fz : array, (space,nZ)
        Fractional abundances across the same grid used by the input ne,Te values.

    """
    # if input arrays are multi-dimensional, flatten them here and restructure at the end
    _ne = np.ravel(ne_cm3)
    _Te = np.ravel(Te_eV) if Te_eV is not None else None
    _Ti = np.ravel(Ti_eV) if Ti_eV is not None else _Te
    _n0_by_ne = np.ravel(n0_by_ne)
    if superstages is None:
        superstages = []

    include_cx = False if not np.any(n0_by_ne) else True

    out = get_cs_balance_terms(atom_data, _ne, _Te, _Ti, include_cx=include_cx)
    Te, Sne, Rne = out[:3]

    Z_imp = Sne.shape[1]

    if include_cx:
        # Get an effective recombination rate by summing radiative & CX recombination rates
        Rne += n0_by_ne[:, None] * out[3]

    rate_ratio = np.hstack((np.ones_like(Te)[:, None], Sne / Rne))
    fz_full = np.cumprod(rate_ratio, axis=1)
    fz_full /= fz_full.sum(1)[:, None]

    # Enable use of superstages
    if len(superstages):

        # superstage_rates expects values in shape (time,nZ,space)
        superstages, Rne, Sne, _ = superstage_rates(
            Rne.T[None], Sne.T[None], superstages
        )
        Rne = Rne[0].T
        Sne = Sne[0].T

        rate_ratio = np.hstack((np.ones_like(Te)[:, None], Sne / Rne))
        fz_super = np.cumprod(rate_ratio, axis=1)
        fz_super /= fz_super.sum(1)[:, None]

        # bundled stages can have very high values -- clip here
        Rne = np.clip(Rne, 1e-25, 1)
        Sne = np.clip(Sne, 1e-25, 1)

        _superstages = np.r_[superstages, Z_imp + 1]

    if plot:
        # plot fractional abundances (only 1D)
        if ax is None:
            fig, axx = plt.subplots()
        else:
            axx = ax

        if rho is None:
            x = Te
            axx.set_xlabel("T$_e$ [eV]")
            axx.set_xscale("log")
        else:
            if rho_lbl is None:
                rho_lbl = "x"
            x = rho
            axx.set_xlabel(rho_lbl)

        axx.set_prop_cycle("color", cm.plasma(np.linspace(0, 1, fz_full.shape[1])))

        css = 0
        for cs in range(fz_full.shape[1]):
            l = axx.semilogy(x, fz_full[:, cs], ls="--")
            imax = np.argmax(fz_full[:, cs])
            axx.text(
                np.max([0.1, x[imax]]),
                fz_full[imax, cs],
                cs,
                horizontalalignment="center",
                clip_on=True,
            )

            if len(superstages) and cs in _superstages:
                axx.semilogy(x, fz_super[:, css], c=l[0].get_color(), ls="-")
                imax = np.argmax(fz_super[:, css])

                if _superstages[css] == Z_imp:
                    lbl = r"\{" + f"{_superstages[css]}" + r"\}"
                else:
                    if _superstages[css] != _superstages[css + 1] - 1:
                        lbl = (
                            r"\{"
                            + f"{_superstages[css]},{_superstages[css+1]-1}"
                            + r"\}"
                        )
                    else:
                        lbl = r"\{" + f"{_superstages[css]}" + r"\}"
                axx.text(
                    np.max([0.05, x[imax]]),
                    fz_super[imax, css],
                    lbl,
                    horizontalalignment="center",
                    clip_on=True,
                    backgroundcolor="w",
                )
                css += 1

        axx.grid("on")
        axx.set_ylim(1e-2, 1.5)
        axx.set_xlim(x[0], x[-1])
        plt.tight_layout()

    if np.size(ne_cm3) > 1:
        # re-structure to original array dimensions
        Te = Te.reshape(np.array(ne_cm3).shape)
        fz_full = fz_full.reshape(*np.array(ne_cm3).shape, fz_full.shape[1])
        if len(superstages):
            fz_super = fz_super.reshape(*np.array(ne_cm3).shape, fz_super.shape[1])

    return [Te,] + [
        fz_super if len(superstages) else fz_full,
    ]


def get_Z_mean(
    atom_data,
    ne_cm3,
    Te_eV=None,
    Ti_eV=None,
    n0_by_ne=0.0,
    plot=True,
    ax=None,
    rho=None,
    rho_lbl=None,
):
    r"""Calculate mean charge state from ionization and recombination equilibrium.
    If n0_by_ne is not 0, radiative recombination and thermal charge exchange are summed.

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
    Ti_eV : float or array, optional
        Bulk ion temperature in units of eV. If left to None, Ti is set to be equal to Te
    n0_by_ne: float or array, optional
        Ratio of background neutral hydrogen to electron density. If not 0, CX is considered.
    plot : bool, optional
        Show fractional abundances as a function of ne,Te profiles parameterization.
    ax : matplotlib.pyplot Axes instance
        Axes on which to plot if plot=True. If False, it creates new axes
    rho : list or array, optional
        Vector of radial coordinates on which ne,Te (and possibly n0_by_ne) are given. 
        This is only used for plotting, if given. 
    rho_lbl: str, optional
        Label to be used for rho. If left to None, defaults to a general "x".

    Returns
    -------
    Te : array
        Electron temperatures as a function of which the mean charge state is given.
    Z_mean : array
        Mean charge state across the same grid used by the input ne,Te values. 

    """    
    
    Te, fz = get_frac_abundances(atom_data, ne_cm3, Te_eV=Te_eV, Ti_eV=Ti_eV, n0_by_ne=n0_by_ne, superstages=[], plot=False)
    Zarr = np.arange(fz.shape[1])
    Z_mean = np.sum(fz * Zarr[None, :], axis=1)
    
    if plot:
        # plot mean charge states
        if ax is None:
            fig, axx = plt.subplots()
        else:
            axx = ax

        if rho is None:
            x = Te
            axx.set_xlabel("T$_e$ [eV]")
            axx.set_xscale("log")
        else:
            if rho_lbl is None:
                rho_lbl = "x"
            x = rho
            axx.set_xlabel(rho_lbl)
            
        axx.grid("on")
        axx.set_xlim(x[0], x[-1])
        plt.tight_layout()
        plt.plot(x, Z_mean)
        
    return Te, Z_mean


def get_cs_balance_terms(
    atom_data, ne_cm3=5e13, Te_eV=None, Ti_eV=None, include_cx=True, metastables=False
):
    """Get S*ne, R*ne and cx*ne rates on the same logTe grid.

    Parameters
    ----------
    atom_data : dictionary of atomic ADAS files (only acd, scd are required; ccd is
        necessary only if include_cx=True)
    ne_cm3 : float or array
        Electron density in units of :math:`cm^{-3}`
    Te_eV : float or array
        Electron temperature in units of eV. If left to None, the Te grid
        given in the atomic data is used.
    Ti_eV : float or array
        Bulk ion temperature in units of eV, only needed for CX. If left to None, Ti is set equal to Te.
    include_cx : bool
        If True, obtain charge exchange terms as well.

    Returns
    -------
    Te : array (n_Te)
        Te grid on which atomic rates are given
    Sne, Rne (,cxne, Qne, Xne): arrays (n_ne,n_Te)
        atomic rates for effective ionization, radiative+dielectronic
        recombination (+ charge exchange, + crosscoupling if requested).
        All terms will be in units of :math:`s^{-1}`.

    Notes
    -----
        The nomenclature with 'ne' at the end of rate names
        indicates that these are rates in units of :math:`m^3\cdot s^{-1}` multiplied by the
        electron density 'ne' (hence, final units of :math:`s^{-1}`).
    """

    if Te_eV is None:
        # find smallest Te grid from all files
        logTe1 = atom_data["scd"].logT
        logTe2 = atom_data["acd"].logT

        minTe = max(logTe1[0], logTe2[0])
        maxTe = min(logTe1[-1], logTe2[-1])  # avoid extrapolation

        if include_cx:
            logTe3 = atom_data["ccd"].logT  # thermal cx recombination
            minTe = max(minTe, logTe3[0])
            maxTe = min(maxTe, logTe3[-1])  # avoid extrapolation

        Te_eV = np.logspace(minTe, maxTe, 200)

    logne = np.log10(ne_cm3)
    logTe = np.log10(Te_eV)

    Sne = interp_atom_prof(atom_data["scd"], logne, logTe, x_multiply=True)
    Rne = interp_atom_prof(atom_data["acd"], logne, logTe, x_multiply=True)
    out = [Te_eV, Sne, Rne]

    if include_cx:
        # this should be neutral temperature? or weighted Ti and T0 temperature?
        logTi = np.log10(Ti_eV) if Ti_eV is not None else logTe
        cxne = interp_atom_prof(atom_data["ccd"], logne, logTi, x_multiply=True)
        # select appropriate number of charge states
        # this allows use of CCD files from higher-Z ions because of simple CX scaling
        out.append(cxne[:, : Sne.shape[1]])

    if metastables:
        Qne = interp_atom_prof(atom_data["qcd"], logne, logTe, x_multiply=True)
        Xne = interp_atom_prof(atom_data["xcd"], logne, logTe, x_multiply=True)
        out += [Qne, Xne]

    return out


def get_atomic_relax_time(
    atom_data,
    ne_cm3,
    Te_eV=None,
    Ti_eV=None,
    n0_by_ne=0.0,
    superstages=[],
    tau_s=np.inf,
    plot=True,
    ax=None,
    ls="-",
):
    r"""Obtain the relaxation time of the ionization equilibrium for a given atomic species.

    If n0_by_ne is not 0, thermal charge exchange is added to radiative and dielectronic recombination.
    This function can work with ne,Te and n0_by_ne arrays of arbitrary dimension.
    It uses a matrix SVD approach in order to find the relaxation rates, as opposed to the simpler
    approach of :py:func:`~aurora.get_frac_abundances`, but fractional abundances produced by the two
    methods should always be the same.

    This function also allows use of superstages as well as specification of a :math:`\tau` value
    representing the effect of transport on the ionization equilibrium. NB: this is only a rough metric
    to characterize complex physics.

    Parameters
    ----------
    atom_data : dictionary of atomic ADAS files (only acd, scd are required; ccd is
        necessary only if n0_by_ne is not 0).
    ne_cm3 : float or array
        Electron density in units of :math:`cm^{-3}`.
    Te_eV : float or array, optional
        Electron temperature in units of eV. If left to None, the Te grid given in the atomic data is used.
    Ti_eV : float or array
        Bulk ion temperature in units of eV, only needed for CX. If left to None, Ti is set equal to Te.
    n0_by_ne: float or array, optional
        Ratio of background neutral hydrogen to electron density. If set to 0, CX is not considered.
    superstages : list or 1D array
        Indices of charge states of chosen ion that should be included. If left empty, all ion stages
        are included. If only some indices are given, these are modeled as "superstages".
    tau_s : float, opt
        Value of the particle residence time [s]. This is a scalar value that can be used to model
        the effect of transport on ionization equilibrium.
        Setting tau=np.inf (default) corresponds to no effect from transport.
    plot : bool, optional
        If True, the atomic relaxation time is plotted as a function of Te. Default is True.
    ax : matplotlib.pyplot Axes instance
        Axes on which to plot if plot=True. If False, new axes are created.
    ls : str, optional
        Line style for plots. Continuous lines are used by default.

    Returns
    -------
    Te : array
        electron temperatures as a function of which the fractional abundances and
        rate coefficients are given.
    fz : array, (space,nZ)
        Fractional abundances across the same grid used by the input ne,Te values.
    rate_coeffs : array, (space, nZ)
        Rate coefficients in units of [:math:`s^{-1}`].

    Examples
    --------
    To visualize relaxation times for a given species:
    >>> atom_data = aurora.atomic.get_atom_data('N', ["scd", "acd"])
    >>> aurora.get_atomic_relax_time(atom_data, [1e14], Te_eV=np.linspace(0.1,200,1000), plot=True)

    To compare ionization balance with different values of ne*tau:
    >>> Te0, fz0, r0 = aurora.get_atomic_relax_time(atom_data, [1e14], Te_eV=np.linspace(0.1,200,1000), tau_s=1e-3, plot=False)
    >>> Te1, fz1, r1 = aurora.get_atomic_relax_time(atom_data, [1e14], Te_eV=np.linspace(0.1,200,1000), tau_s=1e-2, plot=False)
    >>> Te2, fz2, r2 = aurora.get_atomic_relax_time(atom_data, [1e14], Te_eV=np.linspace(0.1,200,1000), tau_s=1e-1, plot=False)
    >>>
    >>> plt.figure()
    >>> for cs in np.arange(fz0.shape[1]):
    >>>     l = plt.plot(Te0, fz0[:,cs], ls='-')
    >>>     plt.plot(Te1, fz1[:,cs], c=l[0].get_color(), ls='--')
    >>>     plt.plot(Te2, fz2[:,cs], c=l[0].get_color(), ls='-.')

    """
    # if input arrays are multi-dimensional, flatten them here and restructure at the end
    _ne = np.ravel(ne_cm3)
    _Te = np.ravel(Te_eV) if Te_eV is not None else None
    _Ti = np.ravel(Ti_eV) if Ti_eV is not None else _Te
    _n0_by_ne = np.ravel(n0_by_ne)

    include_cx = False if not np.any(n0_by_ne) else True

    out = get_cs_balance_terms(atom_data, _ne, _Te, include_cx=include_cx)

    Te, Sne, Rne = out[:3]
    if include_cx:
        # Get an effective recombination rate by summing radiative & CX recombination rates
        Rne += out[3] * _n0_by_ne

    # Enable use of superstages
    if len(superstages):
        _, Rne, Sne, _ = superstage_rates(Rne, Sne, superstages)

    # numerical method that calculates also rate_coeffs
    nion = Rne.shape[1]
    fz = np.zeros((Te.size, nion + 1))
    rate_coeffs = np.zeros(Te.size)

    for it, t in enumerate(Te):
        A = (
            -np.diag(np.r_[Sne[it], 0] + np.r_[0, Rne[it]] + 1.0 / tau_s)
            + np.diag(Sne[it], -1)
            + np.diag(Rne[it], 1)
        )

        N, rate_coeffs[it] = null_space(A)
        fz[it] = N / np.sum(N)

    if np.size(ne_cm3) > 1:
        # re-structure to original array dimensions
        Te = Te.reshape(np.shape(ne_cm3))
        fz = fz.reshape(*np.shape(ne_cm3), fz.shape[1])
        rate_coeffs = rate_coeffs.reshape(np.shape(ne_cm3))

    if plot:
        # Now plot relaxation times
        if ax is None:
            fig, ax = plt.subplots()

        ax.loglog(Te, 1e3 / rate_coeffs, "b")
        ax.set_xlim(Te[0], Te[-1])
        ax.grid("on")
        ax.set_xlabel("T$_e$ [eV]")
        ax.set_ylabel(r"$\tau_\mathrm{relax}$ [ms]")

    return Te, fz, rate_coeffs


class CartesianGrid:
    """Fast linear interpolation for 1D and 2D vector data on equally spaced grids.
    This offers optimal speed in Python for interpolation of atomic data tables such
    as the ADAS ones.

    Parameters
    ----------
    grids: list of arrays, N=len(grids), N=1 or N=2
        List of 1D arrays with equally spaced grid values for each dimension
    values: N+1 dimensional array of values used for interpolation
        Values to interpolate. The first dimension typically refers to different ion stages, for which
        data is provided on the input grids.
        Other dimensions refer to values on the density and temperature grids.
    """

    def __init__(self, grids, values):

        values = np.ascontiguousarray(np.moveaxis(values, 0, -1))
        self.N = values.shape[:-1]

        if len(self.N) > 2:
            raise valueError("Only 1 and 2 dimensional interpolation is supported")

        for g, s in zip(grids, self.N):
            if len(g) != s:
                raise ValueError("wrong size of values array")

        self.eq_spaced_grid = np.all(
            [np.std(np.diff(g)) / np.std(g) < 0.01 for g in grids]
        )
        if self.eq_spaced_grid:
            self.offsets = [g[0] for g in grids]
            self.scales = [(g[-1] - g[0]) / (n - 1) for g, n in zip(grids, self.N)]
        else:
            self.grids = grids

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

        if self.eq_spaced_grid:
            coords = np.array(coords).T
            coords -= self.offsets
            coords /= self.scales
            coords = coords.T
        else:
            # for non-equally spaced grids must be used linear interpolation to map inputs to equally spaced indexes
            coords = [
                np.interp(c, g, np.arange(len(g))) for c, g in zip(coords, self.grids)
            ]

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


def interp_atom_prof(atom_table, xprof, yprof, log_val=False, x_multiply=True):
    r"""Fast interpolate atomic data in atom_table onto the xprof and yprof profiles.
    This function assume that xprof, yprof, x, y, table are all base-10 logarithms,
    and xprof, yprof are equally spaced.

    Parameters
    ----------
    atom_table : list
        object atom_data, containing atomic data from one of the ADAS files.
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

    Notes
    -------
    This function uses `np.log10` and exponential operations to optimize speed, since it has
    been observed that base-e operations are faster than base-10 operations in numpy.
    """
    x = atom_table.logNe
    y = atom_table.logT
    table = atom_table.logdata

    if x_multiply:  # multiplying of logarithms is just adding
        table = table + x  # don't modify original table, create copy

    if (abs(table - table[..., [0]]) < 0.05).all() or xprof is None:
        # 1D interpolation if independent of the last dimension - like SXR radiation data

        reg_interp = CartesianGrid((y,), table[:, :, 0] * np.log(10))
        interp_vals = reg_interp(yprof)

    else:  # 2D interpolation
        # broadcast both variables to the same shape
        xprof, yprof = np.broadcast_arrays(xprof, yprof)

        # perform fast linear interpolation
        reg_interp = CartesianGrid((x, y), table.swapaxes(1, 2) * np.log(10))

        interp_vals = reg_interp(xprof, yprof)

    # reshape to shape(nt,nion,nr)
    interp_vals = interp_vals.swapaxes(0, 1)

    if not log_val:
        # return actual value, not logarithm
        np.exp(interp_vals, out=interp_vals)

    return interp_vals


def gff_mean(Z, Te):
    """
    Total free-free gaunt factor yielding the total radiated bremsstrahlung power
    when multiplying with the result for gff=1.
    Data originally from Karzas & Latter, extracted from STRAHL's atomic_data.f.
    """
    from scipy.constants import e, h, c, Rydberg

    thirteenpointsix = h * c * Rydberg / e

    log_gamma2_grid = [
        -3.000,
        -2.833,
        -2.667,
        -2.500,
        -2.333,
        -2.167,
        -2.000,
        -1.833,
        -1.667,
        -1.500,
        -1.333,
        -1.167,
        -1.000,
        -0.833,
        -0.667,
        -0.500,
        -0.333,
        -0.167,
        0.000,
        0.167,
        0.333,
        0.500,
        0.667,
        0.833,
        1.000,
    ]

    gff_mean_grid = [
        1.140,
        1.149,
        1.159,
        1.170,
        1.181,
        1.193,
        1.210,
        1.233,
        1.261,
        1.290,
        1.318,
        1.344,
        1.370,
        1.394,
        1.416,
        1.434,
        1.445,
        1.448,
        1.440,
        1.418,
        1.389,
        1.360,
        1.336,
        1.317,
        1.300,
    ]

    # set min Te here to 10 eV, because the grid above does not extend to lower temperatures
    Te = np.maximum(Te, 10.0)

    log_gamma2 = np.log10(Z**2 * thirteenpointsix / Te)

    # dangerous/inaccurate extrapolation...
    return np.interp(log_gamma2, log_gamma2_grid, gff_mean_grid)



def impurity_brems(nz, ne, Te, freq="all", cutoff=0.1):
    """Approximate impurity bremsstrahlung in :math:`W/m^3`for a given range
    of frequencies or a specific frequency.

    We apply here a cutoff for Bremsstrahlung at h*c/lambda = cutoff*Te,
    where `cutoff` is an input parameter, conventionally set to 0.1 (default).

    Gaunt factors from :py:func:`~aurora.atomic.gff_mean` are applied.
    NB: recombination is not included.

    Formulation based on Hutchinson's Principles of Plasma Diagnostics,
    p. 196, Eq. (5.3.40).

    Parameters
    ----------
    nz : array (time,nZ,space)
        Densities for each charge state [:math:`cm^{-3}`]
    ne : array (time,space)
        Electron density [:math:`cm^{-3}`]
    Te : array (time,space)
        Electron temperature [:math:`cm^{-3}`]
    freq : float, 1D array, or str
        If a float, calculate bremsstrahlung from all charge states at
        this frequency. If a 1D array, evaluate bremsstrahlung at these wavelengths.
        If set to `all`, then bremsstrahlung is integrated over the whole range from plasma frequency to  cutoff
        Frequencies are expected in units of :math:`s^{-1}`.
    cutoff : float
        Fraction of Te below which bremsstrahlung is set to 0.
        A value of 0.1 is commonly set and is the default.

    Returns
    -------
    array (time,nZ,space):
        Bremsstrahlung for each charge state at the given frequency or, if
        multiple frequences are given (or if `freq='all'`), integrated over frequencies.
        Units of :math:`W/cm^3`.
    """

    Z_imp = nz.shape[1] - 1
    Z = np.arange(Z_imp)[None, :, None] + 1

    gff = gff_mean(Z, Te[:, None])

    # take cutoff frequency to be cutoff*Te
    cut = cutoff * Te * constants.e / (constants.h)

    # plasma frequency (divide by 2pi to have units of Hz)
    fp = np.sqrt( 1e6 * ne * constants.e**2 / (constants.epsilon_0 * constants.m_e)) / (2 * np.pi)
 

    # constant in Hutchinson Eq. 5.3.10
    const = (
        32
        * np.pi**2
        / (3 * np.sqrt(3) * constants.m_e**2 * constants.c**3)
        * (constants.e**2 / (4 * np.pi * constants.epsilon_0)) ** 3
        * np.sqrt(2 * constants.m_e / np.pi)
    )

    # conversion factor to eventually have result in units of W/cm^3 rather than W/m^3
    const *= 1e-6

    a = -constants.h   / (Te * constants.e)
    if freq == 'all':
        intV = (np.exp(a*cut)-np.exp(a*fp))/a
    else:
        freq = np.atleast_1d(freq)[:,None,None]
        intV = np.exp(a* freq)
        # set brems to 0 below plasma frequency and above chosen cutoff frequency
        intV[(freq < fp)|(freq >  cut )] = 0
 
     
    brs = 4* np.pi * Z**2 * nz[:, 1:] * gff * (ne * 1e12 * const / np.sqrt(Te * constants.e) * intV)[... ,None,:]
 
    return  brs



def plot_norm_ion_freq(
    S_z,
    q_prof,
    R_prof,
    imp_A,
    Ti_prof,
    nz_profs=None,
    rhop=None,
    plot=True,
    eps_prof=None,
):
    r"""Compare effective ionization rate for each charge state with the characteristic
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

    """
    nu = np.zeros(
        (S_z.shape[0], S_z.shape[1] - 1)
    )  # exclude neutral states, which have no parallel transport
    for cs in np.arange(nu.shape[1]):
        nu[:, cs] = (
            S_z[:, cs + 1]
            * q_prof
            * R_prof
            * np.sqrt((imp_A * constants.m_p) / (2 * Ti_prof * constants.e))
        )

    if nz_profs is not None:
        # calculate average nu_ioniz_star
        nu_ioniz_star = np.sum(nz_profs[:, 1:] * nu, axis=1) / np.sum(
            nz_profs[:, 1:], axis=1
        )
    else:
        # return normalized ionization rate for each charge state
        nu_ioniz_star = nu

    if plot:
        if rhop is None:
            rhop = np.arange(nu.shape[0])

        fig, ax = plt.subplots()
        if nu_ioniz_star.ndim == 1:
            ax.semilogy(rhop, nu_ioniz_star, label=r"$\nu_{ion}^*$")
        else:
            for cs in np.arange(nu.shape[1]):
                ax.semilogy(rhop, nu_ioniz_star[:, cs], label=f"q={cs+1}")
            ax.set_ylabel(r"$\nu_{ion}^*$")

        ax.set_xlabel(r"$\rho_p$")

        if eps_prof is not None:
            ax.semilogy(rhop, np.sqrt(eps_prof), label=r"$\sqrt{\epsilon}$")

        ax.legend().set_draggable(True)
        ax.set_xlim([0, 1])


def read_adf12(filename, block, Ebeam, ne_cm3, Ti_eV, zeff):
    """Read charge exchange effective emission coefficients from ADAS ADF12 files.

    Files may be automatically downloaded using :py:fun:`~aurora.adas_files.get_adas_file_loc`.

    Parameters
    ----------
    filename : str
        adf12 file name/path
    block : int
        Source block selected
    Ebeam : float
        Energy of the neutral beam population of interest, in units of :math:`eV/amu`.
    ne_cm3 : float or 1D array
        Electron densities at which to evaluate coefficients, in units of :math:`cm^{-3}`.
    Ti_eV : float or 1D array
        Bulk ion temperature at which to evaluate coefficients, in units of :math:`eV`.
    Zeff : float or 1D array
        Effective background charge.

    Returns
    -------
    float or 1D array :
        Interpolated coefficients, units of :math:`cm^3/s`.

    """
    with open(filename, "r") as f:
        nlines = int(f.readline())

        for iline in range(block):
            cer_line = {}
            params = []
            first_line = "0"
            while not first_line[0].isalpha():
                first_line = f.readline()

            cer_line["header"] = first_line
            cer_line["qefref"] = np.float(f.readline()[:63].replace("D", "e"))
            cer_line["parmref"] = np.float_(f.readline()[:63].replace("D", "e").split())
            cer_line["nparmsc"] = np.int_(f.readline()[:63].split())

            for ipar, npar in enumerate(cer_line["nparmsc"]):
                for q in range(2):
                    data = []
                    while npar > len(data):
                        line = f.readline()
                        if len(line) > 63:
                            name = line[63:].strip().lower()
                            cer_line[name] = []
                            if q == 0:
                                params.append(name)

                        values = np.float_(line[:63].replace("D", "E").split())
                        values = values[values > 0]
                        if not len(values):
                            continue
                        data += values.tolist()
                    cer_line[name] = data

    # interpolate in logspace
    lqefref = np.log(cer_line["qefref"])

    lnq = np.zeros(np.broadcast(Ebeam, ne_cm3, Ti_eV, zeff).shape)
    lnq += lqefref * (1 - 4)
    lnq += np.interp(np.log(Ti_eV), np.log(cer_line["tiev"]), np.log(cer_line["qtiev"]))
    lnq += np.interp(
        np.log(ne_cm3), np.log(cer_line["densi"]), np.log(cer_line["qdensi"])
    )
    lnq += np.interp(np.log(Ebeam), np.log(cer_line["ener"]), np.log(cer_line["qener"]))
    lnq += np.interp(np.log(zeff), np.log(cer_line["zeff"]), np.log(cer_line["qzeff"]))

    return np.exp(lnq)


def read_adf21(filename, Ebeam, ne_cm3, Te_eV):
    """Read ADAS ADF21 or ADF22 files.

    ADF21 files contain effective beam stopping/excitation coefficients.
    ADF22 contain effective beam emission/population coefficients.

    Files may be automatically downloaded using :py:fun:`~aurora.adas_files.get_adas_file_loc`.

    Parameters
    ----------
    filename : str
        adf21 or adf22 file name/path
    Ebeam : float
        Energy of the neutral beam, in units of :math:`eV/amu`.
    ne_cm3 : float or 1D array
        Electron densities at which to evaluate coefficients.
    Te_eV : float or 1D array
        Electron temperature at which to evaluate coefficients.

    Returns
    -------
    float or 1D array :
        Interpolated coefficients. For ADF21 files, these have units of :math:`cm^3/s`for ADF21 files.
        For ADF22, they correspond to n=2 fractional abundances.
    """

    with open(filename, "r") as f:
        line = f.readline()
        ref = float(line.split()[1].split("=")[1])
        f.readline()
        line = f.readline()
        nE, nne, Teref = line.split()
        nE, nne = int(nE), int(nne)
        Teref = float(Teref.split("=")[1])
        f.readline()

        E = []
        while len(E) < nE:
            line = f.readline()
            E.extend([float(f) for f in line.split()])
        E = np.array(E)

        ne = []
        while len(ne) < nne:
            line = f.readline()
            ne.extend([float(f) for f in line.split()])
        ne = np.array(ne)
        f.readline()

        Q2 = []
        while len(Q2) < nne * nE:
            line = f.readline()
            Q2.extend([float(f) for f in line.split()])
        Q2 = np.reshape(Q2, (nne, nE))

        f.readline()
        line = f.readline()
        nTe, Eref, Neref = line.split()
        nTe, Eref, Neref = (
            int(nTe),
            float(Eref.split("=")[1]),
            float(Neref.split("=")[1]),
        )

        f.readline()

        Te = []
        while len(Te) < nTe:
            line = f.readline()
            Te.extend([float(f) for f in line.split()])
        Te = np.array(Te)

        f.readline()

        Q1 = []
        while len(Q1) < nTe:
            line = f.readline()
            Q1.extend([float(f) for f in line.split()])
        Q1 = np.array(Q1)

    # clip data in available range to avoid extrapolation
    Ebeam = np.clip(Ebeam, *E[[0, -1]])
    ne_cm3 = np.clip(ne_cm3, *ne[[0, -1]])
    Te_eV = np.clip(Te_eV, *Te[[0, -1]])

    lref = np.log(ref)

    # interpolate on the requested values
    RectInt1 = interp.interp1d(
        np.log(Te), np.log(Q1) - lref, assume_sorted=True, kind="quadratic"
    )
    RectInt2 = interp.RectBivariateSpline(
        np.log(ne), np.log(E), np.log(Q2) - lref, kx=2, ky=2
    )

    adf = RectInt1(np.log(Te_eV)) + RectInt2.ev(np.log(ne_cm3), np.log(Ebeam))
    return np.exp(adf + lref)


def get_natural_partition(ion, plot=True):
    """Identify natural partition of charge states by plotting the variation of ionization energy
    as a function of charge for a given ion, using the ADAS metric :math:`2 (I_{z+1}-I_z)/(I_{z+1}+I_z)`.

    Parameters
    ----------
    ion : str
        Atomic symbol of species of interest.
    plot : bool
        If True, plot the variation of ionization energy.

    Returns
    -------
    q : 1D array
        Metric to identify natural partition.

    Notes
    -----
    A ColRadPy installation must be available for this function to work.
    """

    try:
        # temporarily import this here, until ColRadPy dependency can be set up properly
        import colradpy
    except ImportError:
        raise ValueError(
            "Could not import colradpy. Install this from the Github repo!"
        )

    # find location of files containing NIST data
    colradpy_dist = os.sep.join(colradpy.__file__.split(os.sep)[:-2])
    loc = colradpy_dist + os.sep + "atomic" + os.sep + "nist_energies"

    # find energies for each charge state of interest
    _E_eV = []
    cs = []
    for filename in os.listdir(loc):
        if not filename.startswith("#") and not filename.endswith("~"):
            _ion = filename.split("_")[0]

            if _ion != ion.lower():
                continue
            charge = int(filename.split("_")[1])

            # read energy from file
            with open(loc + os.sep + filename, "r") as f:
                cont = f.readlines()

            # last value is energy in cm^-1
            E_cm_val = float(cont[-1].split(",")[-1].strip("/n"))
            E_eV_val = constants.h * constants.c / constants.e * (E_cm_val * 100.0)

            cs.append(charge)
            _E_eV.append(E_eV_val)

    idx = np.argsort(cs)
    E_eV = np.array(_E_eV)[idx]

    # compute ADAS natural partitioning metric
    q = []
    for i in np.arange(1, len(E_eV) - 1):
        q.append(2 * (E_eV[i + 1] - E_eV[i]) / (E_eV[i + 1] + E_eV[i]))

    if plot:
        fig, ax = plt.subplots()

        ax.plot(np.arange(len(q)), q)
        ax.set_xlabel("Z")
        ax.set_title(r"$2\times (I_{z+1}-I_z)/(I_{z+1}+I_z)$")

        # take running mean over 7 adjacent charge states as indicated by Foster's thesis
        q_mean = np.convolve(q, np.ones(7) / 7, mode="same")
        plt.plot(q_mean)

    return q
