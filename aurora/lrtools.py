'''Tools to analyze spectroscopic line ratios.
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

import numpy as np
import copy
from scipy import optimize
import matplotlib.pyplot as plt
plt.ion()

from . import atomic
from . import radiation

class spline_sum:
    def __init__(self, *args):
        self.res = {}
        for i,arg in enumerate(args):
            self.res[i] = arg

    def ev(self, val1, val2):
        res = 10**self.res[0].ev(val1, val2)
        for i in np.arange(1,len(self.res)):
            res += 10**self.res[i].ev(val1, val2)
        return np.log10(res)
                
class tec:
    ''' calculate total line emission coefficient (at ioniz eqm).
    NB: we use a class rather than a function so that this remains immutable 
    over the for-loop. '''
    def __init__(self, exc, rec, atom_data, cs):
        self.exc = exc
        self.rec = rec
        self.atom_data = atom_data
        self.cs = cs

    def __call__(self, ne_cm3, Te_eV):
        '''Definition of total line emission coefficient for current line signal'''
        vals_exc = 10**self.exc.ev(np.log10(ne_cm3),np.log10(Te_eV))

        if (self.atom_data is not None) and (self.rec is not None):
            # combine exc and rec as total emission coeff
            _Te, fz = atomic.get_frac_abundances(
                self.atom_data, ne_cm3, Te_eV, plot=False
            )
            vals_rec = 10**self.rec.ev(np.log10(ne_cm3),np.log10(Te_eV))
            tec = fz[...,self.cs] * vals_exc + fz[...,self.cs+1] * vals_rec
        else:
            # only excitation included
            tec = vals_exc
        return tec
            
def get_line_ratio_ne_Te(data, adf15_path, cs = 1,
                         atom_data = None, fixed_ne_cm3 = None, p0 = None,
                         min_ne=1e13, max_ne=1e15, min_Te=0.1, max_Te=50):
    '''Generalized method to infer ne and Te from an arbitrary number of spectral lines.
    
    Parameters
    ----------
    data : dict
        Dictionary containing signal values and ISEL/block numbers in the ADF15 file
        in order to model emissivity. See the examples below.
    adf15_path : str
        Path to relevant ADF15 file.
    cs : int
        Charge state of interest. Only used if recombination is included (requires
        atomic data to be provided).
    atom_data : dict
        Dictionary containing ionization and recombination rates, in the format given
        by Aurora, e.g. `atom_data = aurora.atomic.get_atom_data('N', ["scd", "acd"])`
    fixed_ne_cm3 : float
        If provided, this value is used to fix the electron density.
    p0 : list (2,) or (1,)
        Guesses for ne and Te. If `fixed_ne_cm3` is provided, then only one element is
        expected (for Te).
    min_ne, max_ne : floats
        Minimum and maximum values of electron density
    min_Te, max_Te : floats
        Minimum and maximum values of electron temperature

    Examples:
    ---------
    # load some AUG data (replace values for different application)
    shot = 41034; t0 = 2.0; t1 = 3.0
    quants = ['N_1_4041','N_1_3995']
    xvl = div_imp_spec(shot, (t0+t1)/2.)
    xvl.load_data(t0, t1, quants=quants, num_t=1)

    # values and uncertainties of N1+ line intensities (phot/m^2/s/steradian)
    idx_4041 = xvl.data_expt['quants'].index('N_1_4041')
    idx_3995 = xvl.data_expt['quants'].index('N_1_3995')
    val_4041 = xvl.data_expt['sig'][idx_4041]
    unc_4041 = xvl.data_expt['sig_unc'][idx_4041]
    val_3995 = xvl.data_expt['sig'][idx_3995]
    unc_3995 = xvl.data_expt['sig_unc'][idx_3995]

    data = {
        'N_1_4041': {
            'isel': [[21,71]],
            'val': val_4041,
            'unc': unc_4041
            },
        'N_1_3995': {
            'isel': [[15,65]],
            'val': val_3995,
            'unc': unc_3995,
            }
        }

    atom_data = aurora.atomic.get_atom_data('N', ["scd", "acd"])
    ne, ne_unc, Te, Te_unc = get_line_ratio_ne_Te(data, cs=1, atom_data=atom_data)
    '''
    if p0 is None:
        # optimization guess
        p0 = [1, 5] if fixed_ne_cm3 is None else [5]
    else:
        p0 = list(p0)
    if (fixed_ne_cm3 is None) and (len(p0)!=2):
        raise ValueError(
            'The optimization guess must have 2 elements when inferring both ne and Te!')

    # only with ioniz/recom rates we can calculate fractional abundances at ioniz eqm
    include_rec = False if atom_data is None else True
        
    # normalize all signals to mean value of first line signal
    data_norm = copy.deepcopy(data)
    line0 = list(data.keys())[0]
    norm_val = np.nanmean(data[line0]['val'])

    # for each line, load excitation and recombination interpolation functions
    vals = []; uncs = []; tec_funs = []
    for l,line in enumerate(data_norm):
        trs = radiation.read_adf15(adf15_path)
        trans = data_norm[line]['isel'][0]
        if trans[0] is not None:
            exc = trs[trs['isel']==trans[0]]['log10 PEC fun'].iloc[0]
        else:
            exc = None
        if include_rec and trans[1] is not None:
            rec = trs[trs['isel']==trans[1]]['log10 PEC fun'].iloc[0]
        else:
            rec = None

        if len(data_norm[line]['isel'])!=1:
            # sum contributions for any other transition to the same line
            for trans in data_norm[line]['isel'][1:]:
                if trans[0] is not None:
                    _exc = trs[trs['isel']==trans[0]]['log10 PEC fun'].iloc[0]
                    exc = spline_sum(exc, _exc)
                if include_rec and trans[1] is not None:
                    _rec = trs[trs['isel']==trans[1]]['log10 PEC fun'].iloc[0]
                    rec = spline_sum(rec, _rec)

        tec_funs.append(tec(exc, rec, atom_data, cs))
        vals.append(data_norm[line]['val'])
        uncs.append(data_norm[line]['unc'])

    def min_fun(x, vals, uncs, tec_funs):
        '''Minimization helper, allowing an arbitrary number of signals to be given.
        All are normalized to the first (0th) signal.

        Inputs: ne in units of 1e13*cm^-3, Te in eV.
        '''
        if fixed_ne_cm3 is not None:
            # take ne from given argument
            ne_cm3, Te_eV = x[0], fixed_ne_cm3
        else:
            ne_cm3, Te_eV = x[0]*1e13, x[1]
        tec0 = tec_funs[0](ne_cm3, Te_eV)
        
        res = 0
        for val,unc,tec_fun in zip(vals[1:],uncs[1:],tec_funs[1:]):
            # uncertainty on ratio with first signal
            unc0 = np.sqrt((1/vals[1])**2*uncs[0]**2 + (vals[0]/vals[1]**2)**2*uncs[1]**2)
            # metric: chi^2
            res += np.abs((val/vals[0] - tec_fun(ne_cm3, Te_eV)/tec0)/unc0)
        return res

    # bounds in order mins, maxs
    if fixed_ne_cm3 is None:
        # optimize for ne and Te
        res_lsq = optimize.least_squares(lambda x: min_fun(x, vals, uncs, tec_funs),
                                         p0, bounds=[(min_ne/1e13, min_Te),(max_ne/1e13, max_Te)],
                                         loss='linear') # linear: standard least squares
    else:
        # optimize only for Te
        res_lsq = optimize.least_squares(lambda x: min_fun(x, vals, uncs, tec_funs),
                                         p0, bounds=[(min_Te),(max_Te)],
                                         loss='linear') # linear: standard least squares
        
    # robust way of obtaining uncertainties from output jacobian
    U, s, Vh = np.linalg.svd(res_lsq.jac, full_matrices=False)
    tol = np.finfo(float).eps*s[0]*max(res_lsq.jac.shape)
    w = s > tol
    cov = (Vh[w].T/s[w]**2) @ Vh[w]  # robust covariance matrix
    perr = np.sqrt(np.diag(cov))     # 1 sigma uncertainty on fitted parameters

    if fixed_ne_cm3:
        ne_cm3 = fixed_ne_cm3
        Te_eV = res_lsq['x'][0]
        ne_cm3_unc = 0.
        Te_eV_unc = perr[0]
    else:
        ne_cm3 = res_lsq['x'][0]*1e13
        Te_eV = res_lsq['x'][1]
        ne_cm3_unc = perr[0]*1e13
        Te_eV_unc = perr[1]
        
    return ne_cm3, ne_cm3_unc, Te_eV, Te_eV_unc
    
        
def plot_line_ratio_space(imp, cs, fun1_exc, fun2_exc, fun1_rec=None, fun2_rec=None,
                          min_ne=1e13, max_ne=1e15, min_Te=0.1, max_Te=50):
    '''Make a contour plot showing expected values of some line ratios over the (ne,Te) space.
    Note that this only considers ionization equilibrium, i.e. no considerations on transport
    are made.

    Parameters
    ----------
    imp : str
        Atomic symbol of chosen species.
    cs : int
        Integer representing the charge state of interest for excitation.
        This is only used if both excitation and recombination components are considered.
    fun1_exc : scipy.interpolate.RectBivariateSpline instance
        Interpolation function for the excitation component of the first line, 
        as given by :py:fun`~aurora.radiation.read_adf15`
    fun2_exc : scipy.interpolate.RectBivariateSpline instance
        Interpolation function for the excitation component of the second line, 
        as given by :py:fun`~aurora.radiation.read_adf15`
    fun1_rec : scipy.interpolate.RectBivariateSpline instance
        Optional; recombination component of the first line.
    fun2_rec : scipy.interpolate.RectBivariateSpline instance
        Optional; recombination component of the second line.
    min_ne, max_ne : floats
        Bounds of ne grid [:math:`cm^{-3}`]
    min_Te, max_Te : floats
        Bounds of Te grid [:math:`eV`]

    MWE:
    ---
    trs_NII = aurora.radiation.read_adf15('pec98#n_ssh_pju#n1.dat')
    exc3995 = trs_NII[trs_NII['isel']==15]['log10 PEC fun'].iloc[0]
    exc4042 = trs_NII[trs_NII['isel']==21]['log10 PEC fun'].iloc[0]
    plot_line_ratio_space('N',1, exc4042, exc3995)
    '''
    # ne and Te grids  
    ne_grid = np.linspace(min_ne, max_ne, 300)
    Te_grid = np.linspace(min_Te, max_Te, 400)
    NE, TE = np.meshgrid(ne_grid, Te_grid)

    # fractional abundances at ionization equilibrium
    atom_data = atomic.get_atom_data(imp, ["scd", "acd"])
    _Te, fz = atomic.get_frac_abundances(
        atom_data, NE, TE, plot=False
    )

    # excitation components
    vals1_exc = 10**fun1_exc.ev(np.log10(NE),np.log10(TE))
    vals2_exc = 10**fun2_exc.ev(np.log10(NE),np.log10(TE))

    # recombination components (not used if not provided)
    if fun1_rec is None:
        vals1_rec = 0.0
        fz = np.ones_like(fz)
    else:
        vals1_rec = 10**fun1_rec.ev(np.log10(NE), np.log10(TE))
    if fun2_rec is None:
        vals2_rec = 0.0
        fz = np.ones_like(fz)
    else:
        vals2_rec = 10**fun2_rec.ev(np.log10(NE), np.log10(TE))

    # combine exc and rec component via fractional abundances at ioniz eqm
    # ASSUMES NO TRANSPORT!
    tec1 = fz[...,cs] * vals1_exc + fz[...,cs+1] * vals1_rec
    tec2 = fz[...,cs] * vals2_exc + fz[...,cs+1] * vals2_rec
    
    fig,ax = plt.subplots()
    cntr = ax.contourf(ne_grid, Te_grid, tec1/tec2, levels=50)
    cbar = fig.colorbar(cntr)
    cbar.set_label('Line ratio')
    ax.set_xlabel(r'$n_e$ [cm$^{-3}$]')
    ax.set_ylabel(r'$T_e$ [eV]')
    ax.set_xscale('log')
