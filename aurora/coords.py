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

import numpy as np,sys,os
from scipy.interpolate import interp1d, RectBivariateSpline
from . import grids_utils
import copy

def get_rhop_RZ(R,Z, geqdsk):
    '''Find rhop at every R,Z [m] based on the equilibrium in the geqdsk dictionary.
    '''
    return RectBivariateSpline(geqdsk['AuxQuantities']['Z'],
                               geqdsk['AuxQuantities']['R'],
                               geqdsk['AuxQuantities']['RHOpRZ']).ev(Z,R)



def vol_average(quant, rhop, method='omfit', geqdsk=None, device=None, shot=None, time=None, return_geqdsk=False):
    '''Calculate the volume average of the given radially-dependent quantity on a rhop grid. 

    Parameters
    ----------
    quant : array, (space, ...)
        quantity that one wishes to volume-average. The first dimension must correspond to space,
        but other dimensions may be exist afterwards.
    rhop : array, (space,)
        Radial rhop coordinate in cm units. 
    method : {'omfit','fs'}
        Method to evaluate the volume average. The two options correspond to the way to compute
        volume averages via the OMFIT fluxSurfaces classes and via a simpler cumulative sum in r_V 
        coordinates. The methods only slightly differ in their results. Note that 'omfit' will fail if 
        rhop extends beyond the LCFS, while method 'fs' can estimate volume averages also into the SOL.
        Default is method='omfit'. 
    geqdsk : output of the :py:class:`omfit_classes.omfit_eqdsk.OMFITgeqdsk` class, postprocessing the EFIT geqdsk file
        containing the magnetic geometry. If this is left to None, the function internally tries to fetch
        it using MDS+ and `omfit_classes.omfit_eqdsk`. In this case, device, shot and time to fetch the equilibrium 
        are required. 
    device : str
        Device name. Note that routines for this device must be implemented in `omfit_classes.omfit_eqdsk` for this to work. 
    shot : int
        Shot number of the above device, e.g. 1101014019 for C-Mod.
    time : float
        Time at which equilibrium should be fetched in units of ms. 
    return_geqdsk : bool
        If True, `omfit_classes.omfit_eqdsk` dictionary is also returned

    Returns
    -------
    quant_vol_avg : array, (space, ...)
        Volume average of the quantity given as an input, in the same units as in the input.
        If extrapolation beyond the range available from EFIT volume averages over a shorter section
        of the radial grid will be attempted. This does not affect volume averages within the LCFS. 
    geqdsk : dict
        Only returned if return_geqdsk=True. 
    '''
    if time is not None and np.mean(time)<1e2:
        print('Input time was likely in the wrong units! It should be in [ms]!')
    if np.max(rhop)>1.0:
        print("Input rhop goes beyond the LCFS! Results may not be meaningful (and can only be obtained via method=='fs').")
        
    if geqdsk is None:
        # Fetch device geqdsk from MDS+ and post-process it using the OMFIT geqdsk format.
        try:
            from omfit_classes import omfit_eqdsk
        except:
            raise ValueError('Could not import omfit_classes.omfit_eqdsk! Install with pip install omfit_classes')
        geqdsk = omfit_eqdsk.OMFITgeqdsk('').from_mdsplus(
            device=device, shot=shot, time=time, SNAPfile='EFIT01',
            fail_if_out_of_range=False,time_diff_warning_threshold=20)

    if method=='fs':
        # obtain mapping between rhop and r_V coordinates
        rho_pol, r_V_ = grids_utils.get_rhopol_rvol_mapping(geqdsk)
        
        # find r_V corresponding to input rhop (NB: extrapolation in the far SOL should be used carefully)
        r_V = interp1d(rho_pol, r_V_, bounds_error=False)(rhop)
        
        # use convenient volume averaging in r_V coordinates
        if np.any(np.isnan(r_V)):
            print('Ignoring all nan points! These may derive from an attempted extrapolation or from nan inputs')
            
        vol_avg = rV_vol_average(quant[~np.isnan(r_V)], r_V[~np.isnan(r_V)])
        
    elif method=='omfit':
        # use fluxSurfaces classes from OMFIT
        rhopp = np.sqrt(geqdsk['fluxSurfaces']['geo']['psin'])
        quantp = interp1d(rhop, quant, bounds_error=False,fill_value='extrapolate')(rhopp)
        vol_avg = geqdsk['fluxSurfaces'].volume_integral(quantp)

    else:
        raise ValueError('Input method for volume average could not be recognized')

    if return_geqdsk:
        return vol_avg, geqdsk
    else:
        return vol_avg


def rV_vol_average(quant,r_V):
    '''Calculate a volume average of the given radially-dependent quantity on a r_V grid.
    This function makes useof the fact that the r_V radial coordinate, defined as 
    :math:`r_V = \sqrt{ V / (2 \pi^2 R_{axis} }`,
    maps shaped volumes onto a circular geometry, making volume averaging a trivial 
    operation via
    :math:`\langle Q \rangle = \Sigma_i Q(r_i) 2 \pi \ \Delta r_V`
    where :math:`\Delta r_V` is the spacing between radial points in r_V.
    
    Note that if the input r_V coordinate is extended outside the LCFS,
    this function will return the effective volume average also in the SOL, since it is 
    agnostic to the presence of the LCFS. 

    Parameters
    ----------
    quant : array, (space, ...)
        quantity that one wishes to volume-average. The first dimension must correspond to r_V,
        but other dimensions may be exist afterwards. 
    r_V : array, (space,)
        Radial r_V coordinate in cm units. 

    Returns
    -------
    quant_vol_avg : array, (space, ...)
        Volume average of the quantity given as an input, in the same units as in the input
    '''
    quant_vol_avg = 2.*np.cumsum(quant*r_V*np.diff(r_V,prepend=0.0)) / (r_V[-1]**2 )

    return quant_vol_avg


    
def rad_coord_transform(x, name_in, name_out, geqdsk):
    """Transform from one radial coordinate to another. Note that this coordinate conversion is only
    strictly valid inside of the LCFS. A number of common coordinate nomenclatures are accepted, but
    it is recommended to use one of the coordinate names indicated in the input descriptions below.

    Parameters
    ----------
    x: array or float
        input x coordinate
    name_in: str
        input x coordinate name ('rhon','rvol','rhop','rhov','Rmid','rmid','r/a')
    name_out: str
        input x coordinate ('rhon','psin','rvol', 'rhop','rhov','Rmid','rmid','r/a')
    geqdsk: dict
        gEQDSK dictionary, as obtained from the omfit-eqdsk package. 
    
    Returns
    -------
    array
        Conversion of `x` input for the requested radial grid coordinate.
    """
    if name_in == name_out:
        return x
    x = copy.deepcopy(x)

    # avoid confusion with name conventions
    conventions = {'rvol':'rvol', 'r_vol':'rvol', 'r_V':'rvol',
                   'rhon':'rhon', 'rho_tor':'rhon', 'rho_pol':'rhop', 'rhop':'rhop',
                   'r/a':'r/a', 'roa':'r/a', 'rhov':'rhov', 'rho_V':'rhov', 'rho_v':'rhov',
                   'Rmid':'Rmid', 'R_mid':'Rmid', 'rmid':'rmid', 'r_mid':'rmid'}
    name_in = conventions[name_in]
    name_out = conventions[name_out]
        
    if 'rvol' not in geqdsk['fluxSurfaces']['geo']:
        R0 = geqdsk['RMAXIS']
        eq_vol = geqdsk['fluxSurfaces']['geo']['vol']
        rvol = np.sqrt(eq_vol/(2*np.pi**2*R0))
        geqdsk['fluxSurfaces']['geo']['rvol'] = rvol

    # sqrt(norm. tor. flux)
    rhon_ref = geqdsk['fluxSurfaces']['geo']['rhon']
    # norm. pol. flux
    psin_ref = geqdsk['fluxSurfaces']['geo']['psin']
    # sqrt(norm. pol. flux)
    rhop_ref = np.sqrt(psin_ref)
    # volume radius
    rvol = geqdsk['fluxSurfaces']['geo']['rvol']
    # R at midplane
    Rmid = geqdsk['fluxSurfaces']['midplane']['R']
    # r at midplane
    R0 = geqdsk['fluxSurfaces']['R0']
    rmid = Rmid - R0

    # Interpolate to transform coordiantes
    if name_in == 'rhon':
        coord_in = rhon_ref
    elif name_in == 'rhop':
        coord_in = rhop_ref
    elif name_in == 'rvol':
        coord_in = rvol
    elif name_in == 'rhov':
        rvol_lcfs = np.interp(1, rhon_ref, rvol)
        coord_in = rvol/rvol_lcfs
    elif name_in == 'Rmid':
        coord_in = rmid # use rmid since it starts at 0.0, making interpolation easier
        x -= R0 # make x represent a rmid value
    elif name_in == 'rmid':
        coord_in = rmid
    elif name_in == 'r/a':
        rmid_lcfs = np.interp(1, rhon_ref, rmid)
        coord_in = rmid/rmid_lcfs
    else:
        raise ValueError('Input coordinate was not recognized!')

    if name_out == 'rhon':
        coord_out = rhon_ref
    elif name_out == 'psin':
        coord_out = psin_ref
    elif name_out == 'rhop':
        coord_out = rhop_ref
    elif name_out == 'rvol':
        coord_out = rvol
    elif name_out == 'rhov':
        rvol_lcfs = np.interp(1, rhon_ref, rvol)
        coord_out = rvol/rvol_lcfs
    elif name_out == 'Rmid':
        coord_out = rmid # use rmid since it starts at 0.0, making interpolation easier
    elif name_out == 'rmid':
        coord_out = rmid
    elif name_out == 'r/a':
        rmid_lcfs = np.interp(1, rhon_ref, rmid)
        coord_out = rmid/rmid_lcfs
    else:
        raise ValueError('Output coordinate was not recognized!')

    # trick for better extrapolation
    ind0 = coord_in == 0
    out = np.interp(x, coord_in[~ind0], coord_out[~ind0]/coord_in[~ind0])*x

    if (x==coord_in[0]).any() and np.sum(ind0):
        x0 = x == coord_in[0]
        out[x0] = coord_out[ind0]  # give exact magnetic axis

    if name_out=='Rmid':   # interpolation was done on rmid rather than Rmid
        out += R0

    return out

        

