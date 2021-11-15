''' This script contains a number of functions used for interpolation of kinetic profiles and D,V profiles in STRAHL.
Refer to the STRAHL manual for details.
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

from scipy.interpolate import interp1d
import numpy as np


def funct(params,rLCFS,r):
    ''' Function 'funct' in STRAHL manual

    The "params" input is broken down into 6 arguments:
        y0 is core offset
        y1 is edge offset
        y2 (>y0, >y1) sets the gaussian amplification
        p0 sets the width of the inner gaussian
        P1 sets the width of the outer gaussian
        p2 sets the location of the inner and outer peaks
    '''
    params = np.reshape(params,(-1,6))
    out = []
    for param in params:
        y0,y1,y2,p0,p1,p2 = param
        r1 = p2*rLCFS

        rin = r[r<=r1]
        rout = r[r>r1]

        yin  = y0 + (y2-y0) * np.exp(np.maximum(-(rin -r1)**2/p0**2,-50))
        yout = y1 + (y2-y1) * np.exp(np.maximum(-(rout-r1)**2/p1**2,-50))
        out.append(np.concatenate((yin,yout)))

    return np.array(out)


def funct2(params,rLCFS,r):
    '''Function 'funct2' in STRAHL manual.   
    '''
    params_1,params_2 = np.swapaxes(np.reshape(params,(-1,2,6)),0,1)

    funct_1 = funct(params_1,rLCFS,r)
    funct_2 = funct(params_2,rLCFS,r)
    return funct_1+funct_2


def exppol0(params,d,rLCFS,r):
    rho = r[:,None]/rLCFS
    d = np.array(d)/rLCFS
    params = np.array(params).T
    idx = np.searchsorted(r,rLCFS)
    core = params[0]*np.exp(params[1]*rho[:idx]**2 +
                            params[2]*rho[:idx]**4 +
                            params[3]*rho[:idx]**6 +
                            params[4]*rho[:idx]**8)
    edge = core[-1]*np.exp(-(rho[idx:]-rho[idx-1])/d)
    return np.concatenate([core,edge]).T

def exppol1(params,d,rLCFS,r):
    rho = r[:,None]/rLCFS
    d = np.array(d)/rLCFS
    params = np.array(params).T
    idx = np.searchsorted(r,rLCFS)
    core = params[0]*np.exp(params[1]*rho[:idx]**2 +
                            params[2]*rho[:idx]**3 +
                            params[3]*rho[:idx]**4 +
                            params[4]*rho[:idx]**5)
    edge = core[-1]*np.exp(-(rho[idx:]-rho[idx-1])/d)
    return np.concatenate([core,edge]).T


def ratfun(params,d,rLCFS,r):
    rho = r[:,None]/rLCFS
    d = np.array(d)/rLCFS
    params = np.array(params).T
    idx = np.searchsorted(r,rLCFS)
    core = params[0] * ( (1.0 - params[1])*(1.0-rho[:idx]**params[2])**params[3] + params[1])
    edge = core[-1]*np.exp(-(rho[idx:]-rho[idx-1])/d)
    return np.concatenate([core,edge]).T

def interp_quad(x,y,d,rLCFS,r):
    '''Function 'interp' used for kinetic profiles.
    '''
    f = interp1d(x,np.log(y), kind='quadratic', assume_sorted=True, copy=False)
    idx = np.searchsorted(r,rLCFS)
    core = np.exp(f(np.clip(r[:idx]/rLCFS,0, x[-1])))
    edge =  core[...,[idx-1]]*np.exp(-np.outer(1./np.asarray(d),r[idx:]-r[idx-1]))

    return np.concatenate([core,edge],axis=-1)

def interpa_quad(x,y,rLCFS,r):
    '''Function 'interpa' used for kinetic profiles
    '''
    f = interp1d(x,np.log(y),bounds_error=False, kind='quadratic',
                             assume_sorted=True, copy=False)
    return np.exp(f(np.minimum(r/rLCFS, x[-1])))


def interp(x,y,rLCFS,r):
    '''Function 'interp' used in STRAHL for D and V profiles.
    '''
    f = interp1d(x,y,fill_value='extrapolate', assume_sorted=True, copy=False)
    return f(r/rLCFS)
