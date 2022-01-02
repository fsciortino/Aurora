'''
Methods related to impurity source functions.
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
import copy,sys
from scipy.constants import m_p, e as q_electron
from scipy.special import erfc

def get_source_time_history(namelist, Raxis_cm, time):
    '''Load source time history based on current state of the namelist.

    Parameters
    ----------
    namelist : dict
        Aurora namelist dictionary. The field namelist['source_type'] specifies how the 
        source function is being specified -- see the notes below.
    Raxis_cm : float
        Major radius at the magnetic axis [cm]. This is needed to normalize the 
        source such that it is treated as toroidally symmetric -- a necessary
        idealization for 1.5D simulations. 
    time : array (nt,), optional
        Time array the source should be returned on.

    Returns
    -------
    source_time_history : array (nt,)
        The source time history on the input time base.

    Notes
    -----
    There are 4 options to describe the time-dependence of the source:

    #. namelist['source_type'] == 'file': in this case, a simply formatted 
    source file, with one time point and corresponding and source amplitude on each
    line, is read in. This can describe an arbitrary time dependence, e.g. 
    as measured from an experimental diagnostic. 

    #. namelist['source_type'] == 'interp': the time history for the source is 
    provided by the user within the 'explicit_source_time' and 'explicit_source_vals'
    fields of the namelist dictionary and this data is simply interpolated.

    #. namelist['source_type'] == 'const': in this case, a constant source 
    (e.g. a gas puff) is simulated. It is recommended to run the simulation for 
    >100ms in order to see self-similar charge state profiles in time. 

    #. namelist['source_type'] == 'step': this allows the creation of a source
    that suddenly appears and suddenly stops, i.e. a rectangular "step". The 
    duration of this step is given by namelist['step_source_duration']. Multiple 
    step times can be given as a list in namelist['src_step_times']; the amplitude
    of the source at each step is given in namelist['src_step_rates']

    #. namelist['source_type'] == 'synth_LBO': this produces a model source from a LBO
    injection, given by a convolution of a gaussian and an exponential. The required 
    parameters in this case are inside a namelist['LBO'] dictionary:
    namelist['LBO']['t_start'], namelist['LBO']['t_rise'], namelist['LBO']['t_fall'], 
    namelist['LBO']['n_particles']. The "n_particles" parameter corresponds to the 
    amplitude of the source (the number of particles corresponding to the integral over
    the source function. 

    '''
    imp = namelist['imp']

    if namelist['source_type'] == 'file':
        # read time history from a simple file with 2 columns
        src_times, src_rates = read_source(namelist['source_file'])
        
    elif namelist['source_type'] == 'interp' and np.ndim(namelist['explicit_source_vals'])==1:
        # user provided time history, only 1D interpolation is needed
        src_times = namelist['explicit_source_time']
        src_rates = namelist['explicit_source_vals']

    elif namelist['source_type'] == 'const':
        # constant source
        src_times = copy.deepcopy(time)
        src_rates = np.ones(len(time)) * namelist['source_rate']
        src_rates[0] = 0.0 # start with 0

    elif namelist['source_type'] == 'step':
        # Create the time-dependent step source
        tbuf = namelist.get('step_source_duration',1e-6)   # e.g. 1.e-6

        # Start with zero source at t=0
        src_times = [0.]
        src_rates = [0.]

        # Form the source with "tbuf" duration to connect the steps
        for i in range(len(namelist['src_step_times'])):
            src_times.append(namelist['src_step_times'][i]-tbuf)
            src_times.append(namelist['src_step_times'][i])

            src_rates.append(src_rates[-1])
            src_rates.append(namelist['src_step_rates'][i])

        # Append the final rate to maximum time
        src_rates.append(src_rates[-1])
        src_times.append(np.max(time))

    elif namelist['source_type'] == 'synth_LBO':
        # use idealized source function
        src_times, src_rates = lbo_source_function(namelist['LBO']['t_start'],
                                                   namelist['LBO']['t_rise'], 
                                                   namelist['LBO']['t_fall'],
                                                   namelist['LBO']['n_particles'])
    else:
        raise ValueError('Unspecified source function time history!')

    source = np.interp(time, src_times, src_rates, left=0,right=0)
    
    # get number of particles per cm and sec
    circ = 2*np.pi*Raxis_cm

    # For ease of comparison with STRAHL, shift source by one time step
    source_time_history = np.r_[source[1:],0]/circ
    #source_time_history =  source/circ
    
    return np.asfortranarray(source_time_history)



def write_source(t, s, shot, imp='Ca'):
    """Write a STRAHL source file. 
        
    This will overwrite any {imp}flx{shot}.dat locally. 
        
    Parameters
    ----------
    t : array of float, (`n`,)
        The timebase (in seconds).
    s : array of float, (`n`,)
        The source function (in particles/s).
    shot : int
        Shot number, only used for saving to a .dat file
    imp : str, optional
        Impurity species atomic symbol

    Returns
    -------
    contents : str
        Content of the source file written to {imp}flx{shot}.dat

    """
    contents = '{.d}\n'.format(len(t),)
    for tv, sv in zip(t, s):
        contents += '    {5.5f}    {5.5e}\n'.format(tv, sv)
        
    with open(f'{imp}flx{shot}.dat', 'w') as f:
        f.write(contents)

    return contents

def read_source(filename):
    '''Read a STRAHL source file from {imp}flx{shot}.dat locally. 
    
    Parameters
    ----------
    filename : str 
        Location of the file containing the STRAHL source file. 
    
    Returns
    -------
    t : array of float, (`n`,)
        The timebase (in seconds).
    s : array of float, (`n`,)
        The source function (#/s).
    '''
    t = []; s = []
    with open(filename,'r') as f:
        lines = f.read()
    for line in lines.split('\n')[1:]:  # first line contains number of lines
        line = line.strip()
        if line != '':
            t.append(float(line.split()[0]))
            s.append(float(line.split()[1]))
    t = np.array(t)
    s = np.array(s)

    return t,s
        
    

def lbo_source_function(t_start, t_rise, t_fall, n_particles=1.0, time_vec=None):
    ''' Model for the expected shape of the time-dependent source function,
    using a convolution of a gaussian and an exponential decay.

    Parameters
    ----------
    t_start : float or array-like [ms]
        Injection time, beginning of source rise. If multiple values are given, they are
        used to create multiple source functions.
    t_rise : float or array-like [ms]
        Time scale of source rise. Similarly to t_start for multiple values.
    t_fall : float or array-like [ms]
        Time scale of source decay.Similarly to t_start for multiple values.
    n_particles : float, opt
        Total number of particles in source. Similarly to t_start for multiple values.
        Defaults to 1.0.
    time_vec : array-like
        Time vector on which to create source function. If left to None,
        use a linearly spaced time vector including the main features of the function.

    Returns
    -------
    time_vec : array
        Times for the source function of each given impurity
    source : array
        Time history of the synthetized source function.
    '''

    t_start = np.atleast_1d(t_start)
    t_start,t_fall,t_rise,n_particles = np.broadcast_arrays(t_start,t_fall/1e3,t_rise/1e3,n_particles)


    if time_vec is None:
        time_vec = np.hstack([np.linspace(ts-3*tr,ts+6*tf,200) for ts,tr,tf in zip(t_start,t_rise,t_fall)])

    source = np.zeros_like(time_vec)

    for ts,tr,tf,N in zip(t_start,t_rise,t_fall,n_particles):
        tind = slice(*time_vec.searchsorted([ts-3*tr,ts+6*tf]))
        T = time_vec[tind]
        source[tind] = np.exp((1 - 4* tf*(T-ts)/tr**2)/(4*(tf/tr)**2))*(erfc((T-ts)/tr - 1/(2*tf/tr)) - 2)

        # scale source to correspond to the given total number of particles
        source[tind]*= N/np.trapz(source[tind],T)

        # ensure that source function ends with 0 to avoid numerical issues
        source[tind][[0,-1]] = 0


    return time_vec, source




def get_radial_source(namelist, rvol_grid, pro_grid, S_rates, Ti_eV=None):
    '''Obtain spatial dependence of source function.

    If namelist['source_width_in']==0 and namelist['source_width_out']==0, the source
    radial profile is defined as an exponential decay due to ionization of neutrals. This requires
    S_rates, the ionization rate of neutral impurities, to be given with S_rates.shape=(len(rvol_grid),len(time_grid))

    If namelist['imp_source_energy_eV']<0, the neutrals speed is taken as the thermal speed based
    on Ti_eV, otherwise the value corresponding to namelist['imp_source_energy_eV'] is used.

    Parameters
    ----------
    namelist : dict
        Aurora namelist. Only elements referring to the spatial distribution and energy of 
        source atoms are accessed. 
    rvol_grid : array (nr,)
        Radial grid in volume-normalized coordinates [cm]
    pro_grid : array (nr,)
        Normalized first derivatives of the radial grid in volume-normalized coordinates. 
    S_rates : array (nr,nt)
        Ionization rate of neutral impurity over space and time.
    Ti_eV : array, optional (nt,nr)
        Background ion temperature, only used if source_width_in=source_width_out=0.0 and 
        imp_source_energy_eV<=0, in which case the source impurity neutrals are taken to 
        have energy equal to the local Ti [eV]. 

    Returns
    -------
    source_rad_prof : array (nr,nt)
        Radial profile of the impurity neutral source for each time step.
    '''
    r_src = namelist['rvol_lcfs'] + namelist['source_cm_out_lcfs']
    nt = S_rates.shape[1]
    try:
        # TODO: invert order of dimensions of Ti_eV...
        assert S_rates.shape==Ti_eV.T.shape
    except AssertionError as msg:
        raise AssertionError(msg)
    
    source_rad_prof = np.zeros_like(S_rates)

    # find index of radial grid vector that is just greater than r_src
    i_src = rvol_grid.searchsorted(r_src)-1
    # set source to be inside of the wall
    i_src = min(i_src, len(rvol_grid)-1)

    if (namelist['source_width_in'] < 0. and namelist['source_width_out'] < 0.):
        # point source
        source_rad_prof[i_src] = 1.0

    # source with FWHM=source_width_in inside and FWHM=source_width_out outside
    if namelist['source_width_in']>0. or namelist['source_width_out']>0.:

        if namelist['source_width_in']> 0.:
            ff = np.log(2.)/namelist['source_width_in']**2
            source_rad_prof[:i_src] = np.exp(-(rvol_grid[:i_src]-rvol_grid[i_src])**2*ff)[:,None]

        if namelist['source_width_out']>0.:
            ff = np.log(2.)/namelist['source_width_out']**2
            source_rad_prof[i_src:] = np.exp(-(rvol_grid[i_src:]-rvol_grid[i_src])**2*ff)[:,None]


    # decay of neutral density with exp(-Int( [ne*S+dv/dr]/v ))
    if namelist['source_width_in']==0 and namelist['source_width_out']==0:
        #neutrals energy
        if namelist['imp_source_energy_eV'] > 0: 
            E0 = namelist['imp_source_energy_eV']*np.ones_like(rvol_grid)
        else:
            if Ti_eV is not None:
                E0 = copy.deepcopy(Ti_eV)
            else:
                raise ValueError('Could not compute a valid energy of injected ions!')

        # import here to avoid issues with omfit_commonclasses during docs and package creation
        from omfit_classes.utils_math import atomic_element

        # velocity of neutrals [cm/s]
        out = atomic_element(symbol=namelist['imp'])
        spec = list(out.keys())[0]
        imp_ion_A = int(out[spec]['A'])
        v = - np.sqrt(2.*q_electron*E0/(imp_ion_A*m_p))*100 #cm/s
 

        # integration of ne*S for atoms and calculation of ionization length for normalizing neutral density
        source_rad_prof[i_src]= -0.0625*S_rates[i_src]/pro_grid[i_src]/v[i_src]   #1/16
        for i in np.arange(i_src-1,0,-1):
            source_rad_prof[i]=source_rad_prof[i+1]+0.25*(
                S_rates[i+1]/pro_grid[i+1]/v[i+1] + S_rates[i]/pro_grid[i]/v[i])

        # prevents FloatingPointError: underflow encountered
        source_rad_prof[1:i_src] = np.exp(np.maximum(source_rad_prof[1:i_src],-100))

        # calculate relative density of neutrals
        source_rad_prof[1:i_src] *= (rvol_grid[i_src]*v[i_src]/rvol_grid[1:i_src]/v[1:i_src])[:,None]
        source_rad_prof[i_src]=1.0

        # remove promptly redeposited ions
        if namelist['prompt_redep_flag']:
            omega_c = 1.602e-19/1.601e-27*namelist['Baxis']/namelist['main_ion_A']  
            dt = (rvol_grid[i_src]-rvol_grid)/v
            pp = dt*omega_c
            non_redep =  pp**2/(1.+pp**2)
            source_rad_prof*=  non_redep


    # total ion source
    pnorm = np.pi*np.sum(source_rad_prof*S_rates*(rvol_grid/pro_grid)[:,None],0)  # sum over radius
    
    # neutral density for influx/unit-length = 1/cm
    source_rad_prof /= pnorm
    
    # broadcast in right shape if time averaged profiles are used
    source_rad_prof = np.broadcast_to(source_rad_prof, (source_rad_prof.shape[0], nt))

    return source_rad_prof

