'''
Methods related to impurity source functions.

sciortino, 2020
'''

import numpy as np
import copy
from scipy.constants import m_p, e as q_electron
from omfit_commonclasses.utils_math import atomic_element

def get_aurora_source(namelist, time=None):
    '''Load source function based on current state of the namelist.

    The "time" argument is only needed for time-dependent sources
    '''
    imp = namelist['imp']

    if namelist['source_type'] == 'file':
        src_times, src_rates = read_source(namelist['source_file'])  

    elif namelist['source_type'] == 'const':
        src_times = copy.deepcopy(time)
        src_rates = np.ones(len(time)) * namelist['Phi0']
        src_rates[0] = 0.0 # start with 0

    elif namelist['source_type'] == 'step':
        # Create the time-dependent step source
        tbuf = namelist['step_source_duration']   # e.g. 1.e-6

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
                                                   namelist['LBO']['t_rise'], namelist['LBO']['t_fall'], namelist['LBO']['t_fall'],
                                                   namelist['LBO']['n_particles'])
    else:
        raise ValueError('Unspecified source function time history!')
        
    source = np.interp(time,src_times, src_rates, left=0,right=0)
    circ = 2*np.pi*namelist['Raxis']*100   #cm

    # number of particles per cm and sec
    source_function = np.r_[source[1:],0]/circ #NOTE source in STRAHL is by one timestep shifted

    return source_function



def write_source(t, s, shot, imp='Ca'):
    """Write a STRAHL source file. 
        
    This will overwrite any {imp}flx{shot}.dat locally. 
        
    Args:
        t : array of float, (`n`,)
            The timebase (in seconds).
        s : array of float, (`n`,)
            The source function (in particles/s).
        shot : int
            Shot number, only used for saving to a .dat file
        imp : str, optional
            Impurity species atomic symbol

    Returns:
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
    
    Args:
        filename : str 
            Location of the file containing the STRAHL source file. 
    
    Returns:
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
        if line is not '':
            t.append(float(line.split()[0]))
            s.append(float(line.split()[1]))
    t = np.array(t)
    s = np.array(s)

    return t,s
        
    

def lbo_source_function(t_start, t_rise, t_fall, n_particles=1.0, time_vec=None):
    ''' Model for the expected shape of the time-dependent source function,
    using a convolution of a gaussian and an exponential decay.

    Args:
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

    Returns:
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




def get_radial_source(namelist, radius_grid, S, pro, Ti=None):
    ''' Obtain spatial dependence of source function.

    If namelist['source_width_in']==0 and namelist['source_width_out']==0, the source
    radial profile is defined as an exponential decay due to ionization of neutrals. This requires
    S, the ionization rate of neutral impurities, to be given with S.shape=(len(radius_grid),)

    If axkopt=True, the neutrals speed is taken as the thermal speed based on Ti, otherwise
    the value corresponding to the namelist['imp_energy'] energy is used.

    This funtion reproduces the functionality of neutrals.f of STRAHL.
    '''
    r_src = namelist['rvol_lcfs'] + namelist['source_position']

    sint = np.zeros_like(S)

    # find index of radial grid vector that is just greater than r_src
    i_src = radius_grid.searchsorted(r_src)-1
    # set source to be inside of the wall
    i_src = min(i_src, len(radius_grid)-1)

    if (namelist['source_width_in'] < 0. and namelist['source_width_out'] < 0.):
        # point source
        sint[i_src] = 1.0

    # source with FWHM=source_width_in inside and FWHM=source_width_out outside
    if namelist['source_width_in']>0. or namelist['source_width_out']>0.:

        if namelist['source_width_in']> 0.:
            ff = np.log(2.)/namelist['source_width_in']**2
            sint[:i_src] = np.exp(-(radius_grid[:i_src]-radius_grid[i_src])**2*ff)[:,None]

        if namelist['source_width_out']>0.:
            ff = np.log(2.)/namelist['source_width_out']**2
            sint[i_src:] = np.exp(-(radius_grid[i_src:]-radius_grid[i_src])**2*ff)[:,None]


    # decay of neutral density with exp(-Int( [ne*S+dv/dr]/v ))
    if namelist['source_width_in']==0 and namelist['source_width_out']==0:
        #neutrals energy
        if namelist['imp_energy'] > 0: 
            E0 = namelist['imp_energy']*np.ones_like(radius_grid)
        else:
            if Ti is not None:
                E0 = Ti.mean(0)
            else:
                raise ValueError('Could not compute a valid energy of injected ions!')

        # velocity of neutrals [cm/s]
        out = atomic_element(symbol=namelist['main_element'])
        spec = list(out.keys())[0]
        main_ion_A = int(out[spec]['A'])
        v = - np.sqrt(2.*q_electron*E0/(main_ion_A*m_p))

        #integration of ne*S for atoms and calculation of ionization length for normalizing neutral density
        sint[i_src]= -0.0625*S[i_src]/pro[i_src]/v[i_src]   #1/16
        for i in np.arange(i_src-1,0,-1):
            sint[i]=sint[i+1]+0.25*(S[i+1]/pro[i+1]/v[i+1] + S[i]/pro[i]/v[i])

        #prevents FloatingPointError: underflow encountered
        sint[1:i_src] = np.exp(np.maximum(sint[1:i_src],-100))

        # calculate relative density of neutrals
        sint[1:i_src] *= (radius_grid[i_src]*v[i_src]/radius_grid[1:i_src]/v[1:i_src])[:,None]
        sint[i_src]=1.0

        # remove promptly redeposited ions
        if namelist['prompt_redep_flag']:
            omega_c = 1.602e-19/1.601e-27*namelist['Baxis']/namelist['a']    #Baxis = btotav in STRAHL neutrals.f
            dt = (radius_grid[i_src]-radius_grid)/v
            pp = dt*omega_c
            non_redep =  pp**2/(1.+pp**2)
            sint*=  non_redep


    # total ion source
    pnorm = np.pi*np.sum(sint*S*(radius_grid/pro)[:,None],0)

    # neutral density for influx/unitlength = 1/cm
    sint /= pnorm

    return np.asfortranarray(sint)

