"""
Functions to efficiently set a radial profile and a time dependency
for the anomalous transport coefficient (particle diffusivity and
radial pinch velocity) and possibly impose other time-dependent models
(e.g. for ELMs)
"""
# MIT License
#
# Copyright (c) 2022 Antonello Zito
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
import scipy
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator


def interp_coeffs(namelist, asim, data, radial_dependency = False, rhop = None, method = 'linear_interp', time_dependency = False, times = None):
    """
    Routine for interpolating a radial transport profile onto the radial grid, at each user-specified time
    The routine may be called for outputting both a diffusion coefficient D_Z
    and a radial pinch velocity v_Z

    Parameters
    ----------
    namelist: dict
        Dictionary containing aurora inputs. 
    asim: object
        Object containing aurora input methods.
    data: list
        List containing the user-specified values for the transport coefficient. It can be:
            0-dimensional: constant and uniform transport coefficient.
            1-dimensional: constant transport profile (list in function of rhop).
            2-dimensional: different transport profiles for each arbitrary time
                (list of lists, with one sublist for each time).
        If the output is a diffusion coefficient, the value(s) must be in units of :math:`cm^2/s`.
        If the output is a radial pinch velocity, the value(s) must be in units of :math:`cm/s`.
    radial_dependency: bool
        Select whether to impose a radial dependency
        False by default.
    rhop: list
        List containing the user-specified values of rhop at which the transport coefficients are imposed.
        None by default.
    method: str
        Select the interpolation method. It can be:
            'linear_interp': simple 1-D linear interpolation onto the radial grid.
            'cubic_spline': piecewise cubic polynomial and twice continuously differentiable interpolation
                onto the radial grid.
            'Pchip_spline': piecewise cubic Hermite and monotonicity-preserving interpolation
                onto the radial grid.
        'interp' by default.
    time_dependency: bool
        Select whether to impose a time dependency (only in case a radial dependency has been imposed as well)
        False by default.
    times: list
        List containing the user-specified times at which the transport profiles are imposed.
        None by default.
        
    Returns
    -------
    coeffs : array
        Contains the desired output transport coefficients, defined as a:
            Scalar, in case of no space- and time-dependence requested.
            1-D array in case of space-dependence (but not time-dependence) requested,
                i.e. as shape of (space).
            2-D array in case of both space- and time-dependence requested,
                i.e. as shape of (space,time).
        This is to be used as input of the main aurora routine.
    """
                 
    
    # Deduce number of charge states
    imp = namelist["imp"]
    from omfit_classes.utils_math import atomic_element
    out = atomic_element(symbol=imp)
    spec = list(out.keys())[0]
    Z_imp = int(out[spec]["Z"])
    
    # Extract the radial grid
    rhop_grid = asim.rhop_grid
    
    # Convert input list into array
    data = np.array(data)
    
    # Constant coefficient
    if not radial_dependency and not time_dependency:
        
        if data.ndim != 0:
            raise ValueError("The transport coefficient must be a scalar value.")          
        
        coeffs =  data * np.ones(len(rhop_grid))

        return coeffs        
        
    # Space-dependent coefficient
    if radial_dependency and not time_dependency:
        
        if data.ndim != 1:
            raise ValueError("D_Z or v_Z must be set as a one-dimensional list.") 
        if rhop is None or len(rhop) != len(data):
            raise ValueError("rhop must be a list of the same lenght of D_Z or v_Z.")       
            
        # Interpolate the transport coefficient profile onto the radial grid
        coeffs =  interp_transp(rhop,data,rhop_grid,method)
        return coeffs
        
    # Space- and time-dependent coefficient
    if radial_dependency and time_dependency:    
        
        data = np.transpose(data)
        if data.ndim != 2:
            raise ValueError("D_Z or v_Z must be set as a two-dimensional list.") 
        if rhop is None or len(rhop) != data.shape[0]:
            raise ValueError("rhop must be a list of the same lenght of D_Z or v_Z at each time.")       
        if times is None or len(times) != data.shape[1]:
            raise ValueError("times must be a list of the same length of D_Z or v_Z at each radial step.")            
       
        # Interpolate the transport coefficient profile onto the radial grid
        coeffs = np.zeros((len(rhop_grid),len(times)))
        for i in range(0,len(times)):  
            coeffs[:,i] = interp_transp(rhop,data[:,i],rhop_grid,method)
        return coeffs
  
    else:
        raise ValueError("Transport coefficients may only be given as a function of (space,time,charge) or (space,time) or (space), or constant.")



def ELM_model(timing, ELM_model, asim, rhop, data_inter_ELM, data_intra_ELM, method = 'linear_interp'):
    """
    Routine for creating a 2D grid for the transport coefficients where the space dependence is given
    interpolating a user-imposed transport profile onto the radial grid, and the time dependence is
    given according to the user-specified model for ELMs
    The routine may be called for outputting both a diffusion coefficient D_Z
    and a radial pinch velocity v_Z

    Parameters
    ----------
    timing: dict
        Sub-dict "timing" from the main aurora inputs namelist.
    ELM_model: dict
        Sub-dict "ELM_model" from the main aurora inputs namelist.
    asim: object
        Object containing aurora input methods.
    rhop: list
        List containing the user-specified values of rhop at which the transport coefficients are imposed.    
    data_inter_ELM: list
        1-dimensional list containing the user-specified transport profile (over the radial grid with
        values rhop) in the times between ELMs.
        If the output is a diffusion coefficient, the values must be in units of :math:`cm^2/s`.
        If the output is a radial pinch velocity, the values must be in units of :math:`cm/s`.
    data_intra_ELM: list
        1-dimensional list containing the user-specified transport profile (over the radial grid with
        values rhop) in the times within the ELMs.
        If the output is a diffusion coefficient, the values must be in units of :math:`cm^2/s`.
        If the output is a radial pinch velocity, the values must be in units of :math:`cm/s`.
    method: str
        Select the interpolation method. It can be:
            'linear_interp': simple 1-D linear interpolation onto the radial grid.
            'cubic_spline': piecewise cubic polynomial and twice continuously differentiable interpolation
                onto the radial grid.
            'Pchip_spline': piecewise cubic Hermite and monotonicity-preserving interpolation
                onto the radial grid.
        'interp' by default.
        
    Returns
    -------
    times_transport: list
        Contains the values at which the transport coefficients are imposed.
        This is to be used as input of the main aurora routine.
    coeffs : array
        Contains the desired output transport coefficients, defined as a 2-D array,
        i.e. in function of (space,time).
        This is to be used as input of the main aurora routine.
    """
    
    # Extract the radial grid
    rhop_grid = asim.rhop_grid
    
    # Convert input list into array
    data_inter_ELM = np.array(data_inter_ELM)
    data_intra_ELM = np.array(data_intra_ELM)
    
    if len(rhop) != len(data_inter_ELM) or len(rhop) != len(data_intra_ELM):
        raise ValueError("rhop and D_Z / v_Z must have the same length.") 
        
    # Interpolate the transport coefficient profile onto the radial grid
    coeffs_inter_ELM =  interp_transp(rhop,data_inter_ELM,rhop_grid,method)
    coeffs_intra_ELM =  interp_transp(rhop,data_intra_ELM,rhop_grid,method)
        
    ELM_time_windows = ELM_model['ELM_time_windows'] # s
    ELM_frequency = ELM_model['ELM_frequency'] # Hz
    crash_duration = ELM_model['crash_duration'] # ms
    plateau_duration = ELM_model['plateau_duration'] # ms
    recovery_duration = ELM_model['recovery_duration'] # ms
    
    times_transport = [timing['times'][0]]
    coeffs = [coeffs_inter_ELM]
    
    # Assuming that ELMs take place throughout the entire simulation duration
    if ELM_time_windows is None:     
        
        ELM_duration = (crash_duration + plateau_duration + recovery_duration)/1000 # s
        ELM_period = 1/ELM_frequency # s
        
        while times_transport[-1] < timing['times'][-1]:
            times_transport.append(times_transport[-1]+(ELM_period-ELM_duration))
            coeffs.append(coeffs_inter_ELM)
            times_transport.append(times_transport[-1]+crash_duration/1000)
            coeffs.append(coeffs_intra_ELM)
            if plateau_duration > 0:
                times_transport.append(times_transport[-1]+plateau_duration/1000)
                coeffs.append(coeffs_intra_ELM)
            times_transport.append(times_transport[-1]+recovery_duration/1000)
            coeffs.append(coeffs_inter_ELM)
       
    # Assuming that ELMs take place only in some reduced time windows 
    else:
        
        ELM_duration = np.zeros(len(ELM_frequency))
        ELM_period = np.zeros(len(ELM_frequency))
        for i in range(0,len(ELM_time_windows)):
            ELM_duration[i] = (crash_duration[i] + plateau_duration[i] + recovery_duration[i])/1000 # s
            ELM_period[i] = 1/ELM_frequency[i] # s  
            
        if ELM_time_windows[0][0] > timing['times'][0]:
            times_transport.append(ELM_time_windows[0][0]) 
            coeffs.append(coeffs_inter_ELM)

        for i in range(0,len(ELM_time_windows)):

            while times_transport[-1] < ELM_time_windows[i][1]:
                times_transport.append(times_transport[-1]+(ELM_period[i]-ELM_duration[i]))
                coeffs.append(coeffs_inter_ELM)
                times_transport.append(times_transport[-1]+crash_duration[i]/1000)
                coeffs.append(coeffs_intra_ELM)
                if plateau_duration[i] > 0:
                    times_transport.append(times_transport[-1]+plateau_duration[i]/1000)
                    coeffs.append(coeffs_intra_ELM)
                times_transport.append(times_transport[-1]+recovery_duration[i]/1000)
                coeffs.append(coeffs_inter_ELM)

            if i<len(ELM_time_windows)-1:
                if ELM_time_windows[i+1][0]>ELM_time_windows[i][1]:
                    times_transport.append(ELM_time_windows[i+1][0])
                    coeffs.append(coeffs_inter_ELM)
            else:
                if ELM_time_windows[i][1]<timing['times'][-1]:
                    times_transport.append(timing['times'][-1])
                    coeffs.append(coeffs_inter_ELM)
    
    return np.round(times_transport,6), np.transpose(np.array(coeffs))
     


def ELM_time_grid(timing, ELM_model, dt_intra_ELM, dt_increase_inter_ELM):
    """
    Routine for adapting the time grid to imposed ELM time characteristics

    Parameters
    ----------
    timing: dict
        Sub-dict "timing" from the main aurora inputs namelist (to update).
    ELM_model: dict
        Sub-dict "ELM_model" from the main aurora inputs namelist.
    dt_intra_ELM: float
        dt values during the entire intra-ELM cycles
        and at the beginning of each inter-ELM cycle
    dt_increase_inter_ELM: float
        dt multiplier at every time steps in the inter-ELM cycles
        
    Returns
    -------
    timing_update : dict
        Update to sub-dict "timing" from the main aurora inputs namelist.
    """ 
    
    
    ELM_time_windows = ELM_model['ELM_time_windows'] # s
    ELM_frequency = ELM_model['ELM_frequency'] # Hz
    crash_duration = ELM_model['crash_duration'] # ms
    plateau_duration = ELM_model['plateau_duration'] # ms
    recovery_duration = ELM_model['recovery_duration'] # ms
    
    # Assuming that ELMs take place throughout the entire simulation duration
    if ELM_time_windows is None:     
        
        ELM_duration = (crash_duration + plateau_duration + recovery_duration)/1000 # s
        ELM_period = 1/ELM_frequency # s
        
        # Check consistency of time windows with imposed ELM frequency
        if ELM_duration > ELM_period:
            raise ValueError("ELM duration not consistent with imposed ELM frequency.") 
        
        # Create a time grid which allows to distinguish between inter- and intra-ELM time windows   
        
        times = [timing['times'][0]] 
        for i in range(0,round((timing['times'][-1]-timing['times'][0])*ELM_frequency)):
            times.append(round(times[-1]+(ELM_period-ELM_duration),6))
            times.append(round(times[-1]+ELM_duration,6))
        
        dt_start = [dt_intra_ELM] * len(times)
    
        steps_per_cycle = [1] * len(times)
 
        dt_increase = [dt_increase_inter_ELM]
        
        while len(dt_increase)<len(times):
            dt_increase.append(1.000)
            dt_increase.append(dt_increase_inter_ELM)  
       
    # Assuming that ELMs take place only in some reduced time windows 
    else:
        
        if len(ELM_time_windows) != len(ELM_frequency):
            raise ValueError("Number of ELM time windows inconsistent with specification of ELM characteristics")
        ELM_duration = np.zeros(len(ELM_frequency))
        ELM_period = np.zeros(len(ELM_frequency))
        for i in range(0,len(ELM_time_windows)):
            ELM_duration[i] = (crash_duration[i] + plateau_duration[i] + recovery_duration[i])/1000 # s
            ELM_period[i] = 1/ELM_frequency[i] # s  
            
            # Check consistency of time windows with imposed ELM frequency
            if ELM_duration[i] > ELM_period[i]:
                raise ValueError("ELM duration not consistent with imposed ELM frequency.") 
        
        # Create a time grid which allows to distinguish between inter- and intra-ELM time windows
        
        times = [timing['times'][0]]
        
        if ELM_time_windows[0][0] > timing['times'][0]:
            times.append(ELM_time_windows[0][0]) 
        
        for i in range(0,len(ELM_time_windows)):
            
            for j in range(0,round((ELM_time_windows[i][1]-ELM_time_windows[i][0])*ELM_frequency[i])):
                times.append(round(times[-1]+(ELM_period[i]-ELM_duration[i]),6))
                times.append(round(times[-1]+ELM_duration[i],6))
            
            if i<len(ELM_time_windows)-1:
                if ELM_time_windows[i+1][0]>ELM_time_windows[i][1]:
                    times.append(ELM_time_windows[i+1][0])
            else:
                if ELM_time_windows[i][1]<timing['times'][-1]:
                    times.append(timing['times'][-1])
            
        dt_start = [dt_intra_ELM] * len(times)
    
        steps_per_cycle = [1] * len(times)
                
        dt_increase = [dt_increase_inter_ELM]  
        
        if ELM_time_windows[0][0] > timing['times'][0]:
            dt_increase.append(dt_increase_inter_ELM)
        
        for i in range(0,len(ELM_time_windows)):
            
            for j in range(0,round((ELM_time_windows[i][1]-ELM_time_windows[i][0])*ELM_frequency[i])):
                dt_increase.append(1.000)
                dt_increase.append(dt_increase_inter_ELM)
            
            if i<len(ELM_time_windows)-1:
                if ELM_time_windows[i+1][0]>ELM_time_windows[i][1]:
                    dt_increase.append(dt_increase_inter_ELM)
            else:
                if ELM_time_windows[i][1]<timing['times'][-1]:
                    dt_increase.append(dt_increase_inter_ELM)
    
    timing_update = {
        "dt_increase": dt_increase,
        "dt_start": dt_start,
        "steps_per_cycle": steps_per_cycle,
        "times": times,
    } 
    
    return timing_update



def interp_transp(x, y, grid, method):
    """Function 'interp_transp' used for interpolating the user-defined transport
    coefficients onto the radial grid.
    """
    
    if method == "linear_interp": 
        # Simple 1-D linear interpolation onto the radial grid
        f = interp1d(x, y, kind="linear", fill_value="extrapolate", assume_sorted=True)

    elif method == "cubic_spline":
        # Piecewise cubic polynomial and twice continuously differentiable interpolation onto the radial grid
        f = CubicSpline(x, y)

    elif method == "Pchip_spline":
        # Piecewise cubic Hermite and monotonicity-preserving interpolation onto the radial grid
        f = PchipInterpolator(x, y)
    
    return f(grid) 