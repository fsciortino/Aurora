"""
Functions to efficiently set a radial profile and a time dependency
for the anomalous transport coefficient (particle diffusivity and
radial pinch velocity) and possibly impose other time-dependent models
(e.g. for ELMs)
"""
# MIT License
#
# Copyright (c) 2021 Francesco Sciortino
#
# Module provided by Antonello Zito
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
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
from . import plot_tools
from . import surface


def interp_coeffs(namelist, rhop_grid, data, radial_dependency = False, rhop = None, method = 'linear_interp', time_dependency = False, times = None, plot = False, name = None):
    """
    Routine for interpolating a radial transport profile onto the radial grid, at each user-specified time
    The routine may be called for outputting both a diffusion coefficient D_Z
    and a radial pinch velocity v_Z

    Parameters
    ----------
    namelist: dict
        Dictionary containing aurora inputs. 
    rhop_grid: array
        Values of rho_poloidal on the aurora radial grid.
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
    plot: bool
        Select whether to produce plots with the resulting transport profiles imposed in the simulation.
    name: str
        Select the name ('D' or 'v') of the used coefficient, for the plot labels.
        
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
    
        if plot:
            fig, ax = plt.subplots(figsize=(7,5))
            ax.plot(rhop_grid, coeffs)
            ax.set_xlabel(r'$\rho_p$')
            if name == 'D':
                ax.set_ylabel('$D_Z$ [$\mathrm{cm^{2}/s}$]')
            elif name == 'v':
                ax.set_ylabel('$v_Z$ [$\mathrm{cm/s}$]')   
                
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
        
        if plot:
            if name == 'D':
                zlabel = '$D_Z$ [$\mathrm{cm^{2}/s}$]'
            elif name == 'v':
                zlabel = '$v_Z$ [$\mathrm{cm/s}$]'
            plot_tools.slider_plot(
                rhop_grid,
                times,
                np.reshape(coeffs,(1,coeffs.shape[0],coeffs.shape[1])),
                xlabel=r'$\rho_p$',
                ylabel="time [s]",
                zlabel = zlabel,
                labels = name,
                zlim = True,
            )
            
        return coeffs
  
    else:
        raise ValueError("Transport coefficients may only be given as a function of (space,time,charge) or (space,time) or (space), or constant.")



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



def ELM_model(timing, ELM_model, rhop_grid, rhop, data_inter_ELM, data_intra_ELM, method = 'linear_interp', plot = False, name = None):
    """ Routine for creating a 2D grid for the transport coefficients where the space dependence is given
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
    rhop_grid: array
        Values of rho_poloidal on the aurora radial grid.
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
    plot: bool
        Select whether to produce plots with the resulting transport profiles imposed in the simulation.
    name: str
        Select the name ('D' or 'v') of the used coefficient, for the plot labels.
        
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

    if plot:
        if name == 'D':
            zlabel = '$D_Z$ [$\mathrm{cm^{2}/s}$]'
        elif name == 'v':
            zlabel = '$v_Z$ [$\mathrm{cm/s}$]'
        times = np.round(times_transport,6)
        coeff = np.transpose(np.array(coeffs))
        plot_tools.slider_plot(
            rhop_grid,
            times,
            np.reshape(coeff,(1,coeff.shape[0],coeff.shape[1])),
            xlabel=r'$\rho_p$',
            ylabel="time [s]",
            zlabel = zlabel,
            labels = name,
            zlim = True,
        )
        
    return np.round(times_transport,6), np.transpose(np.array(coeffs))

    

def ELM_cycle_SOL_mach(SOL_mach,SOL_mach_ELM,time_grid,ELM_model,timing):
    """
    Generate a time-dependent shape of the Mach number in the SOL during an ELM
    cycle, assigning a peak value with a time dependence which follows the parallel
    ELM flux following the free streaming model.
    For now this works only if the time grid has constant time steps and for ELMs
    present for the entire duration of the simulation.
    """    
    
    # general time-dependent shape following the ELMS onto the entire time grid
    shape = ELM_cycle_shape(time_grid,ELM_model,timing)
    max_shape = np.max(shape)
    
    # normalize the time-dependent shape so that its peaks value equates
    #   the difference between SOL_mach_ELM and SOL_mach
    SOL_mach_shape_ELM = shape / (max_shape / (SOL_mach_ELM - SOL_mach))

    # now add this normalized shape to the inter-ELM value in order to achieve
    #   a Mach number which flattens on its intra-ELM value during inter-ELM phases
    #   but peaks at SOL_mach_ELM in the moment of maximum ELM-carried parallel flux
    mach = SOL_mach + SOL_mach_shape_ELM
    
    return mach   



def ELM_cycle_impact_energy_main_wall(species,time_grid,full_PWI,ELM_model,timing):
    """
    Generate a time-dependent shape of the impact energy of impurity ions on the main
    wall over an ELM cycle.
    For now this works only if the time grid has constant time steps and for ELMs
    present for the entire duration of the simulation.
    """

    # intra- and inter-ELM limiter electron temperatures + sheath parameters
    Te_lim_ELM = full_PWI["Te_lim_ELM"]
    Te_lim = full_PWI["Te_lim"]
    Ti_over_Te = full_PWI["Ti_over_Te"]
    gammai = full_PWI["gammai"]
    
    # Calculate the inter-ELM impact energy
    E0_inter_ELM = surface.get_impact_energy(Te_lim, species, mode = 'sheath' ,Ti_over_Te = Ti_over_Te, gammai = gammai)

    # Calculate the peak intra-ELM impact energy
    E0_peak_intra_ELM = surface.get_impact_energy(Te_lim_ELM, species, mode = 'sheath' ,Ti_over_Te = Ti_over_Te, gammai = gammai)
    
    # general time-dependent shape following the ELMS onto the entire time grid
    shape = ELM_cycle_shape(time_grid,ELM_model,timing)
    max_shape = np.max(shape)
    
    # normalize the time-dependent shape so that its peaks value equates
    #   the difference between E0_peak_intra_ELM and E0_inter_ELM
    E0_shape_ELM = shape / (max_shape / (E0_peak_intra_ELM - E0_inter_ELM))

    # now add this normalized shape to the inter-ELM value in order to achieve
    #   an impact energy which flattens on its intra-ELM value during inter-ELM phases
    #   but peaks at E0_peak_intra_ELM in the moment of maximum ELM-carried flux
    E0 = E0_inter_ELM + E0_shape_ELM
    
    return E0   



def ELM_cycle_impact_energy_div_wall(species,time_grid,full_PWI,ELM_model,timing):
    """
    Generate a time-dependent shape of the impact energy of impurity ions on the divertor
    target over an ELM cycle, assigning a peak value with a time dependence which follows the
    parallel ELM flux following the free streaming model.
    For now this works only if the time grid has constant time steps and for ELMs
    present for the entire duration of the simulation.
    """

    # intra-ELM pedestal electron temperature
    Te_ped_ELM = full_PWI["Te_ped_ELM"]
    
    # inter-ELM electron temperature on the divertor target + sheath parameters
    Te_div = full_PWI["Te_div"]
    Ti_over_Te = full_PWI["Ti_over_Te"]
    gammai = full_PWI["gammai"]
    
    # Calculate the inter-ELM impact energy
    E0_inter_ELM = surface.get_impact_energy(Te_div, species, mode = 'sheath' ,Ti_over_Te = Ti_over_Te, gammai = gammai)

    # Calculate the peak intra-ELM impact energy
    E0_peak_intra_ELM = surface.get_impact_energy(Te_ped_ELM, species, mode = 'FSM')
    
    # general time-dependent shape following the ELMS onto the entire time grid
    shape = ELM_cycle_shape(time_grid,ELM_model,timing)
    max_shape = np.max(shape)
    
    # normalize the time-dependent shape so that its peaks value equates
    #   the difference between E0_peak_intra_ELM and E0_inter_ELM
    E0_shape_ELM = shape / (max_shape / (E0_peak_intra_ELM - E0_inter_ELM))

    # now add this normalized shape to the inter-ELM value in order to achieve
    #   an impact energy which flattens on its intra-ELM value during inter-ELM phases
    #   but peaks at E0_peak_intra_ELM in the moment of maximum ELM-carried parallel flux
    E0 = E0_inter_ELM + E0_shape_ELM
    
    return E0



def ELM_cycle_shape(time_grid,ELM_model,timing):
    """
    Generate a time-dependent shape resembling the ion parallel flux onto the divertor
    target during an ELM cycle following the free streaming model, using empirical
    input parameters.
    For now this works only if the time grid has constant time steps and for ELMs
    present for the entire duration of the simulation.
    """    
    #TODO: adapt for variable time steps during inter-ELM phases
    #      and in case of ELM_time_window not None
    
    # Empiric parameters regulating the shape
    ELM_shape_decay_param = ELM_model["ELM_shape_decay_param"]
    ELM_shape_delay_param = ELM_model["ELM_shape_delay_param"]

    # ELM parameters
    ELM_frequency = ELM_model["ELM_frequency"]
    ELM_period = 1/ELM_frequency
    
    if ELM_shape_delay_param < 0.0:
        ELM_shape_delay_param = ELM_period + ELM_shape_delay_param       
    
    # Number of time steps during an ELM period
    time_step = timing['dt_start'][0]
    num_time_steps = round(ELM_period/time_step)
    
    # build a time-dependent parallel ion flow during an ELM cycle, in a.u.,
    #    (assuming the ELM crash at t = 0) according to the free streaming model
    t = np.linspace(time_step,ELM_period,num_time_steps)
    shape = np.zeros(len(t))
    for j in range(1,len(t)):
        shape[j] = (1/((t[j])**2)) * np.exp(-((1/ELM_shape_decay_param)**2)/(2*((t[j])**2)))
        
    fraction_to_move = ELM_shape_delay_param/ELM_period
    indeces_to_move = round(num_time_steps*fraction_to_move)
    shape_new = np.zeros(len(t))
    shape_new[indeces_to_move:num_time_steps] = shape[0:num_time_steps-indeces_to_move]
    shape_new[0:indeces_to_move] = shape[num_time_steps-indeces_to_move:num_time_steps]
        
    num_periods = int((timing['times'][-1]-timing['times'][0])/ELM_period)
    
    shape_new = np.tile(shape_new,num_periods)
    # make sure that the generated shape has the same length of the time grid
    if len(shape_new)>len(time_grid):      
        shape_new = shape_new[:-1]
 
    return shape_new   