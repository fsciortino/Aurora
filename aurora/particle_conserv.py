import matplotlib.pyplot as plt
import xarray
import numpy as np
from IPython import embed
from netCDF4 import Dataset
import copy
from scipy.integrate import cumtrapz

def vol_int(ds, var,rhop_max=None, R_axis=68.5):
    """
    Perform a volume integral of an input variable. If the variable is f(t,x) then the result is f(t).
    If the variable is f(t,*,x) then the result is f(t,charge) when "*" represents charge, line index, etc...

    :param ds: xarray dataset containing STRAHL result
    :param var: Name of the variable in the strahl_result.cdf file
    :params rhop_max: Maximum normalized poloidal flux for integral
    :params R_axis: major radius on axis [cm]

    :return: Time history of volume integrated variable
    """

    # Coordinates
    pro = ds['pro'].data #2x radial step
    radius_grid = ds['radius_grid'].data
    rhop_grid = ds['rhop_grid'].data

    C = 2.0 * np.pi * R_axis
    zvol = C * np.pi * radius_grid / pro

    # Get our variable
    f = ds[var].data

    # Determine range
    if rhop_max is not None:
        wh = (rhop_grid <= rhop_max)
        zvol = zvol[wh]
        f  = f[...,wh]

    # 2D or 3D array f(t,x)
    fvolint = np.nansum(f*zvol,axis=-1)

    return fvolint


def get_particle_nums(filepath=None, ds=None, R_axis=None):
    ''' Check time evolution and particle conservation in (py)STRAHL output.
    If filepath is None, load data from output STRAHL file, otherwise use xarray Dataset in ds. 

    R_axis [cm] is needed for volume integrals. If None, it is set to a C-Mod default. 
    '''
    if filepath is not None:
        ds = xarray.open_dataset(filepath)  
        # use STRAHL notation for source function
        source_lbl = 'influx_through_valve'
    else:
        # use Aurora notation for source function
        source_lbl = 'source_function'

    if R_axis is None:
        # set to C-Mod default
        R_axis = 68.5

    # calculate total impurity density (summed over charge states)
    ds['total_impurity_density'] = xarray.DataArray(np.nansum(ds['impurity_density'].data, axis=1),
                                                    coords=[ds['time'].data, ds['radius_grid'].data], 
                                                    dims=['time', 'space'])
    
    keys = [source_lbl,'particles_in_divertor','particles_in_pump',
            'parallel_loss','parallel_loss_to_limiter','edge_loss', 'particles_at_wall']

    labels = [r'Influx ($s^{-1}$)','Particles in Divertor','Particles in Pump',
              'Parallel Loss',   #losses between LCFS and limiter
              'Parallel Loss to Limiter',  #losses between limiter and edge
              'Edge Loss',  #losses outside the last grit point
             'Particles Stuck at Wall'] # particles that will never leave the wall

    if 'recycling_from_wall' in ds:   # activated recycling model
        if 'particles_retained_at_wall' in ds:    # activated wall retention model
            keys += ['particles_retained_at_wall']
            labels += [ 'Particles Retained at Wall']

        keys += ['recycling_from_wall','recycling_from_divertor']
        labels += ['Wall Recy. Rate','Divertor Recy. Rate']

    for key in keys:
        # substitute 0 to nan's for particle numbers
        ds[key] = copy.deepcopy(ds[key].fillna(0.0))

    time = ds['time'].data
    vol_int_keys = ['total_impurity_density','impurity_radiation']
    vol_int_rhop = [1.0, 1.0]
    vol_int_labels = ['Core Impurity Particles', 'Core Radiation (W)']

    circ = 2*np.pi*R_axis #cm 

    # Check particle conservation
    plasma_particles = vol_int(ds, 'total_impurity_density')
    total = plasma_particles+(ds['particles_at_wall'].data+\
                              ds['particles_in_divertor'].data)*circ 
    
    if 'recycling_from_wall' in ds:
        total+=ds['particles_in_pump'].data*circ
        
        if 'particles_retained_at_wall' in ds:
            # only plot particles retained at wall if retention model is used
            total+=ds['particles_retained_at_wall'].data*circ

    source = ds[source_lbl].data*circ
    integ_source =  cumtrapz(source,time,initial=0)
    
    # return the relevant quantities for particle conservation
    res = {}
    res['time'] = time
    res['total'] = total
    res['plasma_particles'] = plasma_particles # vol_int(ds, 'total_impurity_density')
    res['particles_at_wall'] = ds['particles_at_wall'].data*circ
    res['particles_in_divertor'] = ds['particles_in_divertor'].data*circ
    if 'recycling_from_wall' in ds:
        res['particles_in_pump'] = ds['particles_in_pump'].data*circ
    if 'recycling_from_wall' in ds and 'particles_retained_at_wall' in ds:
        res['particles_retained_at_wall'] = ds['particles_retained_at_wall'].data*circ
    res['integ_source'] = integ_source

    # close dataset
    ds.close()
    
    return res





def plot_particle_conserv(res, ax=None):
    ''' Plot number of particles in each Aurora simulation reservoir. '''

    if ax is None:
        fig,ax = plt.subplots() 
    else:
        ax = axs[1]

    ax.plot(res['time'], res['plasma_particles'] ,label='Particles in Plasma')
    ax.plot(res['time'], res['particles_at_wall'] ,label='Particles stuck at wall')
    ax.plot(res['time'], res['particles_in_divertor'],label='Particles in Divertor')

    if 'particles_in_pump' in res:
        ax.plot(res['time'], res['particles_in_pump'] ,label='Particles in Pump')

    if 'particles_retained_at_wall' in res:
        # only plot particles retained at wall if retention model is used
        ax.plot(res['time'], res['particles_retained_at_wall'],label='Particles retained at wall')

    ax.plot(res['time'],res['total'],lw=2,label='Total')
    ax.plot(res['time'],res['integ_source'],lw=2,label='Integrated source')

    if abs((res['total'][-1]-res['integ_source'][-1])/res['integ_source'][-1]) > .1:
        print('Warning: significant error in particle conservation! try to increase grid or time resolution')

    ax.set_ylim(0,None)
    ax.legend(loc='best')

    plt.tight_layout()





def check_1d_conserv(Raxis, ds=None, filepath=None, linestyle='-', axs = None):
    ''' Check time evolution and particle conservation in Aurora or STRAHL output.

    Args:
         Raxis : float
             Major radius on axis [m], used for volume integrals. 
         ds : xarray dataset, optional
             Dataset containing Aurora results, created using the xarray package. 
             See :py:meth:`~aurora.core.check_conservation` for an illustration on how
             to use this. 
         filepath : str, optional
             If provided, load results from STRAHL output file and check particle
             particle conservation as for an Aurora run. 
         linestyle : str, optional
             matplotlib linestyle, default is '-' (continuous lines). Use this to 
             overplot lines on the same plots using different linestyles, e.g. to check
             whether some aurora option makes particle conservation better or worse. 
         axs : 2-tuple or array
             array-like structure containing two matplotlib.Axes instances: the first one 
             for the separate particle time variation in each reservoir, the second for 
             the total particle-conservation check. This can be used to plot results 
             from several aurora runs on the same axes. 

    Returns:
         ds : xarray dataset
             Dataset containing results in xarray format. 
         res : dict
             Dictionary containing time histories across all reservoirs, useful for 
             the assessment of particle conservation.
         axs : 2-tuple or array
             array-like structure containing two matplotlib.Axes instances, (ax1,ax2).
             See optional input argument.
    '''
    if filepath is not None:
        ds = xarray.open_dataset(filepath)  
        # use STRAHL notation for source function
        source_lbl = 'influx_through_valve'
    else:
        # use Aurora notation for source function
        source_lbl = 'source_function'

    if Raxis is None:
        Raxis = 0.685 # m 
    
    # calculate total impurity density (summed over charge states)
    ds['total_impurity_density'] = xarray.DataArray(np.nansum(ds['impurity_density'].data, axis=1),
                                                    coords=[ds['time'].data, ds['radius_grid'].data], 
                                                    dims=['time', 'space'])
    
    keys = [source_lbl,'particles_in_divertor','particles_in_pump',
            'parallel_loss','parallel_loss_to_limiter','edge_loss', 'particles_at_wall']

    labels = [r'Influx ($s^{-1}$)','Particles in Divertor','Particles in Pump',
              'Parallel Loss',   #losses between LCFS and limiter
              'Parallel Loss to Limiter',  #losses between limiter and edge
              'Edge Loss',  #losses outside the last grit point
             'Particles Stuck at Wall'] # particles that will never leave the wall

    if 'recycling_from_wall' in ds:   # activated recycling model
        if 'particles_retained_at_wall' in ds:    # activated wall retention model
            keys += ['particles_retained_at_wall']
            labels += [ 'Particles Retained at Wall']

        keys += ['recycling_from_wall','recycling_from_divertor']
        labels += ['Wall Recy. Rate','Divertor Recy. Rate']

    for key in keys:
        # substitute 0 to nan's for particle numbers
        ds[key] = copy.deepcopy(ds[key].fillna(0.0))

    time = ds['time'].data
    vol_int_keys = ['total_impurity_density','impurity_radiation']
    vol_int_rhop = [1.0, 1.0]
    vol_int_labels = ['Core Impurity Particles', 'Core Radiation (W)']

    circ = 2*np.pi*Raxis*100 # cm
    nplots = np.sum([k in ds for k in keys+vol_int_keys])
    ncol = min(3,nplots)
    nrow = int(np.ceil(float(nplots)/ncol))

    if axs is None:
        fig,ax1 = plt.subplots(nrows=nrow,ncols=ncol,sharex=True, figsize=(15,10))
    else:
        ax1 = axs[0]

    axf = ax1.flatten()
    iplot = 0

    for  key,label in zip(keys,labels):
        if key in ds:
            y = ds[key].data
            if key != 'particles_in_pump':
                y = y* circ #normalize it to #/s or # units

            axf[iplot].plot(time,y,'-o',label=label,markersize=2, ls=linestyle)
            iplot += 1

    for i,key,rhop,lab in zip(list(range(len(vol_int_keys))),vol_int_keys,vol_int_rhop,vol_int_labels):
        if key in ds:
            vol_int_data = vol_int(ds, key,rhop_max=rhop)
            
            if key == 'impurity_radiation':
                vol_int_data = vol_int_data[:,-1]
            axf[iplot].plot(time,vol_int_data,'o-',label=lab,markersize=2.0, ls=linestyle)
            iplot += 1

    for axxx in axf[-ncol:]:
        axxx.set_xlabel('Time (s)')
    axxx.set_xlim(time[[0,-1]])

    for aaa in axf:
        aaa.legend(loc='best').set_draggable(True)

    # ---------------------------------------------
    if 'particles_at_wall' in ds:
        # Check particle conservation

        if axs is None:
            fig,ax2 = plt.subplots()
        else:
            ax2 = axs[1]
    
        ax2.set_xlabel('time [s]')
        all_particles = vol_int(ds, 'total_impurity_density')
        total = all_particles+(ds['particles_at_wall'].data+\
                ds['particles_in_divertor'].data)*circ 
    
        ax2.plot(time, all_particles ,label='Particles in Plasma', ls=linestyle)
        ax2.plot(time, ds['particles_at_wall'].data*circ,label='Particles stuck at wall', ls=linestyle)
        ax2.plot(time, ds['particles_in_divertor'].data*circ,label='Particles in Divertor', ls=linestyle)

        if 'recycling_from_wall' in ds:
            ax2.plot(time, ds['particles_in_pump'].data*circ ,label='Particles in Pump', ls=linestyle)
            total+=ds['particles_in_pump'].data*circ

            if 'particles_retained_at_wall' in ds:
                # only plot particles retained at wall if retention model is used
                ax2.plot(time, ds['particles_retained_at_wall'].data*circ,label='Particles retained at wall', ls=linestyle)
                total+=ds['particles_retained_at_wall'].data*circ

        ax2.plot(time,total,lw=2,label='Total', ls=linestyle)
        
        source = ds[source_lbl].data*circ
        integ_source =  cumtrapz(source,time,initial=0)
    
        ax2.plot(time,integ_source,lw=2,label='Integrated source', ls=linestyle)

        if abs((total[-1]-integ_source[-1])/integ_source[-1])> .1:
            print('Warning: significant error in particle conservation!')   #try increasing grid or time resolution

        ax2.set_ylim(0,None)
        ax2.legend(loc='best')
        
        plt.tight_layout()

    # return the relevant quantities for particle conservation
    res = {}
    res['total'] = total
    res['plasma_particles'] = vol_int(ds, 'total_impurity_density')
    res['particles_at_wall'] = ds['particles_at_wall'].data*circ,
    res['particles_in_divertor'] = ds['particles_in_divertor'].data*circ
    if 'recycling_from_wall' in ds:
        res['particles_in_pump'] = ds['particles_in_pump'].data*circ
    if 'recycling_from_wall' in ds and 'particles_retained_at_wall' in ds:
        res['particles_retained_at_wall'] = ds['particles_retained_at_wall'].data*circ
    res['integ_source'] = integ_source
    
    # close dataset
    try:
        ds.close()
    except:
        pass

    return ds, res, (ax1,ax2)

