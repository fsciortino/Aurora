import matplotlib.pyplot as plt
import numpy as np, copy
import matplotlib.gridspec as mplgs
import matplotlib.widgets as mplw
import itertools
from scipy.constants import m_p, e as q_electron

def slider_plot(x, y, z, xlabel='', ylabel='', zlabel='', labels=None, plot_sum=False,
                x_line=None, y_line=None, **kwargs):
    """Make a plot to explore multidimensional data.

    INPUTS
    ----------
    x : array of float, (`M`,)
        The abscissa. (in aurora, often this may be rhop)
    y : array of float, (`N`,)
        The variable to slide over. (in aurora, often this may be time)
    z : array of float, (`P`, `M`, `N`)
        The variables to plot.
    xlabel : str, optional
        The label for the abscissa.
    ylabel : str, optional
        The label for the slider.
    zlabel : str, optional
        The label for the ordinate.
    labels : list of str with length `P`
        The labels for each curve in `z`.
    plot_sum : bool, optional
        If True, will also plot the sum over all `P` cases. Default is False.
    x_line : float, optional
        x coordinate at which a vertical line will be drawn. 
    y_line : float, optional
        y coordinate at which a horizontal line will be drawn.
    """
    if labels is None:
        labels = ['' for v in z]

    # make sure not to modify the z array in place
    zz = copy.deepcopy(z)

    fig = plt.figure()
    fig.set_size_inches(10,7, forward=True)
    
    # separate plot into 3 subgrids
    a_plot = plt.subplot2grid((10,10),(0,0),rowspan = 8, colspan = 8, fig=fig) 
    a_legend = plt.subplot2grid((10,10),(0,8),rowspan = 8, colspan = 2, fig=fig) 
    a_slider = plt.subplot2grid((10,10),(9,0),rowspan = 1, colspan = 8, fig=fig) 
    
    a_plot.set_xlabel(xlabel)
    a_plot.set_ylabel(zlabel)
    if x_line is not None: a_plot.axvline(x_line, c='r',ls=':',lw=0.5)
    if y_line is not None: a_plot.axhline(y_line, c='r',ls=':',lw=0.5)

    ls_cycle = get_ls_cycle()

    l = []

    # plot all lines
    for v, l_ in zip(zz, labels):
        ls = next(ls_cycle)
        tmp, = a_plot.plot(x, v[:, 0], ls, **kwargs)
        _ = a_legend.plot([], [], ls, label=l_, **kwargs)
        l.append(tmp)

    if plot_sum:
        # add sum of the first axis to the plot (and legend)
        ls = next(ls_cycle)
        l_sum, = a_plot.plot(x, zz[:, :, 0].sum(axis=0), ls, **kwargs)
        _ = a_legend.plot([],[], ls, label='total', **kwargs)

    leg=a_legend.legend(loc='best', fontsize=12).set_draggable(True)
    title = fig.suptitle('')
    a_legend.axis('off')
    a_slider.axis('off')


    def update(dum):
        # ls_cycle = itertools.cycle(ls_vals)
        # remove_all(l)
        # while l:
        #     l.pop()

        i = int(slider.val)

        for v, l_ in zip(zz, l):
            l_.set_ydata(v[:, i])
            # l.append(a_plot.plot(x, v[:, i], ls_cycle.next(), label=l_, **kwargs))

        if plot_sum:
            l_sum.set_ydata(zz[:, :, i].sum(axis=0))
            # l.append(a_plot.plot(x, zz[:, :, i].sum(axis=0), ls_cycle.next(), label='total', **kwargs))

        a_plot.relim()
        a_plot.autoscale()

        title.set_text('%s = %.5f' % (ylabel, y[i]) if ylabel else '%.5f' % (y[i],))

        fig.canvas.draw()

    def arrow_respond(slider, event):
        if event.key == 'right':
            slider.set_val(min(slider.val + 1, slider.valmax))
        elif event.key == 'left':
            slider.set_val(max(slider.val - 1, slider.valmin))

    slider = mplw.Slider(
        a_slider,
        ylabel,
        0,
        len(y) - 1,
        valinit=0,
        valfmt='%d'
    )
    slider.on_changed(update)
    update(0)
    fig.canvas.mpl_connect(
        'key_press_event',
        lambda evt: arrow_respond(slider, evt)
    )




def get_ls_cycle():
    color_vals = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    style_vals = ['-', '--', '-.', ':']
    ls_vals = []
    for s in style_vals:
        for c in color_vals:
            ls_vals.append(c + s)
    return itertools.cycle(ls_vals)




def plot_norm_ion_freq(S_z, q_prof, R_prof, m_imp, Ti_prof,
                       nz_profs=None, rhop=None, plot=True, eps_prof=None):
    '''
    Compare effective ionization rate for each charge state with the 
    characteristic transit time that a non-trapped and trapped impurity ion takes
    to travel a parallel distance L = q R. 

    If the normalized ionization rate is less than 1, then flux surface averaging of
    background asymmetries (e.g. from edge or beam neutrals) can be considered in a 
    "flux-surface-averaged" sense; otherwise, local effects (i.e. not flux-surface-averaged)
    may be too important to ignore. 

    This function is inspired by Dux et al. NF 2020. Note that in this paper the ionization 
    rate averaged over all charge state densities is considered. This function avoids the 
    averaging over charge states, unless these are provided as an input. 

    INPUTS:
    -------
    S_z : array (r,cs) [s^-1]
         Effective ionization rates for each charge state as a function of radius. 
         Note that, for convenience within aurora, cs includes the neutral stage.
    q_prof : array (r,)
         Radial profile of safety factor
    R_prof : array (r,) or float [m]
         Radial profile of major radius, either given as an average of HFS and LFS, or also
         simply as a scalar (major radius on axis)
    m_imp : float [amu]
         Mass of impurity of interest in amu units (e.g. 2 for D)
    Ti_prof : array (r,)
         Radial profile of ion temperature [eV]
    nz_profs : array (r,cs), optional
         Radial profile for each charge state. If provided, calculate average normalized 
         ionization rate over all charge states.
    rhop : array (r,), optional
         Sqrt of poloidal flux radial grid. This is used only for (optional) plotting. 
    plot : bool, optional
         If True, plot results.
    eps_prof : array (r,), optional
         Radial profile of inverse aspect ratio, i.e. r/R, only used if plotting is requested.  


    OUTPUTS:
    --------
    nu_ioniz_star : array (r,cs) or (r,)
         Normalized ionization rate. If nz_profs is given as an input, this is an average over
         all charge state; otherwise, it is given for each charge state.
    '''

    nu = np.zeros_like(S_z)
    for cs in np.arange(S_z.shape[1]): # exclude neutral states
        nu[:,cs] = S_z[:,cs] * q_prof * R_prof * np.sqrt((m_imp * m_p)/(2*Ti_prof))

    if nz_profs is not None:
        # calculate average nu_ioniz_star 
        nu_ioniz_star = np.sum(nz_profs[:,1:]*nu[:,1:],axis=1)/np.sum(nz_profs[:,1:],axis=1)
    else:
        # return normalized ionization rate for each charge state
        nu_ioniz_star = nu[:,1:]

    if plot:
        if rhop is None:
            rhop = np.arange(nu.shape[0])
            
        fig,ax = plt.subplots()
        if nu_ioniz_star.ndim==1:
            ax.semilogy(rhop,nu_ioniz_star, label=r'$\nu_{ion}^*$')
        else:
            for cs in np.arange(S_z.shape[1]-1):
                ax.semilogy(rhop, nu_ioniz_star[:,cs], label=f'q={cs+1}')
            ax.set_ylabel(r'$\nu_{ion}^*$')

        ax.set_xlabel(r'$\rho_p$')

        if eps_prof is not None:
            ax.semilogy(rhop, np.sqrt(eps_prof), label=r'$\sqrt{\epsilon}$')

        ax.legend().set_draggable(True)
