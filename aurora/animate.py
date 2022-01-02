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
from matplotlib import pyplot as plt
from matplotlib import animation
plt.ion()
import pickle as pkl
from . import plot_tools


def animate_aurora(x,y,z, xlabel='', ylabel='', zlabel='', 
                   labels=None, plot_sum=False,
                   uniform_y_spacing=True, save_filename=None):
    ''' Produce animation of time- and radially-dependent results from aurora.

    Parameters
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
        The label for the animated coordinate. This is expected in a format such that ylabel.format(y_val)
        will display a good moving label, e.g. ylabel='t={:.4f} s'.
    zlabel : str, optional
        The label for the ordinate.
    labels : list of str with length `P`
        The labels for each curve in `z`.
    plot_sum : bool, optional
        If True, will also plot the sum over all `P` cases. Default is False.
    uniform_y_spacing : bool, optional
        If True, interpolate values in z onto a uniformly-spaced y grid
    save_filename : str
        If a valid path/filename is provided, the animation will be saved here in mp4 format. 
    '''
    if labels is None:
        labels = ['' for v in z]

    if plot_sum:
        labels.append('total')
        z_sum = np.sum(z, axis=0)
        z = np.vstack((z, np.atleast_3d(z_sum).transpose(2,0,1)))

    if uniform_y_spacing:
        from scipy.interpolate import RegularGridInterpolator as rgi
        interp_fun = rgi((np.arange(z.shape[0]), x, y), z)

        y_eq = np.linspace(min(y), max(y), len(y))
        new_grid = np.ix_(np.arange(z.shape[0]), x, y_eq)
        z = interp_fun(new_grid)

    # set up the figure and side space for legend
    fig = plt.figure(figsize=(10,6))
    a_plot = plt.subplot2grid((1,10),(0,0),rowspan = 1, colspan = 8, fig=fig) 
    a_legend = plt.subplot2grid((1,10),(0,8),rowspan = 1, colspan = 2, fig=fig) 
    a_plot.set_xlabel(xlabel)
    a_plot.set_ylabel(zlabel)

    a_plot.set_xlim([np.min(x),np.max(x)])
    a_plot.set_ylim([0.0, 1.1*np.max(z)])

    # get nice sequence of line styles/colors
    ls_cycle = plot_tools.get_ls_cycle()

    lines = []
    for l_ in labels:
        a_legend.plot([],[],'k-' if l_=='total' else next(ls_cycle),
                      lw=2.5 if l_=='total' else 1.0, label=l_)[0]
        lobj = a_plot.plot([],[],'k-' if l_=='total' else next(ls_cycle),
                           lw=2.5 if l_=='total' else 1.0)[0]
        lines.append(lobj)

    # time label (NB: update won't work if this is placed outside axes)
    y_text = a_plot.text(0.75, 0.95, ' ', fontsize=14, transform=a_plot.transAxes) 

    def init():   # initialization function
        for line in lines:
            line.set_data([], [])
        y_text.set_text('')
        return tuple(lines) + (y_text,) 

    def animate(i):   # animation function, called sequentially 
        y_text.set_text(ylabel.format(y_eq[i] if uniform_y_spacing else y[i])) 

        for lnum,line in enumerate(lines):
            line.set_data(x, z[lnum,:,i])
        return tuple(lines) + (y_text,)

    a_legend.legend(loc='center').set_draggable(True)
    a_legend.axis('off')

    # run animation now:
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(y), interval=20, blit=True)

    if save_filename is not None:
        if 'gif' in save_filename:
            anim.save(save_filename,fps=30)
        else:
            anim.save(save_filename+'.mp4',fps=30, extra_args=['-vcodec','libx264'])
        

