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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np, copy
import matplotlib.gridspec as mplgs
import matplotlib.widgets as mplw
from matplotlib.cm import ScalarMappable
import itertools
plt.ion()


def slider_plot(x, y, z, xlabel='', ylabel='', zlabel='', labels=None, plot_sum=False,
                x_line=None, y_line=None, **kwargs):
    """Make a plot to explore multidimensional data.

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
        l_sum, = a_plot.plot(x, zz[:, :, 0].sum(axis=0), c='k', lw=mpl.rcParams['lines.linewidth']*2, **kwargs)
        _ = a_legend.plot([],[],  c='k', lw=mpl.rcParams['lines.linewidth']*2, label='total', **kwargs)

    leg=a_legend.legend(loc='best', fontsize=12).set_draggable(True)
    title = fig.suptitle('')
    a_legend.axis('off')
    a_slider.axis('off')


    def update(dum):

        i = int(slider.val)

        for v, l_ in zip(zz, l):
            l_.set_ydata(v[:, i])

        if plot_sum:
            l_sum.set_ydata(zz[:, :, i].sum(axis=0))

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


def get_color_cycle():
    color_vals = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    return itertools.cycle(color_vals)


def get_line_cycle():
    style_vals = ['-', '--', '-.', ':']
    return itertools.cycle(style_vals)


class DraggableColorbar:
    '''Create a draggable colorbar for matplotlib plots to enable quick changes in color scale. 

    Example:::

        fig,ax = plt.subplots()
        cntr = ax.contourf(R, Z, vals)
        cbar = plt.colorbar(cntr, format='%.3g', ax=ax)
        cbar = DraggableColorbar(cbar,cntr)
        cbar.connect()
    '''
    def __init__(self, cbar, mapimage):
        self.cbar = cbar
        self.mapimage = mapimage
        self.press = None
        self.cycle = sorted([i for i in dir(plt.cm) if hasattr(getattr(plt.cm,i),'N')])

        self.index = self.cycle.index(ScalarMappable.get_cmap(cbar).name)

    def connect(self):
        '''Matplotlib connection for button and key pressing, release, and motion.
        '''
        self.cidpress = self.cbar.patch.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.cbar.patch.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.cbar.patch.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.keypress = self.cbar.patch.figure.canvas.mpl_connect('key_press_event', self.key_press)

    def on_press(self, event):
        '''Button pressing; check if mouse is over colorbar.
        '''
        if event.inaxes != self.cbar.ax: return
        self.press = event.x, event.y

    def key_press(self, event):
        '''Key pressing event
        '''
        if event.key=='down':
            self.index += 1
        elif event.key=='up':
            self.index -= 1

        if self.index<0:
            self.index = len(self.cycle)
        elif self.index>=len(self.cycle):
            self.index = 0

        cmap = self.cycle[self.index]

        self.cbar.set_cmap(cmap)
        self.cbar.draw_all()
        self.mapimage.set_cmap(cmap)
        self.mapimage.get_axes().set_title(cmap)
        self.cbar.patch.figure.canvas.draw()

    def on_motion(self, event):
        '''Move if the mouse is over the colorbar.
        '''
        if self.press is None: return
        if event.inaxes != self.cbar.ax: return
        xprev, yprev = self.press
        dx = event.x - xprev
        dy = event.y - yprev
        self.press = event.x,event.y

        scale = self.cbar.norm.vmax - self.cbar.norm.vmin
        perc = 0.03
        if event.button==1:
            self.cbar.norm.vmin -= (perc*scale)*np.sign(dy)
            self.cbar.norm.vmax -= (perc*scale)*np.sign(dy)
        elif event.button==3:
            self.cbar.norm.vmin -= (perc*scale)*np.sign(dy)
            self.cbar.norm.vmax += (perc*scale)*np.sign(dy)
        self.cbar.draw_all()
        self.mapimage.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()


    def on_release(self, event):
        '''Upon release, reset press data
        '''
        self.press = None
        self.mapimage.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()

    def disconnect(self):
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidpress)
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidrelease)
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidmotion)
