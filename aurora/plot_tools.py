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


def slider_plot(
    x,
    y,
    z,
    xlabel="",
    ylabel="",
    zlabel="",
    plot_title=None,
    labels=None,
    plot_sum=False,
    x_line=None,
    y_line=None,
    zlim = False,
    **kwargs
):
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
    plot_title : None or string, optional
        Title of the plot.
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
        labels = ["" for v in z]

    # make sure not to modify the z array in place
    zz = copy.deepcopy(z)

    fig = plt.figure()
    fig.set_size_inches(10, 7, forward=True)

    # separate plot into 3 subgrids
    a_plot = plt.subplot2grid((10, 10), (0, 0), rowspan=8, colspan=8, fig=fig)
    a_legend = plt.subplot2grid((10, 10), (0, 8), rowspan=8, colspan=2, fig=fig)
    a_slider = plt.subplot2grid((10, 10), (9, 0), rowspan=1, colspan=8, fig=fig)

    a_plot.set_xlabel(xlabel)
    a_plot.set_ylabel(zlabel)
    if x_line is not None:
        a_plot.axvline(x_line, c="r", ls=":", lw=0.5)
    if y_line is not None:
        a_plot.axhline(y_line, c="r", ls=":", lw=0.5)

    ls_cycle = get_ls_cycle()

    l = []

    # plot all lines
    for v, l_ in zip(zz, labels):
        ls = next(ls_cycle)
        (tmp,) = a_plot.plot(x, v[:, 0], ls, **kwargs)
        _ = a_legend.plot([], [], ls, label=l_, **kwargs)
        l.append(tmp)

    if plot_sum:
        # add sum of the first axis to the plot (and legend)
        (l_sum,) = a_plot.plot(
            x,
            zz[:, :, 0].sum(axis=0),
            c="k",
            lw=mpl.rcParams["lines.linewidth"] * 2,
            **kwargs
        )
        _ = a_legend.plot(
            [],
            [],
            c="k",
            lw=mpl.rcParams["lines.linewidth"] * 2,
            label="total",
            **kwargs
        )
        
    if zlim:
        if plot_sum:
            lim_min = np.min(zz.sum(axis=0))
            lim_max = np.max(zz.sum(axis=0))*1.15
        else:
            lim_min = np.min(zz)
            lim_max = np.max(zz)*1.15

    leg = a_legend.legend(loc="best", fontsize=12).set_draggable(True)
    title = fig.suptitle("")
    a_legend.axis("off")
    a_slider.axis("off")

    def update(dum):

        i = int(slider.val)

        for v, l_ in zip(zz, l):
            l_.set_ydata(v[:, i])

        if plot_sum:
            l_sum.set_ydata(zz[:, :, i].sum(axis=0))

        a_plot.relim()
        a_plot.autoscale()

        a_plot.set_xlim(x[0],x[-1]*1.05)
        
        if zlim:
            a_plot.set_ylim(lim_min,lim_max)

        if plot_title is not None:
            title.set_text(f"{plot_title}, %s = %.5f" % (ylabel, y[i]) if ylabel else f"{plot_title}, %.5f" % (y[i],))
        else:
            title.set_text("%s = %.5f" % (ylabel, y[i]) if ylabel else "%.5f" % (y[i],))

        fig.canvas.draw()

    def arrow_respond(slider, event):
        if event.key == "right":
            slider.set_val(min(slider.val + 1, slider.valmax))
        elif event.key == "left":
            slider.set_val(max(slider.val - 1, slider.valmin))

    slider = mplw.Slider(a_slider, ylabel, 0, len(y) - 1, valinit=0, valfmt="%d")
    slider.on_changed(update)
    update(0)
    fig.canvas.mpl_connect("key_press_event", lambda evt: arrow_respond(slider, evt))


def get_ls_cycle():
    color_vals = ["b", "g", "r", "c", "m", "y", "k"]
    style_vals = ["-", "--", "-.", ":"]
    ls_vals = []
    for s in style_vals:
        for c in color_vals:
            ls_vals.append(c + s)
    return itertools.cycle(ls_vals)


def get_color_cycle(num=None, map="plasma"):
    """Get an iterable to select different colors in a loop.
    Efficiently splits a chosen color map into as many (`num`) parts as needed.
    """
    cols = ["b", "g", "r", "c", "m", "y", "k"]
    if num is None or num <= len(cols):
        return itertools.cycle(cols[:num])
    cm = plt.get_cmap(map)
    cols = np.empty(num)
    for j in np.arange(num):
        cols[j] = cm(1.0 * j / num)
    return itertools.cycle(cols)


def load_color_codes_reservoirs():
    '''Get a systematic color code for the reservoirs and particle conservation plots.
    '''
    blue = (0.0000, 0.4470, 0.7410) # color for plasma reservoirs and fluxes, 1st variation
    light_blue = (0.2824, 0.6980, 0.9686) # color for plasma reservoirs and fluxes, 2nd variation
    green = (0.4660, 0.6740, 0.1880) # color for neutrals reservoirs and fluxes, 1st variation
    light_green = (0.7137, 0.9020, 0.4667) # color for neutrals reservoirs and fluxes, 2nd variation
    grey = (0.5098, 0.5098, 0.5098) # color for wall reservoirs, 1st variation
    light_grey = (0.7451, 0.7451, 0.7451) # color for wall reservoirs, 2nd variation
    red = (0.8500, 0.3250, 0.0980) # color code for sources/sinks, 1st variation
    light_red = (1.0000, 0.5059, 0.2902) # color code for sources/sinks, 2nd variation
    
    return (blue,light_blue,green,light_green,grey,light_grey,red,light_red)


def load_color_codes_PWI():
    '''Get a systematic color code for the PWI-related plots.
    '''
    reds = [(0.6, 0, 0), (0.9, 0, 0), (1, 0.4, 0.4), (1, 0.6, 0.6), (1, 0.8, 0.8)]
    blues = [(0, 0, 0.6), (0, 0, 0.9), (0.4, 0.4, 1), (0.6, 0.6, 1), (0.8, 0.8, 1)]
    light_blues = [(0, 0.3608, 0.6), (0, 0.5412, 0.9020), (0.3020, 0.7216, 1), (0.6, 0.8392, 1), (0.8118, 0.9216, 1)]
    greens = [(0.3176, 0.4706, 0.1294), (0.4745, 0.7059, 0.1922), (0.6706, 0.8510, 0.4510), (0.8118, 0.9137, 0.6863), (0.9059, 0.9569, 0.8431)]
    
    return (reds, blues, light_blues, greens)


def get_line_cycle():
    style_vals = ["-", "--", "-.", ":"]
    return itertools.cycle(style_vals)


class DraggableColorbar:
    """Create a draggable colorbar for matplotlib plots to enable quick changes in color scale.

    Example:::

        fig,ax = plt.subplots()
        cntr = ax.contourf(R, Z, vals)
        cbar = plt.colorbar(cntr, format='%.3g', ax=ax)
        cbar = DraggableColorbar(cbar,cntr)
        cbar.connect()
    """

    def __init__(self, cbar, mapimage):
        self.cbar = cbar
        self.mapimage = mapimage
        self.press = None
        self.cycle = sorted(
            [i for i in dir(plt.cm) if hasattr(getattr(plt.cm, i), "N")]
        )

        self.index = self.cycle.index(ScalarMappable.get_cmap(cbar).name)

    def connect(self):
        """Matplotlib connection for button and key pressing, release, and motion."""
        self.cidpress = self.cbar.patch.figure.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.cidrelease = self.cbar.patch.figure.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cidmotion = self.cbar.patch.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )
        self.keypress = self.cbar.patch.figure.canvas.mpl_connect(
            "key_press_event", self.key_press
        )

    def on_press(self, event):
        """Button pressing; check if mouse is over colorbar."""
        if event.inaxes != self.cbar.ax:
            return
        self.press = event.x, event.y

    def key_press(self, event):
        """Key pressing event"""
        if event.key == "down":
            self.index += 1
        elif event.key == "up":
            self.index -= 1

        if self.index < 0:
            self.index = len(self.cycle)
        elif self.index >= len(self.cycle):
            self.index = 0

        cmap = self.cycle[self.index]

        self.cbar.set_cmap(cmap)
        self.cbar.draw_all()
        self.mapimage.set_cmap(cmap)
        self.mapimage.get_axes().set_title(cmap)
        self.cbar.patch.figure.canvas.draw()

    def on_motion(self, event):
        """Move if the mouse is over the colorbar."""
        if self.press is None:
            return
        if event.inaxes != self.cbar.ax:
            return
        xprev, yprev = self.press
        dx = event.x - xprev
        dy = event.y - yprev
        self.press = event.x, event.y

        scale = self.cbar.norm.vmax - self.cbar.norm.vmin
        perc = 0.03
        if event.button == 1:
            self.cbar.norm.vmin -= (perc * scale) * np.sign(dy)
            self.cbar.norm.vmax -= (perc * scale) * np.sign(dy)
        elif event.button == 3:
            self.cbar.norm.vmin -= (perc * scale) * np.sign(dy)
            self.cbar.norm.vmax += (perc * scale) * np.sign(dy)
        self.cbar.draw_all()
        self.mapimage.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()

    def on_release(self, event):
        """Upon release, reset press data"""
        self.press = None
        self.mapimage.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()

    def disconnect(self):
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidpress)
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidrelease)
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidmotion)
