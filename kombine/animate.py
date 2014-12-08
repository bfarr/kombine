#!/usr/bin/env python
import numpy as np

from matplotlib import animation
import triangle


def anim_to_html(anim):
    """
    Function to help with inline animations in ipython notebooks.
    """
    from tempfile import NamedTemporaryFile

    VIDEO_TAG = """<video controls>
     <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
     Your browser does not support the video tag.
    </video>"""

    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=30, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")

    return VIDEO_TAG.format(anim._encoded_video)


def animate_triangle(pos_T, labels=None, truths=None,
                     samps_per_frame=10, fps=30,
                     rough_length=10.0, outname='triangle.mp4'):
    """
    Animate a corner plot.

    :param pos_T:
    A ``T x N x dim`` array, containing ``T`` timesteps of ``N`` evolving
    samples of a ``dim`` dimensional distribution.

    :param labels: (optional)
    A ``dim``-long list of parameter labels.

    :param truths: (optional)
    A ``dim``-long list of true parameters.

    :param samps_per_frame: (optional)
    A rough number of timesteps per frame.

    :param fps: (optional)
    The frame rate for the animation.

    :param rough_length: (optional)
    A rough request for the duration (in seconds) of the animation.

    : param outname: (optional)
    The name to use for saving the animation.

    """
    nframes, nwalkers, ndim = pos_T.shape

    final_bins = 50  # number of bins covering final posterior
    # Use last time step to get y-limits of histograms
    bins = []
    ymaxs = []
    for x in range(ndim):
        dx = (pos_T[-1, :, x].max() - pos_T[-1, :, x].min())/final_bins
        nbins = int((pos_T[0, :, x].max() - pos_T[0, :, x].min())/dx)
        these_bins = np.linspace(pos_T[0, :, x].min(),
                                 pos_T[0, :, x].max(), nbins + 1)[:-1]
        bins.append(these_bins)
        hist, _ = np.histogram(pos_T[-1, :, x], bins=bins[-1], normed=True)
        ymaxs.append(1.1*max(hist))

    # Use the first time sample as the initial frame
    fig = triangle.corner(pos_T[0], labels=labels,
                          plot_contours=False, truths=truths)
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for x in range(ndim):
        axes[x, x].set_ylim(top=ymaxs[x])

    # Determine number of frames
    thin_factor = int(nframes/rough_length)/fps
    if thin_factor > 1:
        pos_T = pos_T[::thin_factor]
        samps_per_frame *= thin_factor
    samps_per_sec = fps * samps_per_frame

    # Make the movie
    anim = animation.FuncAnimation(fig, update_triangle,
                                   frames=xrange(len(pos_T)), blit=True,
                                   fargs=(pos_T, fig, bins, truths))
    return anim


def update_triangle(i, data, fig, bins, truths=None):
    """
    Routine used to update a corner plot when animating.
    """
    ndim = data.shape[-1]
    axes = np.array(fig.axes).reshape((ndim, ndim))

    # Update histograms along diagonal
    for x in range(ndim):
        ax = axes[x, x]

        # Save bins and y-limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Clean current histrogram while keeping ticks
        while len(ax.patches) > 0:
            ax.patches[0].remove()

        ax.hist(data[i, :, x], range=xlim,
                bins=bins[x], histtype='step', normed=True, color='k')
        ax.set_ylim(*ylim)
        if truths is not None:
                ax.axvline(truths[x], color="#4682b4")

    # Update scatter plots
    for x in range(1, ndim):
        for y in range(x):
            ax = axes[x, y]
            line = ax.get_lines()[0]
            line.set_data(data[i, :, y], data[i, :, x])

    return fig,
