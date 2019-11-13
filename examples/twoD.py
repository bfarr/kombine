"""
A 2D example.
"""

import kombine

import numpy as np
from scipy.ndimage.filters import median_filter
from scipy.interpolate import RectBivariateSpline

try:
    from scipy.misc import imread
except ImportError:
    print(
        "This example uses scipy.misc.imread, which depends on Pillow (`pip install pillow`)"
    )

from matplotlib import animation as mpl_animation

try:
    import triangle
except ImportError:
    triangle = None

try:
    import prism
except ImportError:
    prism = None


class Posterior(object):
    """
    Posterior class for sampling a 2-D image.
    """

    def __init__(self, inp_img):
        self.ndim = 2

        # Load up the image (greyscale) and filter to soften it up
        img = median_filter(imread(inp_img, flatten=True), 5)

        # Convert 'ij' indexing to 'xy' coordinates
        self.img = np.flipud(img).T
        self._lower_left = np.array([0.0, 0.0])
        self._upper_right = self.img.shape

        # Construct a spline interpolant to use as a target
        x = np.arange(self._lower_left[0], self._upper_right[0], 1)
        y = np.arange(self._lower_left[1], self._upper_right[1], 1)
        self._interpolant = RectBivariateSpline(x, y, self.img)

    def prior_draw(self, N=1):
        """
        Draw ``N`` samples from the prior.
        """
        p = np.random.ranf(size=(N, self.ndim))
        p = (self._upper_right - self._lower_left) * p + self._lower_left
        return p

    def lnprior(self, X):
        """
        Use a uniform, bounded prior.
        """
        if np.any(X < self._lower_left) or np.any(X > self._upper_right):
            return -np.inf
        else:
            return 0.0

    def lnlike(self, X):
        """
        Use a softened version of the interpolant as a likelihood.
        """
        return -3.5 * np.log(self._interpolant(X[0], X[1], grid=False))

    def lnpost(self, X):
        return self.lnprior(X) + self.lnlike(X)

    def __call__(self, X):
        return self.lnpost(X)


lnprob = Posterior("../docs/_static/kombine.png")

# Initially distribute the ensemble across the prior
nwalkers = 1000
ndim = 2
sampler = kombine.Sampler(nwalkers, ndim, lnprob)
p = lnprob.prior_draw(nwalkers)

# Sample for a bit
p, prob, q = sampler.run_mcmc(200, p)

if triangle is None:
    print("Get triangle.py for some awesome corner plots!")
    print("https://github.com/dfm/triangle.py")

else:
    triangle.corner(p)
    fig = triangle.corner(p)
    fig.savefig("triangle.png")

if prism is None:
    print("Get prism and some popcorn for a sweet movie!")
    print("https://github.com/bfarr/prism")

else:
    # Animate the ensemble's progress
    anim = prism.corner(sampler.chain)

    # Write the animation to file
    writer = mpl_animation.writers["ffmpeg"](fps=30, bitrate=20000)
    anim.save("kombine.mp4", writer=writer)

np.savetxt("samples.dat", p)
