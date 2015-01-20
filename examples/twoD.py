"""
A 2D example.
"""

import kombine

import numpy as np
from scipy.misc import imread
from scipy.ndimage.filters import median_filter
from scipy.interpolate import RectBivariateSpline

from matplotlib import animation as mpl_animation


# Load up the image (greyscale) and filter to soften it up
img = median_filter(imread('./kombine.png', flatten=True), 5)

# Convert 'ij' indexing to 'xy' coordinates
img = np.flipud(img).T
lower_left = np.array([0., 0.])
upper_right = img.shape

# Construct a spline interpolant to use as a target
x = np.arange(lower_left[0], upper_right[0], 1)
y = np.arange(lower_left[1], upper_right[1], 1)
f = RectBivariateSpline(x, y, img)

# Use a uniform (bounded) prior
def lnprior(X):
    if np.any(X < lower_left) or np.any(X > upper_right):
        return np.NINF
    else:
        return 0.0

# Use a softened version of the interpolant as a likelihood
def lnlike(X):
    return -3.5*np.log(f(X[0], X[1], grid=False))

# Construct a pool if multiprocessing is available
try:
    import multiprocessing as mp
    pool = mp.Pool()

except ImportError:
    pool = None


# Initially distribute the ensemble across the prior
nwalkers = 1000
dim = 2
p = np.random.ranf(size=(nwalkers, dim))
p = (upper_right - lower_left) * p + lower_left

sampler = kombine.Sampler(nwalkers, dim, lnprior, lnlike, pool=pool)

# Sample for a bit
p, prior, like, q = sampler.sample(p, iterations=200)

# Animate the ensemble's progress
anim = sampler.animate()

# Write the animation to file
writer = mpl_animation.writers['ffmpeg'](fps=30, bitrate=20000)
anim.save("kombine.mp4", writer=writer)
