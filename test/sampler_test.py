from kombine.sampler import Sampler
import unittest
import numpy as np
from scipy.ndimage.filters import median_filter
from scipy.interpolate import RectBivariateSpline

try:
    from scipy.misc import imread
except ImportError:
    print(
        "This example uses scipy.misc.imread, which depends on Pillow (`pip "
        "install pillow`)"
    )


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


class TestSampler(unittest.TestCase):
    def setUp(self):
        self.lnpost = Posterior("static_test_files/kombine.png")
        self.nwalkers = 1000
        self.ndim = 2
        self.sampler = Sampler(self.nwalkers, self.ndim, self.lnpost)
        self.p0 = self.lnpost.prior_draw(self.nwalkers)

    def tearDown(self):
        del self.sampler

    def test_burnin(self):
        _ = self.sampler.burnin(p0=self.p0)

    def test_run_mcmc(self):
        _ = self.sampler.run_mcmc(200, self.p0)
