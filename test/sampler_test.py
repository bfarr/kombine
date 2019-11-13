from kombine.sampler import Sampler
from ..examples.twoD import Posterior
import unittest


class TestSampler(unittest.TestCase):
    def setUp(self):
        self.lnpost = Posterior("../docs/_static/kombine.png")
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
