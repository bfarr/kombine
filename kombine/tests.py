#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for nose.
"""

from __future__ import division

import numpy as np
from scipy.stats import multivariate_normal

from .clustered_kde import KDE
from .clustered_kde import ClusteredKDE
from .clustered_kde import optimized_kde

from .clustered_kde import TransdimensionalKDE

from .sampler import Sampler

class MultimodalTestDistribution(object):
    def __init__(self, nmodes=1, ndim=2):
        self.nmodes = nmodes
        self.ndim = ndim

        dx = 1./(self.nmodes + 1)
        self.cov = .005*dx*np.eye(self.ndim)
        means = (1.+np.arange(nmodes))*dx
        self.means = np.column_stack([means for dim in range(self.ndim)])

    def draw(self, mode_size=1000):
        return np.vstack([multivariate_normal.rvs(mean, self.cov, size=mode_size) for mean in self.means])

    def logpdf(self, x):
        #x = np.atleast_2d(x)

        log_probs = [multivariate_normal.logpdf(x, mean, self.cov) - np.log(self.nmodes) for mean in self.means]
        return np.logaddexp.reduce(log_probs)

    def kl_divergence(self, kde, size=1000):
        """Compute the KL divergence to see if the distributions are consistent"""
        test_pts = self.draw(size)

        logpk = kde(test_pts)
        logqk = self(test_pts)
        logpk -= np.logaddexp.reduce(logpk)
        logqk -= np.logaddexp.reduce(logqk)
        kl_div = np.sum(np.exp(logpk) * (logpk - logqk))
        return kl_div

    @property
    def ln_marginal_prob(self):
        # This is a normalized distribution
        return 0.0

    def __call__(self, x):
        return self.logpdf(x)

# Check the KDE proposals
def check_kde_estimate(kde, test_dist, kl_thresh=0.02):
    D = test_dist.kl_divergence(kde)
    assert D < kl_thresh, \
        ("KDE is inconsistent with test distribution.  KL divergence {0:g} is above " \
         "the threshold {1:g}").format(D, kl_thresh)

def check_kde_normalization(kde, thresh=1e-5):
    x = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, x)
    positions = np.column_stack([X.ravel(), Y.ravel()])

    pdf = np.exp(np.reshape(kde(positions), X.shape))
    prob = np.trapz(np.trapz(pdf, x), x)
    assert np.abs(1. - prob) < thresh, \
        "KDE not properly normalized.  KDE found to integrate to {:.7f}".format(prob)

def check_kde(kde, test_dist):
    check_kde_normalization(kde)
    check_kde_estimate(kde, test_dist)

def test_base_kde():
    nmodes = 1
    test_dist = MultimodalTestDistribution(nmodes=nmodes)
    training_pts = test_dist.draw()
    kde = KDE(training_pts)
    check_kde(kde, test_dist)

def test_clustered_kde():
    nmodes = 3
    test_dist = MultimodalTestDistribution(nmodes=nmodes)
    training_pts = test_dist.draw()
    kde = ClusteredKDE(training_pts, k=nmodes)
    check_kde(kde, test_dist)

def test_optimized_kde():
    nmodes = 5
    test_dist = MultimodalTestDistribution(nmodes=nmodes)
    training_pts = test_dist.draw()
    kde = optimized_kde(training_pts)
    check_kde(kde, test_dist)

# Check the sampler

log_threshold = -3
std_threshold = 3

class TestSampler:
    def setUp(self):
        self.nwalkers = 128
        self.ndim = 3
        self.nmodes = 2
        self.nsteps = 50
        self.update_interval = 10
        self.split = 0.5

        self.target = MultimodalTestDistribution(self.nmodes, self.ndim)
        self.p0 = np.random.uniform(-1, 1, size=(self.nwalkers, self.ndim))

    def check_sampling(self, nsteps=None, p0=None, update_interval=None):
        if nsteps is None:
            nsteps = self.nsteps
        if p0 is None:
            p0 = self.p0
        if update_interval is None:
            update_interval = self.update_interval

        for i in self.sampler.sample(p0, iterations=nsteps, update_interval=update_interval):
            pass

        p = self.sampler.chain[-1]
        mode_sel = [np.all(p < self.split, axis=1), np.all(p > self.split, axis=1)]
        count_std = self.nwalkers * 1/self.nmodes * (1 - 1/self.nmodes)

        for mean, sel in zip(self.target.means, mode_sel):
            assert np.abs(np.count_nonzero(sel) - self.nwalkers/self.nmodes) < std_threshold * count_std
            assert np.all((np.mean(p[sel], axis=0) - mean) ** 2 < 10. ** log_threshold)
            assert np.all((np.cov(p[sel], rowvar=0) - self.target.cov) ** 2 < 10. ** log_threshold)

        # Check marginal likelihood estimate
        lnZ, dlnZ = self.sampler.ln_ev(self.nwalkers)
        assert np.abs(lnZ - self.target.ln_marginal_prob) < std_threshold * dlnZ

    def test_sampler_serially(self):
        self.sampler = Sampler(self.nwalkers, self.ndim, self.target, processes=1)
        self.check_sampling()

    def test_sampler_parallelly(self):
        self.sampler = Sampler(self.nwalkers, self.ndim, self.target, processes=4)
        self.check_sampling()

    def test_burnin(self):
        self.sampler = Sampler(self.nwalkers, self.ndim, self.target)
        self.sampler.burnin(self.p0)
        self.check_sampling(nsteps=0)

    def test_blobs(self):
        blobby_target = lambda p: (self.target(p), np.random.randn())
        self.sampler = Sampler(self.nwalkers, self.ndim, blobby_target, processes=1)
        self.check_sampling()

        blobs = np.array(self.sampler.blobs)
        assert (self.sampler.chain.shape == (self.nsteps, self.nwalkers, self.ndim)
                and blobs.shape == (self.nsteps, self.nwalkers)), \
                    "You broke the blob!"

        # Make sure some blobs were updated
        assert len(np.unique(blobs)) > self.nwalkers, \
            "blobs repeated: {} != {} {}".format(len(np.unique(blobs)), len(blobs), blobs.shape)
