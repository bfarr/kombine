#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for nose.
"""

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import entropy

from .clustered_kde import KDE
from .clustered_kde import ClusteredKDE
from .clustered_kde import optimized_kde

from .clustered_kde import TransdimensionalKDE

from .sampler import Sampler

class MultimodalTestDistribution(object):
    def __init__(self, nmodes=1, ndim=2):
        self._nmodes = nmodes
        self._ndim = ndim

        dx = 1./(self._nmodes + 1)
        self._cov = .005*dx*np.eye(self._ndim)
        means = (1.+np.arange(nmodes))*dx
        self._means = np.column_stack([means for dim in range(self._ndim)])

    def draw(self, mode_size=1000):
        return np.vstack([np.random.multivariate_normal(mean, self._cov, size=mode_size) for mean in self._means])

    def pdf(self, x):
        x = np.atleast_2d(x)
        prob = np.sum([multivariate_normal.pdf(x, mean, self._cov) for mean in self._means], axis=0)/self._nmodes
        return prob

    def is_consistent(self, kde, size=1000, kl_thresh=0.01):
        """Compute the KL divergence to see if the distributions are consistent"""
        test_pts = self.draw(size)

        D = entropy(np.exp(kde(test_pts)), self(test_pts))
        return D < kl_thresh

    def __call__(self, x):
        return self.pdf(x)


# Check the KDE proposals
def check_kde_estimate(kde, test_dist):
    assert test_dist.is_consistent(kde)

def check_kde_normalization(kde, thresh=1e-3):
    x = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, x)
    positions = np.column_stack([X.ravel(), Y.ravel()])

    pdf = np.exp(np.reshape(kde(positions), X.shape))
    prob = np.trapz(np.trapz(pdf, x), x)
    assert np.abs(1. - prob) < thresh

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
