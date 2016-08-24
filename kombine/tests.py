#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for nose.
"""

import numpy as np
from scipy.stats import kstest

from .clustered_kde import KDE
from .clustered_kde import ClusteredKDE
from .clustered_kde import optimized_kde

from .clustered_kde import TransdimensionalKDE

from .sampler import Sampler

# Check the KDE proposals

def draw_multimodal_samples(nmodes, ndim=2, mode_size=1000):
    dx = 1./(nmodes+1)
    cov = .005*dx*np.eye(ndim)

    means = (1.+np.arange(nmodes))*dx
    means = np.column_stack([means for dim in range(ndim)]) 
    draws = np.vstack([np.random.multivariate_normal(mean, cov, size=mode_size) for mean in means])
    return draws

def check_kde_draws(kde):
    draw_size=1000

    def draw(size=None):
        draws = kde.draw(size)
        probs = np.exp(kde(draws))
        return np.cumsum(probs)/np.sum(probs)

    pval = kstest(draw, 'uniform', N=draw_size)[1]
    assert pval > 0.001

def check_kde_normalization(kde):
    thresh = 1e-3

    x = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, x)
    positions = np.column_stack([X.ravel(), Y.ravel()])

    pdf = np.exp(np.reshape(kde(positions), X.shape))
    prob = np.trapz(np.trapz(pdf, x), x)
    assert np.abs(1. - prob) < thresh

def check_kde(kde):
    check_kde_normalization(kde)
    check_kde_draws(kde)

def test_base_kde():
    training_pts = draw_multimodal_samples(1)
    kde = KDE(training_pts)
    check_kde(kde)

def test_clustered_kde():
    nmodes = 3
    training_pts = draw_multimodal_samples(nmodes)
    kde = ClusteredKDE(training_pts, k=nmodes)
    check_kde(kde)

def test_optimized_kde():
    nmodes = 5
    training_pts = draw_multimodal_samples(nmodes)
    kde = optimized_kde(training_pts)
    check_kde(kde)

# Check the sampler
