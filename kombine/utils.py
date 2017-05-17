#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import textwrap

import numpy as np
import scipy

from scipy.stats.distributions import f

def gelman_rubin(chains, return_cdf=False):
    """
    Compute the Gelman-Rubin R-statistic from an ensemble of chains.  `chains`
    is expected to have shape `(nsteps, nchains)` if samples are one dimensional,
    or `(nsteps, nchains, ndim)` if multidimensional.  For multidimensional samples
    R-statistics will be computed for each dimension.

    :param chains:
        An `(nsteps, nchains)` or `(nsteps, nchains, ndim)`-shaped array.

    :param return_cdf: (optional)
        If ``True``, the CDF of the R-statistic(s), assuming an F-distribution, are
        returned in addition to the R-statistic(s).
    """
    if len(chains.shape) > 2:
        results = [gelman_rubin(chains[..., param], return_cdf=return_cdf)
                   for param in range(chains.shape[-1])]
        if return_cdf:
            return zip(*results)
        else:
            return results

    nchains, nsteps = chains.shape[1], chains.shape[0]

    chain_means = np.mean(chains, axis=0)
    chain_vars = np.var(chains, axis=0)

    # between-chain variance
    interchain_var = np.sum((chain_means - np.mean(chains)) ** 2) / (nchains - 1)

    # within-chain variances
    intrachain_vars = (chains - chain_means)**2 / (nsteps - 1)
    intrachain_var = np.sum(intrachain_vars)/nchains

    var = intrachain_var * (nsteps - 1) / nsteps + interchain_var
    post_var = var + interchain_var / nchains

    # The Statistic
    R = np.sqrt(post_var / intrachain_var)

    if return_cdf:
        # R should be F-distributed
        dof1 = nchains - 1
        dof2 = 2*intrachain_var**2*nchains/np.var(intrachain_vars)
        return R, f.cdf(R, dof1, dof2)
    else:
        return R
