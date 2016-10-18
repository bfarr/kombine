#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import textwrap

import numpy as np
import scipy

from scipy.stats.distributions import f

from .interruptible_pool import disable_openblas_threading as ip_disable_threading

def bad_blas_msg(pkg_name):
    """A warning message to give the user"""
    return textwrap.dedent(
        """
        {0} linked against 'Accelerate.framework', which  doesn't play nicely
        with 'multiprocessing'.

        Building {0} against OpenBLAS can avoid this, e.g.:

            brew tap homebrew/python
            brew update && brew upgrade

            brew install openblas

            brew install {0} --with-openblas

        To maintain stability, multiprocessing won't be used.
        """.format(pkg_name))

def get_config_info(pkg):
    """Extract config info from Scipy or Numpy"""
    try:
        config_info = str([value for key, value in
                           pkg.__config__.__dict__.iteritems()
                           if key.endswith("_info")]).lower()
    except AttributeError:
        config_info = str([value for key, value in
                           pkg.__config__.__dict__.items()
                           if key.endswith("_info")]).lower()
    return config_info

def mp_safe_blas():
    """Check if BLAS implementation is known to not play nicely with :mod:`multiprocessing`."""
    safe_blas = True

    np_config_info = get_config_info(np)
    scipy_config_info = get_config_info(scipy)

    if "accelerate" in np_config_info or "veclib" in np_config_info:
        warnings.warn(bad_blas_msg('numpy'))
        safe_blas = False

    if "accelerate" in scipy_config_info or "veclib" in scipy_config_info:
        warnings.warn(bad_blas_msg('scipy'))
        safe_blas = False

    return safe_blas

def disable_openblas_threading():
    """If openblas is linked, disable threading to avoid extra overhead"""
    np_config_info = get_config_info(np)
    scipy_config_info = get_config_info(scipy)

    if "openblas" in np_config_info or "openblas" in scipy_config_info:
        ip_disable_threading()


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
