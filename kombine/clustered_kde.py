#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The kernel density estimators.
"""

from __future__ import (division, print_function, absolute_import, unicode_literals)

import numpy as np
import numpy.ma as ma
from scipy.misc import logsumexp
from scipy import linalg as la
from scipy.cluster.vq import kmeans, vq

# Avoid log(0) warnings when weights go to 0
np.seterr(divide='ignore')


def optimized_kde(data, pool=None, kde=None, max_samples=None, **kwargs):
    """
    Iteratively run a k-means clustering algorithm, estimating the distibution of each identified
    cluster with an independent kernel density estimate.  Starting with ``k = 1``, the distribution
    is estimated and the Bayes Information criterion (BIC) is calculated.  `k` is increased until
    the BIC stops increasing.

    :param data:
        An `(N, ndim)`-shaped array, containing `N` samples from the target distribution.

    :param pool: (optional)
        A pool of processes with a :func:`map` function to use.

    :param kde: (optional)
        An old KDE to inherit samples from.

    :param max_samples: (optional)
        The maximum number of samples to use for constructing or updating the KDE.  If a KDE is
        supplied and adding the samples from it will go over this, old samples are thinned by
        factors of two until under the limit.

    :param kwargs: (optional)
        Keyword arguments to pass to :class:`ClusteredKDE`.

    :returns: :meth:`ClusteredKDE` that maximizes the BIC.
    """
    # Trim data if too many samples were given
    n_new = len(data)

    if kde is None and n_new == 0:
        return None

    if max_samples is not None and max_samples <= n_new:
        data = data[:max_samples]

    else:
        # Combine data, thinning old data if we need room
        if kde is not None:
            old_data = kde.data

            if max_samples is not None:
                nsamps = len(old_data) + n_new

                while nsamps > max_samples:
                    old_data = old_data[::2]
                    nsamps = len(old_data) + n_new

            if n_new == 0:
                # If there's no new data, just use the old
                data = old_data
            else:
                # Otherwise combine the old and the new
                data = np.concatenate((old_data, data))

    best_bic = -np.inf
    best_kde = None

    k = 1
    while True:
        try:
            kde = ClusteredKDE(data, k, **kwargs)
            bic = kde.bic(pool=pool)
        except la.LinAlgError:
            bic = -np.inf

        if bic > best_bic:
            best_kde = kde
            best_bic = bic
        else:
            break
        k += 1

    return best_kde


class ClusteredKDE(object):
    """
    Run a k-means clustering algorithm, estimating the distibution of each identified cluster with
    an independent kernel density estimate.  The full distibution is then estimated by combining the
    individual KDE's, weighted by the fraction of samples assigned to each cluster.

    :param data:
        An `(N, ndim)`-shaped array, containing `N` samples from the target distribution.

    :param k:
        The number of clusters for k-means clustering.
    """
    def __init__(self, data, k=1):
        self._data = data
        self._nclusters = k

        self._mean = np.mean(data, axis=0)
        self._std = np.std(data, axis=0)

        # Cluster data that's mean 0 and scaled to unit width in each parameter independently
        white_data = self._whiten(data)
        self._centroids, _ = kmeans(white_data, k)
        self._assignments, _ = vq(white_data, self.centroids)

        self._kdes = [KDE(self.data[self.assignments == c]) for c in range(k)]
        self._logweights = np.log([np.count_nonzero(self.assignments == c)/self.size
                                   for c in range(k)])

    def draw(self, size=1):
        """Draw `size` samples from the KDE."""
        # Pick clusters randomly with the assigned weights
        cumulative_weights = np.cumsum(np.exp(self._logweights))
        clusters = np.searchsorted(cumulative_weights, np.random.rand(size))

        draws = np.empty((size, self.ndim))
        for cluster in range(self.nclusters):
            sel = clusters == cluster
            draws[sel] = self._kdes[cluster].draw(np.count_nonzero(sel))

        return draws

    def logpdf(self, pts, pool=None):
        """Evaluate the logpdf of the KDE at `pts`."""
        logpdfs = [logweight + kde(pts, pool=pool)
                   for logweight, kde in zip(self._logweights, self._kdes)]
        if len(pts.shape) == 1:
            return logsumexp(logpdfs)
        else:
            return logsumexp(logpdfs, axis=0)

    def _whiten(self, data):
        """Whiten `data`, probably before running k-means."""
        return (data - self._mean)/self._std

    def _color(self, data):
        """Recolor `data`, reversing :meth:`_whiten`."""
        return data * self._std + self._mean

    def bic(self, pool=None):
        r"""
        Evaluate Bayes Information Criterion for the KDE's estimate of the distribution

        .. math::

            \mathrm{BIC} = \mathrm{ln}\mathcal{L}_\mathrm{max} - \frac{d_m}{2} \mathrm{ln} N

        where :math:`d_m` is the number of dimensions of the KDE model (:math:`n_\mathrm{clusters}
        d` centroid location parameters, :math:`n_\mathrm{clusters} - 1` normalized weights, and
        :math:`n_\mathrm{clusters} (d+1)*d/2` kernel covariance parameters, one matrix for each of
        :math:`n_\mathrm{clusters}` clusters), and :math:`N` is the number of samples used to build
        the KDE.
        """
        log_l = np.sum(self.logpdf(self.data, pool=pool))

        # Determine the total number of parameters in clustered-KDE
        # Account for centroid locations
        nparams = self.nclusters * self.ndim

        # One for each cluster, minus one for constraint that all sum to unity
        nparams += self.nclusters - 1

        # Separate kernel covariances for each cluster
        nparams += self.nclusters * (self.ndim + 1) * self.ndim/2

        return log_l - nparams/2 * np.log(self.size)

    @property
    def data(self):
        """Samples used to build the KDE."""
        return self._data

    @property
    def nclusters(self):
        """The number of clusters used for k-means."""
        return self._nclusters

    @property
    def assignments(self):
        """Cluster assignments from k-means."""
        return self._assignments

    @property
    def centroids(self):
        """Cluster centroids from k-means."""
        return self._centroids

    @property
    def ndim(self):
        """The number of dimensions of the KDE."""
        return self.data.shape[1]

    @property
    def size(self):
        """The number of samples used to build the KDE."""
        return self.data.shape[0]

    __call__ = logpdf

    __len__ = size


class KDE(object):
    """
    A Gaussian kernel density estimator that provides a means for evaluating the estimated
    probability density function, and drawing additional samples from the estimated distribution.
    Cholesky decomposition of the covariance matrix makes this class a bit more stable than the
    :mod:`scipy`'s Gaussian KDE.

    :param data:
        An `(N, ndim)`-shaped array, containing `N` samples from the target distribution.

    """
    def __init__(self, data):
        self._data = np.atleast_2d(data)

        self._mean = np.mean(data, axis=0)
        self._cov = None

        if self.data.shape[0] > 1:
            try:
                self._cov = np.cov(data.T)

                # Try factoring now to see if regularization is needed
                la.cho_factor(self._cov)

            except la.LinAlgError:
                self._cov = oas_cov(data)

        self._set_bandwidth()

    def __enter__(self):
        return self

    def _set_bandwidth(self):
        r"""
        Use Scott's rule to set the kernel bandwidth:

        .. math::

            \mathcal{K} = n^{-1/(d+4)} \Sigma^{1/2}

        Also store Cholesky decomposition for later.
        """
        if self.size > 0 and self._cov is not None:
            self._kernel_cov = self._cov * self.size ** (-2/(self.ndim + 4))

            # Used to evaluate PDF with cho_solve()
            self._cho_factor = la.cho_factor(self._kernel_cov)

            # Make sure the estimated PDF integrates to 1.0
            self._lognorm = self.ndim/2 * np.log(2*np.pi) + np.log(self.size) +\
                np.sum(np.log(np.diag(self._cho_factor[0])))

        else:
            self._lognorm = -np.inf

    def draw(self, size=1):
        """
        Draw samples from the estimated distribution.
        """
        # Return nothing if this is an empty KDE
        if self.size == 0:
            return []

        # Draw vanilla samples from a zero-mean multivariate Gaussian
        draws = np.random.multivariate_normal(np.zeros(self.ndim), self._kernel_cov, size=size)

        # Pick N random kernels as means
        kernels = np.random.randint(0, self.size, size)

        # Shift vanilla draws to be about chosen kernels
        return self.data[kernels] + draws

    def logpdf(self, pts, pool=None):
        """Evaluate the logpdf at `pts` as estimated by the KDE."""
        pts = np.atleast_2d(pts)

        npts, ndim = pts.shape
        assert ndim == self.ndim

        # Apply across the pool if it exists
        if pool:
            this_map = pool.map
        else:
            this_map = map

        # Return -inf if this is an empty KDE
        if np.isinf(self._lognorm):
            results = np.zeros(npts) - np.inf

        else:
            args = [(pt, self.data, self._cho_factor) for pt in pts]
            results = list(this_map(_evaluate_point_logpdf, args))

        # Normalize and return
        return np.array(results) - self._lognorm

    @property
    def data(self):
        """Samples used to build the KDE."""
        return self._data

    @property
    def ndim(self):
        """The number of dimensions of the KDE."""
        return self.data.shape[1]

    @property
    def size(self):
        """The number of samples used to build the KDE."""
        return self.data.shape[0]

    __len__ = size

    __call__ = logpdf


def unique_spaces(mask):
    """
    Determine the unique sets of dimensions based on a mask.  Inverted 1D masks are returned to use
    as selectors.
    """
    ncols = mask.shape[1]

    # Do some magic with views so `np.unique` can be used to find the unique sets of dimensions.
    dtype = mask.dtype.descr * ncols
    struct = mask.view(dtype)

    uniq = np.unique(struct)
    uniq = uniq.view(mask.dtype).reshape(-1, ncols)
    return ~uniq


class TransdimensionalKDE(object):
    """
    A generalized Gaussian kernel density estimator that reads masked arrays, constructs a
    :class:`ClusteredKDE` using :func:`optimized_kde` for each unique parameter space, then weighs
    the KDEs based on the number of samples in each parameter space.

    :param data:
        An `(N, max_dim)`-shaped masked array, containing N samples from the the target distribution.

    :param kde: (optional)
        An old trans-dimensional KDE to inherit samples from.

    :param max_samples: (optional)
        The maximum number of samples to use for constructing or updating the kde in each unique
        parameter space. If a KDE is supplied and adding the samples from `data` will go over this,
        old samples are thinned by factors of two until under the limit in each parameter space.
    """
    def __init__(self, data, kde=None, max_samples=None, pool=None):
        npts_new, max_ndim = data.shape
        self._max_ndim = max_ndim

        if kde is None:
            # Save an (inverted) mask for each unique set of dimensions
            self._spaces = unique_spaces(data.mask)
        else:
            # Inherit old space definitions, in case the new sample has no points in a subspace
            self._spaces = kde.spaces

        # Construct a separate clustered-KDE for each parameter space
        weights = []
        self._kdes = []
        for space_id, space in enumerate(self.spaces):
            # Construct a selector for the samples from this space
            subspace = np.all(~data.mask == space, axis=1)

            # Determine weights from only the new samples
            npts_subspace = np.count_nonzero(subspace)
            weight = npts_subspace/npts_new
            weights.append(weight)

            fixd_data = data[subspace]
            if npts_subspace > 0:
                fixd_data = np.asarray(fixd_data[~fixd_data.mask].reshape((npts_subspace, -1)))

            old_kde = None
            if kde is not None:
                old_kde = kde.kdes[space_id]

            self._kdes.append(optimized_kde(fixd_data, pool, old_kde, max_samples))

        self._logweights = np.log(np.array(weights))

    def draw(self, size=1, spaces=None):
        """
        Draw samples from the transdimensional distribution.
        """
        if spaces is not None:
            if len(spaces) != size:
                raise ValueError('Sample size inconsistent with number of spaces saved')
            space_inds = np.empty(size)
            for space_id, space in enumerate(self.spaces):
                subspace = np.all(spaces == space, axis=1)
                space_inds[subspace] = space_id

        else:
            # Draws spaces randomly with the assigned weights
            cumulative_weights = np.cumsum(np.exp(self._logweights))
            space_inds = np.searchsorted(cumulative_weights, np.random.rand(size))

        draws = ma.masked_all((size, self._max_ndim))
        for space_id in range(len(self.spaces)):
            sel = space_inds == space_id
            n_fixedd = np.count_nonzero(sel)
            if n_fixedd > 0:
                # Populate only the valid entries for this parameter space
                draws[np.ix_(sel, self._spaces[space_id])] = self.kdes[space_id].draw(n_fixedd)

        return draws

    def logpdf(self, pts, pool=None):
        """Evaluate the log-transdimensional-pdf at `pts` as estimated by the KDE."""
        logpdfs = []
        for logweight, space, kde in zip(self._logweights,
                                         self.spaces,
                                         self.kdes):
            # Calculate the probability for each parameter space individually
            if np.all(space == ~pts.mask) and np.isfinite(logweight):
                logpdfs.append(logweight + kde(pts[space], pool=pool))

        return logsumexp(logpdfs, axis=0)

    @property
    def kdes(self):
        """List of fixed-dimension :meth:`ClusteredKDE` s"""
        return self._kdes

    @property
    def spaces(self):
        """Unique sets of dimensions, usable as selectors."""
        return self._spaces

    __call__ = logpdf


def _evaluate_point_logpdf(args):
    """
    Evaluate the Gaussian KDE at a given point `p`.  This lives outside the KDE method to allow for
    parallelization using :mod:`multipocessing`. Since :func:`map` only allows single-argument
    functions, the following arguments to be packed into a single tuple.

    :param p:
        The point to evaluate the KDE at.

    :param data:
        The `(N, ndim)`-shaped array of data used to construct the KDE.

    :param cho_factor:
        A Cholesky decomposition of the kernel covariance matrix.
    """
    point, data, cho_factor = args

    # Use Cholesky decomposition to avoid direct inversion of covariance matrix
    diff = data - point
    tdiff = la.cho_solve(cho_factor, diff.T, check_finite=False).T
    diff *= tdiff

    # Work in the log to avoid large numbers
    return logsumexp(-np.sum(diff, axis=1)/2)


def oas_cov(pts):
    r"""
    Estimate the covariance matrix using the Oracle Approximating Shrinkage algorithm

    .. math::

        (1 - s)\Sigma + s \mu \mathcal{I}_d

    where :math:`\mu = \mathrm{tr}(\Sigma) / d`.  This ensures the covariance matrix estimate is
    well behaved for small sample sizes.

    :param pts:
        An `(N, ndim)`-shaped array, containing `N` samples from the target distribution.


    This follows the implementation in `scikit-learn
    <https://github.com/scikit-learn/scikit-learn/blob/31c5497/
    sklearn/covariance/shrunk_covariance_.py>`_.
    """
    pts = np.atleast_2d(pts)
    npts, ndim = pts.shape

    emperical_cov = np.cov(pts.T)
    mean = np.trace(emperical_cov) / ndim

    alpha = np.mean(emperical_cov * emperical_cov)
    num = alpha + mean * mean
    den = (npts + 1) * (alpha - (mean * mean) / ndim)

    shrinkage = min(num / den, 1)
    shrunk_cov = (1 - shrinkage) * emperical_cov
    shrunk_cov[np.diag_indices(ndim)] += shrinkage * mean

    return shrunk_cov
