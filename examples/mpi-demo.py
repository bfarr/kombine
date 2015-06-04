# coding: utf-8

"""
    Demonstrate how to use MPI with kombine. Run this module with

        mpiexec -n <nprocesses> python mpi-demo.py

    where <nprocesses> is the number of processes to spawn.
 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import sys

# Third-party
import kombine
import numpy as np
from scipy.stats import multivariate_normal

try:
    from mpipool import MPIPool
except ImportError:
    raise ImportError("mpipool (https://github.com/dfm/mpipool/) is required "
                      "for this example, along with a working installation of "
                      "openMPI.")

class Model(object):
    def __init__(self, mean, cov):
        self.mean = np.atleast_1d(mean)
        self.cov = np.array(cov)
        self.ndim = self.cov.shape[0]

    def lnposterior(self, x):
        return multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov)

    def __call__(self, x):
        return self.lnposterior(x)

ndim = 3
A = np.random.rand(ndim, ndim)
mean = np.zeros(ndim)
cov = A*A.T + ndim*np.eye(ndim)

# create an ND Gaussian model
model = Model(mean, cov)

# define an MPI pool
pool = MPIPool()

# # Make sure the thread we're running on is the master
if not pool.is_master():
    pool.wait()
    sys.exit(0)

nwalkers = 500
sampler = kombine.Sampler(nwalkers, ndim, model, pool=pool)

p0 = np.random.uniform(-10, 10, size=(nwalkers, ndim))
p, post, q = sampler.burnin(p0)
p, post, q = sampler.run_mcmc(100)

# close the MPI poll
pool.close()

sys.exit(0)
