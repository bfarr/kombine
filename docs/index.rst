kombine
=======

``kombine`` is an ensemble sampler built for efficiently exploring multimodal distributions.
By using estimates of ensemble's instantaneous distribution as a proposal, it achieves very
fast burnin, followed by sampling with very short autocorrelation times.
Contents:


Example Usage
=============

To sample a simple 10D bimodal distribution::
    
    import numpy as np
    import kombine

    def lnprob(x, inv_var, nmodes=2):
        lnprobs = [-0.5 * np.sum(inv_var * (x + 5*mode*inv_var) ** 2) for mode in range(nmodes)]
        return np.logaddexp.reduce(lnprobs)

    ndim, nwalkers = 10, 1000
    nmodes = 10
    inv_var = 1. / np.random.rand(ndim)

    p0 = nmodes * np.random.rand(nwalkers, ndim)
    sampler = kombine.Sampler(nwalkers, ndim, lnprob)
    sampler.burnin(p0)


.. toctree::
    :maxdepth: 2

    intro
    tutorial


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

