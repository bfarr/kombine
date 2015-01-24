import numpy as np

from .clustered_kde import optimized_kde


class GetLnProbWrapper(object):
    def __init__(self, lnprior, lnlike, kde):
        self.lnprior = lnprior
        self.lnlike = lnlike
        self.kde = kde

    def __call__(self, p):
        lnprior = self.lnprior(p)
        if lnprior == np.NINF:
            lnlike = 0.0
            kde = 0.0
        else:
            lnlike = self.lnlike(p)
            kde = self.kde(p)

        return np.array([lnprior, lnlike, kde])


class Sampler(object):
    """
    An Ensemble sampler.

    The :attr:`chain` member of this object has the shape:
    ``(nsteps, nwalkers, dim)`` where ``nsteps`` is the number of steps

    :param nwalkers:
        The number of individual MCMC chains to include in the ensemble.

    :param dim:
        Number of dimensions in the parameter space.

    :param lnpriorfn:
        A function that takes a vector in the parameter space as input and
        returns the natural logarithm of the prior probability for that
        position.

    :param lnlikefn:
        A function that takes a vector in the parameter space as input and
        returns the natural logarithm of the likelihood for that position.

    """
    def __init__(self, nwalkers, ndim, lnpriorfn, lnlikefn, pool=None):
        self.nwalkers = nwalkers
        self.dim = ndim

        self._get_lnprior = lnpriorfn
        self._get_lnlike = lnlikefn

        self.iterations = 0

        self._kde = None

        self._pool = pool

        self.accepted = np.zeros((0, self.nwalkers))
        self._chain = np.empty((0, self.nwalkers, self.dim))
        self._lnpost = np.empty((0, self.nwalkers))
        self._lnprop = np.empty((0, self.nwalkers))

    def sample(self, p0, lnprior0=None, lnlike0=None, lnq0=None,
               iterations=1, update_interval=10):
        """
        Advance the ensemble ``iterations`` steps.

        :param p0:
            A list of the initial walker positions.  It should have the
            shape ``(nwalkers, dim)``.

        :param lnprior0: (optional)
            The list of log prior probabilities for the walkers at
            positions ``p0``. If ``lnprior0 is None``, the initial
            values are calculated. It should have the shape
            ``(nwalkers, dim)``.

        :param lnlike0: (optional)
            The list of log likelihoods for the walkers at
            positions ``p0``. If ``lnlike0 is None``, the initial
            values are calculated. It should have the shape
            ``(nwalkers, dim)``.

        :param lnq0: (optional)
            The list of log proposal densities for the walkers at
            positions ``p0``. If ``lnq0 is None``, the initial
            values are calculated. It should have the shape
            ``(nwalkers, dim)``.

        :param iterations: (optional)
            The number of steps to run.

        :param update_interval: (optional)
            The number of steps between proposal updates.

        After ``iteration`` steps, this method returns:

        * ``p`` - A list of the current walker positions, the shape of which
            will be ``(nwalkers, dim)``.

        * ``lnprior`` - The list of log prior probabilities for the
          walkers at positions ``p``, with shape ``(nwalkers, dim)``.

        * ``lnlike`` - The list of log likelihoods for the
          walkers at positions ``p``, with shape ``(nwalkers, dim)``.

        * ``lnq`` - The list of log proposal densities for the
          walkers at positions ``p``, with shape ``(nwalkers, dim)``.

        """
        p = np.array(p0)

        if self._pool is None:
            m = map
        else:
            m = self._pool.map

        # Build a proposal if one doesn'g already exist
        if self._kde is None:
            self._kde = optimized_kde(p, pool=self._pool)

        lnprior = lnprior0
        lnlike = lnlike0
        lnq = lnq0

        if lnprior is None or lnlike is None or lnq is None:
            results = np.array(m(GetLnProbWrapper(self._get_lnprior,
                                                  self._get_lnlike,
                                                  self._kde), p))
            lnprior = results[:, 0]
            lnlike = results[:, 1]
            lnq = results[:, 2]

        # Prepare arrays for storage ahead of time
        self._chain = np.concatenate(
            (self._chain, np.zeros((iterations, self.nwalkers, self.dim))))
        self._lnpost = np.concatenate(
            (self._lnpost, np.zeros((iterations, self.nwalkers))))
        self._lnprop = np.concatenate(
            (self._lnprop, np.zeros((iterations, self.nwalkers))))
        self.accepted = np.concatenate(
            (self.accepted, np.zeros((iterations, self.nwalkers))))

        for i in xrange(int(iterations)):
            # Draw new walker locations from the proposal
            p_p = self._kde.draw(N=self.nwalkers)

            # Calculate the prior, likelihood, and proposal density
            # at the proposed locations
            results = np.array(m(GetLnProbWrapper(self._get_lnprior,
                                                  self._get_lnlike,
                                                  self._kde), p_p))
            lnprior_p = results[:, 0]
            lnlike_p = results[:, 1]
            lnq_p = results[:, 2]

            # Calculate the (ln) Metropolis-Hastings ration
            ln_mh_ratio = lnprior_p + lnlike_p - lnprior - lnlike + lnq - lnq_p

            # Accept if ratio is greater than 1
            acc = ln_mh_ratio > 0

            # Decide which of the remainder will be accepted
            worse = ~acc
            nworse = np.sum(worse)
            acc[worse] = ln_mh_ratio[worse] > np.log(np.random.rand(nworse))

            # Update locations and probability densities
            if np.any(acc):
                p[acc] = p_p[acc]
                lnprior[acc] = lnprior_p[acc]
                lnlike[acc] = lnlike_p[acc]
                lnq[acc] = lnq_p[acc]

            # Store stuff
            self._chain[self.iterations, :, :] = p
            self._lnpost[self.iterations, :] = lnprior + lnlike
            self._lnprop[self.iterations, :] = lnq
            self.accepted[self.iterations, :] = acc

            self.iterations += 1

            # Update the proposal at the requested interval
            if self.iterations % update_interval == 0:
                self._kde = optimized_kde(p, pool=self._pool)

        return p, lnprior, lnlike, lnq

    def animate(self, labels=None):
        from .animate import animate_triangle

        if not labels:
            labels = [r'$x_{}$'.format(i) for i in range(self.dim)]

        return animate_triangle(self._chain, labels=labels)
