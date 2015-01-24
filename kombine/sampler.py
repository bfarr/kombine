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

        self._chain = np.empty((0, self.nwalkers, self.dim))
        self._lnprior = np.empty((0, self.nwalkers))
        self._lnlike = np.empty((0, self.nwalkers))
        self._lnprop = np.empty((0, self.nwalkers))
        self._acceptance = np.zeros((0, self.nwalkers))

        self._failed_p = None

    def sample(self, p0, lnprior0=None, lnlike0=None, lnprop0=None,
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

        :param lnprop0: (optional)
            The list of log proposal densities for the walkers at
            positions ``p0``. If ``lnprop0 is None``, the initial
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

        * ``lnprop`` - The list of log proposal densities for the
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
        lnprop = lnprop0

        if lnprior is None or lnlike is None or lnprop is None:
            results = np.array(m(GetLnProbWrapper(self._get_lnprior,
                                                  self._get_lnlike,
                                                  self._kde), p))
            lnprior = results[:, 0]
            lnlike = results[:, 1]
            lnprop = results[:, 2]

        # Prepare arrays for storage ahead of time
        self._chain = np.concatenate(
            (self._chain, np.zeros((iterations, self.nwalkers, self.dim))))
        self._lnprior = np.concatenate(
            (self._lnprior, np.zeros((iterations, self.nwalkers))))
        self._lnlike = np.concatenate(
            (self._lnlike, np.zeros((iterations, self.nwalkers))))
        self._lnprop = np.concatenate(
            (self._lnprop, np.zeros((iterations, self.nwalkers))))
        self._acceptance = np.concatenate(
            (self._acceptance, np.zeros((iterations, self.nwalkers))))

        for i in xrange(int(iterations)):
            # Draw new walker locations from the proposal
            p_p = self._kde.draw(N=self.nwalkers)

            # Calculate the prior, likelihood, and proposal density
            # at the proposed locations
            try:
                results = np.array(m(GetLnProbWrapper(self._get_lnprior,
                                                      self._get_lnlike,
                                                      self._kde), p_p))
                lnprior_p = results[:, 0]
                lnlike_p = results[:, 1]
                lnprop_p = results[:, 2]

            # Catch any exceptions and exit gracefully
            except Exception as e:
                self._failed_p = p_p
                self._chain = self._chain[:self.iterations]
                self._lnprior = self._lnprior[:self.iterations]
                self._lnlike = self._lnlike[:self.iterations]
                self._lnprop = self._lnprop[:self.iterations]
                self._acceptance = self._acceptance[:self.iterations]

                print "Offending samples stored in ``failed_p``."
                raise

            # Calculate the (ln) Metropolis-Hastings ration
            ln_mh_ratio = lnprior_p + lnlike_p - lnprior - lnlike
            ln_mh_ratio += lnprop - lnprop_p

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
                lnprop[acc] = lnprop_p[acc]

            # Store stuff
            self._chain[self.iterations, :, :] = p
            self._lnprior[self.iterations, :] = lnprior
            self._lnlike[self.iterations, :] = lnlike
            self._lnprop[self.iterations, :] = lnprop
            self._acceptance[self.iterations, :] = acc

            self.iterations += 1

            # Update the proposal at the requested interval
            if self.iterations % update_interval == 0:
                self._kde = optimized_kde(p, pool=self._pool)

        return (p, lnprior, lnlike, lnprop)

    @property
    def failed_p(self):
        """
        Sample that caused the last exception.
        """
        return self._failed_p

    @property
    def chain(self):
        """
        Ensemble's past samples,
            with shape ``(iterations, nwalkers, ndim)``.
        """
        return self._chain

    @property
    def lnprior(self):
        """
        Ensemble's past prior probabilities,
            with shape ``(iterations, nwalkers)``.
        """
        return self._lnprior

    @property
    def lnlike(self):
        """
        Ensemble's past likelihoods,
            with shape ``(iterations, nwalkers)``.
        """
        return self._lnlike

    @property
    def lnprop(self):
        """
        Ensemble's past proposal probabilities,
            with shape ``(iterations, nwalkers)``.
        """
        return self._lnprop

    @property
    def acceptance(self):
        """
        Boolean array of ensemble's past acceptances,
            with shape ``(iterations, nwalkers)``.
        """
        return self._acceptance

    def animate(self, labels=None):
        from .animate import animate_triangle

        if not labels:
            labels = [r'$x_{}$'.format(i) for i in range(self.dim)]

        return animate_triangle(self._chain, labels=labels)
