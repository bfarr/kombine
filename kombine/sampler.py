import numpy as np
import numpy.ma as ma
from scipy.stats import ks_2samp

from .interruptible_pool import Pool
from .clustered_kde import optimized_kde, TransdimensionalKDE


class GetLnProbWrapper(object):
    def __init__(self, lnpost, kde):
        self.lnpost = lnpost
        self.kde = kde

    def __call__(self, p):
        result = self.lnpost(p)
        kde = self.kde(p)

        # allow posterior function to optionally
        # return additional metadata
        try:
            lnpost = result[0]
            blob = result[1]
            return lnpost, kde, blob
        except (IndexError, TypeError):
            lnpost = result
            return lnpost, kde


class Sampler(object):
    """
    An Ensemble sampler.

    The :attr:`chain` member of this object has the shape:
    ``(nsteps, nwalkers, dim)`` where ``nsteps`` is the number of steps

    :param nwalkers:
        The number of individual MCMC chains to include in the ensemble.

    :param dim:
        Number of dimensions in the parameter space.  If ``transd`` is ``True``
        this is the maximum number of dimensions.

    :param lnpostfn:
        A function that takes a vector in the parameter space as input and
        returns the natural logarithm of the posterior probability for that
        position.

    :param transd:
        If ``True``, the sampler will operate across parameter spaces using
        a ``TransdimensionalKDE`` proposal distribution. In this mode a masked
        array with samples in each of the possible sets of dimensions must
        be given for the initial ensemble distribution.

    """
    def __init__(self, nwalkers, ndim, lnpostfn, transd=False,
                 processes=None, pool=None):
        self.nwalkers = nwalkers
        self.dim = ndim
        self._kde = None

        self._get_lnpost = lnpostfn

        self.iterations = 0
        self.stored_iterations = 0

        self.pool = pool
        self.processes = processes
        if self.processes is not None and self.processes > 1 and self.pool is None:
            self.pool = Pool(self.processes)

        self._transd = transd
        if self._transd:
            self._chain = ma.masked_all((0, self.nwalkers, self.dim))
            self.update_proposal = TransdimensionalKDE
        else:
            self._chain = np.zeros((0, self.nwalkers, self.dim))
            self.update_proposal = optimized_kde

        self._lnpost = np.empty((0, self.nwalkers))
        self._lnprop = np.empty((0, self.nwalkers))
        self._acceptance = np.zeros((0, self.nwalkers))
        self._blobs = []

        self._last_run_mcmc_result = None
        self._failed_p = None

    def burnin(self, p0=None, lnpost0=None, lnprop0=None, blob0=None,
               update_interval=10, max_steps=None, critical_pval=0.05):
        """
        Use two-sample K-S tests to determine when burnin is complete.  The
        interval over which distributions are compared will be adapted based
        on the average acceptance rate of the walkers.

        :param p0: (optional)
            A list of the initial walker positions.  It should have the
            shape ``(nwalkers, dim)``.  If ``None`` and the sampler has been
            run previously, it'll pick up where it left off.

        :param lnpost0: (optional)
            The list of log posterior probabilities for the walkers at
            positions ``p0``. If ``lnpost0 is None``, the initial
            values are calculated. It should have the shape
            ``(nwalkers, dim)``.

        :param lnprop0: (optional)
            The list of log proposal densities for the walkers at
            positions ``p0``. If ``lnprop0 is None``, the initial
            values are calculated. It should have the shape
            ``(nwalkers, dim)``.

        :param blob0: (optional)
            The list of blob data for walkers at positions ``p0``.

        :param update_interval: (optional)
            The number of steps between proposal updates.

        :param max_steps: (optional)
            An absolute maximum number of steps to take, in case burnin
            is too painful.

        :param critical_pval: (optional)
            Burnin proceeds until all two-sample K-S p-values exceed this
            threshold (default 0.05)

        After burning in, this method returns:

        * ``p`` - A list of the current walker positions, the shape of which
            will be ``(nwalkers, dim)``.

        * ``lnpost`` - The list of log posterior probabilities for the
          walkers at positions ``p``, with shape ``(nwalkers, dim)``.

        * ``lnprop`` - The list of log proposal densities for the
          walkers at positions ``p``, with shape ``(nwalkers, dim)``.

        * ``blob`` - The list of blob data for walkers at position ``p``,
          with shape ``(nwalkers,)`` if returned by ``lnpostfn`` else None

        """
        if self._transd:
            raise NotImplementedError('Auto-burnin not implemented for trans-dimensional sampling.')

        if p0 is not None:
            p0 = np.atleast_2d(p0).reshape(self.nwalkers, -1)
        else:
            p0 = self.draw(self.nwalkers)

        # Start the K-S testing interval at the update interval length
        test_interval = update_interval

        # Determine the maximum iteration to look for
        start = self.iterations
        max_iter = np.inf
        if max_steps is not None:
            max_iter = start + max_steps

            # If max_steps < update interval, at least run to max_steps
            test_interval = min(test_interval, max_steps)

        burned_in = False
        while not burned_in:
            # Give up if we're about to exceed the maximum number of iterations
            if self.iterations + test_interval > max_iter:
                break

            results = self.run_mcmc(test_interval, p0, lnpost0, lnprop0, blob0,
                                    update_interval=update_interval, storechain=True)
            try:
                p, lnpost, lnprop, blob = results
            except ValueError:
                blob = None
                p, lnpost, lnprop = results

            burned_in = True
            for par in range(self.dim):
                KS, pval = ks_2samp(p0[:, par], p[:, par])

                if pval < critical_pval:
                    burned_in = False
                    break

            # Adjust the interval so >~ 90% of walkers accept a jump
            if not burned_in:
                # Use the average acceptance of the last step to window
                #   over the last 10 accepted jumps (on average).  A floor
                #   acceptance rate of 1% is used in case no jumps were accepted
                #   in the last step.
                avg_nacc = 10
                floor_acc_rate = 0.01

                # Don't look back too far early in the run
                nacc_window = min(avg_nacc, self.iterations)
                window = int(nacc_window * 1/max(np.mean(self.acceptance[-1]), floor_acc_rate))

                acceptance_rates = np.mean(self.acceptance[-window:], axis=0)

                # Use the first decile of the walkers' acceptance rates to
                #  decide the next test interval
                index = int(.1*self.nwalkers)
                low_rate = np.sort(acceptance_rates)[index]

                # If there is a lot of variance in acceptance rates, get
                #   another 10 acceptances (on average) and check again.
                if low_rate > 0.:
                    test_interval = int(1./low_rate)
                else:
                    test_interval = window

                p0, lnpost0, lnprop0, blob0 = p, lnpost, lnprop, blob

        if not burned_in:
            print "Burnin unsuccessful."

        if blob is None:
            return p, lnpost, lnprop
        else:
            return p, lnpost, lnprop, blob

    def sample(self, p0=None, lnpost0=None, lnprop0=None, blob0=None,
               iterations=1, kde=None, update_interval=10, kde_size=None,
               uniform_transd=False, storechain=True):
        """
        Advance the ensemble ``iterations`` steps as a generator.

        :param p0 (optional):
            A list of the initial walker positions of shape
            ``(nwalkers, dim)``.  If ``None`` and a proposal distribution
            exists, walker positions will be drawn from the proposal.

        :param lnpost0: (optional)
            The list of log posterior probabilities for the walkers at
            positions ``p0``. If ``lnpost0 is None``, the initial
            values are calculated. It should have the shape
            ``(nwalkers, dim)``.

        :param lnprop0: (optional)
            The list of log proposal densities for the walkers at
            positions ``p0``. If ``lnprop0 is None``, the initial
            values are calculated. It should have the shape
            ``(nwalkers, dim)``.

        :param blob0: (optional)
            The list of blob data for the walkers at positions ``p0``.
            If ``blob0 is None`` but ``lnpost0`` and ``lnprop0`` are
            given, the likelihood function is assumed not
            to return blob data and it is not recomputed.

        :param iterations: (optional)
            The number of steps to run.

        :param kde: (optional)
            An already-constucted KDE with `__call__` and `draw` methods.

        :param update_interval: (optional)
            The number of steps between proposal updates.

        :param kde_size: (optional)
            Maximum sample size for KDE construction.  When the KDE is updated,
            existing samples are thinned by factors of two until there's enough
            room for ``nwalkers`` new samples.  The default is 2*``nwalkers``,
            and must be greater than ``nwalkers`` if specified.

        :param uniform_trans: (optional)
            If `True` when transdimensional sampling, weight is assigned uniformly
            across parameter spaces in the proposal distribution.  This helps to
            avoid races for burnin of each parameter space.

        :param storechain: (optional)
            Whether to keep the chain and posterior values in memory or
            return them step-by-step as a generator

        After ``iteration`` steps, this method returns (if storechain=True):

        * ``p`` - A list of the current walker positions, the shape of which
            will be ``(nwalkers, dim)``.

        * ``lnpost`` - The list of log posterior probabilities for the
          walkers at positions ``p``, with shape ``(nwalkers, dim)``.

        * ``lnprop`` - The list of log proposal densities for the
          walkers at positions ``p``, with shape ``(nwalkers, dim)``.

        * ``blob`` - The list of blob data for the walkers at positions ``p``
          if provided by `lnpostfn`` else None

        """
        if p0 is None:
            p = self.draw(self.nwalkers)
        else:
            if not isinstance(p0, np.ndarray):
                p = np.array(p0)
            else:
                p = p0

        if self.pool is None:
            m = map
        else:
            m = self.pool.map

        # Build a proposal if one doesn't already exist
        self._kde_size = kde_size
        if self._kde_size is None:
            self._kde_size = 2*self.nwalkers

        if kde is None and self._kde is None:
            self._kde = self.update_proposal(p, uniform_weight=uniform_transd,
                                             max_samples=self._kde_size, pool=self.pool)

        lnpost = lnpost0
        lnprop = lnprop0
        blob = blob0

        if lnpost is None or lnprop is None:
            results = list(m(GetLnProbWrapper(self._get_lnpost, self._kde), p))
            lnpost = np.array([r[0] for r in results]) if lnpost is None else lnpost
            lnprop = np.array([r[1] for r in results]) if lnprop is None else lnprop

            if blob is None:
                try:
                    blob = [r[2] for r in results]
                except IndexError:
                    blob = None

        # ensure blob has the right shape
        if blob is None:
            blob = [None]*self.nwalkers

        # Prepare arrays for storage ahead of time
        if storechain:
            # Make sure to mask things if the stored chain has a mask
            if hasattr(self._chain, "mask"):
                self._chain = ma.concatenate((self._chain,
                                              ma.masked_all((iterations, self.nwalkers, self.dim))))
            else:
                self._chain = np.concatenate((self._chain,
                                              np.zeros((iterations, self.nwalkers, self.dim))))

            self._lnpost = np.concatenate((self._lnpost, np.zeros((iterations, self.nwalkers))))
            self._lnprop = np.concatenate((self._lnprop, np.zeros((iterations, self.nwalkers))))
            self._acceptance = np.concatenate((self._acceptance,
                                               np.zeros((iterations, self.nwalkers))))

        for i in xrange(int(iterations)):
            try:
                # Draw new walker locations from the proposal
                p_p = self.draw(self.nwalkers)

                # Calculate the posterior probability and proposal density
                # at the proposed locations
                try:
                    results = list(m(GetLnProbWrapper(self._get_lnpost, self._kde), p_p))

                    lnpost_p = np.array([r[0] for r in results])
                    lnprop_p = np.array([r[1] for r in results])
                    try:
                        blob_p = [r[2] for r in results]
                    except IndexError:
                        blob_p = None

                # Catch any exceptions and exit gracefully
                except Exception as e:
                    self.rollback(self.stored_iterations)
                    self._failed_p = p_p

                    print "Offending samples stored in ``failed_p``."
                    raise

                # Calculate the (ln) Metropolis-Hastings ratio
                ln_mh_ratio = lnpost_p - lnpost + lnprop - lnprop_p

                # Accept if ratio is greater than 1
                acc = ln_mh_ratio > 0

                # Decide which of the remainder will be accepted
                worse = ~acc
                nworse = np.sum(worse)
                uniform_draws = np.random.rand(nworse)
                acc[worse] = ln_mh_ratio[worse] > np.log(uniform_draws)

                # Update locations and probability densities
                if np.any(acc):
                    p[acc] = p_p[acc]
                    lnpost[acc] = lnpost_p[acc]
                    lnprop[acc] = lnprop_p[acc]

                    if blob_p is None:
                        blob = None
                    else:
                        blob = [blob_p[i] if a else blob[i] for i,a in enumerate(acc)]

                if storechain:
                    # Store stuff
                    self._chain[self.stored_iterations, :, :] = p
                    self._lnpost[self.stored_iterations, :] = lnpost
                    self._lnprop[self.stored_iterations, :] = lnprop
                    self._acceptance[self.stored_iterations, :] = acc

                    if blob:
                        self._blobs.append(blob)

                    self.stored_iterations += 1

                self.iterations += 1

                # Update the proposal at the requested interval
                if self.iterations % update_interval == 0:
                    self._kde = self.update_proposal(p, uniform_weight=uniform_transd,
                                                     pool=self.pool, kde=self._kde,
                                                     max_samples=self._kde_size)

                # create generator for sampled points
                if blob:
                    yield p, lnpost, lnprop, blob
                else:
                    yield p, lnpost, lnprop

            except KeyboardInterrupt:
                self.rollback(self.stored_iterations)
                raise

    def draw(self, N):
        """
        Draw ``N`` samples from the current proposal distribution.
        """
        return self._kde.draw(N)

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
    def blobs(self):
        """
        Ensemble's past metadata
        """
        return self._blobs

    @property
    def lnpost(self):
        """
        Ensemble's past posterior probabilities,
            with shape ``(iterations, nwalkers)``.
        """
        return self._lnpost

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

    @property
    def acceptance_fraction(self):
        """
        An array (length: ``iterations``) of the fraction of walkers that
            accepted each step.
        """
        return np.mean(self.acceptance, axis=1)

    @property
    def acceptance_rate(self, window=None):
        """
        """
        N = len(self.acceptance)

        # Use the mean acceptance rate of the last step to set the window
        if window is None:
            window = int(20 * 1.0/np.mean(self.acceptance[-1]))

        rates = np.empty((self.nwalkers, N - window + 1))
        weights = np.ones(window)/window

        for w in range(self.nwalkers):
            rates[w] = np.convolve(self.acceptance[:, w], weights, 'valid')

        return rates

    def rollback(self, iteration):
        """
        Shrink arrays down to a length of ``iteration`` and reset the
        pool if there is one.
        """
        self._chain = self._chain[:iteration]
        self._lnpost = self._lnpost[:iteration]
        self._lnprop = self._lnprop[:iteration]
        self._acceptance = self._acceptance[:iteration]
        self._blobs = self._blobs[:iteration]

        # Close the old pool and open a new one
        if self.processes != 1 and isinstance(self.pool, Pool):
            self.pool.close()
            self.pool = Pool(self.processes)

    def run_mcmc(self, N, p0=None, lnpost0=None, lnprop0=None, blob0=None, **kwargs):
        """
        Iterate `sample` for ``N`` iterations and return the result.
        :param N:
            The number of steps to take.

        :param p0 (optional):
            A list of the initial walker positions of shape
            ``(nwalkers, dim)``.  If ``None`` and a proposal distribution
            exists, walker positions will be drawn from the proposal.

        :param lnpost0: (optional)
            The list of log posterior probabilities for the walkers at
            positions ``p0``. If ``lnpost0 is None``, the initial
            values are calculated. It should have the shape
            ``(nwalkers, dim)``.

        :param lnprop0: (optional)
            The list of log proposal densities for the walkers at
            positions ``p0``. If ``lnprop0 is None``, the initial
            values are calculated. It should have the shape
            ``(nwalkers, dim)``.

        :param blob0: (optional)
            The list of blob data for the walkers at positions ``p0``.
            If ``blob0 is None`` but ``lnpost0`` and ``lnprop0`` are
            given, the likelihood function is assumed not
            to return blob data and it is not recomputed.

        :param kwargs: (optional)
            The rest is passed to the `sample` method.

        Results of the final sample in the form that `sample` yields are
        returned.  Usually you'll get:
        ``p``, ``lnpost``, ``lnprop``, ``blob``(optional)
        """
        if self.pool is None:
            m = map
        else:
            m = self.pool.map

        if p0 is None:
            if self._last_run_mcmc_result is None:
                try:
                    p0 = self.chain[-1]
                    if lnpost0 is None:
                        lnpost0 = self.lnpost[-1]
                    if lnprop0 is None:
                        lnprop0 = self.lnprop[-1]
                except IndexError:
                    raise ValueError("Cannot have p0=None if the sampler hasn't been called.")
            else:
                p0 = self._last_run_mcmc_result[0]
                if lnpost0 is None:
                    lnpost0 = self._last_run_mcmc_result[1]
                if lnprop0 is None:
                    lnprop0 = self._last_run_mcmc_result[2]

        if self._kde is not None:
            if self._last_run_mcmc_result is None and (lnpost0 is None or lnprop0 is None):
                results = list(m(GetLnProbWrapper(self._get_lnpost, self._kde), p0))

                if lnpost0 is None:
                    lnpost0 = np.array([r[0] for r in results])
                if lnprop0 is None:
                    lnprop0 = np.array([r[1] for r in results])

        for results in self.sample(p0, lnpost0, lnprop0, blob0, N, **kwargs):
            pass

        # Store the results for later continuation and toss out the blob
        self._last_run_mcmc_result = results[:3]

        return results
