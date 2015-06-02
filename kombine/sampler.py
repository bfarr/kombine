import numpy as np
import numpy.ma as ma
from scipy.stats import chi2_contingency

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

    :param ndim:
        Number of dimensions in the parameter space.  If ``transd`` is ``True``
        this is the number of unique dimensions across the parameter spaces.

    :param lnpostfn:
        A function that takes a vector in the parameter space as input and
        returns the natural logarithm of the posterior probability for that
        position.

    :param transd:
        If ``True``, the sampler will operate across parameter spaces using
        a ``TransdimensionalKDE`` proposal distribution. In this mode a masked
        array with samples in each of the possible sets of dimensions must
        be given for the initial ensemble distribution.

    :param processes: (optional)
        The number of processes to use with `multiprocessing`.  By default,
        all available cores will be used.

    :param pool: (optional)
        A pre-constructed pool with a map method. If `None` a pool will be created
        using multiprocessing.

    """
    def __init__(self, nwalkers, ndim, lnpostfn, transd=False,
                 processes=None, pool=None):
        self.nwalkers = nwalkers
        self.dim = ndim
        self._kde = None
        self._kde_size = self.nwalkers
        self.updates = np.array([])

        self._get_lnpost = lnpostfn

        self.iterations = 0
        self.stored_iterations = 0

        self.pool = pool
        self.processes = processes
        if self.processes != 1 and self.pool is None:
            self.pool = Pool(self.processes)

        self._transd = transd
        if self._transd:
            self._chain = ma.masked_all((0, self.nwalkers, self.dim))
        else:
            self._chain = np.zeros((0, self.nwalkers, self.dim))

        self._lnpost = np.empty((0, self.nwalkers))
        self._lnprop = np.empty((0, self.nwalkers))
        self._acceptance = np.zeros((0, self.nwalkers))
        self._blobs = []

        self._last_run_mcmc_result = None
        self._failed_p = None

    def burnin(self, p0=None, lnpost0=None, lnprop0=None, blob0=None,
               test_steps=16, max_steps=None, **kwargs):
        """
        Evolve an ensemble until the acceptance rate becomes roughly constant.  This is done
        by splitting acceptances in half and checking for consistency.  This isn't guaranteed to
        return a fully burned-in ensemble, but it will get most of the way there.

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

        :param test_steps: (optional)
            The (rough) number of accepted steps over which to check for acceptance
            rate consistency. If you find burnin repeatedly ending prematurely try increasing this.

        :param max_steps: (optional)
            An absolute maximum number of steps to take, in case burnin
            is too painful.

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
        if p0 is not None:
            p0 = np.atleast_2d(p0)

        # Determine the maximum iteration to look for
        start = self.iterations

        max_iter = np.inf
        if max_steps is not None:
            max_iter = start + max_steps

        step_size = 2
        while step_size <= test_steps:
            # Update the proposal
            self.update_proposal(p0, uniform_weight=True,
                                 pool=self.pool, max_samples=self.nwalkers)

            # Take one step to estimate acceptance rate
            test_interval = 1
            results = self.run_mcmc(test_interval, p0, lnpost0, lnprop0, blob0, **kwargs)
            try:
                p, lnpost, lnprop, blob = results
            except ValueError:
                p, lnpost, lnprop = results
                blob = None

            # Use the fraction of acceptances in the last step to estimate acceptance rate
            #   Bottom out at 1% if acceptances are really bad
            last_acc_rate = np.max(np.mean(self.acceptance[-1]), 0.01)

            # Estimate ACT based on acceptance
            act = 2./last_acc_rate - 1

            # Use the ACT to set the new test interval, but avoid overstepping a specified max
            test_interval = min(int(step_size*act), max_iter - self.iterations)

            # Give up if we're about to exceed the maximum number of iterations
            if self.iterations + test_interval > max_iter:
                break

            results = self.run_mcmc(test_interval, p, lnpost, lnprop, blob, **kwargs)
            try:
                p, lnpost, lnprop, blob = results
            except ValueError:
                p, lnpost, lnprop = results
                blob = None

            # Quit if we hit the max
            if self.iterations >= max_iter:
                print "Burnin hit {} iterations before completing.".format(max_iter)
                break

            if self.consistent_acceptance_rate():
                step_size *= 2

            p0, lnpost0, lnprop0, blob0 = p, lnpost, lnprop, blob

        if blob is None:
            return p, lnpost, lnprop
        else:
            return p, lnpost, lnprop, blob

    def sample(self, p0=None, lnpost0=None, lnprop0=None, blob0=None,
               iterations=1, kde=None, update_interval=None, kde_size=None,
               uniform_transd=False, storechain=True, **kwargs):
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
            try:
                # Try copying to preserve array type (i.e.masked or not)
                p = p0.copy()
            except AttributeError:
                # If not already an array, make it a non-masked array by default.
                #   Operations with masked arrays can be slow.
                p = np.array(p0, copy=True)

        if self.pool is None:
            m = map
        else:
            m = self.pool.map

        if kde_size:
            self._kde_size = kde_size

        # Build a proposal if one doesn't already exist
        if kde is not None:
            self._kde = kde
        elif self._kde is None:
            self.update_proposal(p, max_samples=self._kde_size, pool=self.pool, **kwargs)

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
        self._acceptance = np.concatenate((self._acceptance,
                                           np.zeros((iterations, self.nwalkers))))
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
                        blob = [blob_p[i] if a else blob[i] for i, a in enumerate(acc)]

                self._acceptance[self.iterations, :] = acc
                if storechain:
                    # Store stuff
                    self._chain[self.stored_iterations, :, :] = p
                    self._lnpost[self.stored_iterations, :] = lnpost
                    self._lnprop[self.stored_iterations, :] = lnprop

                    if blob:
                        self._blobs.append(blob)

                    self.stored_iterations += 1

                # Update the proposal at the requested interval
                if self.trigger_update(update_interval):
                    self.update_proposal(p, max_samples=self._kde_size, pool=self.pool, **kwargs)

                self.iterations += 1

                # create generator for sampled points
                if blob:
                    yield p, lnpost, lnprop, blob
                else:
                    yield p, lnpost, lnprop

            except KeyboardInterrupt:
                if storechain:
                    self.rollback(self.stored_iterations)
                raise

    def draw(self, N):
        """
        Draw ``N`` samples from the current proposal distribution.
        """
        return self._kde.draw(N)

    def trigger_update(self, interval=None):
        """
        Decide whether a proposal update should be triggered, given the requested interval.
        If `interval` is `None`, no updates will be done.  If it's `auto` acceptances are split
        in half and checked for consistency.  If `interval` is an integer, the proposal will be
        updated every `interval` iterations.
        """
        trigger = False
        if interval is None:
            trigger = False
        elif interval == 'auto':
            trigger = not self.consistent_acceptance_rate()
        elif isinstance(interval, int):
            if self.iterations % interval == 0:
                trigger = True
        else:
            raise RuntimeError("Unexpected `interval` in `trigger_update`.")

        return trigger

    def update_proposal(self, p, pool=None, max_samples=None, **kwargs):
        """
        Update the proposal density with points `p`.

        :param p:
            Samples to update the proposal with.

        :param pool: (optional)
            A pool of processes with `map` function to use.

        :param max_samples: (optional)
            The maximum number of samples to use for constructing or updating the kde.
            If a KDE is supplied and adding the samples from `data` will go over this,
            old samples are thinned by factors of two until under the limit.
        """
        self.updates = np.concatenate((self.updates, [self.iterations]))

        # Ignore the uniform-transd arg when fixed-D sampling
        uniform_weight = None
        if "uniform_weight" in kwargs:
            uniform_weight = kwargs.pop("uniform_weight")

        if self._transd:
            self._kde = TransdimensionalKDE(p, pool=self.pool, kde=self._kde,
                                            max_samples=self._kde_size,
                                            uniform_weight=uniform_weight, **kwargs)
        else:
            self._kde = optimized_kde(p, pool=self.pool, kde=self._kde,
                                      max_samples=self._kde_size, **kwargs)

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

    def consistent_acceptance_rate(self, window_size=None, critical_pval=0.05):
        """
        A convenience funcion for `burnin`.  Returns `True` if the acceptances of the two halves
        of the window are consistent with having the same acceptance rates.  This is done using
        a chi-squared contingency test.
        """
        if window_size is None:
            if len(self.updates) == 0:
                return False
            else:
                window_start = self.updates[-1]
        else:
            window_start = self.iterations - window_size

        window_length = self.iterations - window_start

        # If window is really small, return `consistent` to avoid gratuitous updating
        consistent = True
        if window_length > 2:
            windowed_acceptances = self.acceptance[window_start:self.iterations].flatten()
            X1, X2 = np.array_split(windowed_acceptances, 2)

            n1, n2 = len(X1), len(X2)
            k1, k2 = np.sum(X1), np.sum(X2)

            # Use chi^2 contingency test to test whether the halves have consistent acceptances
            table = [[k1, k2], [n1 - k1, n2 - k2]]
            p_val = chi2_contingency(table)[1]

            if p_val < critical_pval:
                consistent = False

        return consistent

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
