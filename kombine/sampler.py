#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A kernel-density-based, embarrassingly parallel ensemble sampler.
"""

from __future__ import (division, print_function, absolute_import, unicode_literals)
from .pool import choose_pool
import numpy as np
import numpy.ma as ma
from scipy.stats import chisquare
from .clustered_kde import optimized_kde, TransdimensionalKDE


def dynamic_pbar_update(step_size, last_step_size, acc, pbar=None):
    if pbar is not None:
        pbar.set_postfix_str('| Single-Step Acceptence Rate: {0}'.format(acc))
        pbar.update(step_size-last_step_size)


def in_notebook():
    import __main__ as main
    return not hasattr(main, '__file__')


class _GetLnProbWrapper(object):
    """Convenience class for evaluating multiple probability densities at a single point."""
    def __init__(self, lnpost, kde, *args):
        self.lnpost = lnpost
        self.kde = kde
        self.args = args

    def update_kde(self, kde):
        self.kde = kde

    def lnprobs(self, p):
        """
        Evaluate the log probability density of the stored target distribution fuction
        and KDE at `p`.

        :param p: Location to evaluate probability densties at.

        :returns: ``lnpost(p)``, ``kde(p)``
        """
        result = self.lnpost(p, *self.args)
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

    __call__ = lnprobs


_lnprob_wrapper = None


def _set_global_lnprob_wrapper(wrapper_instance):
    """Sets `_lnprob_wrapper` to a global variable equal to the provided
    instance of `_GetLnProbWrapper`. Making the wrapper a global variable
    allows chilren processes to access their own copy of the wrapper without
    needing to push out the same data on every call. For this to work it must
    be done *before* a multiprocessing pool is initialized. See
    http://stackoverflow.com/a/10118250 for details.
    """
    global _lnprob_wrapper
    _lnprob_wrapper = wrapper_instance


def _get_lnprob_from_wrapper(p):
    return _lnprob_wrapper.lnprobs(p)

def _update_wrapper_kde(kde):
    _lnprob_wrapper.update_kde(kde)


class Sampler(object):
    """
    An Ensemble sampler.

    The :attr:`chain` member of this object has the shape: `(nsteps, nwalkers, ndim)` where
    `nsteps` is the stored number of steps taken thus far.

    :param nwalkers:
        The number of individual MCMC chains to include in the ensemble.

    :param ndim:
        Number of dimensions in the parameter space.  If `transd` is ``True`` this is the number of
        unique dimensions across the parameter spaces.

    :param lnpostfn:
        A function that takes a vector in the parameter space as input and returns the natural
        logarithm of the posterior probability for that position.

    :param transd:
        If ``True``, the sampler will operate across parameter spaces using a
        :class:`.clustered_kde.TransdimensionalKDE` proposal distribution. In this mode a masked
        array with samples in each of the possible sets of dimensions must be given for the initial
        ensemble distribution.

    :param processes: (optional)
        The number of processes to use with :mod:`multiprocessing`.  If ``None``, all available
        cores are used.

    :param pool: (optional)
        A pre-constructed pool with a map method. If ``None`` a pool will be created using
        :mod:`multiprocessing`.

    """
    def __init__(self, nwalkers, ndim, lnpostfn, transd=False,
                 processes=None, pool=None, mpi=False, args=[]):
        self.nwalkers = nwalkers
        self.dim = ndim
        self._kde = None
        self._kde_size = self.nwalkers
        self._updates = []
        self._burnin_length = None

        self._get_lnpost = lnpostfn
        self._lnpost_args = args

        self.iterations = 0
        self.stored_iterations = 0
        self.mpi = mpi
        self.processes = processes
        if processes is None and pool is not None:
            raise ValueError("Please provide the number of processes if "
                             " if also providing a pool instance.")
        self._managing_pool = False
        if pool is not None:
            self._pool = pool
        else:
            self._managing_pool = True
            self._pool = choose_pool(processes, mpi=mpi)

        if not hasattr(self._pool, 'map'):
            raise AttributeError("Pool object must have a map() method.")

        self._transd = transd
        if self._transd:
            self._chain = ma.masked_all((0, self.nwalkers, self.dim))
        else:
            self._chain = np.zeros((0, self.nwalkers, self.dim))

        self._lnpost = np.empty((0, self.nwalkers))
        self._lnprop = np.empty((0, self.nwalkers))
        self._acceptance = np.zeros((0, self.nwalkers))
        self._blobs = []
        self.tqdm = None

        self._last_run_mcmc_result = None
        self._burnin_spaces = None
        self._failed_p = None
        self._set_wrapper()

    def _set_wrapper(self):
        """ Determine the lnprob wrapper """
        # If the pool can guarantee a call to all processes we can update
        # the data explicitly, which is much faster
        if hasattr(self._pool, 'broadcast'):
            wrapper = _GetLnProbWrapper(self._get_lnpost, self._kde,

                                        *self._lnpost_args)
            self._pool.broadcast(_set_global_lnprob_wrapper, wrapper)

    def _get_wrapper(self):
        """ Retunr the lnprob wrapper function call """
        if hasattr(self._pool, 'broadcast'):
            return _get_lnprob_from_wrapper
        else:
            return _GetLnProbWrapper(self._get_lnpost, self._kde,

                                     *self._lnpost_args)

    def _get_dynamci_pbar(self, progress, t):
        pbar = None
        if progress:
            import tqdm
            self.tqdm = tqdm
            if in_notebook():
                pbar = self.tqdm.tqdm_notebook(desc="Burning in until Constant Acceptence Rate over {} ACLs".format(t), position=0, leave=True,
                                               total=t)
            else:
                pbar = self.tqdm.tqdm(desc="Burning in until Constant Acceptence Rate over {} ACLs".format(t), position=0, leave=True,
                                      total=t)
        return pbar, dynamic_pbar_update

    def _get_finite_pbar(self, progress, N):
        pbar = None
        if progress:
            if self.tqdm is None:
                import tqdm
                self.tqdm = tqdm
            if in_notebook():
                pbar = self.tqdm.tqdm_notebook(total=N, desc="Running MCMC")
            else:
                pbar = self.tqdm.tqdm(total=N, desc="Running MCMC")
        return pbar

    def burnin(self, p0=None, lnpost0=None, lnprop0=None, blob0=None,
               test_steps=16, critical_pval=0.05, max_steps=None,
               verbose=False, callback=None, progress=False, **kwargs):
        """
        Evolve an ensemble until the acceptance rate becomes roughly constant.  This is done by
        splitting acceptances in half and checking for statistical consistency.  This isn't
        guaranteed to return a fully burned-in ensemble, but usually does.

        :param p0: (optional)
            A list of the initial walker positions.  It should have the shape `(nwalkers, ndim)`.
            If ``None`` and the sampler has been run previously, it'll pick up where it left off.

        :param lnpost0: (optional)
            The list of log posterior probabilities for the walkers at positions `p0`. If ``lnpost0
            is None``, the initial values are calculated. It should have the shape `(nwalkers, )`.

        :param lnprop0: (optional)
            List of log proposal densities for walkers at positions `p0`. If ``lnprop0 is None``,
            the initial values are calculated. It should have the shape `(nwalkers, )`.

        :param blob0: (optional)
            The list of blob data for walkers at positions `p0`.

        :param test_steps: (optional)
            The (rough) number of accepted steps over which to check for acceptance rate
            consistency. If you find burnin repeatedly ending prematurely try increasing this.

        :param critical_pval: (optional)
            The critial p-value for considering the acceptance distribution
            consistent.  If the calculated p-value is over this, then the acceptance rate
            is considered to be stationary.  Lower this if you want burnin criteria to be
            less strict.

        :param max_steps: (optional)
            An absolute maximum number of steps to take, in case burnin is too painful.

        :param verbose: (optional)
            Print status messages each time a milestone is reached in the burnin.

        :param kwargs: (optional)
            The rest is passed to :meth:`run_mcmc`.

        After burnin...

        :returns:
            * ``p`` - A list of the current walker positions with shape `(nwalkers, ndim)`.

            * ``lnpost`` - Array of log posterior probabilities for walkers at positions `p`; has
              shape `(nwalkers, )`.

            * ``lnprop`` - Array of log proposal densities for walkers at positions `p`; has shape
              `(nwalkers, )`.

            * ``blob`` - (if `lnprobfn` returns blobs) The list of blob data for the walkers at
              positions `p`.
        """
        if p0 is not None:
            p0 = np.atleast_2d(p0)

        # Determine the maximum iteration to look for
        start = self.iterations
        # Confine walkers to their space during burnin
        freeze_transd = False
        if self._transd:
            freeze_transd = True
            self._burnin_spaces = ~p0.mask
        pbar, pbar_status = self._get_dynamci_pbar(progress, test_steps)
        max_iter = np.inf
        if max_steps is not None:
            max_iter = start + max_steps
        if progress:
            verbose = False

        sampling_ct = 0
        step_size = 2
        last_step_size = 0
        while step_size <= test_steps:
            # Update the proposal
            if p0 is not None:
                self.update_proposal(p0, max_samples=self.nwalkers)
                lnprop0 = self._kde(p0)
            if verbose:
                print('Updated proposal')

            # Take one step to estimate acceptance rate
            test_interval = 1
            results = self.run_mcmc(test_interval, p0, lnpost0, lnprop0, blob0,
                                    freeze_transd=freeze_transd, spaces=self._burnin_spaces,
                                    **kwargs)
            try:
                p, lnpost, lnprop, blob = results
            except ValueError:
                p, lnpost, lnprop = results
                blob = None

            # Use the fraction of acceptances in the last step to estimate acceptance rate
            #   Bottom out at 1% if acceptances are really bad
            last_acc_rate = max(np.mean(self.acceptance[-1]), 0.01)

            # Estimate ACT based on acceptance
            act = int(np.ceil(2.0/last_acc_rate - 1.0))

            if verbose:
                print('Single-step acceptance rate is ', last_acc_rate)
                print('Producing ACT of ', act)

            # Use the ACT to set the new test interval, but avoid
            # overstepping a specified max.  We throw away the first
            # 2*act worth of steps as an initial burnin when comparing
            # acceptance rates
            test_interval = min((step_size+2)*act, max_iter - self.iterations)

            # Make sure we're taking at least one step
            test_interval = max(test_interval, 1)
            sampling_ct += 1
            pbar_status(step_size, last_step_size, last_acc_rate, pbar)

            if verbose:
                if ~np.isinf(max_iter):
                    print("Time {} running mcmc during burnin with {}/{} Iterations and test_interval = {}:".format(
                        sampling_ct, self.iterations, max_iter, test_interval))
                else:
                    print("Time {} running mcmc during burnin with test_interval = {}:".format(
                        sampling_ct, test_interval))

            results = self.run_mcmc(test_interval, p, lnpost, lnprop, blob,
                                    freeze_transd=freeze_transd, spaces=self._burnin_spaces, progress=progress,
                                    **kwargs)
            try:
                p, lnpost, lnprop, blob = results
            except ValueError:
                p, lnpost, lnprop = results
                blob = None

            if callback is not None:
                callback(self)

            # Quit if we hit the max
            if self.iterations >= max_iter:
                print("Burnin hit {} iterations before completing.".format(max_iter))
                break

            # Only check for consistency past the burn-in stage of 2*act
            if self.consistent_acceptance_rate(window_size=step_size*act, critical_pval=critical_pval):
                if verbose:
                    print('Acceptance rate constant over ', step_size, ' ACTs')
                last_step_size = step_size
                step_size *= 2
            else:
                if verbose:
                    print('Acceptance rate varies, trying again')

            if verbose:
                print('') # Newline

            p0, lnpost0, lnprop0, blob0 = p, lnpost, lnprop, blob

        self._burnin_length = self.updates[-1]
        if pbar is not None:
            pbar.close()

        if blob is None:
            return p, lnpost, lnprop
        else:
            return p, lnpost, lnprop, blob

    def sample(self, p0=None, lnpost0=None, lnprop0=None, blob0=None,
               iterations=1, kde=None, update_interval=None, kde_size=None,
               freeze_transd=False, spaces=None, storechain=True, **kwargs):
        """
        Advance the ensemble `iterations` steps as a generator.

        :param p0: (optional)
            A list of the initial walker positions.  It should have the shape `(nwalkers, ndim)`.
            If ``None`` and a proposal distribution exists, walker positions will be drawn from the
            proposal.

        :param lnpost0: (optional)
            The list of log posterior probabilities for the walkers at positions `p0`. If ``lnpost0
            is None``, the initial values are calculated. It should have the shape `(nwalkers, )`.

        :param lnprop0: (optional)
            List of log proposal densities for walkers at positions `p0`. If ``lnprop0 is None``,
            the initial values are calculated. It should have the shape `(nwalkers, )`.

        :param blob0: (optional)
            The list of blob data for walkers at positions `p0`.

        :param iterations: (optional)
            The number of steps to run.

        :param kde: (optional)
            An already-constucted, evaluatable KDE with a ``draw`` method.

        :param update_interval: (optional)
            Number of steps between proposal updates.

        :param kde_size: (optional)
            Maximum sample size for KDE construction.  When the KDE is updated, existing samples are
            thinned by factors of two until there's enough room for `nwalkers` new samples.  The
            default is `nwalkers`, and must be greater than :math:`\geq``nwalkers` if specified.

        :param freeze_transd: (optional)
            If ``True`` when transdimensional sampling, walkers are confined to their parameter
            space.  This is helpful during burnin, and allows fox fixed-D burnin before
            transdimensional sampling.

        :param spaces: (optional)
            Confine walkers to the requested parameter spaces. Expects an inverted mask with shape
            `(nwalkers, ndim)`.

        :param storechain: (optional)
            Flag for disabling chain and probability density storage in :attr:`chain`,
            :attr:`lnpost`, and :attr:`lnprop`.

        :param kwargs: (optional)
            The rest is passed to :meth:`update_proposal`.

        After each iteration...

        :yields:
            * ``p`` - An array of current walker positions with shape `(nwalkers, ndim)`.

            * ``lnpost`` - The list of log posterior probabilities for the walkers at positions
              ``p``, with shape `(nwalkers, )`.

            * ``lnprop`` - The list of log proposal densities for the walkers at positions `p`, with
              shape `(nwalkers, )`.

            * ``blob`` - (if `lnprobfn` returns blobs) The list of blob data for the walkers at
              positions `p`.
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

        if kde_size:
            self._kde_size = kde_size

        # Build a proposal if one doesn't already exist
        if kde is not None:
            self.set_kde(kde)
        elif self._kde is None:
            self.update_proposal(p, max_samples=self._kde_size, **kwargs)
            lnprop0 = self._kde(p)

        lnpost = lnpost0
        lnprop = lnprop0
        blob = blob0

        if lnpost is None or lnprop is None:
            results = list(self._pool.map(self._get_wrapper(), p))
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


        for i in range(iterations):
            try:
                if freeze_transd and spaces is None:
                    spaces = ~p.mask
                # Draw new walker locations from the proposal
                p_p = self.draw(self.nwalkers, spaces=spaces)

                # Calculate the posterior probability and proposal density
                # at the proposed locations
                try:
                    results = list(self._pool.map(self._get_wrapper(), p_p))
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

                    print("Offending samples stored in ``failed_p``.")
                    raise

                # Calculate the (ln) Metropolis-Hastings ratio
                ln_mh_ratio = lnpost_p - lnpost + lnprop - lnprop_p

                # Accept if ratio is greater than 1
                acc = ln_mh_ratio > 0

                # Decide which of the remainder will be accepted
                worse = ~acc
                nworse = np.count_nonzero(worse)
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

                # Update the proposal at the requested interval
                if self.trigger_update(update_interval):
                    self.update_proposal(p, max_samples=self._kde_size, **kwargs)
                    lnprop = self._kde(p)

                if storechain:
                    # Store stuff
                    self._chain[self.stored_iterations, :, :] = p
                    self._lnpost[self.stored_iterations, :] = lnpost
                    self._lnprop[self.stored_iterations, :] = lnprop

                    if blob:
                        self._blobs.append(blob)

                    self.stored_iterations += 1
                self.iterations += 1

                # create generator for sampled points
                if blob:
                    yield p, lnpost, lnprop, blob
                else:
                    yield p, lnpost, lnprop

            except KeyboardInterrupt:
                # Resize arrays to remove allocated but unfilled elements
                if storechain:
                    self.rollback(self.stored_iterations)

                # If the sampler is handling the pool, reset it automatically
                if self._managing_pool:
                    self._pool.close()
                    self._pool = choose_pool(self.processes, mpi=self.mpi)

                raise

    def ln_ev(self, ndraws):
        """Produces a Monte-Carlo estimate of the evidence integral using the
        current propasal.

        :param ndraws: The number of draws to make from the proposal
          for the evidence estimate.

        :return: ``(lnZ, dlnZ)``.  Evidence estimate and associated
          uncertainty.
        """

        pts = self.draw(ndraws)
        results = list(self._pool.map(self._get_wrapper(), pts))
        lnpost = np.array([r[0] for r in results])
        lnprop = np.array([r[1] for r in results])

        lninteg = lnpost - lnprop
        lnZ = np.logaddexp.reduce(lninteg) - np.log(lninteg.shape[0])
        lnZ2 = np.logaddexp.reduce(2.0*lninteg) - np.log(lninteg.shape[0])

        # sigma^2 = <Z^2> - <Z>^2
        # log(sigma^2) = log(<Z^2>) + log(1 - <Z>^2/<Z^2>)
        # Standard error = sqrt(sigma^2/N)
        lnsZ = 0.5*(lnZ2 + np.log1p(-np.exp(2.0*lnZ - lnZ2)) - np.log(lninteg.shape[0]))

        # dlnZ = sigma / Z
        dlnZ = np.exp(lnsZ - lnZ)

        return lnZ, dlnZ

    def draw(self, size, spaces=None):
        """
        Draw `size` samples from the current proposal distribution.

        :param size:
            Number of samples to draw.

        :param spaces:
            If not ``None`` while transdimensional sampling, draws are confined to the requested
            spaces.  Such a thing is useful for burnin (e.g. ``spaces = ~p.mask``).

        :returns: `size` draws from the proposal distribution.
        """
        if self._transd:
            draws = self._kde.draw(size, spaces=spaces)
        else:
            draws = self._kde.draw(size)
        return draws

    def trigger_update(self, interval=None):
        """
        Decide whether to trigger a proposal update.

        :param interval:
            Interval between proposal updates.  If ``None``, no updates will be done.  If
            ``"auto"``, acceptances are split in half and checked for consistency (see
            :meth:`consistent_acceptance_rate`).  If an ``int``, the proposal will be updated
            every `interval` iterations.

        :returns: ``bool`` indicating whether a proposal update is due.
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

    def update_proposal(self, p, max_samples=None, **kwargs):
        """
        Update the proposal density with points `p`.

        :param p:
            Samples to update the proposal with.

        :param max_samples: (optional)
            The maximum number of samples to use for constructing or updating the kde.  If a KDE is
            supplied and adding the samples from it will go over this, old samples are thinned by
            factors of two until under the limit.

        :param kwargs: (optional)
            The rest is passed to the KDE constructor.
        """
        self._updates.append(self.iterations)

        if self._transd:
            self.set_kde(TransdimensionalKDE(p, pool=self._pool, kde=self._kde,
                                            max_samples=self._kde_size, **kwargs))
        else:
            self.set_kde(optimized_kde(p, pool=self._pool, kde=self._kde,
                                      max_samples=self._kde_size, **kwargs))


    def set_kde(self, kde):
        """Sets self's kde and re-creates the pool to use it."""
        self._kde = kde

        if hasattr(self._pool, 'broadcast'):
            self._pool.broadcast(_update_wrapper_kde, kde)

    @property
    def pool(self):
        """Returns the pool instance."""
        return self._pool

    @property
    def failed_p(self):
        """
        Sample that caused the last exception.
        """
        return self._failed_p

    @property
    def chain(self):
        """
        Ensemble's past samples, with shape `(iterations, nwalkers, ndim)`.
        """
        return self._chain

    @property
    def blobs(self):
        """
        Ensemble's past metadata.
        """
        return self._blobs

    @property
    def lnpost(self):
        """
        Ensemble's past posterior probabilities, with shape `(iterations, nwalkers)`.
        """
        return self._lnpost

    @property
    def lnprop(self):
        """
        Ensemble's past proposal probabilities, with shape `(iterations, nwalkers)`.
        """
        return self._lnprop

    @property
    def updates(self):
        """
        Step numbers where the proposal density was updated.
        """
        return self._updates

    @property
    def acceptance(self):
        """
        Boolean array of ensemble's past acceptances, with shape `(iterations, nwalkers)`.
        """
        return self._acceptance

    @property
    def acceptance_fraction(self):
        """
        A 1-D array of length :attr:`stored_iterations` of the fraction of walkers that accepted
        each step.
        """
        return np.mean(self.acceptance, axis=1)

    @property
    def acceptance_rate(self):
        """
        An `(nwalkers, )`-shaped array of the windowed acceptance rate for each walker.  The size of
        the window is chosen automatically based on the fraction of acceptances in the ensembles
        last step.  See :meth:`windowed_acceptance_rate` if you want more control.
        """
        return self.windowed_acceptance_rate()

    @property
    def autocorrelation_times(self, lookback=None):
        """
        An `nwalkers`-long vector of the estimated autocorrelation time of each walker, estimated
        using the number of step acceptances over the last `lookback` steps.

        This function leverages the convenience that the proposal density doesn't functionally depend
        on walkers' current locations, which means an accepted step *must* be independent.  This allows
        for an analytic estimate of the autocorrelation time :math:\tau that depends only on the
        acceptance rate

        .. math:: \tau = \frac{2}{r_\mathrm{acc}} - 1

        NOTE: This method can only be used if the chain is being stored.

        :param lookback:
            Number of previous steps to use for the autocorrelation time estimates.  If ``None``,
            all steps since last proposal update will be used.
        """
        if lookback is None:
            lookback = self.iterations - self.updates[-1]

        # Turn lookback into a negative index to count from the end of the array
        lookback *= -1

        acc_rates = np.mean(self.acceptance[lookback:], axis=0)

        return 2.0/acc_rates - 1.0

    def windowed_acceptance_rate(self, window=None):
        """
        An `(nwalkers, -1)`-shaped array of the windowed acceptance rate for each walker.

        :param window:
            Number of iterations to calculate acceptance rate over.  If ``None``, the fraction of
            accepances across the ensemble's last step are used for scale.
        """
        N = len(self.acceptance)

        # Use the mean acceptance rate of the last step to set the window
        if window is None:
            window = 20 * int(1/np.mean(self.acceptance[-1]))

        rates = np.empty((self.nwalkers, N - window + 1))
        weights = np.ones(window)/window

        for w in range(self.nwalkers):
            rates[w] = np.convolve(self.acceptance[:, w], weights, 'valid')

        return rates

    def consistent_acceptance_rate(self, window_size=None, critical_pval=0.05):
        """
        A convenience function for :meth:`burnin` and :meth:`trigger_update`.  Returns ``True``
        if the number of acceptances each step are consistent with the acceptance rate of the
        last step.  This is done using a chi-squared test.

        :param window_size:
            Number of iterations to look back for acceptances.  If ``None``, the iteration of the
            last proposal update (from :attr:`updates`) is used.

        :param critical_pval:
            The critial p-value for considering the distribution consistent.  If the calculated
            p-value is over this, then ``True`` is returned.
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
            last_acc_rate = self.acceptance_fraction[-1]

            nacc = self.nwalkers * self.acceptance_fraction[window_start:self.iterations]
            expected_freqs = last_acc_rate * self.nwalkers * np.ones_like(nacc)

            _, p_val = chisquare(nacc, expected_freqs)

            if p_val < critical_pval:
                consistent = False

        return consistent

    def rollback(self, iteration):
        """
        Shrink internal arrays down to a length of `iteration` and reset the :attr:`pool` if there
        is one.  This is helpful for keeping things consistent after a :exc:`KeyboardInterrupt`.
        """
        self._chain = self._chain[:iteration]
        self._lnpost = self._lnpost[:iteration]
        self._lnprop = self._lnprop[:iteration]
        self._acceptance = self._acceptance[:iteration]
        self._blobs = self._blobs[:iteration]

    def run_mcmc(self, N, p0=None, lnpost0=None, lnprop0=None, blob0=None, progress=False, **kwargs):
        """
        Iterate :meth:`sample` for `N` iterations and return the result.

        :param N:
            The number of steps to take.

        :param p0: (optional)
            A list of the initial walker positions.  It should have the shape `(nwalkers, ndim)`.
            If ``None`` and the sampler has been run previously, it'll pick up where it left off.

        :param lnpost0: (optional)
            The list of log posterior probabilities for the walkers at positions `p0`. If ``lnpost0
            is None``, the initial values are calculated. It should have the shape `(nwalkers, )`.

        :param lnprop0: (optional)
            List of log proposal densities for walkers at positions `p0`. If ``lnprop0 is None``,
            the initial values are calculated. It should have the shape `(nwalkers, )`.

        :param blob0: (optional)
            The list of blob data for walkers at positions `p0`.

        :param kwargs: (optional)
            The rest is passed to :meth:`sample`.

        After `N` steps...

        :returns:
            * ``p`` - An array of current walker positions with shape `(nwalkers, ndim)`.

            * ``lnpost`` - The list of log posterior probabilities for the walkers at positions
              ``p``, with shape `(nwalkers, )`.

            * ``lnprop`` - The list of log proposal densities for the walkers at positions `p`, with
              shape `(nwalkers, )`.

            * ``blob`` - (if `lnprobfn` returns blobs) The list of blob data for the walkers at
              positions `p`.
        """

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
                results = list(self._pool.map(self._get_wrapper(), p0))
                if lnpost0 is None:
                    lnpost0 = np.array([r[0] for r in results])
                if lnprop0 is None:
                    lnprop0 = np.array([r[1] for r in results])
                if blob0 is None:
                    try:
                        blob0 = [r[2] for r in results]
                    except IndexError:
                        blob0 = None

        pbar = self._get_finite_pbar(progress, N)
        iter = self.iterations
        for results in self.sample(p0, lnpost0, lnprop0, blob0, N, **kwargs):
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str("| {}/{} Walkers Accepted | Last step Acc Rate: {}".format(np.count_nonzero(self.acceptance[iter]),
                                                                                             self.nwalkers, self.acceptance_fraction[-1]))
                iter += 1

        # Store the results for later continuation and toss out the blob
        self._last_run_mcmc_result = results[:3]
        if pbar is not None:
            pbar.close()
        return results

    @property
    def burnin_length(self):
        """
        If :meth:`burnin` was not used, and ``burnin_length`` is
        ``None``, the iteration of the last proposal update will
        be used.
        """
        if self._burnin_length is None:
            return self.updates[-1]
        else:
            return self._burnin_length

    def get_samples(self, burnin_length=None):
        """
        Extract the independent samples collected after burnin.
        If :meth:`burnin` was not used, and ``burnin_length`` is
        ``None``, the iteration of the last proposal update will
        be used.

        :param burnin_length: (optional)
            The step after which to extract samples from.

        :returns: An `(nsamples, ndim)` array of independent samples.
        """
        if burnin_length is None:
            burnin_length = self.burnin_length
        ACTs = np.ceil(self.autocorrelation_times).astype(int)
        samples = [self.chain[burnin_length::ACT, walker]
                   for walker, ACT in zip(range(self.nwalkers), ACTs)]
        return np.vstack(samples)