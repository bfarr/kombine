import numpy as np
import numpy.linalg as nl
import numpy.random as nr
import rv_model as rv
import correlated_likelihood as cl

class DataGenerator(object):
    """Generate synthetic data for some number of telescopes observing
    some number of planets."""

    def __init__(self, params, tsundowns=None, cadences=None, tstarts=None, tends=None):
        """Initialize a data generator for the given system
        parameters.  

        :param params: Parameters describing the system in question.

        :param tsundown: time of sundown at each observatory, in hours.

        :param cadence: mean time between observations (Poisson
          distributed) at each observatory.

        :param tstarts: time (in hours) of start of observations at each
          observatory.

        :param tends: time (in hours) of the end of observations at each
          observatory."""

        self._params = params

        if tsundowns is None:
            tsundowns=24.0*nr.rand(params.nobs)
        else:
            assert tsundowns.ndim == 1, 'tsundowns must be 1D'
            assert tsundowns.shape[0] == params.nobs, 'tsundowns shape does not match number of observatories'
        self._tsundowns = tsundowns

        if cadences is None:
            cadences=0.9*nr.rand(params.nobs)+0.1
        else:
            assert cadences.ndim == 1, 'cadences must be 1D'
            assert cadences.shape[0] == params.nobs, 'cadences shape does not match number of observatories'
        self._cadences = cadences

        if tstarts is None:
            tstarts=365.0*24.0*nr.rand(params.nobs)
        else:
            assert tstarts.ndim == 1, 'tstarts must be 1D'
            assert tstarts.shape[0] == params.nobs, 'tstarts shape does not match number of observatories'
        self._tstarts = tstarts

        if tends is None:
            tends=tstarts + 30.0*24.0 + 365.0*24.0*nr.rand(params.nobs)
        else:
            assert tends.ndim == 1, 'tends must be 1D'
            assert tends.shape[0] == params.nobs, 'tends shape does not match number of observatories'
        self._tends = tends

    @property
    def params(self):
        return self._params

    @property
    def tsundowns(self):
        return self._tsundowns

    @property
    def cadences(self):
        return self._cadences

    @property
    def tstarts(self):
        return self._tstarts

    @property 
    def tends(self):
        return self._tends

    def generate_poisson(self, tstart, tend, cadence):
        n=int((tend-tstart)/cadence*2 + 20)

        dts=cadence*nr.exponential(size=n)

        ts=tstart + np.cumsum(dts)

        return ts[ts<tend]

    def filter_times(self, ts, tsundown):
        tsmod=np.fmod(ts, 24.0)

        tsmod = tsmod - np.fmod(tsundown, 24.0)

        tsmod[tsmod < 0] += 24.0

        return ts[tsmod < 12.0]

    def generate_tobs(self):
        tobs=[]
        for tstart,tend,sundown,cadence in zip(self.tstarts, self.tends, self.tsundowns, self.cadences):
            tobs.append(self.filter_times(self.generate_poisson(tstart, tend, cadence), sundown))

        return tobs

    def generate_noise(self, ts):
        
        noise=[]
        for t, V, sigma0, sigma, tau in zip(ts, self.params.V, self.params.sigma0, self.params.sigma, self.params.tau):
            cov = cl.generate_covariance(t, sigma0, sigma, tau)
            A = nl.cholesky(cov)
            xs = nr.normal(size=len(t))
            noise.append(V + np.dot(A, xs))

        return noise

    def generate(self):
        """Returns (ts, rvs), where ts is a list of arrays of
        observation times (one array for each observatory), and rvs is
        a corresponding list of total radial velocity measurements."""

        ts=self.generate_tobs()
        noise=self.generate_noise(ts)
        
        rvs=[]
        for t,n in zip(ts, noise):
            rvs.append(n + np.sum(rv.rv_model(t, self.params), axis=0))

        return ts,rvs
        
