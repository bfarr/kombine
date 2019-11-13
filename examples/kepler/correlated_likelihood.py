import numpy as np
import numpy.linalg as nl
import numpy.random as nr
import rv_model as rv
import scipy.linalg as sl
import scipy.stats as ss


def generate_covariance(ts, sigma, tau):

    r"""Generates a covariance matrix according to an
    squared-exponential autocovariance
    
    .. math::
    
      \left\langle x_i x_j \right\rangle = \sigma_0^2 \delta_{ij} + \sigma^2 \exp\left[ - \frac{\left| t_i - t_j\right|^2}{2 \tau^2} \right]
    """

    ndim = ts.shape[0]

    tis = ts[:, np.newaxis]
    tjs = ts[np.newaxis, :]

    return sigma * sigma * np.exp(-np.square(tis - tjs) / (2.0 * tau * tau))


params_dtype = np.dtype(
    [
        ("mu", np.float),
        ("K", np.float),
        ("e", np.float),
        ("omega", np.float),
        ("chi", np.float),
        ("P", np.float),
        ("nu", np.float),
        ("sigma", np.float),
        ("tau", np.float),
    ]
)


class Log1PPosterior(object):
    """Log of the posterior for a single planet system observed with a
    single telescope.   """

    def __init__(self, ts, vs, dvs):
        self.ts = np.sort(ts)
        self.vs = vs
        self.dvs = dvs

        self.T = self.ts[-1] - self.ts[0]
        self.dt_min = np.min(np.diff(self.ts))

    def to_params(self, p):
        p = np.atleast_1d(p)
        return p.view(params_dtype)

    def log_prior(self, p):
        p = self.to_params(p)

        # Bounds
        if (
            p["K"] < 0.0
            or p["e"] < 0.0
            or p["e"] > 1.0
            or p["omega"] < 0.0
            or p["omega"] > 2.0 * np.pi
            or p["P"] < 0.0
            or p["nu"] < 0.1
            or p["nu"] > 10.0
            or p["sigma"] < 0.0
            or p["tau"] < 0.0
            or p["tau"] > self.T
        ):
            return np.NINF

        # Otherwise, flat prior on everything.
        return 0.0

    def log_likelihood(self, p):
        p = self.to_params(p)

        v = self.rvs(p)

        res = self.vs - v - p["mu"]

        cov = p["nu"] * p["nu"] * np.diag(self.dvs * self.dvs)
        cov += generate_covariance(self.ts, p["sigma"], p["tau"])

        cfactor = sl.cho_factor(cov)
        cc, lower = cfactor

        n = self.ts.shape[0]

        return (
            -0.5 * n * np.log(2.0 * np.pi)
            - np.sum(np.log(np.diag(cc)))
            - 0.5 * np.dot(res, sl.cho_solve(cfactor, res))
        )

    def __call__(self, p):
        lp = self.log_prior(p)

        if lp == np.NINF:
            return np.NINF
        else:
            return lp + self.log_likelihood(p)

    def rvs(self, p):
        p = self.to_params(p)

        return rv.rv_model(self.ts, p["K"], p["e"], p["omega"], p["chi"], p["P"])
