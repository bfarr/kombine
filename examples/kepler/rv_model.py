import kepler as kp
import numpy as np
import parameters as params
import scipy.optimize as so

def kepler_f(M, E, e):
    """Returns the residual of Kepler's equation with mean anomaly M,
    eccentric anomaly E and eccentricity e."""
    return E - e*np.sin(E) - M

def kepler_fp(E, e):
    """Returns the derivative of kepler_f with respect to E."""
    return 1.0 - e*np.cos(E)

def kepler_fpp(E, e):
    """Returns the second derivative of kepler_f with respect to E."""
    return e*np.sin(E)

def kepler_fppp(E, e):
    """Returns the third derivative of kepler_f with respect to E."""
    return e*np.cos(E)

def kepler_solve_ea(n, e, t):
    """Solve for the eccentric anomaly for an orbit with mean motion
    n, eccentricity e, and time since pericenter passage t."""

    t=np.atleast_1d(t)

    M = np.fmod(n*t, 2.0*np.pi)

    while np.any(M < 0.0):
        M[M<0.0] += 2.0*np.pi

    # Method taken from Danby, J.M.A.  The Solution of Kepler's
    # Equations - Part Three.  Celestial Mechanics, Vol. 40,
    # pp. 303-312, 1987.
    E=np.zeros_like(M)
    E[M<np.pi]=M[M<np.pi] + 0.85*e
    E[M>np.pi]=M[M>np.pi] - 0.85*e

    f = kepler_f(M, E, e)

    while np.any(np.abs(f) > 1e-8):
        fp = kepler_fp(E,e)
        disc = np.sqrt(np.abs(16.0*fp*fp - 20.0*f*kepler_fpp(E,e)))
        d = -5.0*f / (fp + np.sign(fp)*disc)

        E += d
        f = kepler_f(M,E,e)

    return E
    

def kepler_solve_ta(n, e, t):
    """Solve for the true anomaly of a Keplerian orbit with mean
    motion n, eccentricity e at time t since pericenter passage."""

    E=kepler_solve_ea(n,e,t)

    f = 2.0*np.arctan(np.sqrt((1.0+e)/(1.0-e))*np.tan(E/2.0))

    # Get positive f, either [0, pi/2] or [3pi/2, 2pi]
    if np.any(f < 0.0):
        f[f<0.0] += 2.0*np.pi

    return f

def rv_model(ts, ps):
    """Returns the radial velocity measurements associated with the
    planets in parameters ps at times ts.  The returned array has
    shape (Npl, Nts)."""

    assert ts.ndim == 1, 'ts must be one-dimensional'

    return kp.rv_model(ts, ps.K, ps.e, ps.omega, ps.chi, ps.n)

def old_rv_model(ts, ps):
    """Returns the radial velocity measurements associated with the
    planets in parameters ps at times ts.  The returned array has
    shape (Npl, Nts)."""

    rvs=np.zeros((ps.npl, ts.shape[0]))

    for i,(K,e,omega,chi,n) in enumerate(zip(ps.K, ps.e, ps.omega, ps.chi, ps.n)):

        ecw=e*np.cos(omega)

        t0 = -chi*2.0*np.pi/n

        fs=kepler_solve_ta(n, e, (ts-t0))

        rvs[i,:]=K*(np.cos(fs + omega) + ecw)

    return rvs

    
