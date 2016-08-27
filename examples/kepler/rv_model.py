import kepler as kp

def rv_model(ts, K, e, omega, chi, P):
    """Returns the radial velocity measurements associated with the
    planets in parameters ps at times ts.  The returned array has
    shape (Nts,)."""

    assert ts.ndim == 1, 'ts must be one-dimensional'

    return kp.rv_model(ts, K, e, omega, chi, P)
