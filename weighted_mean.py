import numpy as np
import uncertainties as un
from uncertainties import unumpy as unp

def weighted_mean(y, covy=None):
    """
    Weighted mean (with covariance matrix).
    
    Parameters
    ----------
    y : array of numbers or ufloats
    covy : None or matrix
        If covy is None, y must be an array of ufloats.
    
    Returns
    -------
    a : ufloat
        Weighted mean of y.
    Q : float
        Chisquare (the value of the minimized quadratic form at the minimum).
    """
    # get covariance matrix
    if covy is None:
        covy = un.covariance_matrix(y)
    else:
        y = un.correlated_values(y, covy)
    
    # compute weighted average
    inv_covy = np.linalg.inv(covy)
    vara = 1 / np.sum(inv_covy)
    a = vara * np.sum(np.dot(inv_covy, y))
    
    # check computation of uncertainties module against direct computation
    assert np.allclose(vara, a.s ** 2)
    
    # compute chisquare
    res = unp.nominal_values(y) - a.n
    Q = float(res.reshape(1,-1) @ inv_covy @ res.reshape(-1,1))
    
    return a, Q
