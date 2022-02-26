# Copyright (C) 2022 Giacomo Petrillo
# Released under the MIT license

from __future__ import division

import numpy as np
    
def uevuev(x, axis=None, kurtosis=None):
    """
    Unbiased Estimator of the Variance of the Unbiased Estimator of the
    Variance.
    
    This function computes an unbiased estimate of the variance of the sample
    variance. The sample variance is the one with N - 1, i.e. you would compute
    it in numpy with np.var(x, ddof=1).
    
    The argument axis works like usual in numpy, see for example np.mean or
    np.var documentation.
    
    The result can be negative! But it is improbable for large enough sample.
    
    If you assume the shape of the parent distribution, specify the kurtosis.
    In this case the result will be strictly positive, apart from the corner
    case in which all the values in x, apart at most one, are identical (it
    gives 0 then). It may be useful to remember that, for any distribution,
    kurtosis >= 1 + skewness^2, so in particular kurtosis >= 1.
    
    You must assume an exact value for the kurtosis. If you pass in an unbiased
    estimate of the kurtosis, the output will not be unbiased (still a good
    estimate anyway).
    
    The formulas are taken from https://stats.stackexchange.com/questions/307537/unbiased-estimator-of-the-variance-of-the-sample-variance.
    
    Parameters
    ----------
    x : array
        The sample. Must have at least 4 elements for the result to be finite.
    axis : None, integer or tuple of integers
        The variance is computed only along the axes specified. If None, it is
        computed on the flattened array.
    kurtosis : None, number or array
        If None, no assumption is made on the parent distribution of x. If a
        number, the kurtosis of x is assumed to be the one given. The kurtosis
        is defined as E[(x - E[x])^4] / E[(x - E[x])^2]^2 and yields 3 for the
        normal distribution. In any case the kurtosis is >= 1. If an array,
        it must broadcast correctly against the shape of the result.
    
    Returns
    -------
    var : float or array
        It may be an array if axis is not None. The shape is the same of x
        but with the specified axes removed.
    
    See also
    --------
    https://xkcd.com/2110/
    
    Examples
    --------
    Compute the mean, the error on the mean, and the error on the error on
    the mean, using first-order error propagation:
    >>> x = np.random.randn(1000)
    >>> m = np.mean(x)
    >>> var_m = np.var(x, ddof=1) / len(x)
    >>> m_err = np.sqrt(var_m)
    >>> var_var_m = uevuev(x)
    >>> m_err_err = 1/2 * np.sqrt(var_var_m / var_m) / np.sqrt(len(x))
    """
    x = np.asarray(x)
    xavg = np.mean(x, axis=axis, keepdims=True)
    N = np.prod(x.shape) // np.prod(xavg.shape)
    delta = (x - xavg) ** 2
    m2 = np.sum(delta, axis=axis) / N
    delta *= delta
    m4 = np.sum(delta, axis=axis) / N
    denom = (N-1) * (N-2) * (N-3)
    if kurtosis is None:
        return N/(N-1) * ((N-1)**2 * m4 - (N**2-3) * m2**2) / denom
    else:
        assert(np.all(kurtosis >= 1))
        return (kurtosis - (N-3)/(N-1)) * ((N**2-3*N+3) * m2**2 - (N-1) * m4) / denom

if __name__ == '__main__':
    import unittest
    
    class TestUevuev(unittest.TestCase):
        
        def test_run(self):
            # Just check it runs on 1D input.
            uevuev(np.random.randn(1000))
        
        def test_fixed(self):
            """Check with fixed values I computed once."""
            x = np.array([
                -1.33732893,  0.56876361, -0.08921175, -1.98455305, -1.42938293,
                -0.60432474, -0.69840249, -0.15742432,  0.80079734,  0.93249997
            ])
            self.assertTrue(np.allclose(uevuev(x), 0.08691819049888123))
            self.assertTrue(np.allclose(uevuev(x, kurtosis=3.1415), 0.21014786463251045))
        
        def test_axis(self):
            # Check it works on given axis and gives same result if flattened.
            x = np.random.randn(100, 100)
            y1 = uevuev(x, axis=1)
            y2 = np.array([uevuev(x) for x in x])
            self.assertTrue(np.allclose(y1, y2, rtol=1e-15, atol=1e-15))
    
        def test_multidim(self):
            # Check with multidim axis.
            x = np.random.randn(10, 10, 10, 10)
            y1 = uevuev(x, axis=(0, 2))
            flat_y2 = [uevuev([x[:, i, :, j]]) for i, j in np.ndindex(10, 10)]
            y2 = np.array(flat_y2).reshape(10, 10)
            assert(np.allclose(y1, y2, rtol=1e-15, atol=1e-15))
        
        def test_kurtosis_broadcast(self):
            # Check kurtosis broadcasts correctly.
            x = np.random.randn(10, 10, 10, 10)
            y1 = uevuev(x, axis=(0, 2), kurtosis=1 + np.arange(100).reshape(10, 10))
            flat_y2 = [
                uevuev([x[:, i, :, j]], kurtosis=1 + j + 10 * i)
                for i, j in np.ndindex(10, 10)
            ]
            y2 = np.array(flat_y2).reshape(10, 10)
            assert(np.allclose(y1, y2, rtol=1e-15, atol=1e-15))
    
        def test_normal(self):
            # Check it gives sensible result with normal distribution.
            x = np.random.randn(1000, 100)
            true_value = 2 / (x.shape[1] - 1)
            estimate = uevuev(x, axis=-1)
            delta = np.abs(np.mean(estimate) - true_value)
            sigma = np.std(estimate, ddof=1) / np.sqrt(len(estimate))
            assert(delta < 5 * sigma)
    
        def test_normal_assumed(self):
            # Check it gives sensible result with normal distribution assuming
            # kurtosis.
            x = np.random.randn(1000, 100)
            true_value = 2 / (x.shape[1] - 1)
            estimate = uevuev(x, axis=-1, kurtosis=3)
            delta = np.abs(np.mean(estimate) - true_value)
            sigma = np.std(estimate, ddof=1) / np.sqrt(len(estimate))
            assert(delta < 5 * sigma)
    
        def test_corner(self):
            # Check it gives 0 for corner case.
            x = np.ones((1000, 10))
            x[:, 0] = np.linspace(-10, 10, 1000)
            vv = uevuev(x, axis=-1, kurtosis=3.14)
            assert(np.allclose(0, vv, atol=1e-13, rtol=1e-13))
    
        def test_positive(self):
            # Check it yields positive result when kurtosis is specified.
            x = np.random.binomial(1, 0.7, size=(1000, 10))
            vv = uevuev(x, axis=-1, kurtosis=1)
            assert(np.all(vv >= -1e-15))
        
        def test_kurtosis_bound(self):
            # Check it raises if kurtosis less than 1.
            x = np.random.randn(1000)
            with self.assertRaises(AssertionError):
                vv = uevuev(x, kurtosis=0.999)
    
    unittest.main()
