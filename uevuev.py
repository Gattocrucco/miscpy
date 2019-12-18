import numpy as np
    
def uevuev(x, axis=None, kurtosis=None):
    """
    Unbiased Estimator of the Variance of the Unbiased Estimator of the Variance
    
    This function computes an unbiased estimate of the variance of the sample
    variance. The sample variance is the one with N - 1, i.e. you would compute
    it in numpy with np.var(x, ddof=1).
    
    The argument axis works like usual in numpy, see for example np.mean or
    np.var documentation.
    
    Note: it can be negative! But it is improbable for large enough sample. If
    you assume the shape of the parent distribution, specify the kurtosis. In
    this case the result will be strictly positive, apart from the corner case
    in which all the values in x, apart at most one, are identical. It may be
    useful to remember that, for any distribution, kurtosis >= 1 + skewness^2.
    
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
    """
    x = np.asarray(x)
    xavg = np.mean(x, axis=axis, keepdims=True)
    N = np.prod(x.shape) // np.prod(xavg.shape)
    delta2 = (x - xavg) ** 2
    delta4 = delta2 ** 2
    m2 = np.sum(delta2, axis=axis) / N
    m4 = np.sum(delta4, axis=axis) / N
    if kurtosis is None:
        return N/(N-1) * ((N-1)**2 * m4 - (N**2-3) * m2**2) / ((N-1) * (N-2) * (N-3))
    else:
        assert(np.all(kurtosis >= 1))
        return (kurtosis - (N-3)/(N-1)) * ((N**2-3*N+3) * m2**2 - (N-1) * m4) / ((N-1) * (N-2) * (N-3))

if __name__ == '__main__':
    # Test the function.
    
    # Test 1.
    # Just check it runs on 1D input.
    uevuev(np.random.randn(1000))
    
    # Test 2.
    # Check it works on given axis and gives same result if flattened.
    x = np.random.randn(100, 100)
    y1 = uevuev(x, axis=1)
    y2 = np.array([uevuev(x) for x in x])
    assert(np.allclose(y1, y2, rtol=1e-15, atol=1e-15))
    
    # Test 3.
    # Check with multidim axis.
    x = np.random.randn(10, 10, 10, 10)
    y1 = uevuev(x, axis=(0, 2))
    y2 = np.array([uevuev([x[:, i, :, j]]) for i, j in np.ndindex(10, 10)]).reshape(10, 10)
    assert(np.allclose(y1, y2, rtol=1e-15, atol=1e-15))
    
    # Test 3a.
    # Check kurtosis broadcasts correctly.
    x = np.random.randn(10, 10, 10, 10)
    y1 = uevuev(x, axis=(0, 2), kurtosis=1 + np.arange(100).reshape(10, 10))
    y2 = np.array([uevuev([x[:, i, :, j]], kurtosis=1 + j + 10 * i) for i, j in np.ndindex(10, 10)]).reshape(10, 10)
    assert(np.allclose(y1, y2, rtol=1e-15, atol=1e-15))
    
    # Test 4.
    # Check it gives sensible result with normal distribution.
    x = np.random.randn(1000, 100)
    true_value = 2 / (x.shape[1] - 1)
    estimate = uevuev(x, axis=-1)
    delta = np.abs(np.mean(estimate) - true_value)
    sigma = np.std(estimate, ddof=1) / np.sqrt(len(estimate))
    assert(delta < 5 * sigma)
    
    # Test 5.
    # Check it gives sensible result with normal distribution assuming kurtosis.
    x = np.random.randn(1000, 100)
    true_value = 2 / (x.shape[1] - 1)
    estimate = uevuev(x, axis=-1, kurtosis=3)
    delta = np.abs(np.mean(estimate) - true_value)
    sigma = np.std(estimate, ddof=1) / np.sqrt(len(estimate))
    assert(delta < 5 * sigma)
    
    # Test 6.
    # Check it gives 0 for corner case.
    x = np.ones((1000, 10))
    x[:, 0] = np.linspace(-10, 10, 1000)
    vv = uevuev(x, axis=-1, kurtosis=3.14)
    assert(np.allclose(0, vv, atol=1e-13, rtol=1e-13))
    
    # Test 7.
    # Check it yields positive result when kurtosis is specified.
    x = np.random.binomial(1, 0.7, size=(1000, 10))
    vv = uevuev(x, axis=-1, kurtosis=1)
    assert(np.all(vv >= -1e-15))
