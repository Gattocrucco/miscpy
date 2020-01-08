import numpy as np
from scipy import signal

def neff(a, splitaxis=0, sampleaxis=1):
    """
    Compute n_eff as defined in Andrew Gelman, John B. Carlin, Hal S.
    Stern, David B. Dunson, Aki Vehtari, Donald B. Rubin - Bayesian Data
    Analysis-Chapman and Hall_CRC (2013), 3rd edition, page 286-287.
    
    It is an estimate of the "effective number of samples" of a markov chain,
    defined as the ratio between the variance and the variance of the sample
    mean.
    
    This estimate is defined for multiple chains. You need to have at least 2
    independent markov chains of the same length. The returned value is an
    estimate for the *total* effective number of samples in the chains, so to
    get the n_eff per chain you have to divide by the number of chains.
    
    If the input array has axes other than the chain and the sample axes, n_eff
    is computed separately along these other axes.
    
    Parameters
    ----------
    a : array
        An at least 2D array.
    splitaxis : integer, default: 0
        The axis of a that runs along chains.
    sampleaxis : integer, default: 1
        The axis of a that runs along samples in the chains.
    
    Returns
    -------
    n_eff : float or array
        If a is 2D, a float, otherwise an array.
    
    Example
    -------
    >>> chain1 = mymarkovchain(initial_state=0, samples=10000)
    >>> chain2 = mymarkovchain(initial_state=1, samples=10000)
    >>> chain1 = chain1[5000:]
    >>> chain2 = chain2[5000:]
    >>> n = neff([chain1[:2500], chain1[2500:], chain2[:2500], chain2[2500:]])
    
    Notes
    -----
    This implementation uses FFT to compute autocorrelation so it's O(NlogN).
    """
    a = np.asarray(a)
    assert(len(a.shape) >= 2)
    
    assert(splitaxis != sampleaxis)
    m = a.shape[splitaxis]
    n = a.shape[sampleaxis]
    assert(m >= 2)
    assert(n >= 2)
    
    barpsidotj = np.mean(a, axis=sampleaxis, keepdims=True)
    B = n * np.var(barpsidotj, ddof=1, axis=splitaxis, keepdims=True)
    sjsquared = np.var(a, ddof=1, axis=sampleaxis, keepdims=True)
    W = np.mean(sjsquared, axis=splitaxis, keepdims=True)
    hatvarplus = (n-1)/n * W + 1/n * B
    
    V = _variogram_fft(a, axis=sampleaxis)
    V = np.mean(V, axis=splitaxis % len(a.shape) + 1, keepdims=True)
    
    hatrho = 1 - V / (2 * hatvarplus)
    hatrho_even = hatrho[:(n // 2) * 2]
    hatrho_block = hatrho_even.reshape(n // 2, 2, *hatvarplus.shape)
    hatrho_block_sum = np.sum(hatrho_block, axis=1, keepdims=True)
    hatrho_sum = np.cumprod(hatrho_block_sum >= 0, axis=0, dtype=bool)
    hatrho_block[0, 0] = 0
    tau_int = np.sum(hatrho_block * hatrho_sum, axis=(0, 1))
    
    neff = m * n / (1 + 2 * tau_int)
    return np.squeeze(neff, axis=(splitaxis, sampleaxis))

def _variogram_fft(x, axis=-1):
    n = x.shape[axis]

    flipped_x = np.flip(x, axis=axis)
    squared_x = x ** 2
    flipped_squared_x = np.flip(squared_x, axis=axis)
    
    a = np.flip(np.cumsum(squared_x, axis=axis), axis=axis)
    b = np.flip(np.cumsum(flipped_squared_x, axis=axis), axis=axis)
    c = signal.fftconvolve(x, flipped_x, mode='full', axes=axis)
    c = np.moveaxis(np.moveaxis(c, axis, 0)[n - 1:], 0, axis)
    
    x_shape = [1] * len(x.shape)
    x_shape[axis] = n
    V = (a + b - 2 * c) / (n - np.arange(n)).reshape(x_shape)
    return np.expand_dims(np.moveaxis(V, axis, 0), axis % len(x.shape) + 1)

def _variogram_direct(x, axis=-1):
    n = x.shape[axis]
    x_shape = list(x.shape)
    x_shape[axis] = 1
    V = np.empty([n] + x_shape)
    x_view = np.moveaxis(x, axis, 0)
    for t in range(n):
        shift_diff = np.moveaxis(x_view[:n-t] - x_view[t:], 0, axis) ** 2
        V[t] = np.mean(shift_diff, axis=axis, keepdims=True)
    return V

if __name__ == '__main__':
    import unittest
    import numba
    
    def metropolis(start, target_pdf, proposal_sampler, nsamples):
        x = np.empty(nsamples)
        x[0] = start
        last_pdf = target_pdf(x[0])
        acc = np.random.rand(nsamples)
        for i in range(1, nsamples):
            x_prop = proposal_sampler(x[i - 1])
            prop_pdf = target_pdf(x_prop)
            p_acc = prop_pdf / last_pdf
            if acc[i] < p_acc:
                x[i] = x_prop
                last_pdf = prop_pdf
            else:
                x[i] = x[i - 1]
        return x
    
    def normal_metropolis(delta, nsamples):
        return metropolis(
            np.random.randn(),
            lambda x: np.exp(-1/2 * x**2),
            lambda x: x + np.random.randn() * delta,
            nsamples
        )
    
    def blocking_bootstrap_single(v, f, n, m, out=None):
        if out is None:
            out = np.empty(n)
        nblocks = len(v) // m
        tail_shape = v.shape[1:]
        v = v[:nblocks * m].reshape(nblocks, m, *tail_shape)
        for i in range(n):
            w = v[np.random.randint(0, nblocks, size=nblocks)]
            out[i] = f(w.reshape(nblocks * m, *tail_shape))
        return out
    
    @numba.jit(nopython=True)
    def ar(start, lamda, iid_samples):
        x = np.empty(1 + len(iid_samples))
        x[0] = start
        for i in range(len(iid_samples)):
            x[i + 1] = lamda * x[i] + (1 - lamda) * iid_samples[i]
        return x

    class TestVariogram(unittest.TestCase):
        
        def test_variogram(self):
            """Check that the two implementations of variogram give the same
            results."""
            x = np.random.randn(1000)
            v1 = _variogram_fft(x)
            v2 = _variogram_direct(x)
            self.assertTrue(np.allclose(v1, v2, atol=1e-13, rtol=1e-13))
        
        def test_variogram_nd(self):
            """Check variogram with multidimensional input."""
            x = np.random.randn(2, 1000, 3)
            v1 = _variogram_fft(x, 1)
            v2 = _variogram_fft(x, -2)
            v3 = _variogram_direct(x, 1)
            v4 = _variogram_direct(x, -2)
            self.assertTrue(np.allclose(v1, v2, atol=1e-13, rtol=1e-13))
            self.assertTrue(np.allclose(v1, v3, atol=1e-13, rtol=1e-13))
            self.assertTrue(np.allclose(v1, v4, atol=1e-13, rtol=1e-13))
        
    class TestNeff(unittest.TestCase):
        
        def test_run(self):
            # Check it runs without crashing.
            a = np.random.randn(2, 100)
            neff(a)
        
        def test_iid(self):
            # Check it gives original sample size for independent samples.
            a = np.random.randn(2, 100, 100)
            n = neff(a)
            mn = np.mean(n)
            en = np.std(n, ddof=1) / np.sqrt(len(n))
            true_n = a.shape[0] * a.shape[1]
            self.assertTrue(np.abs(mn - true_n) < 10 * en)
        
        def test_axes_params(self):
            """Check that it gives the same result using non-default axes."""
            a = np.random.randn(4, 100)
            n1 = neff(a)
            n2 = neff(a.T, 1, 0)
            self.assertTrue(np.allclose(n1, n2, rtol=1e-15, atol=1e-15))
        
        def test_broadcast(self):
            """Check that it works correctly with multidimensional input."""
            a = np.random.randn(4, 2, 100, 3)
            n1 = neff(a, splitaxis=0, sampleaxis=2)
            n2_flat = [neff(a[:, i, :, j]) for i, j in np.ndindex(2, 3)]
            n2 = np.array(n2_flat).reshape(2, 3)
            self.assertTrue(np.allclose(n1, n2, rtol=1e-14, atol=1e-14))
        
        def test_broadcast2(self):
            """Check that it works with splitaxis not first axis."""
            a = np.random.randn(2, 4, 100, 3)
            n1 = neff(a, splitaxis=1, sampleaxis=2)
            n2_flat = [neff(a[i, :, :, j]) for i, j in np.ndindex(2, 3)]
            n2 = np.array(n2_flat).reshape(2, 3)
            self.assertTrue(np.allclose(n1, n2, rtol=1e-14, atol=1e-14))
        
        def test_low_dim(self):
            """Check that an exception is raised if input is not 2D."""
            with self.assertRaises(AssertionError):
                neff(np.array(0))
            with self.assertRaises(AssertionError):
                neff(np.arange(10))
        
        def test_overlap(self):
            """Check that an exception is raised if the split and sample axes
            are the same."""
            with self.assertRaises(AssertionError):
                neff(np.random.randn(4, 10), 0, 0)
        
        def test_too_short(self):
            """Check that an exception is raised if there are not enough
            elements to compute the variances."""
            with self.assertRaises(AssertionError):
                neff(np.random.randn(1, 10))
            with self.assertRaises(AssertionError):
                neff(np.random.randn(2, 1))
        
        def test_ar(self):
            """Check with a simple autoregressive chain:
            x_n = \lambda x_{n-1} + (1 - \lambda) r_n."""
            lamda = 0.5
            variance = (1 - lamda) / (1 + lamda)
            neff_over_n = variance
            
            x = np.array([
                ar(np.random.randn() * variance, lamda, np.random.randn(99))
                for _ in range(4000)
            ]).reshape(1000, 4, 100)
            
            n_eff = neff(x, splitaxis=1, sampleaxis=2) / x.shape[1]
            true_neff = neff_over_n * x.shape[2]
            delta = np.abs(np.mean(n_eff) - true_neff)
            sigma = np.std(n_eff, ddof=1) / np.sqrt(x.shape[0])
            self.assertTrue(delta < 10 * sigma)
        
        def test_mcmc(self):
            # Check neff with an actual mcmc.
            # This is all too empirical, but whatever.
            x = np.array([normal_metropolis(1, 10000) for _ in range(4)])
            n_eff = neff(x) / x.shape[0]
            var_x = np.mean(1 / (1 - 1 / n_eff) * np.var(x, axis=1))
            var_mean_x_1 = var_x / n_eff
            mean_x_sample = np.array([
                blocking_bootstrap_single(chain, np.mean, 100, 100)
                for chain in x
            ])
            var_mean_x_2 = np.mean(np.var(mean_x_sample, ddof=1, axis=1))
            self.assertTrue(np.allclose(np.sqrt(var_mean_x_2),
                                        np.sqrt(var_mean_x_1), rtol=0.1))
    
    unittest.main()
