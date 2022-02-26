# Copyright (C) 2022 Giacomo Petrillo
# Released under the MIT license

import numpy as np

def rhat(a, splitaxis=0, sampleaxis=1):
    """
    Computes $\hat R$ as defined in Andrew Gelman, John B. Carlin, Hal S.
    Stern, David B. Dunson, Aki Vehtari, Donald B. Rubin - Bayesian Data
    Analysis-Chapman and Hall_CRC (2013), 3rd edition, page 284.
    
    Use it to check convergence of a Markov chain Monte Carlo in the following
    way. Run more than once the Markov chain with different and independently
    extracted random initial states, but with the same number of samples.
    Remove burnin samples by your judgement (Gelman's default advice is to
    halve the chains). Split in two what remains of the chains. So if you
    started with M chains, you now have 2 * M chain pieces. Put everything into
    a multidimensional array with one index running over pieces and one over
    samples in the pieces. Pass the array to this function indicating the piece
    and sample axes with the corresponding parameters. If the returned number
    is near 1, you have reached convergence (Gelman checks that \hat R < 1.1).
    
    If you have multiple scalar quantities, compute \hat R for each of them,
    and check that they are all near 1.
    
    Note: this works for near-normal distributed scalar quantities, and has
    problems detecting 1) chains with same mean but different variances and
    2) chains with different locations when the variance is not defined or for
    heavy-tailed distributions. See Aki Vehtari et al. arxiv.org/abs/1903.08008.
    
    Axes other than splitaxis and sampleaxis are kept, i.e. \hat R is
    computed separately along additional axes.
    
    Parameters
    ----------
    a : numpy array
        Must be at least 2 dimensional.
    splitaxis : integer, default: 0
        The axis of `a` that runs along chain pieces.
    sampleaxis : integer, default: 1
        The axis of `a` that runs along samples. Must be different from
        splitaxis.
    
    Returns
    -------
    rhat : float or numpy array
        The Gelman-Rubin \hat R statistics.
    
    Example
    -------
    >>> chain1 = mymarkovchain(initial_state=0, samples=10000)
    >>> chain2 = mymarkovchain(initial_state=1, samples=10000)
    >>> chain1 = chain1[5000:]
    >>> chain2 = chain2[5000:]
    >>> r = rhat([chain1[:2500], chain1[2500:], chain2[:2500], chain2[2500:]])
    >>> if r > 1.1:
    >>>     raise RuntimeError("no convergence")
    """
    a = np.asarray(a)
    assert(len(a.shape) >= 2)
    assert(splitaxis != sampleaxis)
    
    n = a.shape[sampleaxis]
    m = a.shape[splitaxis]
    assert(n >= 2)
    assert(m >= 2)
    if n < m:
        raise RuntimeWarning("samples < chains, are you sure?")
    
    barpsidotj = np.mean(a, axis=sampleaxis, keepdims=True)
    B = n * np.var(barpsidotj, ddof=1, axis=splitaxis, keepdims=True)
    sjsquared = np.var(a, ddof=1, axis=sampleaxis, keepdims=True)
    W = np.mean(sjsquared, axis=splitaxis, keepdims=True)
    hatvarpluspsiy = (n-1)/n * W + 1/n * B
    rhat = np.sqrt(hatvarpluspsiy / W)
    return np.squeeze(rhat, axis=(splitaxis, sampleaxis))

if __name__ == "__main__":
    import unittest

    class TestRhat(unittest.TestCase):
        
        def test_run(self):
            """Just check it runs without crashing."""
            a = np.arange(20).reshape(4, -1)
            rhat(a)
        
        def test_near_1(self):
            """Check rhat is near 1 when everything comes from the same
            distribution."""
            a = np.random.randn(10, 100)
            r = rhat(a)
            self.assertTrue(r < 1.1)
        
        def test_far_1(self):
            """Check rhat is not near 1 when chains come from different
            distributions."""
            a = np.random.randn(4, 100) + np.arange(4).reshape(-1, 1)
            r = rhat(a)
            self.assertTrue(r > 1.1)
        
        def test_axes_params(self):
            """Check that it gives the same result using non-default axes."""
            a = np.random.randn(4, 100)
            r1 = rhat(a)
            r2 = rhat(a.T, 1, 0)
            self.assertEqual(r1, r2)
        
        def test_broadcast(self):
            """Check that it works correctly with multidimensional input."""
            a = np.random.randn(4, 2, 100, 3)
            r1 = rhat(a, splitaxis=0, sampleaxis=2)
            r2_flat = [rhat(a[:, i, :, j]) for i, j in np.ndindex(2, 3)]
            r2 = np.array(r2_flat).reshape(2, 3)
            self.assertTrue(np.allclose(r1, r2, rtol=1e-15, atol=1e-15))
        
        def test_low_dim(self):
            """Check that an exception is raised if input is not 2D."""
            with self.assertRaises(AssertionError):
                rhat(np.array(0))
            with self.assertRaises(AssertionError):
                rhat(np.arange(10))
        
        def test_overlap(self):
            """Check that an exception is raised if the split and sample axes
            are the same."""
            with self.assertRaises(AssertionError):
                rhat(np.random.randn(4, 10), 0, 0)
        
        def test_too_short(self):
            """Check that an exception is raised if there are not enough
            elements to compute the variances."""
            with self.assertRaises(AssertionError):
                rhat(np.random.randn(1, 10))
            with self.assertRaises(AssertionError):
                rhat(np.random.randn(2, 1))
        
        def test_transpose(self):
            """Check that an exception is raised if there are more chains
            than samples."""
            with self.assertRaises(RuntimeWarning):
                rhat(np.random.randn(10, 9))
    
    unittest.main()
