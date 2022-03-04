"""

Module to do inference with a BART (Bayesian Regression Tree) using Gaussian
processes. Run as a script to do a test.

The notation follows the original BART article:

Hugh A. Chipman, Edward I. George, Robert E. McCulloch "BART: Bayesian additive
regression trees," The Annals of Applied Statistics, Ann. Appl. Stat. 4(1),
266-298, (March 2010).

Functions
---------
sample_bart_prior :
    Draw samples from the (exact) BART prior.
bart_correlation_extremity :
    Compute the exact BART prior correlation under some limitations.
bart_correlation_maxd :
    Compute an approximation of the BART prior correlation.

Classes
-------
BartGP :
    Do inference approximating the BART with a Gaussian process. Note: this
    does not use the Laplace kernel.

"""

import functools

import numpy as np
import numba
from scipy import stats, linalg, optimize

default_bart_params = dict(m=200, alpha=0.95, beta=2, k=2, nu=3, q=0.9)

def sample_bart_prior(X_data, X_test, y_data, size, *, seed=0, check=True, error=True, **bart_params):
    """
    Sample from the prior distribution of the model
    
        f(x) = sum_{j=1}^m g(x; T_j, M_j) + eps(x)
    
    where each g(x; T_j, M_j) is a decision tree with random rules T and random
    leaves M, and eps(x) is a i.i.d. random error.
    
    Parameters
    ----------
    X_data : (p, n) array
        Data, n vectors of p covariates. Used for splitting points.
    X_test : (p, N) array
        Points where the prior is computed.
    y_data : (n,) array
        Outcome. Used for the hyperparameters of the priors on mu_ij|T_j and
        sigma.
    size : int
        Number of generated samples.
    seed : int
        Seed for the random number generator, default 0.
    check : bool
        Check arguments, default True.
    error : bool
        Wether to sample and add the error term. Default True.
    
    Keyword arguments
    -----------------
    These are the hyperparameters of the BART. The default values are in the
    global variable `default_bart_params`.
    m : int
        Number of trees.
    alpha, beta : scalar
        Parameters of p(T_j), P(node nonterminal) = alpha (1 + d)^(-beta).
    k : scalar
        Inverse scale for p(mu_ij|T_j).
    nu : int
        Degrees of freedom of the chisquare for the prior on the error term.
    q : scalar
        Order of the quantile of y used to scale the prior on the error term
        sigma2_epsilon = lambda / (chi2_nu/nu), p(sigma2_epsilon < var(y)) = q.
        If q = 1 the error is disabled.
    
    Return
    ------
    y_test : (size, N) array
        Samples from the prior on f(X_test)
    """
    
    # extract BART hyperparameters
    params = dict(default_bart_params)
    params.update(**bart_params)
    m = params['m']
    alpha = params['alpha']
    beta = params['beta']
    k = params['k']
    nu = params['nu']
    q = params['q']

    if check:   # check all arguments
        X_data = np.asarray(X_data)
        p, n = X_data.shape
        assert n >= 2, n
        # for dim in range(p):
        #     u = np.unique(X_data[dim])
        #     assert len(u) >= 2, f'no splits for coordinate {dim} (0-based), unique value = {u.item()}'
        # TODO maybe it works even if there are degenerate coordinates with
        # no splits, but maybe it is still appropriate to warn the user
    
        X_test = np.asarray(X_test)
        p2, N = X_test.shape
        assert p == p2, (p, p2)
    
        y_data = np.asarray(y_data)
        n2, = y_data.shape
        assert n == n2, (n, n2)
    
        assert size == int(size), size
        assert size > 0, size
    
        assert m == int(m), m
        assert m > 0, m
    
        assert alpha == float(alpha), alpha
        assert alpha >= 0, alpha
    
        assert beta == float(beta), beta
        assert beta >= 0, beta
    
        assert k == float(k), k
        assert k >= 0
    
        assert nu == int(nu), nu
        assert nu > 0, nu
    
        assert q == float(q), q
        assert 0 <= q <= 1, q
    
        assert seed == int(seed)
    
    seeds = make_seeds(seed, size + 1)
    gen = np.random.default_rng(seeds[0])
    if not error:
        q = 1
    return sample_bart_prior_impl(X_data, X_test, y_data, size, m, alpha, beta, k, nu, q, seeds[1:], gen)

def make_seeds(seed, size):
    s = np.random.SeedSequence(seed)
    out = np.empty(size, np.uint32)
    for i in range(len(out)):
        out[i] = s.spawn(1)[0].pool[0]
    return out

def sample_bart_prior_impl(X_data, X_test, y_data, size, m, alpha, beta, k, nu, q, seeds, gen):
    p, N = X_test.shape
    y_test = np.zeros((size, N))
    
    # pre-computed/allocated stuff for tree generation
    sort_map = make_sort_map(X_test)
    sp_map = make_sp_map(X_data, X_test)
    sp_slice = root_sp_slice(X_data)
    active_dims = np.zeros(p, bool)
    
    # cycle over samples and trees, generate trees and leaves
    size = int(size)
    m = int(m)
    alpha = float(alpha)
    beta = float(beta) # casting to avoid jit overload
    sample_bart_prior_hellpit(size, m, y_test, sort_map, sp_map, sp_slice, alpha, beta, active_dims, seeds)
    
    # set mean and sdev of p(mu_ij|T_j)
    mu_mu, sigma_mu = mumu_sigmamu(y_data, k, m)
    y_test *= sigma_mu
    y_test += mu_mu
    
    # add error term
    dist = var_epsilon_dist(y_data, nu, q)
    if dist is not None:
        sigma2_eps = dist.rvs((size, 1), gen)
        y_test += np.sqrt(sigma2_eps) * gen.standard_normal((size, N))
    
    return y_test

@numba.jit(nopython=True, cache=True, parallel=True)
def sample_bart_prior_hellpit(size, m, y_test, sort_map, sp_map, sp_slice, alpha, beta, active_dims, seeds):
    for isamp in numba.prange(size):
        np.random.seed(seeds[isamp]) # explicit seeds because numba draws
                                     # them at random for each thread
        cycle_sp_slice = np.copy(sp_slice)
        cycle_active_dims = np.copy(active_dims)
        # copies to avoid parallel modification, not necessary in inner cycle
        for itree in range(m):
            recursive_tree_descent(y_test[isamp], 0, sort_map, sp_map, cycle_sp_slice, alpha, beta, cycle_active_dims)
        # assert np.all(cycle_sp_slice == sp_slice)
        # assert np.all(cycle_active_dims == active_dims)
        ## (!) asserts break parallelization

def var_epsilon_dist(y_data, nu, q):
    hat_sigma2 = np.var(y_data)
    dist = stats.invgamma(a=nu / 2)
    lamda = hat_sigma2 / dist.ppf(q)
    return None if lamda == 0 else stats.invgamma(a=nu / 2, scale=lamda)

def mumu_sigmamu(y_data, k, m):
    ymin = np.min(y_data)
    ymax = np.max(y_data)
    mu_mu = (ymax + ymin) / 2
    sigma_mu = (ymax - ymin) / (2 * k * np.sqrt(m))
    return mu_mu, sigma_mu
    
def make_sort_map(X_test):
    p, k = X_test.shape
    sort_map = np.empty((p, k), int)
    for dim in range(p):
        sort_map[dim] = np.argsort(X_test[dim])
    return sort_map

def make_sp_map(X_data, X_test):
    p, n = X_data.shape
    sp_map = np.full((2, p, n + 1), np.iinfo(int).max)
    for dim in range(p):
        x_data = X_data[dim]
        x_test = X_test[dim]
        x_data = np.unique(x_data)
        x_test = np.sort(x_test)
        split = (x_data[1:] + x_data[:-1]) / 2
        split = np.block([-np.inf, split, np.inf])
        sp_map[0, dim, :len(split)] = np.searchsorted(x_test, split, side='left')
        sp_map[1, dim, :len(split)] = np.searchsorted(x_test, split, side='right')
    return sp_map

def root_sp_slice(X_data):
    p, n = X_data.shape
    sp_slice = np.empty((p, 2), int)
    for dim in range(p):
        x = X_data[dim]
        u = np.unique(x)
        sp_slice[dim] = [0, len(u)]
    return sp_slice

@numba.jit(nopython=True, cache=True)
def recursive_tree_descent(y_test, d, sort_map, sp_map, sp_slice, alpha, beta, active_dims):
    """
    y_test : (N,) float array
        The tree output is accumulated here
    d : int
        node depth (root = 0)
    sort_map : (p, N) int array
        argsort separately for each coordinate of X_test
    sp_map : (2, p, n + 1) int array
        map from splitting points to sorted X_test for each coordinate
    sp_slice : (p, 2) int array
        current slice (as start:end) in the splitting points indices for each
        dimension. index i = split to the left of the ith element in the
        sorted unique p coordinate of X_data
    alpha, beta : scalar
        parameters of termination probability
    active_dims : (p,) bool array
        dimensions used in ancestors' splits
    """
    p, N = sort_map.shape
    
    # decide if node is nonterminal
    pnt = alpha * (1 + d) ** -np.float64(beta)
    ## (!) float64 needed otherwise result of power is casted to int64 when
    #      beta is an integer (fuck numba)
    u = np.random.uniform(0, 1)
    nt = u < pnt
    splittable_dims, = np.nonzero(sp_slice[:, 0] + 1 < sp_slice[:, 1])
    
    if nt and len(splittable_dims) > 0:     # split and recurse
        dim_restricted = np.random.randint(0, len(splittable_dims))
        dim = splittable_dims[dim_restricted]
        start, end = sp_slice[dim]
        isplit = np.random.randint(start + 1, end - 1 + 1)

        pa = active_dims[dim]
        active_dims[dim] = True
        
        sp_slice[dim, 1] = isplit
        recursive_tree_descent(y_test, d + 1, sort_map, sp_map, sp_slice, alpha, beta, active_dims)
        sp_slice[dim, 1] = end
        
        sp_slice[dim, 0] = isplit
        recursive_tree_descent(y_test, d + 1, sort_map, sp_map, sp_slice, alpha, beta, active_dims)
        sp_slice[dim, 0] = start
        
        active_dims[dim] = pa

    else:       # generate leaf value and accumulate
        # note: preallocating cum makes no difference
        # note: updating cum at each node instead of computing in each leaf is
        #       slower
        cum = np.zeros(N, np.intp)
        dims, = np.nonzero(active_dims)
        for dim in dims:
            ldata, rdata = sp_slice[dim]
            ltest = sp_map[0, dim, ldata]
            rtest = sp_map[1, dim, rdata]
            indices = sort_map[dim, ltest:rtest]
            cum[indices] += 1
        mask = cum == len(dims)
        y_test[mask] += np.random.normal()

class BartGP:
    """
    Class to fit a BART model using an approximately equivalent Gaussian
    process.
    
    Methods
    -------
    fit :
        Fit the BART model to given data.
    prior_mean :
        Get the mean of the BART prior distribution.
    prior_covariance_matrix :
        Get the covariance matrix of the BART prior distribution.
    posterior_mean :
        Get the mean of the BART posterior distribution conditional on the
        error standard deviation.
    posterior_covariance_matrix :
        Get the covariance matrix of the BART posterior distribution
        conditional on the error standard deviation.
    sample_sigma_prior :
        Draw samples from the prior on the error standard deviation.
    sample_sigma_posterior :
        Draw samples from the posterior on the error standard deviation.
    sample_prior :
        Draw samples from the BART prior.
    sample_posterior :
        Draw samples from the BART posterior.
    log_marginal_likelihood :
        The probability of the data conditional on the error standard
        deviation.
    log_hyper_marginal_likelihood :
        The probability of the data, computed with the Laplace approximation
        on the logarithm of the error standard deviation.
    error_log_sdev_posterior_sdev :
        The standard deviation of the posterior on the logarithm of the error
        standard deviation.
    
    Attributes
    ----------
    sigma : scalar
        Either a fixed value of the error standard deviation provided by the
        user, or the mode of the posterior of its logarithm.
    """
    
    def __init__(self, **bart_params):
        """
        Parameters
        ----------
        bart_params :
            The BART hyperparameters. See `sample_bart_prior` for a
            description.
        """
        self.bart_params = dict(default_bart_params)
        self.bart_params.update(bart_params)
    
    def fit(self, X_data, y_data, X_test, sigma=None, n_prior_samples=None):
        """
        Fit a Gaussian process equivalent to a BART.
        
        The Gaussian process prior covariance matrix is approximated sampling
        the BART prior.
        
        Parameters
        ----------
        X_data : (p, n) array
            The n datapoints for p covariates.
        y_data : (n,) array
            Data outcome.
        X_test : (p, N) array
            The points where the posterior is desired. It can overlap with
            X_data.
        sigma : scalar or None
            The error standard deviation. If provided, the inference is
            conditional on the fixed value. If not (default), the posterior on
            the error is computed approximating it with a log-normal. The mode
            is found with numerical minimization.
        n_prior_sample : int or None
            The number of samples used to computed the prior covariance matrix.
            Should be greater than N. If not provided, defaults to 2 * N.
        """
        
        # size variables
        _, N = X_test.shape
        _, n = X_data.shape
        size = n_prior_samples
        if size is None:
            size = 2 * N
        elif size <= N:
            raise ValueError('number of samples {size} shall be greater than number of test points {N}')
        
        # sample BART prior
        X_total = np.concatenate([X_test, X_data], axis=1)
        y_prior = sample_bart_prior(X_data, X_total, y_data, size, error=False, **self.bart_params)
        
        # split sample
        y_prior_test = y_prior[:, :N]
        y_prior_data = y_prior[:, N:]
        
        # check mean of the prior on y_data
        mu_mu = (np.max(y_data) + np.min(y_data)) / 2
        dev = np.mean(y_prior_data, axis=0) - mu_mu
        sdev = np.std(y_prior_data, axis=0) / np.sqrt(size)
        assert np.all(np.abs(dev) < 5 * sdev)
        self.mu_mu = mu_mu
        # TODO add option in sample_bart_prior to keep the mean at zero

        # SVD of prior on y_data
        U, s, Vh = linalg.svd(y_prior_data - mu_mu, full_matrices=False)
        cut = max(size, n) * np.finfo(float).eps * s[0]
        mask = s > cut
        s = s[mask]
        U = U[:, mask]
        Vh = Vh[mask, :]
        s /= np.sqrt(size) # because cov = Y.T @ Y / size
        
        # define function to compute log p(y|sigma^2)
        Vy = Vh @ (y_data - mu_mu)
        s2 = s ** 2
        norm = n * np.log(2 * np.pi)
        def log_prob_y(sigma2):
            ss2 = s2 + sigma2
            logdet = np.sum(np.log(ss2)) + norm
            quad = (Vy / ss2) @ Vy
            return -1/2 * (logdet + quad)
        
        # define function to compute -log p(y, log(sigma))
        q = self.bart_params['q']
        nu = self.bart_params['nu']
        dist = var_epsilon_dist(y_data, nu, q)
        def minus_log_prob_y_log_sigma(log_sigma):
            sigma2 = np.exp(2 * log_sigma)
            cond = log_prob_y(sigma2)
            prior = np.log(2) + 2 * log_sigma + dist.logpdf(sigma2)
            return -cond - prior
        
        # fit sigma if not given
        nosigma = sigma is None
        if nosigma and dist is None: # BART parameters constrain sigma = 0
            sigma = 0
        elif nosigma:
            
            # find maximum w.r.t. log(sigma) of p(y, log(sigma))
            bracket = (
                1/2 * np.log(dist.interval(0.5)[0]),
                1/2 * np.log(dist.interval(0.0)[0]),
            )
            result = optimize.minimize_scalar(minus_log_prob_y_log_sigma, bracket)
            if not result.success:
                raise RuntimeError(result.message)
            sigma = np.exp(result.x)
            
            # Laplace approximation
            interv = 1/2 * np.log(dist.interval(0.5))
            eps = (interv[1] - interv[0]) * 1e-4
            fc = result.fun
            fr = minus_log_prob_y_log_sigma(result.x + eps)
            fl = minus_log_prob_y_log_sigma(result.x - eps)
            deriv = (fr + fl - 2 * fc) / eps ** 2
            self.y_data_prior_log_marginal_prob = 1/2 * np.log(2 * np.pi / deriv) - fc
            self.log_sigma_posterior_sdev = 1 / np.sqrt(deriv)
        
        # compute prior mean and covariance matrix
        Y = (y_prior_test - mu_mu) / np.sqrt(size)
        self.y_test_prior_mean = np.full(N, mu_mu)
        self.y_test_prior_covariance = Y.T @ Y
        
        # save stuff needed to compute stuff
        self.s = s
        self.Vy = Vy
        self.UY = U.T @ Y
        self.sigma = sigma
        self.N = N
        self.dist = dist
        
        # compute marginal likelihood
        self.y_data_prior_log_conditional_prob = log_prob_y(sigma ** 2)
    
    def posterior_mean(self, sigma=None):
        """
        Compute the posterior mean of the Gaussian process.
        
        Parameters
        ----------
        sigma : scalar or None
            Value of the error standard deviation the posterior is conditioned
            on. Defaults to the attribute `sigma`.
        
        Return
        ------
        mean : (N,) array
            The mean evaluated on X_test.
        """
        if sigma is None:
            sigma = self.sigma
        s = self.s
        ss2 = s ** 2 + sigma ** 2
        UY = self.UY
        Vy = self.Vy
        return self.mu_mu + UY.T @ (Vy * s / ss2)
    
    def posterior_covariance_matrix(self, sigma=None):
        """
        Compute the posterior covariance matrix of the Gaussian process.
        
        Parameters
        ----------
        sigma : scalar or None
            Value of the error standard deviation the posterior is conditioned
            on. Defaults to the attribute `sigma`.
        
        Return
        ------
        cov : (N, N) array
            The covariance matrix evaluated on X_test.
        """
        if sigma is None:
            sigma = self.sigma
        s = self.s
        ss2 = s ** 2 + sigma ** 2
        UY = self.UY
        return self.y_test_prior_covariance - ((UY.T * (s ** 2 / ss2)) @ UY)
        
    def _gen(self, gen):
        if gen is None:
            return np.random.default_rng()
        elif not isinstance(gen, np.random.Generator):
            return np.random.default_rng(gen)
        else:
            return gen
    
    def sample_prior(self, n_samples, gen=None, error=False, sample_sigma=True):
        """
        Draw random samples from the prior.
        
        Parameters
        ----------
        n_samples : int
            The number of samples to draw.
        gen : seed or numpy.random.Generator
            The random generator to be used. Defaults to randomly seeded new
            generator.
        error : bool
            Whether to add the error term (default false).
        sample_sigma : bool
            If the error term is added, whether to sample the standard
            deviation from its prior (default) or fix it to the attribute
            `sigma`.
        
        Return
        ------
        samples : (n_samples, N) array
            Samples from the prior evaluated on X_test.
        """
        gen = self._gen(gen)
        samples = self._sample(n_samples, gen, self.prior_cholesky, self.y_test_prior_mean)
        if error:
            sigma = self.sample_sigma_prior((n_samples, 1), gen) if sample_sigma else self.sigma
            samples += sigma * gen.standard_normal((n_samples, self.N))
        return samples
        
    def sample_posterior(self, n_samples, gen=None, error=False, sample_sigma=None):
        """
        Draw random samples from the posterior.
        
        Parameters
        ----------
        n_samples : int
            The number of samples to draw.
        gen : seed or numpy.random.Generator
            The random generator to be used. Defaults to randomly seeded new
            generator.
        error : bool
            Whether to add the error term (default false).
        sample_sigma : bool
            If the error term is added, whether to sample the standard
            deviation from its posterior or fix it to the attribute `sigma`.
            The default is sampling but only if sigma was not fixed when
            calling `fit`.
        
        Return
        ------
        samples : (n_samples, N) array
            Samples from the posterior evaluated on X_test.
        """
        gen = self._gen(gen)
        sample_sigma = hasattr(self, 'log_sigma_posterior_sdev') if sample_sigma is None else sample_sigma
        if sample_sigma:
            sigmas = self.sample_sigma_posterior(n_samples, gen)
            samples = np.empty((n_samples, self.N))
            for i, sigma in enumerate(sigmas):
                mean = self.posterior_mean(sigma)
                cov = self.posterior_covariance_matrix(sigma)
                U = self.cholesky(cov)
                samples[i] = self._sample(1, gen, U, mean)[0]
        else:
            mean = self.posterior_mean()
            cov = self.posterior_covariance_matrix()
            U = self.cholesky(cov)
            samples = self._sample(n_samples, gen, U, mean)
        if error:
            sigma = sigmas[:, None] if sample_sigma else self.sigma
            samples += sigma * gen.standard_normal((n_samples, self.N))
        return samples
    
    def sample_sigma_prior(self, n_samples, gen=None):
        """
        Draw random samples from the prior on the error standard deviation.
        
        Parameters
        ----------
        n_samples : int
            The number of samples to draw.
        gen : seed or numpy.random.Generator
            The random generator to be used. Defaults to randomly seeded new
            generator.
        
        Return
        ------
        samples : (n_samples,) array
            Samples from the prior.
        """
        gen = self._gen(gen)
        sigma2 = self.dist.rvs(n_samples, gen)
        return np.sqrt(sigma2)
    
    def sample_sigma_posterior(self, n_samples, gen=None):
        """
        Draw random samples from the posterior on the error standard deviation.
        
        Parameters
        ----------
        n_samples : int
            The number of samples to draw.
        gen : seed or numpy.random.Generator
            The random generator to be used. Defaults to randomly seeded new
            generator.
        
        Return
        ------
        samples : (n_samples,) array
            Samples from the posterior.
        
        Raises
        ------
        ValueError :
            If the value of the standard deviation (`sigma`) was fixed when
            calling `fit`.
        """
        if not hasattr(self, 'log_sigma_posterior_sdev'):
            raise ValueError('Cannot sample sigma, it was fixed')
        gen = self._gen(gen)
        mean = np.log(self.sigma)
        sdev = self.log_sigma_posterior_sdev
        log_sigma = mean + sdev * gen.standard_normal(n_samples)
        return np.exp(log_sigma)
    
    def _sample(self, n_samples, gen, U, mean):
        samples_iid = gen.standard_normal((n_samples, len(U)))
        return mean + samples_iid @ U
    
    @functools.cached_property
    def prior_cholesky(self):
        return self.cholesky(self.y_test_prior_covariance)
    
    def cholesky(self, M):
        eigmax_bound = np.max(np.sum(np.abs(M), axis=1)) # Gershgorin thorem
        eps = len(M) * np.finfo(M.dtype).eps * eigmax_bound
        A = np.copy(M)
        indices = np.diag_indices_from(M)
        A[indices] += eps
        while True:
            try:
                U = linalg.cholesky(A)
            except linalg.LinAlgError:
                A[indices] += eps
                eps *= 2
            else:
                break
        return U

    def prior_mean(self):
        """
        Compute the prior mean of the Gaussian process.
        
        Parameters
        ----------
        sigma : scalar or None
            Value of the error standard deviation the prior is conditioned
            on. Defaults to the attribute `sigma`.
        
        Return
        ------
        mean : (N,) array
            The mean evaluated on X_test.
        """
        return self.y_test_prior_mean
        
    def prior_covariance_matrix(self):
        """
        Compute the prior covariance matrix of the Gaussian process.
        
        Parameters
        ----------
        sigma : scalar or None
            Value of the error standard deviation the prior is conditioned
            on. Defaults to the attribute `sigma`.
        
        Return
        ------
        cov : (N, N) array
            The covariance matrix evaluated on X_test.
        """
        return self.y_test_prior_covariance
    
    def log_marginal_likelihood(self):
        """
        Get the logarithm of the probability of the data conditioned on the
        fitted or fixed value of the error standard deviation.
        """
        # TODO add option to specify sigma
        return self.y_data_prior_log_conditional_prob
    
    def log_hyper_marginal_likelihood(self):
        """
        Get the logarithm of the probability of the data. Available only if the
        error standard deviation was not fixed.
        """
        attr = 'y_data_prior_log_marginal_prob'
        return getattr(self, attr, None)
    
    def error_log_sdev_posterior_sdev(self):
        """
        Get the standard deviation of the posterior of the logarithm of the
        error standard deviation. Available only if the error standard
        deviation was not fixed.
        
        Note: with first order propagation, the error on the logarithm is the
        relative error.
        """
        attr = 'log_sigma_posterior_sdev'
        return getattr(self, attr, None)
    
    # TODO
    # compare with posterior obtained with BayesTree

@numba.jit(nopython=True, cache=True, fastmath=True)
def bart_correlation_extremity_recursive(n0, nplus, d, alpha, beta, cache):
    cached_value = cache[nplus, d]
    if cached_value >= 0:
        return cached_value
    
    summation = 0
    for k in range(nplus):
        summation += bart_correlation_extremity_recursive(n0, k, d + 1, alpha, beta, cache)
    pnt = alpha * (1 + d) ** -beta
    value = 1 - pnt + pnt / (n0 + nplus) * summation
    
    cache[nplus, d] = value
    return value

@numba.jit(nopython=True, cache=True, parallel=True)
def bart_correlation_extremity_vectorized(n0, nplus, alpha, beta):
    out = np.empty(len(n0))
    for i in numba.prange(len(out)):
        n0i = n0[i]
        if n0i == 0:
            out[i] = 1
        else:
            nplusi = nplus[i]
            n = n0i + nplusi + 1
            cache = np.full((n, n), -1.0)
            out[i] = bart_correlation_extremity_recursive(n0i, nplusi, 0, alpha, beta, cache)
    return out

def bart_correlation_extremity(splitsbetween, totalsplits, alpha, beta):
    """
    Compute the BART prior correlation between two points.
    
    The correlation is computed exactly but only in 1D and one of the points
    must lie outside the data range (the "extremity").
    
    Parameters
    ----------
    splitsbetween : int (n,) array or scalar
        The number of splitting points between the two points.
    totalsplits : int (n,) array or scalar
        The total number of splitting points, i.e., the number of unique
        datapoints minus one.
    alpha, beta : scalar
        The hyperparameters of the branching probability.
    
    Return
    ------
    corr : (n,) array or scalar
        The prior correlation.
    """
    n0 = np.asarray(splitsbetween, int)
    if n0.ndim == 0:
        n0 = n0[None]
    assert n0.ndim == 1
    nplus = int(totalsplits) - n0
    d = 0
    alpha = float(alpha)
    beta = float(beta)
    out = bart_correlation_extremity_vectorized(n0, nplus, alpha, beta)
    if np.isscalar(splitsbetween):
        out = out.item()
    return out

@numba.jit(nopython=True, cache=True, fastmath=True)
def bart_correlation_maxd_recursive(nminus, n0, nplus, d, alpha, beta, maxd):
    sump = 0.0
    p = len(nminus)
    for i in range(p):
        nminusi = nminus[i]
        n0i = n0[i]
        nplusi = nplus[i]
        
        if n0i == 0:
            sump += 1
        elif d >= maxd:
            sump += (nminusi + nplusi) / (nminusi + n0i + nplusi)
        else:
            sumn = 0
            for k in range(nminusi):
                nminus[i] = k
                sumn += bart_correlation_maxd_recursive(nminus, n0, nplus, d + 1, alpha, beta, maxd)
            nminus[i] = nminusi
            for k in range(nplusi):
                nplus[i] = k
                sumn += bart_correlation_maxd_recursive(nminus, n0, nplus, d + 1, alpha, beta, maxd)
            nplus[i] = nplusi
            sump += sumn / (nminusi + n0i + nplusi)
    
    pnt = alpha * (1 + d) ** -beta
    return 1 - pnt * (1 - sump / p)

@numba.guvectorize('(i8[:], i8[:], i8[:], f8, f8, i8, f8[:])', '(p),(p),(p),(),(),()->()', nopython=True)
def bart_correlation_maxd_vectorized(nminus, n0, nplus, alpha, beta, maxd, out):
    assert np.all(nminus >= 0)
    assert np.all(n0 >= 0)
    assert np.all(nplus >= 0)
    assert maxd >= 0
    if maxd == 0:
        out[0] = 1
    else:
        out[0] = bart_correlation_maxd_recursive(nminus, n0, nplus, 0, alpha, beta, maxd - 1)

def bart_correlation_maxd(splitsbefore, splitsbetween, splitsafter, alpha, beta, maxd):
    """
    Compute the BART prior correlation between two points.
    
    The correlation is computed approximately by limiting the maximum depth
    of the trees. Limiting trees to depth 1 is equivalent to setting beta to
    infinity.
    
    The function is fully vectorized.
    
    Parameters
    ----------
    splitsbefore : int (p,) array
        The number of splitting points less than the two points, separately
        along each coordinate.
    splitsbetween : int (p,) array
        The number of splitting points between the two points, separately along
        each coordinate.
    totalsplits : int (p,) array
        The total number of splitting points, i.e., the number of unique
        datapoints minus one, separately along each coordinate.
    alpha, beta : scalar
        The hyperparameters of the branching probability.
    maxd : int
        The maximum depth of the trees. The root has depth zero.
    
    Return
    ------
    corr : scalar
        The prior correlation.
    """
    return bart_correlation_maxd_vectorized(splitsbefore, splitsbetween, splitsafter, alpha, beta, maxd)

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    p = 2 # coordinates
    n = 10 # data samples
    N = 1000 # test samples
    size = 100 # prior samples
    maxplotsamples = 100 # prior samples plotted
    coord = 0 # plotted X coordinate
    sigma_eps = 0.02 # data error
    bart_params = dict(q=0.99) # BART hyperparameters
    
    # generate data from linear model
    gen = np.random.default_rng(202202212324)
    # X_data = gen.uniform(0, 1, size=(p, n))
    # X_data = np.repeat(gen.uniform(0, 1, size=(1, n)), p, axis=0)
    X_data = np.full((p, n), 0.5)
    X_data += gen.uniform(-0.1, 0.1, size=(p, n))
    X_data[coord] = gen.uniform(0, 1, size=n)
    beta = gen.standard_normal(size=p)
    intercept = gen.standard_normal()
    error = sigma_eps * gen.standard_normal(size=n)
    y_data = beta @ X_data + intercept + error
    
    # choose test set

    X_test = np.full((p, N), 0.5)
    X_test[coord] = np.linspace(-0.1, 1.1, N)

    # X_test = gen.uniform(-0.1, 1.1, size=(p, N))
    # indices = np.argsort(X_test[coord])
    # X_test = np.take(X_test, indices, 1)
    
    # X_test = np.repeat(np.linspace(-0.1, 1.1, N)[None], p, axis=0)
    
    # compute equivalent GP
    gp = BartGP(**bart_params)
    gp.fit(X_data, y_data, X_test)
    
    # generate prior and posterior samples
    samplist = []
    for error in [False, True]:
        y_test_prior = sample_bart_prior(X_data, X_test, y_data, size, seed=gen.integers(2 ** 32), error=error, **bart_params)
        y_test_prior_gp = gp.sample_prior(size, gen, error=error)
        y_test_posterior_gp = gp.sample_posterior(size, gen, error=error)
        samplist.append([y_test_prior, y_test_prior_gp, y_test_posterior_gp])
    
    fig, axs = plt.subplots(3, 2, num='bart', clear=True, figsize=[11, 7.5], sharex=True, sharey='col')
    
    # plot the samples
    for axcol, samples in zip(axs.T, samplist):
        for ax, samp in zip(axcol, samples):
            ax.plot(X_data[coord], y_data, '.k', zorder=10)
            for sample in samp[:maxplotsamples]:
                ax.plot(X_test[coord], sample, '-r', alpha=0.1)
    
    axs[0, 0].set_title('BART prior')
    axs[1, 0].set_title('Equivalent Gaussian process prior')
    axs[2, 0].set_title('Equivalent Gaussian process posterior')
    for ax in axs[:, 1]:
        ax.set_title('(with error term)')
    for ax in axs[-1]:
        ax.set_xlabel(f'X[{coord}]')
    for ax in axs[:, 0]:
        ax.set_ylabel('y')
    
    fig.tight_layout()
    fig.show()
