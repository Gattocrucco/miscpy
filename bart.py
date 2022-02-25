import functools

import numpy as np
import numba
from scipy import stats, linalg

def sample_bart_prior(X_data, X_test, y_data, size, *, m=200, alpha=0.95, beta=2, k=2, nu=3, q=0.9, seed=0, check=True):
    """
    Sample from the prior distribution of the model
    
        f(x) = sum_{j=1}^m g(x; T_j, M_j) + eps(x)
    
    Parameters
    ----------
    X_data : (p, n) array
        Data, n vectors of p covariates (used for splitting points)
    X_test : (p, N) array
        Points where the prior is computed
    y_data : (n,) array
        Outcome (used for prior on mu_ij|T_j and sigma)
    size : int
        Number of generated samples
    m : int
        Number of trees
    alpha, beta : scalar
        Parameters of p(T_j), p(node nonterminal) = alpha (1 + d)^(-beta)
    k : scalar
        Inverse scale for p(mu_ij|T_j)
    nu : int
        Degrees of freedom of the chisquare for the prior on the error term
    q : scalar
        Order of the quantile of y used to scale the prior on the error term
        sigma2_epsilon = lambda / (chi2_nu/nu), p(sigma2_epsilon < var(y)) = q.
        Set q = 1 to disable error.
    seed : int
        Seed for the random number generator
    check : bool
        Check arguments
    
    Return
    ------
    y_test : (size, N) array
        Samples from the prior on f(X_test)
    """

    if check:
        X_data = np.asarray(X_data)
        p, n = X_data.shape
        assert n >= 2, n
        for dim in range(p):
            u = np.unique(X_data[dim])
            assert len(u) >= 2, (dim, u)
    
        X_test = np.asarray(X_test)
        p2, N = X_test.shape
        assert p == p2, (p, p2)
        for dim in range(p):
            x = X_test[dim]
            u = np.unique(x)
            assert len(u) == len(x), len(x) - len(u)
    
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
    
    seeds = make_seeds(seed, size)
    return sample_bart_prior_impl(X_data, X_test, y_data, size, m, alpha, beta, k, nu, q, seeds)

def make_seeds(seed, size):
    s = np.random.SeedSequence(seed)
    out = np.empty(size, np.uint32)
    for i in range(len(out)):
        out[i] = s.spawn(1)[0].pool[0]
    return out

@numba.jit(nopython=True, cache=True)
def random_normal(shape):
    return np.random.randn(*shape)

def sample_bart_prior_impl(X_data, X_test, y_data, size, m, alpha, beta, k, nu, q, seeds):
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
    sigma_eps = sigma_epsilon(y_data, nu, q)
    if sigma_eps != 0:
        y_test += sigma_eps * random_normal((size, N))
    
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

@numba.jit(nopython=True, cache=True)
def set_seed(seed):
    np.random.seed(seed)

def sigma_epsilon(y_data, nu, q):
    hat_sigma2 = np.var(y_data)
    dist = stats.invgamma(a=nu / 2)
    lamda = hat_sigma2 / dist.ppf(q)
    sigma2 = lamda * dist.rvs()
    return np.sqrt(sigma2)

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
    sp_map = np.full((p, n + 1), np.iinfo(int).max)
    for dim in range(p):
        x_data = X_data[dim]
        x_test = X_test[dim]
        x_data = np.unique(x_data)
        x_test = np.sort(x_test)
        split = (x_data[1:] + x_data[:-1]) / 2
        split = np.block([-np.inf, split, np.inf])
        sp_map[dim, :len(split)] = np.searchsorted(x_test, split)
        # TODO this may not work if there are non-unique values in X_test,
        # maybe it is necessary to make two sp_maps, one with side='left' and
        # one with side='right'
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
    sp_map : (p, n + 1) int array
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
            ltest = sp_map[dim, ldata]
            rtest = sp_map[dim, rdata]
            indices = sort_map[dim, ltest:rtest]
            cum[indices] += 1
        mask = cum == len(dims)
        y_test[mask] += np.random.normal()

class BartGP:
    
    def __init__(self, **bart_params):
        self.bart_params = bart_params
    
    def fit(self, X_data, y_data, X_test, n_prior_samples=None):
        _, N = X_test.shape
        X_total = np.concatenate([X_test, X_data], axis=1)
        size = n_prior_samples
        if size is None:
            size = 2 * N
        y_prior = sample_bart_prior(X_data, X_total, y_data, size, **self.bart_params)
        
        # split sample
        y_prior_test = y_prior[:, :N]
        y_prior_data = y_prior[:, N:]
        
        # center mean of the prior on y_data
        mu_mu = (np.max(y_data) + np.min(y_data)) / 2
        dev = np.mean(y_prior_data, axis=0) - mu_mu
        sigma = np.std(y_prior_data, axis=0) / np.sqrt(size)
        assert np.all(np.abs(dev) < 5 * sigma)

        # SVD of prior on y_data
        U, s, Vh = linalg.svd(y_prior_data - mu_mu, full_matrices=False)
        _, n = X_data.shape
        cut = max(size, n) * np.finfo(float).eps * s[0]
        mask = s > cut
        s = s[mask]
        U = U[:, mask]
        
        # compute prior and posterior means and covariance matrices
        Y = y_prior_test - mu_mu
        UY = U.T @ Y
        YY = Y.T @ Y
        self.y_test_prior_mean = np.full(N, mu_mu)
        self.y_test_prior_covariance = YY / size
        self.y_test_posterior_mean = (UY.T / s) @ (Vh @ y_data)
        self.y_test_posterior_covariance = (YY - UY.T @ UY) / size
    
    def sample_prior(self, n_samples, gen=None):
        return self.sample(n_samples, gen, self.prior_cholesky, self.y_test_prior_mean)
        
    def sample_posterior(self, n_samples, gen=None):
        return self.sample(n_samples, gen, self.posterior_cholesky, self.y_test_posterior_mean)
    
    def sample(self, n_samples, gen, U, mean):
        samples_iid = self.rnorm(gen, (n_samples, len(U)))
        return mean + samples_iid @ U
    
    def rnorm(self, gen, size):
        if gen is None:
            gen = np.random.default_rng()
        elif not isinstance(gen, np.random.Generator):
            gen = np.random.default_rng(gen)
        return gen.standard_normal(size)
    
    @functools.cached_property
    def prior_cholesky(self):
        return self.cholesky(self.y_test_prior_covariance)
    
    @functools.cached_property
    def posterior_cholesky(self):
        return self.cholesky(self.y_test_posterior_covariance)
    
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
        return self.y_test_prior_mean
        
    def prior_covariance_matrix(self):
        return self.y_test_prior_covariance
    
    def posterior_mean(self):
        return self.y_test_posterior_mean
    
    def posterior_covariance_matrix(self):
        return self.y_test_posterior_covariance
    
    # TODO
    # add error term
    # method to compute marginal likelihood
    # method to optimize error term using BART's inverse chisquare error
    #   prior and marginal likelihood, do laplace to have the full posterior
    # compare with posterior obtained with BayesTree

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    p = 100 # coordinates
    n = 10 # data samples
    N = 1000 # test samples
    size = 100 # prior samples
    maxplotsamples = 100
    coord = 0 # plotted coordinate
    sigma_eps = 0.01 # data error
    
    # generate data from linear model
    gen = np.random.default_rng(202202212324)
    # X_data = gen.uniform(0, 1, size=(p, n))
    X_data = np.repeat(gen.uniform(0, 1, size=(1, n)), p, axis=0)
    beta = gen.standard_normal(size=p)
    intercept = gen.standard_normal()
    error = sigma_eps * gen.standard_normal(size=n)
    y_data = beta @ X_data + intercept + error
    
    # choose test set

    # X_test = np.full((p, N), 0.5)
    # X_test[coord] = np.linspace(-0.1, 1.1, N)

    # X_test = gen.uniform(-0.1, 1.1, size=(p, N))
    # indices = np.argsort(X_test[coord])
    # X_test = np.take(X_test, indices, 1)
    
    X_test = np.repeat(np.linspace(-0.1, 1.1, N)[None], p, axis=0)
    
    # generate prior samples
    bart_params = dict(q=1)
    y_test_prior = sample_bart_prior(X_data, X_test, y_data, size, seed=gen.integers(2 ** 32), **bart_params)
    
    # compute equivalent GP and sample its prior and posterior
    gp = BartGP(**bart_params)
    gp.fit(X_data, y_data, X_test)
    y_test_prior_gp = gp.sample_prior(size, gen)
    y_test_posterior_gp = gp.sample_posterior(size, gen)
    
    fig, axs = plt.subplots(3, 1, num='bart', clear=True, figsize=[8, 7.5], sharex=True, sharey=True)
    
    samplist = [y_test_prior, y_test_prior_gp, y_test_posterior_gp]
    for ax, samples in zip(axs, samplist):
        ax.plot(X_data[coord], y_data, '.k', zorder=10)
        for sample in samples[:maxplotsamples]:
            ax.plot(X_test[coord], sample, '-r', alpha=0.1)
    
    axs[0].set_title('BART prior')
    axs[1].set_title('Equivalent Gaussian process prior')
    axs[2].set_title('Equivalent Gaussian process posterior')
    axs[-1].set_xlabel(f'X[{coord}]')
    for ax in axs:
        ax.set_ylabel('y')
    
    fig.tight_layout()
    fig.show()
