# Copyright (C) 2022 Giacomo Petrillo
# Released under the MIT license

import numpy as np

__doc__ = """
Functions for dealing with autocorrelated single-chain samples.
"""

def blocking(v, f, dtype=float, min_groups=2, axis=0):
    """
    Compute f(v), then averages v in groups of two elements and compute again
    f(v), etc. until v has only min_groups elements. When v has odd length, the
    last element is discarded.
    
    If v is not 1D, the blocking is done along the axis specified, the first by
    default.
    
    Parameters
    ----------
    v : array
    f : function
        Function that accepts v or a shorter but otherwise similar array as
        input.
    dtype : type of the output array
    min_groups : integer
        Minimum number of groups v is reduced to.
    axis : integer, default: 0
        The axis along which v is averaged.
    
    Returns
    -------
    out : array
        out[0] = f(v), out[1] = f((v[1::2] + v[::2]) / 2), etc.
    ngroups : array
        The size of v at each grouping/averaging step.
    """
    v = np.moveaxis(v, axis, 0)
    num_levels = int(np.floor(np.log2(len(v) / min_groups))) + 1
    out = np.empty(num_levels, dtype=dtype)
    ngroups = np.empty(num_levels, dtype=int)
    
    i = 0
    while len(v) >= min_groups:
        out[i] = f(np.moveaxis(v, 0, axis))
        ngroups[i] = len(v)
        i = i + 1
        N = 2 * (len(v) // 2)
        v = (v[1:N:2] + v[0:N-1:2]) / 2
        
    assert(i == len(out))
    assert(len(v) < 2 * min_groups)
    
    return out, ngroups

def bootstrap(v, f, n, dtype=float):
    """
    Create an array w with the same length of v by extracting elements at random
    from v (repetitions allowed), and computes f(w). Repeat n times. Return
    the array with the f(w)s.
    
    Parameters
    ----------
    v : array
    f : function that accepts v as input
    n : number of times the array is resampled
    dtype : element type of the output array
    
    Returns
    -------
    out : array of length n with type dtype
    """
    out = np.empty(n, dtype=dtype)
    for i in range(n):
        w = np.random.choice(v, size=v.shape)
        out[i] = f(w)
    return out

def blocking_bootstrap_single(v, f, n, m, out=None):
    """
    The array v is divided into contiguous blocks each of length m (any
    leftover elements are discarded). Then, for n times, a new array w with the
    same length as v (counting out leftovers) is constructed by extracting the
    blocks in random order, with repetitions allowed, and concatenating them.
    For each w, f(w) is computed, and finally the array of the f(w)s is
    returned.
    
    If v is not 1D, the blocking is done along the first axis.
    
    Parameters
    ----------
    v : array
    f : function that accepts v as input
    n : number of times v is resampled
    m : size of the blocks
    out : None or array of length at least n
        If None, a float array of length n is created to store the outputs from
        calling f. Otherwise, out must be an array compatible with the return
        type of f and at least long n.
    
    Returns
    -------
    out : array
        The out parameter if not None, otherwise a newly created float array of
        length n.
    """
    if out is None:
        out = np.empty(n)
    nblocks = len(v) // m
    tail_shape = v.shape[1:]
    v = v[:nblocks * m].reshape(nblocks, m, *tail_shape)
    for i in range(n):
        w = v[np.random.randint(0, nblocks, size=nblocks)]
        out[i] = f(w.reshape(nblocks * m, *tail_shape))
    return out

def blocking_bootstrap_vectorized(v, f, n, m, dtype=float):
    """
    Version of blocking_bootstrap_single vectorized over the parameter m.
    
    Parameters
    ----------
    v : array
    f : function that accepts v as input
    n : number of times v is resampled
    m : size of the blocks or array of sizes of the blocks
    dtype : type of the output array, must be compatible with f's return type
    
    Returns
    -------
    out : array with shape (*m.shape, n)
    """
    m = np.asarray(m)
    out = np.empty(m.shape + (n,), dtype=dtype)
    for idx, single_m in np.ndenumerate(m):
        blocking_bootstrap_single(v, f, n, single_m, out[idx + (...,)])
    return out

def blocking_bootstrap(v, f, n, dtype=float, min_groups=2, min_group_size=1, axis=0):
    """
    Let m be a positive integer. The array v is divided into contiguous blocks
    each of length m (any leftover elements are discarded). Then, for n times,
    a new array w with the same length as v (counting out leftovers) is
    constructed by extracting the blocks in random order, with repetitions
    allowed, and concatenating them. For each w, f(w) is computed, and finally
    the array of the f(w)s is returned.
    
    This procedure is repeated for an array of values of m, which are generated
    in this way: the first m is min_group_size, each m is twice the previous
    one, and the last one is such that v is not divided in more than min_groups
    blocks.
    
    If v is not 1D, the blocking is done along the axis specified, the first by
    default.
    
    Parameters
    ----------
    v : array
    f : function
        Function that accepts v, or an array with the same shape but shorter
        along the specified axis, as input.
    n : integer
        Number of times v is resampled.
    dtype : something convertible to a numpy dtype, default: float
        Type of the output array, must be compatible with f's return type. If
        f's output is multidimensional, you can use dtype=(type, *shape), and
        the shape will be appended to the shape of the output.
    min_groups : integer, default: 2
        Lower bound for the number of groups v is divided into, the
        actual minimum number of blocks may be larger.
    min_group_size : integer, default: 1
        The minimum group size. Useful to avoid computing the bootstrap for
        small groups sizes if you already know that they won't be sufficient.
    axis : integer, default: 0
        The axis along which to resample the array.
    
    Returns
    -------
    out : array with shape (len(ngroups), n)
        The values of f on the resampled v.
    ngroups : array
        The number of blocks into which v is divided.
    """
    v = np.moveaxis(v, axis, 0)
    fun = lambda x: f(np.moveaxis(x, 0, axis))
    max_exp = np.floor(np.log2(len(v) / (min_group_size * min_groups)))
    m = min_group_size * 2 ** np.arange(int(max_exp) + 1)
    ngroups = len(v) // m
    assert(min_groups <= np.min(ngroups) < 2 * min_groups)
    return blocking_bootstrap_vectorized(v, fun, n, m, dtype=dtype), ngroups
