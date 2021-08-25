import numpy as np
import numba

def clusterargsort(weight, time, radius):
    """
    Sort an array removing elements that are too "close" to higher elements.
    
    Parameters
    ----------
    weight : array (N, K)
        The values to be sorted (along the last axis).
    time : array (N, K)
        The times corresponding to each weight. Must be already sorted along
        the last axis.
    radius : scalar
        A weight is dropped if its time is within this radius from the time of
        an higher weight, unless the higher weight has been already dropped.
    
    Return
    ------
    indices : array (M,)
    length : array (N + 1,)
        M <= N * K. The subarray indices[length[i]:length[i+1]] contains the
        sorting indices for the i-th subarray of weight. To obtain the indices
        for fancy indexing on the first axis, use
        `repeat(arange(N), diff(length))`.
    """
    
    # TODO (ideally)
    # arbitrary shape instead of N with broadcasting
    # arbitrary axis instead of last
    # time optional, defaults to array indices
    # time not already sorted
    # radius depends on array element
    # propagate option
    
    time = np.asarray(time)
    weight = np.asarray(weight)
    assert time.shape == weight.shape
    nevents, _ = time.shape
    
    indices = np.empty(time.size, int)
    length = np.empty(nevents + 1, int)
    
    _clusterargsort(time, weight, radius, indices, length)
        
    return np.copy(indices[:length[-1]]), length

@numba.njit(cache=True)
def _clusterargsort(time, weight, radius, indices, length):
    length[0] = 0
    for i in range(len(length) - 1):
        length[i + 1] = length[i] + _clusterargsort_nv(time[i], weight[i], radius, indices[length[i]:])

@numba.njit(cache=True)
def _clusterargsort_nv(time, weight, radius, out):
    N = len(time)
    keep = np.ones(N, np.bool8)
    sort = np.argsort(weight)[::-1]
    iout = 0
    
    for i in sort:
        if keep[i]:
            out[iout] = i
            iout += 1
            
            t = time[i]
            
            j = i + 1
            while j < N and time[j] - t <= radius:
                keep[j] = False
                j += 1
            
            j = i - 1
            while j >= 0 and t - time[j] <= radius:
                keep[j] = False
                j -= 1
    
    out[:iout] = out[:iout][::-1]
    return iout

if __name__ == '__main__':
    
    from matplotlib import pyplot as plt
    
    gen = np.random.default_rng(202101271237)
    
    N = 100
    t = np.sort(gen.uniform(0, 1, size=(1, N)))
    x = gen.standard_normal(size=(1, N))
    i, l = clusterargsort(x, t, 10 / N)
    assert np.array_equal(l, [0, len(i)])
    
    fig, ax = plt.subplots(num='clusterargsort', clear=True)
    
    ax.plot(t[0], x[0], 'o-', markerfacecolor='none')
    ax.plot(t[0, i], x[0, i], 'xk')
    
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle=':')
    
    fig.show()
