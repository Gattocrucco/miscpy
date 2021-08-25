import numpy as np
import numba

def ccdelta(f, t, tout, left, right, w=None):
    """
    Compute the cross correlation of a given function with a mixture of dirac
    deltas, i.e., evaluate g(t) = sum_i w_i f(t_i - t).
    
    Parameters
    ----------
    f : function
        A function with signature scalar->scalar than can be compiled with
        numba.njit (it can be provided already compiled).
    t : array (..., N)
        The locations of deltas. Must be sorted along the last axis.
    tout : array (..., K)
        The points where the cross correlation is evaluated. Must be sorted
        along the last axis.
    left, right : scalar
        The support of f. f is assumed to be zero outside of it and not
        evaluated.
    w : array (..., N), optional
        The amplitudes of deltas. If not provided they are set to 1.
    
    Return
    ------
    out : array (..., K)
        The cross correlation evaluated on tout. The shape is determined by
        broadcasting together t, tout, and w along all axes but the last.
    """
    assert callable(f)
    if not hasattr(f, '__numba__'):
        f = numba.njit(f)
    t = np.asarray(t)
    assert len(t.shape) >= 1
    tout = np.asarray(tout)
    assert len(tout.shape) >= 1
    assert np.isscalar(left)
    assert np.isscalar(right)
    if w is None:
        w = np.broadcast_to(np.ones(1), t.shape[-1:])
    w = np.asarray(w)
    assert len(w.shape) >= 1
    assert w.shape[-1] == t.shape[-1]
    
    shape = np.broadcast(t[..., 0], tout[..., 0], w[..., 0]).shape
    
    t = np.broadcast_to(t, shape + t.shape[-1:]).reshape(-1, t.shape[-1])
    tout = np.broadcast_to(tout, shape + tout.shape[-1:]).reshape(-1, tout.shape[-1])
    w = np.broadcast_to(w, shape + w.shape[-1:]).reshape(-1, w.shape[-1])
    
    out = np.zeros(shape + tout.shape[-1:])
    
    _ccdelta(f, t, w, tout, left, right, out.reshape(-1, out.shape[-1]))
    
    return out

@numba.njit
def _ccdelta(f, t, w, tout, left, right, out):
    """
    Compiled implementation of ccdelta. The out array must be initialized to
    zero. The shapes of t, tout, w, out must be (M, N), (M, K), (M, N), (M, K).
    """
    for iouter in numba.prange(len(out)):
        td = t[iouter]
        wd = w[iouter]
        tg = tout[iouter]
        g = out[iouter]
        
        igmin = 0
        for i in range(len(td)):
            ti = td[i]
            wi = wd[i]
            
            igmin += np.searchsorted(tg[igmin:], ti - right)
            ig = igmin
            while ig < len(tg) and tg[ig] <= ti - left:
                g[ig] += wi * f(ti - tg[ig])
                ig += 1
        
if __name__ == '__main__':
    
    from matplotlib import pyplot as plt
    from scipy import interpolate
    
    fig = plt.figure('ccdelta')
    fig.clf()
    
    ax = fig.subplots(1, 1)
    
    def f(t):
        return 1 if 0 < t < 1 else 0
    
    gen = np.random.default_rng(202012201928)
    t = np.sort(gen.uniform(0, 5, size=25))
    w = -1 + 2 * gen.integers(0, 2, size=len(t))
    tout = np.linspace(np.min(t) - 1, np.max(t) + 1, 10000)
    
    out = ccdelta(f, t, tout, 0, 1, w=w)
    out2 = ccdelta(numba.njit(f), t[None, None], tout[None], 0, 1, w=w)[0, 0]
    assert np.array_equal(out, out2)
    
    y = interpolate.interp1d(tout, out)
    ax.plot(tout, out, color='lightgray')
    for weight, color in [(1, 'black'), (-1, 'red')]:
        s = w == weight
        ax.plot(t[s], y(t[s]), 'x', color=color)
        ax.plot(t[s] - 1, y(t[s] - 1), '+', color=color)
        
    fig.tight_layout()
    plt.show()
