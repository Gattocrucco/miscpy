# Copyright (C) 2022 Giacomo Petrillo
# Released under the MIT license

import numpy as np
import numba

def argminrelmin(a, axis=None, out=None):
    """
    Return the index of the minimum relative minimum.
    
    A relative minimum is an element which is lower than its neigbours, or the
    central element of a series of contiguous elements which are equal to each
    other and lower than their external neighbours.
    
    If there are more relative minima with the same value, return the first. If
    there are no relative minima, return -1.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.
    """
        
    a = np.asarray(a)
    if axis is None:
        a = a.reshape(-1)
    else:
        a = np.moveaxis(a, axis, -1)
    return _argminrelmin(a, out=out)

@numba.guvectorize(['(f8[:],i8[:])'], '(n)->()', cache=True)
def _argminrelmin(a, out):
    idx = -1
    val = 0
    wide = -1
    for i in range(1, len(a) - 1):
        if a[i - 1] > a[i] < a[i + 1]:
            if a[i] < val or idx < 0:
                idx = i
                val = a[i]
        elif a[i - 1] > a[i] == a[i + 1]:
            wide = i
        elif wide >= 0 and a[i - 1] == a[i] < a[i + 1]:
            if a[i] < val or idx < 0:
                idx = (wide + i) // 2
                val = a[i]
        elif a[i] != a[i + 1]:
            wide = -1
    out[0] = idx
    
if __name__ == '__main__':
    
    import unittest
    
    class TestArgMinRelMin(unittest.TestCase):
        
        def test_argminrelmin(self):
            a = np.concatenate([
                np.linspace(-1, 1, 20),
                np.linspace(1, 0, 10),
                np.linspace(0, 1, 10),
                np.linspace(1, 0.5, 10),
                np.linspace(0.5, 1, 10),
            ])
            i = argminrelmin(a)
            self.assertEqual(i, 29)
            self.assertEqual(a[i], 0)
            i = argminrelmin([3,2,1,2,3])
            self.assertEqual(i, 2)
            i = argminrelmin([3,2,1,1,2,3])
            self.assertEqual(i, 2)
            i = argminrelmin([3,2,1,1,1,2,3])
            self.assertEqual(i, 3)
            i = argminrelmin([3,2,1,1,0])
            self.assertEqual(i, -1)
            i = argminrelmin([0,1,1,2,3])
            self.assertEqual(i, -1)
    
        def test_axis_out(self):
            a = np.repeat([[1, 0, 1]], 2, axis=0)
            out = np.empty(2, 'i8')
            i = argminrelmin(a, axis=1, out=out)
            self.assertIs(i, out)
            self.assertTrue(np.array_equal(out, [1, 1]))

    unittest.main()
