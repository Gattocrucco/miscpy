# Copyright (C) 2022 Giacomo Petrillo
# Released under the MIT license

import numpy as np

def cartprod(**fields):
    """
    Fill a structured numpy array with the cartesian product of input arrays.
    
    Parameters
    ----------
    **fields : arrays
        The argument names are used as field names in the output array.

    Return
    ------
    out : structured array
        The shape of the array is the concatenation of the shapes of the input
        arrays, in the order in which they appear in the arguments.
    
    Examples
    --------
    >>> cartprod(a=[1, 2, 3], b=["X", "Y"])
    array([[(1, 'X'), (1, 'Y')],
           [(2, 'X'), (2, 'Y')],
           [(3, 'X'), (3, 'Y')]], dtype=[('a', '<i8'), ('b', '<U1')])
    """
    # TODO for completeness, add *args to do an unnamed cartesian product.
    # Only one of *args, **fields can be non-empty.
    
    fields = {k: np.asarray(v) for k, v in fields.items()}
    
    shape = sum((array.shape for array in fields.values()), start=())
    dtype = np.dtype([
        (name, array.dtype)
        for name, array in fields.items()
    ])
    out = np.empty(shape, dtype)
    
    offset = 0
    for name, array in fields.items():
        index = np.full(len(shape), None)
        length = len(array.shape)
        index[offset:offset + length] = slice(None)
        out[name] = array[tuple(index)]
        offset += length
    
    return out

if __name__ == '__main__':
    
    import unittest
    
    class TestCartProd(unittest.TestCase):
        
        def test_empty(self):
            x = cartprod()
            y = np.array((), [])
            self.assertTrue(np.all(x == y))
        
        def test_one(self):
            x = cartprod(a=[1, 2, 3])
            y = np.array([1, 2, 3], [('a', int)])
            self.assertTrue(np.all(x == y))
            
        def test_two(self):
            x = cartprod(f0=[1,2,3], f1=[4,5])
            y = np.array([
                [(1, 4), (1, 5)],
                [(2, 4), (2, 5)],
                [(3, 4), (3, 5)],
            ], 'i8,i8')
            self.assertTrue(all(np.all(x[k] == y[k]) for k in x.dtype.names))
        
    unittest.main()
