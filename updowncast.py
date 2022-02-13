import numpy as np

def downcast(dtype, *shorttypes):
    """
    Downcast a numpy data type, in the sense of converting it to a similar type
    but of smaller size. Works recursively for structured/array data types.
    
    Parameters
    ----------
    dtype : numpy data type
        The data type to downcast.
    *shorttypes : numpy data types
        The types that the dtype can be downcasted to.
    
    Return
    ------
    dtype : numpy data type
        The downcasted data type. Fields and shapes are preserved, but not the
        memory layout.
    
    Examples
    --------
    >>> downcast('f8', 'f4')    # shorter floating type
    dtype('float32')
    >>> downcast('f8', 'i4')    # no downcasting from floating to integer
    dtype('float64')
    >>> downcast('f4', 'f8')    # no upcasting
    dtype('float32')
    >>> downcast('S4', 'S2')    # strings are truncated
    dtype('S2')
    >>> downcast('f8,i8', 'f4', 'i4')           # structured data type
    dtype([('f0', '<f4'), ('f1', '<i4')])
    >>> x = np.zeros(5, [('a', float), ('b', float)])
    >>> x
    array([(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
          dtype=[('a', '<f8'), ('b', '<f8')])
    >>> x.astype(downcast(x.dtype, 'f4'))       # downcast an array
    array([(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
          dtype=[('a', '<f4'), ('b', '<f4')])
    """
    dtype = np.dtype(dtype)
    downlist = [np.dtype(t) for t in shorttypes]
    return recursive_cast(dtype, downlist, [])

def upcast(dtype, *longtypes):
    """
    Like downcast but for upcasting to longer equivalent types.
    """
    dtype = np.dtype(dtype)
    uplist = [np.dtype(t) for t in longtypes]
    return recursive_cast(dtype, [], uplist)

def recursive_cast(dtype, downlist, uplist):
    if dtype.names is not None:
        # structured dtype
        return np.dtype([
            (name, recursive_cast(field[0], downlist, uplist))
            for name, field in dtype.fields.items()
        ])
    elif dtype.subdtype is not None:
        # array dtype
        # note: has names => does not have subdtype
        return np.dtype((recursive_cast(dtype.base, downlist, uplist), dtype.shape))
    else:
        # simple dtype, do the casting
        for downtype in downlist:
            if shorter_dtype(downtype, dtype):
                return downtype
        for uptype in uplist:
            if shorter_dtype(dtype, uptype):
                return uptype
    
    return dtype

def shorter_dtype(t1, t2):
    """
    Returns true iff t1 is a numpy dtype of the same kind of t2 and t1 is
    strictly smaller than t2. Does not work with composite dtypes
    (structured/shaped).
    """
    return np.can_cast(t1, t2, 'safe') and np.can_cast(t2, t1, 'same_kind') and t1.itemsize < t2.itemsize

if __name__ == '__main__':
    
    import unittest
    
    class TestCast(unittest.TestCase):
        
        def test_examples_downcast(self):
            IO = [
                [('f8', 'f4'), np.float32],
                [('f8', 'i4'), np.float64],
                [('f4', 'f8'), np.float32],
                [('S4', 'S2'), np.dtype('S2')],
                [('f8,i8', 'f4', 'i4'), np.dtype('f4,i4')],
                [(np.dtype('2f8'), 'f4'), np.dtype('2f4')],
                [(np.dtype('2f8,3i8'), 'f4', 'i4'), np.dtype('2f4,3i4')],
            ]
            for args, out in IO:
                self.assertEqual(downcast(*args), out)
        
        def test_examples_upcast(self):
            IO = [
                [('f4', 'f8'), np.float64],
                [('i4', 'f8'), np.int32],
                [('f8', 'f4'), np.float64],
                [('S2', 'S4'), np.dtype('S4')],
                [('f4,i4', 'f8', 'i8'), np.dtype('f8,i8')],
                [(np.dtype('2f4'), 'f8'), np.dtype('2f8')],
                [(np.dtype('2f4,3i4'), 'f8', 'i8'), np.dtype('2f8,3i8')],
            ]
            for args, out in IO:
                self.assertEqual(upcast(*args), out)
    
    unittest.main()
