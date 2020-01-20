import numpy as np

def thin_indices(source_n, target_n, order='random'):
    """
    Return indices to get target_n elements from a source_n long array, almost
    evenly spaced, including the first and last elements.
    
    Parameters
    ----------
    source_n : integer >= 2
        The length of the array to be thinned.
    target_n : integer >= 2, <= source_n
        The target length.
    order : one of 'random', 'long-first', 'short-first'
        The distance from an index to the next is not uniform in general. There
        is a "short" distance and a "long" distance = short + 1. This parameter
        specifies how the distances are distributed in the array.
    """
    assert(source_n == int(source_n))
    assert(target_n == int(target_n))
    source_n = int(source_n)
    target_n = int(target_n)
    
    assert(target_n <= source_n)
    assert(source_n >= 2)
    assert(target_n >= 2)
    
    short_skip = (source_n - 1) // (target_n - 1)
    long_skip = short_skip + 1
    long_count = (source_n - 1) % (target_n - 1)
    short_count = (target_n - 1) - long_count
    
    if order == 'random':
        skip = np.zeros(target_n - 1, dtype=int)
        skip[:long_count] = long_skip
        skip[long_count:] = short_skip
        np.random.shuffle(skip)
        indices = np.concatenate([[0], np.cumsum(skip)])
    elif order == 'long-first':
        long_idxs = np.arange(long_count) * long_skip
        short_idxs = np.arange(short_count) * short_skip
        short_idxs += long_count * long_skip
        indices = np.concatenate([long_idxs, short_idxs, [source_n - 1]])
    elif order == 'short-first':
        long_idxs = np.arange(long_count) * long_skip
        short_idxs = np.arange(short_count) * short_skip
        long_idxs += short_count * short_skip
        indices = np.concatenate([short_idxs, long_idxs, [source_n - 1]])
    else:
        raise KeyError(order)
    
    assert(len(indices) == target_n)
    assert(indices[0] == 0)
    assert(indices[-1] == source_n - 1)
    
    return indices

if __name__ == '__main__':
    import unittest
    
    class TestThinIndices(unittest.TestCase):
        
        def test_100(self):
            for order in ['random', 'long-first', 'short-first']:
                for i in range(2, 101):
                    for j in range(2, i + 1):
                        indices = thin_indices(i, j, order='random')
                        self.assertTrue(np.all(np.diff(indices) >= 1))
                        self.assertTrue(len(np.unique(np.diff(indices))) <= 2)
        
        def test_order(self):
            with self.assertRaises(KeyError):
                thin_indices(100, 50, order='cippalippa')
        
        def test_n(self):
            args = [
                (-1, -1),
                (2, 3),
                (1, 1),
                (0, 1),
                (2, 1),
                (100, 101)
            ]
            for sn, tn in args:
                with self.assertRaises(AssertionError):
                    thin_indices(sn, tn)
    
    unittest.main()
