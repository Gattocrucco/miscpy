# Copyright (C) 2022 Giacomo Petrillo
# Released under the MIT license

import tqdm

def runsliced(fun, ntot, n=None, pbar=None, args=()):
    """
    Do a task in batches.
    
    Run a cycle which calls a given function with a progressing slice as sole
    argument until a range is covered, printing a progressbar.
    
    Parameters
    ----------
    fun : function
        A function with a single parameter which is a slice object.
    ntot : int
        The end of the range covered by the sequence of slices.
    n : int, optional
        The length of each slice (the last slice may be shorter). If None, the
        function is called once with the slice 0:ntot.
    pbar : bool, optional
        By default, a progressbar is printed iff n is specified. Force the
        progressbar to be printed or not with this option.
    args : tuple
        Additional arguments passed to `fun`.
    """
    
    # TODO parallel processing, possible code:
    # with multiprocessing.Pool(nprocesses) as pool:
    #      for _ in tqdm.tqdm(pool.imap_unordered(task, slices), total=nslices):
    #          pass
    # or use joblib
    
    if pbar is None:
        pbar = n is not None
    if n is None:
        n = ntot

    it = range(ntot // n + bool(ntot % n))
    if pbar:
        it = tqdm.tqdm(it)

    for i in it:
        start = i * n
        end = min((i + 1) * n, ntot)
        fun(slice(start, end), *args)

if __name__ == '__main__':
    
    import unittest
    
    class TestRunSliced(unittest.TestCase):
        
        def test_unaligned(self):
            slices = []
            def task(slice):
                slices.append(slice)
            runsliced(task, 5, 2, False)
            self.assertEqual(slices, [slice(0, 2), slice(2, 4), slice(4, 5)])
        
        def test_aligned(self):
            slices = []
            def task(slice):
                slices.append(slice)
            runsliced(task, 4, 2, False)
            self.assertEqual(slices, [slice(0, 2), slice(2, 4)])

    unittest.main()
