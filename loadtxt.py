# Copyright (C) 2022 Giacomo Petrillo
# Released under the MIT license

import pandas

def loadtxt(fname, dtype=float, skiprows=0, usecols=None, unpack=False):
    """
    Implements a subset of numpy.loadtxt's functionality,
    using pandas.read_csv (which is much faster).
    
    This may not be needed any more in numpy 1.23 (to be released yet).
    """
    kw = dict(
        skiprows = skiprows,
        delim_whitespace = True,
        comment = '#',
    )
    
    # find number of columns
    if usecols is None:
        guess = pandas.read_csv(fname, header=0, nrows=2, **kw)
        ncolumns = guess.shape[1]
    else:
        ncolumns = len(usecols)
    names = list(map(str, range(ncolumns)))
    
    # read data
    dataframe = pandas.read_csv(
        fname, usecols=usecols, dtype=dtype, header=None, names=names,
        **kw
    )
    
    # convert to array
    array = dataframe.values
    if unpack:
        array = array.T
    
    return array
