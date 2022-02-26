# Copyright (C) 2022 Giacomo Petrillo
# Released under the MIT license

# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import uncertainties
from uncertainties import unumpy
import math
import copy

__doc__ = """
The principal function in this module is formatcov.

Functions
---------
formatcov : format a vector with a covariance matrix.
uformat : format a number with uncertainty.
num2si : format a number using a multiple of 3 exponent or SI suffixes.
num2sup, num2sub : format a number as unicode superscript or subscript.

Classes
-------
TextMatrix : simple class for representing tables.
"""

def normcov(cov):
    """
    normalize a square matrix so that the diagonal is 1:
    ncov[i,j] = cov[i,j] / sqrt(cov[i,i] * cov[j,j])

    Parameters
    ----------
    cov : (N,N)-shaped array-like
        the matrix to normalize

    Returns
    -------
    ncov : (N,N)-shaped array-like
        the normalized matrix
    """
    cov = np.copy(np.asarray(cov, dtype='float64'))
    s = np.sqrt(np.diag(cov))
    for i in range(len(s)):
        for j in range(i + 1):
            p = s[i] * s[j]
            if p != 0:
                cov[i, j] /= p
            elif i != j:
                cov[i, j] = np.nan
            cov[j, i] = cov[i, j]
    return cov

_d = lambda x, n: int(("%.*e" % (n - 1, abs(x)))[0])
_ap = lambda x, n: float("%.*e" % (n - 1, x))
_nd = lambda x: math.floor(math.log10(abs(x))) + 1
def _format_epositive(x, e, errsep=True, minexp=3, dot=True):
    # DECIDE NUMBER OF DIGITS
    if _d(e, 2) < 3:
        n = 2
        e = _ap(e, 2)
    elif _d(e, 1) < 3:
        n = 2
        e = _ap(e, 1)
    else:
        n = 1
    # FORMAT MANTISSAS
    dn = int(_nd(x) - _nd(e)) if x != 0 else -n
    nx = n + dn
    if nx > 0:
        ex = _nd(x) - 1
        if nx > ex and abs(ex) <= minexp:
            xd = nx - ex - 1
            ex = 0
        else:
            xd = nx - 1
        sx = "%.*f" % (xd, x / 10**ex)
        se = "%.*f" % (xd, e / 10**ex)
    else:
        ex = _nd(e) - n
        sx = '0'
        se = "%.*g" % (n, e / 10**ex)
    # RETURN
    if errsep:
        return sx, se, ex
    short_se = se[-(n+1):] if '.' in se[-n:] else se[-n:]
    # ("%#.*g" % (n, e * 10 ** (n - _nd(e))))[:n]
    if not dot:
        short_se = short_se.replace('.', '')
    return sx + '(' + short_se + ')', '', ex

def uformat(x, e, pm=None, percent=False, comexp=True, nicexp=False, dot=True):
    """
    Format a value with its uncertainty.

    Parameters
    ----------
    x : number (or something understood by float(), e.g. string representing number)
        The value.
    e : number (or as above)
        The uncertainty.
    pm : string, optional
        The "plusminus" symbol. If None, use compact notation.
    percent : bool
        If True, also format the relative error as percentage.
    comexp : bool
        If True, write the exponent once.
    nicexp : bool
        If True, format exponent like ×10¹²³.
    dot : bool
        If True, eventually put decimals separator in uncertainty when pm=None.

    Returns
    -------
    s : string
        The formatted value with uncertainty.

    Examples
    --------
    uformat(123, 4) --> '123(4)'
    uformat(10, .99) --> '10.0(10)'
    uformat(1e8, 2.5e6) --> '1.000(25)e+8'
    uformat(1e8, 2.5e6, pm='+-') --> '(1.000 +- 0.025)e+8'
    uformat(1e8, 2.5e6, pm='+-', comexp=False) --> '1.000e+8 +- 0.025e+8'
    uformat(1e8, 2.5e6, percent=True) --> '1.000(25)e+8 (2.5 %)'
    uformat(nan, nan) --> 'nan +- nan'

    See also
    --------
    xe, xep
    """
    x = float(x)
    e = abs(float(e))
    if not math.isfinite(x) or not math.isfinite(e) or e == 0:
        return "%.3g %s %.3g" % (x, '+-', e)
    sx, se, ex = _format_epositive(x, e, errsep=not (pm is None), dot=dot)
    if ex == 0:
        es = ''
    elif nicexp:
        es = "×10" + num2sup(ex, format='%d')
    else:
        es = "e%+d" % ex
    if pm is None:
        s = sx + es
    elif comexp and es != '':
        s = '(' + sx + ' ' + pm + ' ' + se + ')' + es
    else:
        s = sx + es + ' ' + pm + ' ' + se + es
    if (not percent) or sx.split('(')[0] == '0':
        return s
    pe = e / abs(x) * 100.0
    return s + " (%.*g %%)" % (2 if pe < 100.0 else 3, pe)

unicode_pm = u'±'
unicode_sigma = u'σ'

# this function taken from stackoverflow and modified
# http://stackoverflow.com/questions/17973278/python-decimal-engineering-notation-for-mili-10e-3-and-micro-10e-6
def num2si(x, format='%.15g', si=True, space=' '):
    """
    Returns x formatted using an exponent that is a multiple of 3.

    Parameters
    ----------
    x : number
        the number to format
    format : string
        printf-style string used to format the mantissa
    si : boolean
        if true, use SI suffix for exponent, e.g. k instead of e3, n instead of
        e-9 etc. If the exponent would be greater than 24, numerical exponent is
        used anyway.
    space : string
        string interposed between the mantissa and the exponent

    Returns
    -------
    fx : string
        the formatted value

    Example
    -------
         x     | num2si(x)
    -----------|----------
       1.23e-8 |  12.3 n
           123 |  123
        1230.0 |  1.23 k
    -1230000.0 |  -1.23 M
             0 |  0

    See also
    --------
    util_format, xe, xep
    """
    x = float(x)
    if x == 0:
        return format % x + space
    exp = int(math.floor(math.log10(abs(x))))
    exp3 = exp - (exp % 3)
    x3 = x / (10 ** exp3)

    if si and exp3 >= -24 and exp3 <= 24 and exp3 != 0:
        exp3_text = space + 'yzafpnμm kMGTPEZY'[(exp3 - (-24)) // 3]
    elif exp3 == 0:
        exp3_text = space
    else:
        exp3_text = 'e%s' % exp3 + space

    return (format + '%s') % (x3, exp3_text)

_subscr  = '₀₁₂₃₄₅₆₇₈₉₊₋ₑ․'
_subscrc = '0123456789+-e.'
_supscr  = '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ᵉ·'

def num2sub(x, format=None):
    """
    Format a number as subscript.

    Parameters
    ----------
    x : string or number
        The number to format.
    format : None or string
        If None, x is interpreted as string and formatted subscript as-is.
        If string, it is a %-format used to format x before converting to subscript.

    Returns
    -------
    s : string
        x written in subscript.
    """
    if format is None:
        x = str(x)
    else:
        x = format % float(x)
    for i in range(len(_subscrc)):
        x = x.replace(_subscrc[i], _subscr[i])
    return x

def num2sup(x, format=None):
    """
    Format a number as superscript.

    Parameters
    ----------
    x : string or number
        The number to format.
    format : None or string
        If None, x is interpreted as string and formatted superscript as-is.
        If string, it is a %-format used to format x before converting to superscript.

    Returns
    -------
    s : string
        x written in superscript.
    """
    if format is None:
        x = str(x)
    else:
        x = format % float(x)
    for i in range(len(_subscrc)):
        x = x.replace(_subscrc[i], _supscr[i])
    return x

_uformat_vect = np.vectorize(uformat, otypes=[str], excluded={2, 3, 4, 5, 6})

def formatcov(x, cov=None, labels=None, corrfmt='.0f'):
    """
    Format an estimate with a covariance matrix as an upper triangular matrix
    with values on the diagonal (with uncertainties) and correlations
    off-diagonal.
    
    Parameters
    ----------
    x : M-length array
        Values to be written on the diagonal.
    cov : (M, M) matrix or None
        Covariance matrix from which uncertainties and correlations are
        computed. If None, a covariance matrix is extracted from x with
        uncertainties.covariance_matrix(x).
    labels : list of strings
        Labels for the header of the matrix. If there are less than M labels,
        only the first elements are given labels.
    corrfmt : str
        Format for the correlations.
    
    Returns
    -------
    matrix : TextMatrix
        A TextMatrix instance. Can be converted to a string with str(). Has a
        method latex() to format as a LaTeX table.
    
    See also
    --------
    TextMatrix
    
    Examples
    --------
    >>> popt, pcov = scipy.optimize.curve_fit(f, x, y, ...)
    >>> print(formatcov(popt, pcov))
    >>> print(formatcov(popt, pcov).latex()) # LaTeX table
    """
    if cov is None:
        cov = uncertainties.covariance_matrix(x)
        x = unumpy.nominal_values(x)
   
    if isinstance(x, dict) and isinstance(cov, dict):
        keys = list(x.keys())
        x = [x[key] for key in keys]
        cov = [[float(cov[keyi, keyj]) for keyj in keys] for keyi in keys]
        if labels is None:
            labels = keys
    
    pars = _uformat_vect(x, np.sqrt(np.diag(cov)))
    corr = normcov(cov) * 100
    
    matrix = []
    if not (labels is None):
        if len(labels) < len(x):
            labels = list(labels) + [''] * (len(x) - len(labels))
        elif len(labels) > len(x):
            labels = labels[:len(x)]
        matrix.append(labels)
    
    for i in range(len(corr)):
        matrix.append([pars[i]])
        for j in range(i + 1, len(corr)):
            c = corr[i, j]
            cs = ('{:' + corrfmt + '} %').format(c) if math.isfinite(c) else str(c)
            matrix[-1].append(cs)
    
    return TextMatrix(matrix, fill_side='left')

def _array_like(obj):
    return isinstance(obj, tuple) or isinstance(obj, list) or isinstance(obj, np.ndarray)

class TextMatrix(object):
    """
    Object to format tables.
    
    Methods
    -------
    text : generic formatter.
    latex : format as latex table.
    transpose : transpose the matrix.
    __mul__ : multiplication stacks matrices horizontally.
    __truediv__ : division stacks matrices vertically.
    """
    
    def __init__(self, matrix, fill='', fill_side='right'):
        """
        Create a 2D matrix of arbitrary objects.
        
        Parameters
        ----------
        matrix : tipically list of lists
            An object that can be interpreted as 2D matrix. If it can not,
            the resulting TextMatrix is 1x1. If it can be interpreted as
            a 1D array, then each row has length 1.
        fill :
            If the rows are uneven, shorter ones are filled with this object.
        fill_side : 'right' or 'left'
            Specify on which side shorter rows are filled.
        """
        # make sure matrix is at least 1D
        if not _array_like(matrix):
            matrix = [matrix]
        matrix = copy.copy(matrix)
        
        # make sure each element of matrix is at least 1D,
        # and get lengths of all elements
        lengths = []
        for i in range(len(matrix)):
            if not _array_like(matrix[i]):
                matrix[i] = [matrix[i]]
            lengths.append(len(matrix[i]))
            
        # make sure each element of matrix has the same length
        maxlength = max(lengths, default=0)
        for i in range(len(matrix)):
            if len(matrix[i]) != maxlength:
                if fill_side == 'right':
                    matrix[i] = list(matrix[i]) + [fill] * (maxlength - len(matrix[i]))
                elif fill_side == 'left':
                    matrix[i] = [fill] * (maxlength - len(matrix[i])) + list(matrix[i])
                else:
                    raise KeyError(fill_side)
            else:
                matrix[i] = copy.copy(matrix[i])
        
        self._matrix = matrix

    def __repr__(self):
        return self.text(before=' ')
    
    def __str__(self):
        return self.text(before=' ')
    
    def text(self, before='', after='', between='', newline='\n', subs={}):
        """
        Format the matrix as a string. Each element in the matrix
        is converted to a string using str(), then elements are concatenated
        in left to right, top to bottom order.
        
        Parameters
        ----------
        before, after, between : string
            Strings placed respectively before, after, between the elements.
        newline : string
            String placed after each row but the last.
        subs : dictionary
            Dictionary specifying substitutions applied to each element,
            but not to the parameters <before>, <after>, etc. The keys
            of the dictionary are the strings to be replaced, the values
            are the replacement strings. If you want the substitutions
            to be performed in a particular order, use a OrderedDict
            from the <collections> module.
        
        Returns
        -------
        s : string
            Matrix formatted as string.
        """
        nrows = len(self._matrix)
        if nrows == 0:
            return ''
        ncols = len(self._matrix[0])
        
        # convert matrix elements to strings, applying subs
        str_matrix = []
        for row in range(nrows):
            str_matrix.append([])
            for col in range(ncols):
                element = str(self._matrix[row][col])
                for sub, rep in subs.items():
                    element = element.replace(sub, rep)
                str_matrix[row].append(element)
        col_maxlength = np.max([list(map(len, row)) for row in str_matrix], axis=0)
        
        # convert string matrix to text
        s = ''
        for row in range(nrows):
            for col in range(ncols):
                element = str_matrix[row][col]
                formatter = '{:>%d}' % col_maxlength[col]
                s += before + formatter.format(element) + after
                if col < ncols - 1:
                    s += between
            if row < nrows - 1:
                s += newline
        return s
    
    def latex(self, **kwargs):
        """
        Format the matrix as a LaTeX table.
        
        Keyword arguments
        -----------------
        Keyword arguments are passed to the <text> method, taking
        precedence on settings for LaTeX formatting. The `subs` argument is
        treated separately: the default one is updated with the contents of the
        one from **kwargs.
        
        Returns
        -------
        s : string
            Matrix formatted as LaTeX table.
        
        See also
        --------
        TextMatrix.text
        """
        subs = {
            '%': r'\%',
            '&': r'\&'
        }
        kw = dict(before='', after='', between=' & ', newline=' \\\\\n', subs=subs)
        kw['subs'].update(kwargs.pop('subs', dict()))
        kw.update(kwargs)
        return self.text(**kw)
    
    def transpose(self):
        """
        Returns a transposed copy of the matrix. The elements
        are not copied.
        """
        return type(self)([[self._matrix[row][col] for row in range(len(self._matrix))] for col in range(len(self._matrix[0]))])
    
    def __mul__(self, other):
        """Multiplication concatenates two matrices horizontally."""
        assert(isinstance(other, TextMatrix))
        assert(len(other._matrix) == len(self._matrix))
        return self.__class__([l + r for l, r in zip(self._matrix, other._matrix)])
    
    def __truediv__(self, other):
        """Division concatenates two matrices vertically."""
        assert(isinstance(other, TextMatrix))
        assert(len(self._matrix) == 0 or len(other._matrix) == 0 or len(other._matrix[0]) == len(self._matrix[0]))
        return self.__class__(self._matrix + other._matrix)

if __name__ == '__main__':
    import unittest
    
    class TestTextMatrix(unittest.TestCase):
        
        def test_empty(self):
            TextMatrix([])
    
    class TestFormat(unittest.TestCase):

        def test_num2si(self):
            # generic check
            self.assertEqual(num2si(1312), '1.312 k')
            # check crash on 0
            self.assertEqual(num2si(0), '0 ')
            # check that format options are respected
            self.assertEqual(num2si(1, format='%+g'), '+1 ')
            # check that default rounding is sufficient
            self.assertEqual(num2si(0.7), '700 m')
    
        def test_uformat(self):
            # check that big-exponent values use exponential notation
            self.assertEqual(uformat(1.23456789e-8, 1.1111e-10, pm=None, percent=False), '1.235(11)e-8')
            # check that number of digits is chosen correctly in case of uncertainty rounding
            self.assertEqual(uformat(10, 0.99, pm=None, percent=False), '10.0(1.0)')
            # check that percentual error is not negative
            self.assertEqual(uformat(-1, 1, pm=None, percent=True), '-1.0(1.0) (100 %)')
            # check that percentual error is suppressed if mantissa is 0 when we are using compact error notation
            self.assertEqual(uformat(0.001, 1, pm=None, percent=True), '0(10)e-1')
            # check that if mantissa is 0 and compact notation is not used the error has correct exponent
            self.assertEqual(uformat(0, 1, pm='+-'), '(0 +- 10)e-1')
            self.assertEqual(uformat(0, 10, pm='+-'), '0 +- 10')
            self.assertEqual(uformat(0, 1e5, pm='+-'), '(0 +- 10)e+4')
        
        def test_normcov(self):
            # just check that it works because there's always someone willing to rewrite this stupid function
            cov = [[4, -3], [-3, 16]]
            normalized_cov = [[1, -0.375], [-0.375, 1]]
            self.assertTrue(np.array_equal(normcov(cov), normalized_cov))

    unittest.main()
