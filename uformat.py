# Copyright (C) 2022 Giacomo Petrillo
# Released under the MIT license

import numpy as np
import uncertainties
from uncertainties import unumpy
import math
import copy

__doc__ = """
Functions to format uncertainties and covariance matrices.

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

def exponent(x):
    return int(math.floor(math.log10(abs(x))))

def int_mantissa(x, n, e):
    return round(x * 10 ** (n - 1 - e))

def naive_ndigits(x, n):
    log10x = math.log10(abs(x))
    n_int = int(math.floor(n))
    n_frac = n - n_int
    log10x_frac = log10x - math.floor(log10x)
    return n_int + (log10x_frac < n_frac)

def ndigits(x, n):
    ndig = naive_ndigits(x, n)
    xexp = exponent(x)
    rounded_x = int_mantissa(x, ndig, xexp) * 10 ** xexp
    if rounded_x > x:
        rounded_ndig = naive_ndigits(rounded_x, n)
        if rounded_ndig > ndig:
            x = rounded_x
            ndig = rounded_ndig
    return x, ndig

def mantissa(x, n, e):
    m = int_mantissa(x, n, e)
    s = str(abs(int(m)))
    assert len(s) == n or len(s) == n + 1 or (m == 0 and n < 0)
    if n >= 1 and len(s) == n + 1:
        e = e + 1
        s = s[:-1]
    return s, e

def insert_dot(s, n, e, addzeros=True):
    e = e + len(s) - n
    n = len(s)
    if e >= n - 1:
        s = s + '0' * (e - n + 1)
    elif e >= 0:
        s = s[:1 + e] + '.' + s[1 + e:]
    elif e <= -1 and addzeros:
        s = '0' * -e + s
        s = s[:1] + '.' + s[1:]
    return s

def tostring(x):
    return '0' if x == 0 else f'{x:#.6g}'

def uformat(mu, s, errdig=1.5, sep=None, *, shareexp=True, outersign=False, uniexp=False, minnegexp=6, minposexp=4, padzeros=False, possign=False):
    """
    Format a number with uncertainty.
    
    Parameters
    ----------
    mu : number
        The central value.
    s : number
        The error.
    errdig : number
        The number of digits of the error to be shown. Must be >= 1. It can be
        a noninteger, in which case the number of digits switches between the
        lower nearest integer to the upper nearest integer as the first decimal
        digit (after rounding) crosses 10 raised to the fractional part of
        `errdig`. Default 1.5.
    sep : None or str
        The separator put between the central value and the error. Eventual
        spaces must be included. If None, put the error between parentheses,
        sharing decimal places/exponential notation with the central value.
        Default None.
    shareexp : bool
        Applies if sep is not None. When using exponential notation, whether to
        share the exponent between central value and error with outer
        parentheses. Default True.
    outersign : bool
        Applied when sep is not None and shareexp is True. Whether to put the
        sign outside or within the parentheses. Default False
    uniexp : bool
        When using exponential notation, whether to use unicode characters
        instead of the standard ASCII notation. Default False.
    minnegexp : int
        The number of places after the comma at which the notation switches
        to exponential notation. Default 6. The number of places from the
        greater between central value and error is considered.
    minposexp : int
        The power of ten of the least significant digit at which exponential
        notation is used. Default 4. Setting higher values may force padding
        the error with zeros, depending on `errdig`.
    padzeros : bool
        Whether to pad with zeros when not using exponential notation due to
        `minposexp` even if the least significant digits is not on the units.
        Default False, i.e., show more digits than those specified.
    possign : bool
        Whether to put a `+` before the central value when it is positive.
        Default False.
    """
    if errdig < 1:
        raise ValueError('errdig < 1')
    if not math.isfinite(mu) or not math.isfinite(s) or s <= 0:
        if sep is None:
            return f'{tostring(mu)}({tostring(s)})'
        else:
            return f'{tostring(mu)}{sep}{tostring(s)}'
    
    s, sndig = ndigits(s, errdig)
    sexp = exponent(s)
    muexp = exponent(mu) if mu != 0 else sexp - sndig - 1
    smant, sexp = mantissa(s, sndig, sexp)
    mundig = sndig + muexp - sexp
    mumant, muexp = mantissa(mu, mundig, muexp)
    musign = '-' if mu < 0 else '+' if possign else ''
    
    if mundig >= sndig:
        use_exp = muexp >= mundig + minposexp or muexp <= -minnegexp
        base_exp = muexp
    else:
        use_exp = sexp >= sndig + minposexp or sexp <= -minnegexp
        base_exp = sexp
    
    if use_exp:
        mumant = insert_dot(mumant, mundig, muexp - base_exp)
        smant = insert_dot(smant, sndig, sexp - base_exp, sep is not None)
    elif base_exp >= max(mundig, sndig) and not padzeros:
        mumant = str(abs(round(mu)))
        smant = str(abs(round(s)))
    else:
        mumant = insert_dot(mumant, mundig, muexp)
        smant = insert_dot(smant, sndig, sexp, sep is not None)
    
    if not outersign:
        mumant = musign + mumant
    
    if use_exp:
        if uniexp:
            asc = '0123456789+-'
            uni = '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻'
            table = str.maketrans(asc, uni)
            exp = str(base_exp).translate(table)
            suffix = '×10' + exp
        else:
            suffix = f'e{base_exp:+}'
        if sep is None:
            r = mumant + '(' + smant + ')' + suffix
        elif shareexp:
            r = '(' + mumant + sep + smant + ')' + suffix
        else:
            r = mumant + suffix + sep + smant + suffix
    elif sep is None:
        r = mumant + '(' + smant + ')'
    else:
        r = mumant + sep + smant
    
    if outersign:
        r = musign + r
    
    return r

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

_uformat_vect = np.vectorize(uformat, otypes=[str])

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
    
    def check(n, s, string, *args, **kw):
        defaults = dict(minnegexp=2, minposexp=0)
        defaults.update(kw)
        f = uformat(n, s, *args, **defaults)
        if f != string:
            raise RuntimeError(f'{f!r} != {string!r}')

    def allchecks():
        check(1, 0.2, "1.00 pm 0.20", 1.5, " pm ")
        check(1, 0.3, "1.00 pm 0.30", 1.5, " pm ")
        check(1, 0.31, "1.00 pm 0.31", 1.5, " pm ")
        check(1, 0.32, "1.0 pm 0.3", 1.5, " pm ")
        check(-1, 0.34, "-1.00 pm 0.34", 2, " pm ")
        check(0, 0, "0 pm 0", 2, " pm ")
        check(123456, 0, "123456. pm 0", 2, " pm ")
        check(12345.6, 0, "12345.6 pm 0", 2, " pm ")
        check(12345.67, 0, "12345.7 pm 0", 2, " pm ")
        check(1e8, 0, "1.00000e+08 pm 0", 2, " pm ")
        check(1e-2, 0, "0.0100000 pm 0", 2, " pm ")
        check(1e-1, 0, "0.100000 pm 0", 2, " pm ")
        check(12345.99, 0, "12346.0 pm 0", 2, " pm ")
        check(0, 0.001, "(0.0 pm 1.0)e-3", 2, " pm ")
        check(0, 0.01, "(0.0 pm 1.0)e-2", 2, " pm ")
        check(0, 0.1, "0.00 pm 0.10", 2, " pm ")
        check(0, 1, "0.0 pm 1.0", 2, " pm ")
        check(0, 10, "0 pm 10", 2, " pm ")
        check(0, 100, "(0.0 pm 1.0)e+2", 2, " pm ")
        check(0, 1000, "(0.0 pm 1.0)e+3", 2, " pm ")
        check(0, 0.0196, "(0.0 pm 2.0)e-2", 2, " pm ")
        check(0, 0.196, "0.00 pm 0.20", 2, " pm ")
        check(0, 1.96, "0.0 pm 2.0", 2, " pm ")
        check(0, 19.6, "0 pm 20", 2, " pm ")
        check(0, 196, "(0.0 pm 2.0)e+2", 2, " pm ")
        check(0, 0.00996, "(0.0 pm 1.0)e-2", 2, " pm ")
        check(0, 0.0996, "0.00 pm 0.10", 2, " pm ")
        check(0, 0.996, "0.0 pm 1.0", 2, " pm ")
        check(0, 9.96, "0 pm 10", 2, " pm ")
        check(0, 99.6, "(0.0 pm 1.0)e+2", 2, " pm ")
        check(0.025, 3, "0.0 pm 3.0", 2, " pm ")
        check(0.0251, 0.3, "0.03 pm 0.30", 2, " pm ")
        check(0.025, 0.03, "(2.5 pm 3.0)e-2", 2, " pm ")
        check(0.025, 0.003, "(2.50 pm 0.30)e-2", 2, " pm ")
        check(0.0025, 0.003, "(2.5 pm 3.0)e-3", 2, " pm ")
        check(0.251, 3, "0.3 pm 3.0", 2, " pm ")
        check(2.5, 3, "2.5 pm 3.0", 2, " pm ")
        check(25, 3, "25.0 pm 3.0", 2, " pm ")
        check(2500, 300, "(2.50 pm 0.30)e+3", 2, " pm ")
        check(1, 0.99, "1.0 pm 1.0", 1.5, " pm ")
        check(math.inf, 1.0, "inf pm 1.00000", 2, " pm ")
        check(-math.inf, 1.0, "-inf pm 1.00000", 2, " pm ")
        check(0, math.inf, "0 pm inf", 2, " pm ")

        check(1, 0.2, "1.00(20)", 1.5, None)
        check(1, 0.3, "1.00(30)", 1.5, None)
        check(1, 0.31, "1.00(31)", 1.5, None)
        check(1, 0.32, "1.0(3)", 1.5, None)
        check(-1, 0.34, "-1.00(34)", 2, None)
        check(0, 0, "0(0)", 2, None)
        check(123456, 0, "123456.(0)", 2, None)
        check(12345.6, 0, "12345.6(0)", 2, None)
        check(12345.67, 0, "12345.7(0)", 2, None)
        check(1e8, 0, "1.00000e+08(0)", 2, None)
        check(1e-2, 0, "0.0100000(0)", 2, None)
        check(1e-1, 0, "0.100000(0)", 2, None)
        check(12345.99, 0, "12346.0(0)", 2, None)
        check(0, 0.001, "0.0(1.0)e-3", 2, None)
        check(0, 0.01, "0.0(1.0)e-2", 2, None)
        check(0, 0.1, "0.00(10)", 2, None)
        check(0, 1, "0.0(1.0)", 2, None)
        check(0, 10, "0(10)", 2, None)
        check(0, 100, "0.0(1.0)e+2", 2, None)
        check(0, 1000, "0.0(1.0)e+3", 2, None)
        check(0, 0.0196, "0.0(2.0)e-2", 2, None)
        check(0, 0.196, "0.00(20)", 2, None)
        check(0, 1.96, "0.0(2.0)", 2, None)
        check(0, 19.6, "0(20)", 2, None)
        check(0, 196, "0.0(2.0)e+2", 2, None)
        check(0, 0.00996, "0.0(1.0)e-2", 2, None)
        check(0, 0.0996, "0.00(10)", 2, None)
        check(0, 0.996, "0.0(1.0)", 2, None)
        check(0, 9.96, "0(10)", 2, None)
        check(0, 99.6, "0.0(1.0)e+2", 2, None)
        check(0.025, 3, "0.0(3.0)", 2, None)
        check(0.0251, 0.3, "0.03(30)", 2, None)
        check(0.025, 0.03, "2.5(3.0)e-2", 2, None)
        check(0.025, 0.003, "2.50(30)e-2", 2, None)
        check(0.0025, 0.003, "2.5(3.0)e-3", 2, None)
        check(0.251, 3, "0.3(3.0)", 2, None)
        check(2.5, 3, "2.5(3.0)", 2, None)
        check(25, 3, "25.0(3.0)", 2, None)
        check(2500, 300, "2.50(30)e+3", 2, None)
        check(1, 0.99, "1.0(1.0)", 1.5, None)
        check(math.inf, 1.0, "inf(1.00000)", 2, None)
        check(-math.inf, 1.0, "-inf(1.00000)", 2, None)
        check(0, math.inf, "0(inf)", 2, None)
    
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
            
        def test_normcov(self):
            # just check that it works because there's always someone willing to rewrite this stupid function
            cov = [[4, -3], [-3, 16]]
            normalized_cov = [[1, -0.375], [-0.375, 1]]
            self.assertTrue(np.array_equal(normcov(cov), normalized_cov))
    
    allchecks()
    unittest.main()
