import numpy as np

def histogram(a, bins=10, range=None, weights=None, density=None):
    """
    Same as numpy.histogram, but returns uncertainties.
    The uncertainty of a bin with density=False is:
    Case 1 (no weights):
        Square root of the count.
    Case 2 (weights):
        Quadrature sum of the weights (the same as case 1 if the weights are unitary).
    If density=True, the uncertainty is simply divided by the same factor
    as the counts.
    Note: empty bins get a zero uncertainty.
    Note: when using weights and 32 bit floating point, underflow may occur.
    
    Returns
    -------
    hist
    bin_edges
    unc_hist :
        Uncertainty of hist.
    
    See also
    --------
    numpy.histogram
    """
    hist, bin_edges = np.histogram(a, bins=bins, range=range, weights=weights, density=density)
    
    if weights is None and not density:
        unc_hist = np.sqrt(hist)
    elif weights is None:
        counts, _ = np.histogram(a, bins=bins, range=range)
        unc_hist = np.sqrt(counts) / (len(a) * np.diff(bin_edges))
    else:
        unc_hist, _ = np.histogram(a, bins=bins, range=range, weights=weights ** 2)
        unc_hist = np.sqrt(unc_hist)
        if density:
            unc_hist /= (np.sum(weights) * np.diff(bin_edges))
    
    return hist, bin_edges, unc_hist
