import uncertainties
import collections
import numpy as np

def errorsummary(x):
    """
    Returns error components of a ufloat as a ordered dictionary
    where the keys are the tags of the components. Components
    with the same tag are summed over. The ordering is greater
    component first.
    
    See also
    --------
    gvar.fmt_errorbudget
    """
    comps = x.error_components()
    
    # sum variance for each tag
    var = collections.defaultdict(int)
    for v, sd in comps.items():
        var[v.tag] += sd ** 2
    
    # sort by variance
    tags = list(map(lambda v: v.tag, comps.keys()))
    sds = np.sqrt(np.array([var[tag] for tag in tags]))
    idx = np.argsort(sds)[::-1]
    
    # fill ordered dictionary
    d = collections.OrderedDict()
    for i in idx:
        d[tags[i]] = sds[i]
    
    return d

