# Copyright (C) 2022 Giacomo Petrillo
# Released under the MIT license

import numba

@numba.jit('u4(f8,f8[:],f8[:],f8[:],f8)', nopython=True, cache=True)
def _coinc(T, r, tc, tm, tand):
    """
    T = total time
    r = rates (!) > 0
    tc = durations (!) > 0
    tm = dead times (!) >= tc
    tand = superposition time for coincidence (!) > 0
    """
    tau = 1 / r
    n = len(tau)
    
    ncoinc = 0
    
    # generate an event on each sequence
    t = -np.ones(n) * tm
    for i in range(n):
        nt = 0
        while nt < tm[i]:
            nt += np.random.exponential(scale=tau[i])
        t[i] += nt
    
    # first sequence for which a new event is generated after a coincidence
    first = 0
        
    # check if total time elapsed
    while t[first] < T:
        
        # intersection interval of events
        il = t[first]
        ir = t[first] + tc[first]
        
        # minimum of right endpoints in case of coincidence
        rmin = ir
        rmini = first
        
        # iterate over sequences
        for i in range(n):
            if i != first:
                coinc_found = False
            
                # check if coincidence is still possible
                while t[i] < ir - tand:
                
                    # check for coincidence
                    nil = max(il, t[i])
                    nir = min(ir, t[i] + tc[i])
                    if nir - nil >= tand:
                        il = nil
                        ir = nir
                        coinc_found = True
                        if t[i] + tc[i] < rmin:
                            rmin = t[i] + tc[i]
                            rmini = i
                        break
                    
                    # generate a new event
                    nt = 0
                    while nt < tm[i]:
                        nt += np.random.exponential(scale=tau[i])
                    t[i] += nt
                
                if not coinc_found:
                    break
                
        if coinc_found and il < T:
            ncoinc += 1
            first = rmini
        
        # generate a new event on the first sequence
        nt = 0
        while nt < tm[first]:
            nt += np.random.exponential(scale=tau[first])
        t[first] += nt
    
    return ncoinc

def coinc(T, tand, *seqs):
    """
    Simulate logical signals and count coincidences.
    
    Arguments
    ---------
    T : number >= 0
        Total time.
    tand : number >= 0
        Minimum superposition time to yield a coincidence.
    *seqs : r1, tc1, tm1, r2, tc2, tm2, ...
        r = Rate of signals.
        tc = Duration of signal.
        tm = Non restartable dead-time. If tm < tc, tc is used instead.
    
    Returns
    -------
    N : integer
        Number of coincidences. Since it is a count, an estimate of the
        variance is N itself.
    """
    T = np.float64(T)
    if T < 0:
        raise ValueError('Total time is negative.')
    
    tand = np.float64(tand)
    if tand < 0:
        raise ValueError('Superposition time is negative.')
    
    if len(seqs) % 3 != 0:
        raise ValueError('Length of seqs is not a multiple of 3.')
    if len(seqs) / 3 < 2:
        raise ValueError('There are less than 2 sequences in seqs.')
    
    seqs = np.array(seqs, dtype=np.float64)
    r = seqs[::3]
    tc = seqs[1::3]
    tm = np.max((seqs[2::3], tc), axis=0)
    
    if not all(r > 0):
        ValueError('All rates must be positive.')
    if not all(tc > 0):
        ValueError('All durations must be positive.')
    
    return _coinc(T, r, tc, tm, tand)
