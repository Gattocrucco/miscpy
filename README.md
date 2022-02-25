# Miscpy (miscellaneous python code)

Mostly self-contained, tested and documented python functions for scientific
computing. In case you don't know: you can import a python file as a module by
just placing the file in the same directory as your script.

## Files

  * `argminrelmin.py`: index of the minimum local minimum.
  
  * `bart.py`: Gaussian process equivalent to a BART (Bayesian Regression Tree).
  
  * `breaklines.py`: break a string in lines before or after certain characters.
  
  * `cartprod.py`: cartesian product of numpy arrays. Useful to prepare an
    array of parameter values for a grid search on a slow procedure.
  
  * `ccdelta.py`: compute the cross-correlation of dicrete points with a
    continuous function.
    
  * `clusterargsort.py`: filter away values which are close to an higher value
    in a signal.

  * `coinc.py`: simulates poissonian digital signals and counts coincidences,
    i.e., how many times it happens they are all 1 simultaneously.
    
  * `colormap.py`: make a perceptually uniform colormap.
  
  * `errorsummary.py`: takes a `ufloat` variable (from the
    [uncertainties](https://github.com/lebigot/uncertainties) module) and
    separate its error components by tag. Useful to look at different
    contributions to the error.
  
  * `histogram.py`: version of `numpy.histogram` that also computes
    uncertainties.
    
  * `ising.py`: diagonalize the 1D quantum Ising hamiltonian.
  
  * `loadtxt.py`: faster (a lot) version of `numpy.loadtxt`.
  
  * `maxprominencedip.py`: find local minima sorted by higher negative
    prominence. Fully vectorized.
  
  * `mcmc.py`: functions for blocking and bootstrapping.
  
  * `meanmedian.py`: quick less discrete median for ADC waveforms.
  
  * `neff.py`: compute the effective sample size for an autocorrelated
    sequence, defined as the asymptotic ratio between the variance and variance
    of the sample mean.
    
  * `npzload.py`: superclass to add serialization from/to numpy archives to an
    object (alternatives are
    [joblib](https://joblib.readthedocs.io/en/latest/index.html) and pickle
    protocol 5, but they can execute arbitrary code).
  
  * `rhat.py`: compute the Gelman-Rubin split-$\hat R$ statistics for assessing
    convergence of Markov chains.

  * `runsliced.py`: minimal batching.
  
  * `textbox.py`: put a legend-like box with arbitrary text on a matplotlib
    plot.
    
  * `textmatrix.py`: table manipulation object. Can print in LaTeX format.
  
  * `thin.py`: decimate an array almost evenly with randomized disuniformity.
  
  * `uevuev.py`: compute the variance of the variance.
  
  * `uformat.py`: format numbers with uncertainties.
  
  * `updowncast.py`: recursively cast fields of numpy data type to longer/
    shorter equivalent types. Useful for saving data with shorter types after
    checking it's within bounds.
  
  * `weighted_mean.py`: compute a weighted mean with `ufloat`s (correctly takes
    into account covariance).
