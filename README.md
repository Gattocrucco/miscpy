# Miscpy (miscellaneous python code)

Mostly self-contained documented python functions for scientific computing. In
case you don't know: you can import a python file as a module by just placing
the file in the same directory as your script.

## Files

* `errorsummary.py`: takes a `ufloat` variable (from the
  [uncertainties](https://github.com/lebigot/uncertainties) module) and separate
  its error components by tag. Useful to look at different contributions to the
  error.

* `histogram.py`: version of `numpy.histogram` that also computes uncertainties.

* `loadtxt.py`: faster (a lot) version of `numpy.loadtxt`.

* `mcmc.py`: functions for blocking and bootstrapping.

* `uevuev.py`: function to compute the variance of the variance.

* `weighted_mean.py`: function to compute a weighted mean with `ufloat`s
  (correctly takes into account covariance).
