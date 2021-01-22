# tsia: a Python package for time series analysis through images
---
tsia (time series images analysis) is a package to perform time series
analysis and diagnostics thanks to imaging techniques.

## Installation
### Dependencies
tsia requires:

* python (>= 3.7)
* matplotlib (>= 3.3.0)
* networkx (>= 2.5)
* numba (>= 0.48.0)
* numpy (>= 1.17.5)
* python-louvain (>= 0.14)
* pyts (>= 0.11.0)

### User installation
If you already have a working installation of the aforementioned modules,
you can easily install tsia using `pip`:

```
pip install tsia
```

## Implemented features
tsia consists of the following modules:

* `markov`: this module provides some methods to compute statistics from Markov
transition fields as implemented in the `pyts` package. It also implements
functions to map back MTF output to the original time series.

* `network_graph`: this module provides some methods to compute statistics and
several encoding techniques to interpret network graphs built upon MTF.

* `plot`: this module implements several useful visualization techniques to
provide insights into univariate timeseries. This module implement MTF, network
graph visualization and several line graphs (vanilla time series, colored 
time series with different color encodings, quantile discretization overlay...).

* `utils`: this modules includes some utilities leveraged by the other ones.