"""
The :mod:`tsia.plot` module includes useful methods to plot easier to interpret 
time series.
"""

from .plot import *

__all__ = [
    'get_style_colors',
    'get_categories_colors',
    'plot_mtf_metrics', 
    'plot_markov_transition_field',
    'plot_timeseries_signal',
    'plot_multivariate_timeseries',
    'plot_timeseries_quantiles',
    'plot_colored_timeseries',
    'plot_network_graph'
]