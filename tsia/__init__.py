"""Time series image analysis module for Python.
tisa is a Python package for time series analysis through images. It
aims to enable advanced diagnostic tools for time series through imaging
and visualization capabilities.
"""

from .plot import plot
from .markov import mtf as markov
from .network_graph import ng as network_graph

__version__ = '0.1'
__all__ = ['markov', 'network_graph', 'plot']