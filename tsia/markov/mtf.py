import numpy as np
from numba import njit, prange
from pyts.preprocessing.discretizer import KBinsDiscretizer

MTF_N_BINS = 8
MTF_STRATEGY = 'quantile'

def compute_mtf_statistics(mtf):
    """
    Extract statistics from the Markov transition fields passed as an argument.
    This function extract the mean and and standard deviation of the main
    diagonal of the transition field.
    
    PARAMS
    ======
        mtf (numpy.ndarray)
            Numpy array containing a Markov transition field.
            
    RETURNS
    =======
        statistics (dict)
            Dictionnary containing the mean and standard deviation of the
            self-transition probabilities of the Markov transition fields
            (contained in the main diagonal)
    """
    avg_self_transition_prob = np.diag(mtf).mean()
    std_self_transition_prob = np.diag(mtf).std()

    statistics = {
        'Average self-transition prob': avg_self_transition_prob,
        'Std self-transition prob': std_self_transition_prob
    }
    
    return statistics
    
def discretize(timeseries, n_bins=MTF_N_BINS, strategy=MTF_STRATEGY):
    """
    Discretize a given time series into a certain number of bins.
    
    PARAMS
    ======
        timeseries: numpy.ndarray
            The time series data to discretize
        n_bins: int (default = 8)
            Number of bins to discretize the data into. Also known as the size
            of the alphabet.
        strategy: 'uniform', 'quantile' or 'normal' (default = 'quantile')
            Strategy used to define the widths of the bins:
            - 'uniform': All bins in each sample have identical widths
            - 'quantile': All bins in each sample have the same number of points
            - 'normal': Bin edges are quantiles from a standard normal distribution
            
    RETURNS
    =======
        X_binned: numpy.ndarray
            The binned signal
        bin_edges: list
            A list with all the bin edges
    """
    # Build a discretizer and discretize the signal passed as argument:
    discretizer = KBinsDiscretizer(n_bins=n_bins, strategy=strategy)
    X = timeseries.values.reshape(1, -1)
    X_binned = discretizer.fit_transform(X)[0]
    
    # Extract the bin edges computed during the discretization process:
    bin_edges = discretizer._compute_bins(X, 1, n_bins, strategy)
    bin_edges = np.hstack((timeseries.values.min(), 
                           bin_edges[0], 
                           timeseries.values.max()))
    
    return X_binned, bin_edges

@njit()
def markov_transition_matrix(X_binned):
    """
    Build a Markov transition matrix from a binned signal passed as an argument.
    
    PARAMS
    ======
        X_binned: numpy.ndarray
            An array with the bin number associated to each value of an 
            original time series.
    
    RETURNS
    =======
        X_mtm: numpy.ndarray
            The Markov transition matrix
    """
    # 0-initialize the Markov transition matrix:
    n_bins = np.max(X_binned) + 1
    X_mtm = np.zeros((n_bins, n_bins))
    
    # We loop through each timestamp:
    n_timestamps = X_binned.shape[0]
    for j in prange(n_timestamps - 1):
        # For each timestamp 'j', we count the transition from the bin 
        # associated to the current timestamp to the bin associated to 
        # the next timestamp 'j+1':
        X_mtm[X_binned[j], X_binned[j + 1]] += 1
        
    return X_mtm