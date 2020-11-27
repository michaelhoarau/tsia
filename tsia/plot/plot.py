import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from matplotlib import gridspec

# Useful constants definition
COLORMAP = 'jet'

def get_style_colors():
    """
    Get list of colors of the current style selected with
    matplotlib.pyplot.style.use()
    
    RETURNS
    =======
        colors: list
            Returns a list of the main colors of the current style
    """
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    return colors
    
def plot_markov_transition_field(mtf, 
                                 ax=None, 
                                 title='', 
                                 colormap=COLORMAP,
                                 reversed_cmap=False
                                 ):
    """
    Plot a Markov transition field using imshow()
    
    PARAMS
    ======
        mtf: numpy.ndarray
            A numpy array containing a Markov transition field
        ax: matplotlib.axes
            An ax to use to plot the MTF. This function will create one if 
            this argument is set to None. Default: None
        title: string
            A string to use as a title for the plot. Default to ''
        colormap: string
            Name of the colormap to use
        reversed_cmap: boolean
            Indicate if the colormap must be reversed
            
    RETURNS
    =======
        ax: matplotlib.axes
            Returns the ax used or created back to the caller.
        mappable_image: matplotlib.image.AxesImage
            An AxesImage object that can be used to map a colorbar
    """
    if reversed_cmap == True:
        colormap = plt.cm.get_cmap(colormap).reversed()
    else:
        colormap = plt.cm.get_cmap(colormap)
    
    # If no ax is passed as argument, we create a new plot:
    if ax is None:
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
    
    mappable_image = ax.imshow(mtf, cmap=colormap, origin='lower')
    
    if title is not None:    
        ax.set_title(title, fontsize=10)
    
    ax.axis("off")
    
    return ax, mappable_image
    
def plot_network_graph(graph, 
                       ax=None, 
                       title='', 
                       colormap=COLORMAP,
                       reversed_cmap=False,
                       encoding=None):
    """
    Plot a network graph using the draw_networkx() method from the networkx
    package.
    
    PARAMS
    ======
        graph: networkx.classes.graph.Graph
            The network graph to encode
        ax: matplotlib.axes
            An ax to use to plot the network graph. This function will create 
            one if this argument is set to None. Default: None
        title: string
            A title for this plot
        colormap: string
            Name of the colormap to use
        reversed_cmap: boolean
            Indicate if the colormap must be reversed
        encoding: dict
            A dictionnary containing the encoded values for each node size,
            edge color and node color. We need at least the following keys to
            be present:
                - 'node_size': list of integers (one per node)
                - 'edge_color': list of colors (one per edge)
                - 'node_color': list of colors (one per node)
    
    RETURNS
    =======
        ax: matplotlib.axes
            Returns the ax used or created back to the caller.
    """
    # Use the encoding passed as input:
    if encoding is not None:
        options = encoding
        
    # Or build a neutral one:
    else:
        options = dict({
            'node_size': 100,
            'edge_color': '#AAAAAA',
            'node_color': '#999999'
        })
        
    options.update({
        'edgecolors': '#000000',    # Color of the node edges
        'linewidths': 1,            # Width of the node edges
        'width': 0.1,               # Width of the edges
        'alpha': 0.6,               # Transparency of both edges and nodes
        'with_labels': False,
        'cmap': colormap
    })

    if reversed_cmap == True:
        colormap = plt.cm.get_cmap(colormap).reversed()
    else:
        colormap = plt.cm.get_cmap(colormap)
    
    # If no ax is passed as argument, we create a new plot:
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)

    nx.draw_networkx(graph, **options, pos=nx.spring_layout(graph), ax=ax)
    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=10)
    
    return ax

def plot_mtf_metrics(mtf):
    """
    Plot several graphs to help characterize a Markov transition field:
    - A histogram for the self-transition probabilities distribution
    - A plot of all transition probabilities across all the MTF diagonals. This
      shows the transition probabilities at different scales of the time series
    - The means and standard deviation of all transition probabilities across
      all the diagonals of the MTF
      
    PARAMS
    ======
        mtf: numpy.ndarray
            A numpy array containing a Markov transition field
    """
    # Initialization:
    colors = get_style_colors()
    image_size = mtf.shape[0]
    fig = plt.figure(figsize=(26.5, 4))
    gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[1, 1, 1])

    ax = fig.add_subplot(gs[0])
    ax.hist(np.diag(mtf), color=colors[1], edgecolor='#FFFFFF')
    ax.set_title('Distribution of self-transition probabilities')

    ax = fig.add_subplot(gs[1])
    means, stds = [], []
    for diag_index in range(0, image_size):
        diag = np.diag(mtf, k=diag_index)
        means.append(diag.mean())
        stds.append(diag.std())

        diag = np.hstack((np.zeros(shape=(diag_index,)), diag))
        ax.plot(diag, linewidth=1.5 - diag_index/10, alpha=1 - diag_index/10)
    ax.set_title('Transition probabilities across the MTF diagonals')

    ax = fig.add_subplot(gs[2])
    ax.plot(means, label='MTF diagonals mean')
    ax.plot(stds, label='MTF diagonals std')
    ax.set_title('Probability means and standard deviation across the MTF diagonals')
    ax.legend()

    plt.show()
    
def plot_timeseries_signal(timeseries, ax=None, label=''):
    """
    Plot a given timeseries.
    
    PARAMS
    ======
        timeseries: pandas.Series or pandas.DataFrame
            The timeseries to plot
        ax: matplotlib.axes
            An ax to use to plot the time series. This function will create one 
            if this argument is set to None. Default: None
        label: string
            A label for this time series. Default: ''
            
    RETURNS
    =======
        ax: matplotlib.axes
            Returns the ax used or created back to the caller.
    """
    # If no ax is passed as argument, we create a new plot:
    if ax is None:
        fig = plt.figure(figsize=(18,4))
        ax = fig.add_subplot(111)
        
    # Build the plot parameters:
    params = {
        'linewidth': 0.5,
        'alpha': 0.8
    }
    if label != '':
        params.update({'label': label})
        
    # Plot the time series:
    ax.plot(timeseries, **params)
    
    return ax

def plot_timeseries_quantiles(timeseries, bin_edges, label=''):
    """
    This methods superimpose the quantiles stored in the bin edges with the 
    actual time series signal. This visualization helps understanding the 
    transition probabilities from a given value (associated to a certain
    quantile bin) to the next time series value (potentially associated to
    another quantile bin or the same one for the particular case of the 
    self-transition probabilities)
    
    PARAMS
    ======
        timeseries: pandas.Series or pandas.DataFrame
            The timeseries to plot
        bin_edges: numpy.ndarray
            A numpy array with all the bin edges of the quantiles to plot
        label: string
            A label for this time series. Default: ''
            
    RETURNS
    =======
        ax: matplotlib.axes
            Returns the ax created to plot this timeseries and quantiles
    """
    colors = get_style_colors()

    fig = plt.figure(figsize=(28,8))
    ax = fig.add_subplot(111)
    plot_timeseries_signal(timeseries, ax=ax, label=label)

    for index in range(len(bin_edges) - 1):
        ax.fill_between(timeseries.index, 
                        y1=bin_edges[index], 
                        y2=bin_edges[index+1], 
                        alpha=0.1, 
                        color=colors[(index + 1) % len(colors)])
        
    return ax
    
def plot_colored_timeseries(timeseries, colormap, ax=None):
    """
    Map the colors of the colormap to the associated section of the time series.
    
    PARAMS
    ======
        timeseries: pandas.Series or pandas.DataFrame
            The timeseries to plot
        colormap: list of dict
            A list of dictionaries for each slice of data with a color and the 
            associated time series subset, as generated by the get_mtf_map()
            function of this package.
        ax: matplotlib.axes
            An ax to use to plot the time series. This function will create one 
            if this argument is set to None. Default: None

    RETURNS
    =======
        ax: matplotlib.axes
            Returns the ax used or created back to the caller.
    """
    # If no ax is passed as argument, we create a new plot:
    if ax is None:
        fig = plt.figure(figsize=(28, 4))
        ax = fig.add_subplot(111)
        
    for timeseries_slice in colormap:
        ax.plot(timeseries_slice['slice'], 
                linewidth=3, 
                alpha=0.5, 
                color='#000000')
        ax.plot(timeseries_slice['slice'], 
                linewidth=1.5, 
                color=timeseries_slice['color'])
        
    return ax