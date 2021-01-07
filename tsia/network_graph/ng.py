import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from matplotlib.colors import to_hex

# Useful constants definition
COLORMAP = 'jet'

def get_network_graph(mtf):
    """
    A Markov transition field can be interpreted as an adjacency matrix. This
    function computes the network graph associated to the MTF passed as an 
    argument.
    
    PARAMS
    ======
        mtf: numpy.ndarray
            Numpy array containing a Markov transition field.

    RETURNS
    =======
        graph: networkx.graph
            The network graph built from the MTF passed as input. We match
            each edge with a weight attribute equal to the Markov transition
            associated to this edge.
    """
    # Build the graph with networkx:
    graph = nx.from_numpy_matrix(mtf)
    
    # Loops through the edges to get associate each of them with the
    # corresponding Markov transition probability:
    weights = [mtf[u,v] for u,v in graph.edges()]
    for index, e in enumerate(graph.edges()):
        graph[e[0]][e[1]]['weight'] = weights[index]
        
    return graph
    
def get_all_network_graphs(list_mtfs):
    """
    A Markov transition field can be interpreted as an adjacency matrix. This
    function computes all the network graphs associated to the list of MTFs 
    passed as an argument.

    PARAMS
    ======
        list_mtfs: list of numpy.ndarray
            List of numpy arrays, each containing a Markov transition field.

    RETURNS
    =======
        list_graphs: list of networkx.graph
            The network graphs built from the list of MTF passed as input.
    """
    list_graphs = []
    
    for mtf in list_mtfs:
        list_graphs.append(get_network_graph(mtf))
        
    return list_graphs
    
def compute_network_graph_statistics(graph=None, mtf=None):
    """
    Extract the following statistics from the network graph passed as an
    argument (or from the MTF directly):
        - diameter
        - average degree
        - average weighted degree
        - density
        - average path length
        - average clustering coefficient
        - modularity
        - number of partitions found in the graph
        
    PARAMS
    ======
        graph: networkx.classes.graph.Graph (default to None)
            The network graph to extract statistics from. Either graph or
            mtf is mandatory.
        mtf: numpy.ndarray (default to None)
            Numpy array containing a Markov transition field. Either graph
            or mtf is mandatory.
    
    RETURNS
    =======
        statistics (dict)
            Dictionnary containing the aforementionned statistics from the 
            network graph.

    """
    if (graph is None) and (mtf is not None):
        graph = get_network_graph(mtf)
        
    # TODO:
    # Error if graph is None and mtf is None
        
    partitions = community.best_partition(graph)
    nb_partitions = len(set(partitions.values()))
    modularity = community.modularity(partitions, graph)
    diameter = nx.diameter(graph)
    node_size = list(nx.clustering(graph, weight='weight').values())
    avg_clustering_coeff = np.array(node_size).mean()
    density = nx.density(graph)
    avg_path_length = nx.average_shortest_path_length(graph, weight='weight', method='dijkstra')
    
    average_degree = nx.average_degree_connectivity(graph)
    average_degree = np.mean(list(average_degree.values()))
    avg_weighted_degree = nx.average_degree_connectivity(graph, weight='weight')
    avg_weighted_degree = np.mean(list(avg_weighted_degree.values()))
    
    statistics = {
        'Diameter': diameter,
        'Average degree': average_degree,
        'Average weighted degree': avg_weighted_degree,
        'Density': density,
        'Average path length': avg_path_length,
        'Average clustering coefficient': avg_clustering_coeff,
        'Modularity': modularity,
        'Partitions': nb_partitions
    }
    
    return statistics
    
def compute_all_network_graph_statistics(list_mtfs):
    """
    Extract the following statistics from a list of MTF passed as an argument
        - diameter
        - average degree
        - average weighted degree
        - density
        - average path length
        - average clustering coefficient
        - modularity
        - number of partitions found in the graph

    PARAMS
    ======
        list_mtfs: list of numpy.ndarray
            List of numpy arrays, each containing a Markov transition field.

    RETURNS
    =======
        statistics (dict)
            Dictionnary containing the aforementionned statistics for each
            MTF from the input list.
    """
    statistics = dict()
    
    for index, mtf in enumerate(list_mtfs):
        statistics.update({index: compute_network_graph_statistics(mtf=mtf)})
        
    return statistics
    
def get_modularity_encoding(graph, colormap=COLORMAP, reversed_cmap=False):
    """
    This function uses the modularity encoding to build a representation for the
    network graph. Modularity encoding means:
        - Each edge color is mapped to the partitions of the destination node
        - Edge node color is encoded with the partition of the source node
        - Each node size is encoded with the MTF transition probability, which
          is already stored in the 'weight' property of each edge
          
    PARAMS
    ======
        graph: networkx.classes.graph.Graph
            The network graph to encode
        colormap: string
            Name of the colormap to use
        reversed_cmap: boolean
            Indicate if the colormap must be reversed

    RETURNS
    =======
        encoding: dict
            A dictionnary containing the encoded values for each node size,
            edge color and node color.
    """
    if reversed_cmap == True:
        colormap = plt.cm.get_cmap(colormap).reversed()
    else:
        colormap = plt.cm.get_cmap(colormap)
    
    # Get the node partitions and number of partitions found with the Louvain
    # algorithm, as implemented in the `community` package:
    partitions = community.best_partition(graph)
    nb_partitions = len(set(partitions.values()))
    
    # Compute node colors and edges colors for the modularity encoding:
    edge_colors = [to_hex(colormap(partitions.get(v)/(nb_partitions - 1))) for u,v in graph.edges()]
    node_colors = [partitions.get(node) for node in graph.nodes()]
    node_size = list(nx.clustering(graph, weight='weight').values())
    node_size = list((node_size - np.min(node_size)) * 2000 + 10)
    
    # Store the encoding to return in a dictionnary:
    encoding = {
        'node_size': node_size,
        'edge_color': edge_colors,
        'node_color': node_colors
    }
    
    return encoding
    
def get_network_graph_map(timeseries, encoding, colormap=COLORMAP, reversed_cmap=False):
    """
    This function extracts the color associated to each partition (a.k.a. 
    community) of a network graph and map them back to the original time series.
    
    PARAMS
    ======
        timeseries: numpy.ndarray
            The time series data to map the MTF data to
        encoding: dict
            A dictionnary containing the encoded values for each node size,
            edge color and node color, as output by get_modularity_encoding()
        colormap: string. Default = 'jet'
            Name of the matplotlib colormap to use
        reversed_cmap: boolean. Default = False
            Indicate if the colormap must be reversed
            
    RETURNS
    =======
        network_graph_map: list of dict
            A list of dictionaries for each slice of data with a color and the 
            associated time series subset.
    """
    # Get encoding definitions:
    node_colors = encoding['node_color']
    image_size = len(node_colors)
    partition_color = node_colors / np.max(node_colors)
    
    # Define the color map:
    if reversed_cmap == True:
        colormap = plt.cm.get_cmap(colormap).reversed()
    else:
        colormap = plt.cm.get_cmap(colormap)

    # Plot each subset of the signal with the color associated to the network
    # graph partition it belongs to:
    network_graph_map = []
    sequences_width = timeseries.shape[0] / image_size
    for i in range(image_size):
        c = colormap(partition_color[i])
        start = int(i * sequences_width)
        end = int((i+1) * sequences_width - 1)
        data = timeseries.iloc[start:end, :]
        
        current_map = dict()
        current_map.update({
            'color': c,
            'slice': data
        })
        network_graph_map.append(current_map)
        
    return network_graph_map