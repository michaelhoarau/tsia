def network_graph(mtf):
    G = nx.from_numpy_matrix(mtf)
    
def compute_network_graph_statistics(graph):
    partitions = community.best_partition(G)
    nb_partitions = len(set(partitions.values()))
    modularity = community.modularity(partitions, G)
    diameter = nx.diameter(G)
    node_size = list(nx.clustering(G, weight='weight').values())
    avg_clustering_coeff = np.array(node_size).mean()
    avg_weighted_degree = nx.average_degree_connectivity(G, weight='weight')
    average_degree = nx.average_degree_connectivity(G)
    average_degree = np.mean(list(average_degree.values()))
    density = nx.density(G)
    avg_path_length = nx.average_shortest_path_length(G, weight='weight', method='dijkstra')
    
    statistics = {
        'Diameter': diameter,
        'Average degree': average_degree,
        'Average weighted degree': avg_weighted_degree,
        'Density': density,
        'Average path length': avg_path_length,
        'Average clustering coefficient': avg_clustering_coeff,
        'Modularity': modularity,
        '# Communities': nb_partitions
    }
    
    return statistics